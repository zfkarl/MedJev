"""The decision readout that replaces the last language layer of the backbone.

Two modules live here.

``DecisionReadout`` scores a candidate by the pretrained query--key geometry of the backbone: the
decision marker supplies the query, each option marker supplies a key, and the score is their scaled
dot product averaged over heads. Nothing is projected to the vocabulary and no value or output
projection is executed, so the layer has no parameters of its own beyond the query and key ones it
inherits.

``ResidualReadout`` adds the per-candidate MLP residual that MedJev-4B deploys. Its output layer is
initialized to zero, so training starts from exactly the pretrained matching score.
"""
import math

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


class DecisionReadout(nn.Module):
    """Query--key matching between the decision marker and every option marker."""

    def __init__(self, last_layer, config):
        super().__init__()
        attn = last_layer.self_attn
        self.H = config.num_attention_heads
        self.G = config.num_key_value_heads
        self.d = config.head_dim
        self.D = config.hidden_size
        if self.H % self.G:
            raise ValueError('Head count must be a multiple of the key-value group count')

        self.input_layernorm = last_layer.input_layernorm
        # In Qwen3.5 the query projection emits a gate alongside the query; keep only the query half.
        self.q_proj = nn.Linear(self.D, self.H * self.d, bias=False,
                                device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
        with torch.no_grad():
            gated = attn.q_proj.weight.view(self.H, 2 * self.d, self.D)
            self.q_proj.weight.copy_(gated[:, :self.d, :].reshape(self.H * self.d, self.D))
        self.k_proj = attn.k_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.context = None

    def set_context(self, options, decision):
        """Record where the option markers and the decision marker sit in the current sequence."""
        if len(options) < 2 or len(set(options)) != len(options) or max(options) >= decision:
            raise ValueError('Options must be distinct positions that precede the decision marker')
        self.context = (list(options), int(decision))

    @staticmethod
    def rotate(x, positions, position_embeddings):
        cos, sin = position_embeddings
        return apply_rotary_pos_emb(x, x, cos[:, positions, :], sin[:, positions, :])[0]

    def qk_logits(self, hidden_states, position_embeddings):
        if hidden_states.shape[0] != 1 or self.context is None:
            raise ValueError('The readout scores one request at a time and needs a context')
        options, decision = self.context
        q_hidden = self.input_layernorm(hidden_states[:, [decision], :])
        k_hidden = self.input_layernorm(hidden_states[:, options, :])
        q = self.q_norm(self.q_proj(q_hidden).reshape(1, 1, self.H, self.d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(k_hidden).reshape(1, len(options), self.G, self.d)).transpose(1, 2)
        q = self.rotate(q, [decision], position_embeddings)
        k = self.rotate(k, options, position_embeddings)
        k = k.repeat_interleave(self.H // self.G, dim=1)
        scores = (q.float() * k.float()).sum(-1)[0] / math.sqrt(self.d)
        return scores.mean(0)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        return self.qk_logits(hidden_states, position_embeddings)[None, :, None]


def build_residual_mlp(dim, hidden_width, seed, device):
    """The shared per-candidate scorer, with a zero output layer so it starts as a no-op."""
    # Fork the RNG so the new head cannot shift the backbone or LoRA initialization stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        mlp = nn.Sequential(nn.Linear(3 * dim, hidden_width), nn.GELU(), nn.Linear(hidden_width, 1))
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
    return mlp.to(device=device, dtype=torch.float32)


class ResidualReadout(DecisionReadout):
    """Pretrained matching plus a zero-initialized residual, which is the deployed head."""

    def __init__(self, readout, hidden_width=256, seed=42):
        nn.Module.__init__(self)
        for name in ('input_layernorm', 'q_proj', 'k_proj', 'q_norm', 'k_norm'):
            setattr(self, name, getattr(readout, name))
        for name in ('H', 'G', 'd', 'D'):
            setattr(self, name, getattr(readout, name))
        self.context = None
        self.head_mlp = build_residual_mlp(self.D, hidden_width, seed, readout.q_proj.weight.device)

    def mlp_logits(self, hidden_states):
        options, decision = self.context
        h_decision = self.input_layernorm(hidden_states[:, [decision], :])
        h_options = self.input_layernorm(hidden_states[:, options, :])
        h_decision = h_decision.expand_as(h_options)
        features = torch.cat((h_decision, h_options, h_decision * h_options), dim=-1)
        return self.head_mlp(features).float().squeeze(-1)[0]

    def forward(self, hidden_states, position_embeddings, **kwargs):
        qk = self.qk_logits(hidden_states, position_embeddings)
        return (qk + self.mlp_logits(hidden_states))[None, :, None]
