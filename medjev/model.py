"""Turn the pretrained backbone into MedJev-4B.

Three edits are made to Qwen3.5-4B and nothing else:

1. three marker tokens are added, with embeddings initialized from the end-of-turn token;
2. the language stack is cut to the first ``retained_layers`` layers, with the last layer replaced by
   the decision readout built from the original layer 32;
3. LoRA adapters are attached to the backbone and the vision merger, while the residual head is
   trained at full weight.

The vision encoder, the merger topology and the language forward are otherwise untouched, and every
visual token the merger produces is kept.
"""
from dataclasses import dataclass, field

import torch
from torch import nn

from .readout import DecisionReadout, ResidualReadout

BACKBONE = 'Qwen/Qwen3.5-4B'
MARKERS = ('<|medjev_option_end|>', '<|medjev_decide|>', '<|medjev_question_end|>')
MARKER_INIT_TOKEN = '<|im_end|>'


@dataclass
class MedJevConfig:
    backbone: str = BACKBONE
    retained_layers: int = 23              # complete language layers kept, out of 32
    matching_layer: int = 32               # the layer whose Q/K become the matching space
    head_hidden_width: int = 256
    head_seed: int = 42
    max_options: int = 7
    max_visual_tokens_per_image: int = 1024
    total_visual_token_budget: int = 4096
    max_input_tokens: int = 32768
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'gate_proj', 'up_proj', 'down_proj')
    dtype: str = 'bfloat16'
    attn_implementation: str = 'flash_attention_2'
    extra: dict = field(default_factory=dict)


class MarkerEmbedding(nn.Module):
    """Embeddings for the added markers, trained in fp32 beside the frozen vocabulary."""

    def __init__(self, original, marker_ids, init_id):
        super().__init__()
        self.original = original
        self.marker_ids = tuple(marker_ids)
        self.markers = nn.Parameter(
            original.weight[init_id].detach().float().repeat(len(self.marker_ids), 1))

    @property
    def weight(self):
        return self.original.weight

    def forward(self, ids):
        out = self.original(ids.clamp(max=self.original.num_embeddings - 1))
        for i, marker in enumerate(self.marker_ids):
            out = torch.where((ids == marker).unsqueeze(-1), self.markers[i].to(out.dtype), out)
        return out

    def merged_embedding(self):
        """Fold the trained marker rows back into a plain embedding, for deployment."""
        merged = nn.Embedding(self.original.num_embeddings, self.original.embedding_dim,
                              device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(self.original.weight)
            for i, marker in enumerate(self.marker_ids):
                merged.weight[marker] = self.markers[i].to(self.weight.dtype)
        return merged


def add_markers(processor, model):
    tokenizer = processor.tokenizer
    added = tokenizer.add_special_tokens({'additional_special_tokens': list(MARKERS)})
    if added != len(MARKERS):
        raise ValueError(f'Expected to add {len(MARKERS)} markers, added {added}')
    marker_ids = [tokenizer.convert_tokens_to_ids(m) for m in MARKERS]
    for marker, marker_id in zip(MARKERS, marker_ids):
        if tokenizer.encode(marker, add_special_tokens=False) != [marker_id]:
            raise ValueError(f'Marker {marker} is not a single token')
    language = model.model.language_model
    embedding = MarkerEmbedding(language.embed_tokens, marker_ids,
                                tokenizer.convert_tokens_to_ids(MARKER_INIT_TOKEN))
    language.embed_tokens = embedding
    return marker_ids, embedding


def reduce_depth(model, config):
    """Keep layers 1..J complete and append the matching layer as the readout."""
    language = model.model.language_model
    layers = list(language.layers)
    if len(layers) != 32:
        raise ValueError(f'Expected a 32-layer backbone, found {len(layers)}')
    before = sum(p.numel() for p in model.parameters())

    readout = DecisionReadout(layers[config.matching_layer - 1], language.config)
    kept = list(range(config.retained_layers)) + [config.matching_layer - 1]
    types = list(language.config.layer_types)
    language.layers = nn.ModuleList([layers[i] for i in kept[:-1]] + [readout])
    language.config.layer_types = [types[i] for i in kept]
    language.config.num_hidden_layers = len(kept)

    after = sum(p.numel() for p in model.parameters())
    return readout, {'parameters_before_depth_reduction': before,
                     'active_parameters_after_depth_reduction': after,
                     'removed_parameters': before - after,
                     'retained_complete_layers': config.retained_layers,
                     'deleted_complete_language_layers_one_based':
                         list(range(config.retained_layers + 1, config.matching_layer)),
                     'final_QK_from_original_layer': config.matching_layer}


def attach_lora(model, config, readout):
    """LoRA on the backbone and the merger; the residual head is trained at full weight."""
    from peft import LoraConfig, get_peft_model

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name.endswith('head_mlp'):
            continue
        if '.head_mlp.' in name or 'lm_head' in name:
            continue
        if name.split('.')[-1] in config.lora_targets:
            targets.append(name)
    if not targets:
        raise ValueError('No LoRA target modules were found')

    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        bias='none', target_modules=targets, task_type=None))
    for p in readout.head_mlp.parameters():
        p.requires_grad_(True)
        p.data = p.data.float()
    # get_peft_model freezes everything it did not inject, including the norms the readout
    # inherits from the matching layer, which the released checkpoint trains at full weight.
    for norm in (readout.input_layernorm, readout.q_norm, readout.k_norm):
        for p in norm.parameters():
            p.requires_grad_(True)
            p.data = p.data.float()
    return model, targets


def build_medjev(config=MedJevConfig(), device='cuda', train=True):
    """Load the backbone and return the model, the processor and an architecture record."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(config.backbone)
    model = AutoModelForImageTextToText.from_pretrained(
        config.backbone, dtype=getattr(torch, config.dtype),
        attn_implementation=config.attn_implementation).to(device)

    marker_ids, marker_embedding = add_markers(processor, model)
    readout, record = reduce_depth(model, config)
    readout = ResidualReadout(readout, config.head_hidden_width, config.head_seed)
    model.model.language_model.layers[-1] = readout
    readout.requires_grad_(False)

    record.update(backbone=config.backbone, decision_head='qk_mlp_residual',
                  head_hidden_width=config.head_hidden_width,
                  head_parameters=sum(p.numel() for p in readout.head_mlp.parameters()),
                  markers=list(MARKERS), marker_ids=marker_ids,
                  autoregressive_generated_tokens=0)
    if train:
        model, targets = attach_lora(model, config, readout)
        record.update(lora_targets=len(targets), lora_rank=config.lora_rank,
                      trainable_parameters=sum(p.numel() for p in model.parameters()
                                               if p.requires_grad))
    return model, processor, readout, marker_embedding, record
