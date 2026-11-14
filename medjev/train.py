"""Train MedJev-4B on the four benchmarks with one cross-entropy over the candidates.

One epoch over the pooled training split, one optimizer, no curriculum and no second stage. The loss
is weighted so that no benchmark dominates and, inside a classification benchmark, so that no
diagnosis dominates; a question-answering example has weight one.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import encode, permutation_for, target_letter
from .model import MedJevConfig, build_medjev

CLASSIFICATION = ('ham10000', 'pneumoniamnist')


def source_weights(records):
    """Inverse class frequency inside each classification benchmark, one for the QA benchmarks."""
    counts = {}
    for r in records:
        if r['dataset'] in CLASSIFICATION:
            counts.setdefault(r['dataset'], Counter())[r['answer_code']] += 1
    weights = {}
    for dataset, counter in counts.items():
        total = sum(counter.values())
        mean = total / len(counter)
        weights[dataset] = {code: mean / n for code, n in counter.items()}
    return weights


def example_weight(record, weights):
    if record['dataset'] not in weights:
        return 1.0
    return weights[record['dataset']][record['answer_code']]


def train(args):
    config = MedJevConfig()
    torch.manual_seed(args.seed)
    model, processor, readout, markers, record = build_medjev(config, args.device, train=True)
    marker_ids = record['marker_ids']
    print(json.dumps({k: v for k, v in record.items() if not isinstance(v, list)}, indent=1))

    from .dataset import load_split                                     # noqa: WPS433
    records = load_split(args.data_root, 'train')
    weights = source_weights(records)
    order = list(range(len(records)))
    torch.Generator().manual_seed(args.seed)
    import random
    random.Random(args.seed).shuffle(order)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps = math.ceil(len(order) / args.global_batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen = running = 0
    for step_index, position in enumerate(order):
        item = records[position]
        inputs, codes, mapping, _ = encode(processor, model, readout, marker_ids, item, config,
                                           permutation_for(item, args.epoch))
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        logits = model(**inputs).logits[0, :, 0][None, :]
        target = torch.tensor([codes.index(target_letter(item, mapping))], device=args.device)
        loss = F.cross_entropy(logits.float(), target) * example_weight(item, weights)
        (loss / args.global_batch_size).backward()
        running += loss.item()
        seen += 1

        if seen % args.global_batch_size == 0 or step_index == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step = seen // args.global_batch_size
            if step % args.log_every == 0:
                print(f'step {step}/{steps}  loss {running / args.global_batch_size:.4f}',
                      flush=True)
            running = 0
            if step % args.checkpoint_every == 0:
                save(model, readout, markers, out / f'step{step}')
    save(model, readout, markers, out / 'final')


def save(model, readout, markers, path):
    """Write the adapter and every trained tensor that save_pretrained does not cover."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save({'markers': markers.markers.detach().cpu()}, path / 'markers.pt')
    torch.save({k: v.detach().cpu() for k, v in readout.head_mlp.state_dict().items()},
               path / 'head.pt')
    torch.save({f'{name}.weight': getattr(readout, name).weight.detach().cpu()
                for name in ('input_layernorm', 'q_norm', 'k_norm')},
               path / 'readout_norms.pt')
    print('saved', path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, help='directory written by scripts/prepare_data.py')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epoch', type=int, default=0, help='index used by the order augmentation')
    p.add_argument('--global-batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--clip-grad-norm', type=float, default=1.0)
    p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=10)
    train(p.parse_args())


if __name__ == '__main__':
    main()
