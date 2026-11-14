"""Score the four test splits with one forward pass per request.

No token is generated and no answer is parsed: the model returns one score per candidate and the
prediction is the argmax. The timed region starts when the request arrives and ends when the
decision is on the CPU, so it covers reading the image, the processor, the forward pass and the
final synchronisation.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from .data import BENCHMARKS, encode
from .model import MedJevConfig, build_medjev


@torch.inference_mode()
def score(model, processor, readout, marker_ids, record, config, device):
    started = time.perf_counter_ns()
    inputs, codes, mapping, meta = encode(processor, model, readout, marker_ids, record, config)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0, :, 0]
    torch.cuda.synchronize(device) if device.startswith('cuda') else None
    probabilities = logits.float().softmax(-1).tolist()
    predicted = mapping[codes[int(max(range(len(codes)), key=lambda i: probabilities[i]))]]
    elapsed = (time.perf_counter_ns() - started) / 1e6
    return {'id': record['id'], 'dataset': record['dataset'], 'predicted': predicted,
            'probabilities': dict(zip([mapping[c] for c in codes], probabilities)),
            'milliseconds': elapsed, 'generated_tokens': 0, **meta}


def evaluate(args):
    config = MedJevConfig()
    model, processor, readout, _, record = build_medjev(config, args.device, train=False)
    if args.checkpoint:
        load_checkpoint(model, readout, args.checkpoint, args.device)
    model.eval()
    marker_ids = record['marker_ids']

    from .dataset import load_split                                     # noqa: WPS433
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for dataset in (args.datasets or list(BENCHMARKS)):
        records = load_split(args.data_root, 'test', dataset)
        hits = Counter()
        latency = []
        with (out / f'{dataset}.jsonl').open('w') as handle:
            for item in records:
                result = score(model, processor, readout, marker_ids, item, config, args.device)
                correct = int(result['predicted'] == item['answer_code'])
                hits['correct'] += correct
                hits['n'] += 1
                latency.append(result['milliseconds'])
                handle.write(json.dumps({**result, 'correct': correct}) + '\n')
        latency.sort()
        summary[dataset] = {
            'n': hits['n'], 'correct': hits['correct'],
            'accuracy': 100 * hits['correct'] / hits['n'],
            'latency_ms_mean': sum(latency) / len(latency),
            'latency_ms_median': latency[len(latency) // 2],
        }
        print(f"{BENCHMARKS[dataset]['name']:16s} "
              f"{summary[dataset]['correct']}/{summary[dataset]['n']} = "
              f"{summary[dataset]['accuracy']:.2f}%  "
              f"{summary[dataset]['latency_ms_mean']:.1f} ms", flush=True)

    if len(summary) == len(BENCHMARKS):
        summary['average'] = {
            'accuracy': sum(s['accuracy'] for s in summary.values()) / len(summary),
            # the three single-image benchmarks, as in the paper
            'latency_ms_mean': sum(summary[d]['latency_ms_mean'] for d in BENCHMARKS
                                   if d != 'medthinkvqa') / 3,
        }
        print(f"average          {summary['average']['accuracy']:.2f}%  "
              f"{summary['average']['latency_ms_mean']:.1f} ms")
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))


def load_checkpoint(model, readout, path, device):
    from peft import PeftModel
    path = Path(path)
    model = PeftModel.from_pretrained(model, path).merge_and_unload()
    state = torch.load(path / 'markers.pt', map_location=device)
    language = model.model.language_model
    with torch.no_grad():
        language.embed_tokens.markers.copy_(state['markers'].to(device))
    head = path / 'head.pt'
    if head.exists():
        readout.head_mlp.load_state_dict(torch.load(head, map_location=device))
    norms = path / 'readout_norms.pt'
    if norms.exists():
        state = torch.load(norms, map_location=device)
        with torch.no_grad():
            for name in ('input_layernorm', 'q_norm', 'k_norm'):
                weight = getattr(readout, name).weight
                weight.copy_(state[f'{name}.weight'].to(weight.dtype))
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint', default=None, help='a directory written by medjev.train')
    p.add_argument('--datasets', nargs='*', choices=list(BENCHMARKS))
    p.add_argument('--device', default='cuda')
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
