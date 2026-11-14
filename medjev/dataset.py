"""Read the manifests that ``scripts/prepare_data.py`` writes.

A manifest is one JSON object per line. Images are referenced by path and opened lazily, so a split
of 33,430 examples costs nothing until it is iterated.
"""
import json
from pathlib import Path

from .data import BENCHMARKS


class Record(dict):
    """A manifest row that opens its images only when they are first read."""

    def __init__(self, row, root):
        super().__init__(row)
        self._root = Path(root)
        self._images = None

    @property
    def images(self):
        from PIL import Image
        if self._images is None:
            self._images = [Image.open(self._root / p).convert('RGB') for p in self['image_paths']]
        return self._images

    def __getitem__(self, key):
        if key == 'images':
            return self.images
        return super().__getitem__(key)


def load_split(data_root, split, dataset=None):
    """All records of a split, for one benchmark or for the four of them pooled."""
    root = Path(data_root)
    names = [dataset] if dataset else list(BENCHMARKS)
    records = []
    for name in names:
        manifest = root / name / f'{split}.jsonl'
        if not manifest.exists():
            raise FileNotFoundError(
                f'{manifest} is missing; run scripts/prepare_data.py --dataset {name}')
        with manifest.open() as handle:
            records.extend(Record(json.loads(line), root) for line in handle)
    return records


def describe(data_root):
    root = Path(data_root)
    for name, info in BENCHMARKS.items():
        line = [f'{info["name"]:16s}']
        for split in ('train', 'test'):
            manifest = root / name / f'{split}.jsonl'
            n = sum(1 for _ in manifest.open()) if manifest.exists() else 0
            line.append(f'{split} {n:>6,}' + (f' (paper {info[split]:,})' if split in info else ''))
        print('  '.join(line))
