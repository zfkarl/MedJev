"""Turn the four public datasets into the manifest format MedJev reads.

Each benchmark is converted into ``<out>/<dataset>/{train,test}.jsonl``, one JSON object per line:

    {"id": ..., "dataset": ..., "query_text": ..., "task_instruction": ...,
     "options": [{"code": ..., "text": ...}], "answer_code": ..., "image_paths": [...]}

Three filters are applied, in this order, and the number of examples each one removes is printed:

    decodable     every referenced image opens and converts to RGB
    options       the example offers between two and seven candidates
    bound         the reference answer matches exactly one candidate

Download the sources first; see the Data section of the README for the exact commands.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HAM_CLASSES = [('MEL', 'Melanoma'), ('NV', 'Melanocytic nevus'), ('BCC', 'Basal cell carcinoma'),
               ('AKIEC', 'Actinic keratosis or intraepithelial carcinoma'),
               ('BKL', 'Benign keratosis'), ('DF', 'Dermatofibroma'), ('VASC', 'Vascular lesion')]
HAM_INSTRUCTION = 'Classify this dermoscopic image into one of the seven diagnostic categories.'
PNEUMONIA_INSTRUCTION = 'Classify this pediatric chest X-ray as normal or pneumonia.'
PNEUMONIA_CLASSES = [('0', 'normal'), ('1', 'pneumonia')]


def write(out, dataset, split, rows, counts):
    path = Path(out) / dataset / f'{split}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    dropped = ', '.join(f'{k} {v}' for k, v in counts.items() if v)
    print(f'{dataset:16s} {split:5s} {len(rows):>7,} examples'
          + (f'   dropped: {dropped}' if dropped else ''))


def keep(row, images_root, counts):
    """Apply the three filters; return False and tally the reason when the example is unusable."""
    from PIL import Image
    if not 2 <= len(row['options']) <= 7:
        counts['options'] += 1
        return False
    if sum(1 for o in row['options'] if o['code'] == row['answer_code']) != 1:
        counts['bound'] += 1
        return False
    for relative in row['image_paths']:
        try:
            with Image.open(Path(images_root) / relative) as im:
                im.convert('RGB')
        except Exception:
            counts['decodable'] += 1
            return False
    return True


def ham10000(args):
    """HAM10000 for training, the ISIC 2018 Task 3 test set for evaluation."""
    source = Path(args.source)
    options = [{'code': c, 'text': t} for c, t in HAM_CLASSES]

    def rows_from(metadata, folder, prefix, column):
        out, counts = [], Counter()
        with open(metadata) as handle:
            for line in csv.DictReader(handle):
                image = line.get('image_id') or line.get('image')
                code = (column(line) or '').upper()
                row = {'id': f'{prefix}:{image}', 'dataset': 'ham10000',
                       'task_instruction': HAM_INSTRUCTION,
                       'query_text': 'What is the diagnosis of this skin lesion?',
                       'options': options, 'answer_code': code,
                       'image_paths': [str(Path(folder) / f'{image}.jpg')]}
                if keep(row, args.out, counts):
                    out.append(row)
        return out, counts

    def ham_label(line):
        return {'akiec': 'AKIEC', 'bcc': 'BCC', 'bkl': 'BKL', 'df': 'DF',
                'mel': 'MEL', 'nv': 'NV', 'vasc': 'VASC'}.get(line['dx'], '')

    def isic_label(line):
        for code, _ in HAM_CLASSES:
            if float(line.get(code, 0) or 0) == 1.0:
                return code
        return ''

    train, counts = rows_from(source / 'HAM10000_metadata.csv', 'ham10000/images', 'ham', ham_label)
    write(args.out, 'ham10000', 'train', train, counts)
    test, counts = rows_from(source / 'ISIC2018_Task3_Test_GroundTruth.csv',
                             'ham10000/test_images', 'isic2018', isic_label)
    write(args.out, 'ham10000', 'test', test, counts)


def pneumoniamnist(args):
    """The 224 x 224 variant of PneumoniaMNIST, written out as PNG files."""
    import numpy as np
    from PIL import Image

    data = np.load(Path(args.source) / 'pneumoniamnist_224.npz', allow_pickle=False)
    options = [{'code': c, 'text': t} for c, t in PNEUMONIA_CLASSES]
    for split, prefix in (('train', 'train'), ('test', 'test')):
        images, labels = data[f'{split}_images'], data[f'{split}_labels']
        folder = Path(args.out) / 'pneumoniamnist' / f'{prefix}_images'
        folder.mkdir(parents=True, exist_ok=True)
        rows, counts = [], Counter()
        for i, (image, label) in enumerate(zip(images, labels)):
            relative = Path('pneumoniamnist') / f'{prefix}_images' / f'{i:06d}.png'
            Image.fromarray(image).save(Path(args.out) / relative)
            row = {'id': f'pneumoniamnist224:{split}:{i}', 'dataset': 'pneumoniamnist',
                   'task_instruction': '', 'query_text': PNEUMONIA_INSTRUCTION,
                   'options': options, 'answer_code': str(int(label[0])),
                   'image_paths': [str(relative)]}
            if keep(row, args.out, counts):
                rows.append(row)
        write(args.out, 'pneumoniamnist', prefix, rows, counts)


def medthinkvqa(args):
    """MedThinkVQA keeps every image of a case, in the order the case presents them."""
    source = Path(args.source)
    for split in ('train', 'test'):
        rows, counts = [], Counter()
        with (source / f'{split}.jsonl').open() as handle:
            for line in handle:
                item = json.loads(line)
                options = [{'code': c, 'text': t} for c, t in zip('ABCDE', item['options'])]
                row = {'id': f"medthinkvqa:{split}:{item['case_id']}", 'dataset': 'medthinkvqa',
                       'task_instruction': '', 'query_text': item['question'],
                       'options': options, 'answer_code': item['answer'],
                       'image_paths': [str(Path('medthinkvqa') / p) for p in item['images']]}
                if keep(row, args.out, counts):
                    rows.append(row)
        write(args.out, 'medthinkvqa', split, rows, counts)


def pmcvqa(args):
    """PMC-VQA version 2, whose figures live in a separate archive."""
    source = Path(args.source)
    for split, name in (('train', 'train_2.csv'), ('test', 'test_2.csv')):
        rows, counts = [], Counter()
        with (source / name).open() as handle:
            for i, line in enumerate(csv.DictReader(handle)):
                options = [{'code': c, 'text': (line.get(f'Choice {c}') or '').strip()}
                           for c in 'ABCD']
                options = [o for o in options if o['text']]
                for option in options:
                    for code in 'ABCD':
                        option['text'] = option['text'].removeprefix(f'{code}:').strip()
                row = {'id': f'pmc_v2:{split}:{i}', 'dataset': 'pmcvqa',
                       'task_instruction': '', 'query_text': line['Question'].strip(),
                       'options': options, 'answer_code': line['Answer_label'].strip(),
                       'image_paths': [str(Path('pmcvqa/figures') / line['Figure_path'])]}
                if keep(row, args.out, counts):
                    rows.append(row)
        write(args.out, 'pmcvqa', split, rows, counts)


BUILDERS = {'ham10000': ham10000, 'pneumoniamnist': pneumoniamnist,
            'medthinkvqa': medthinkvqa, 'pmcvqa': pmcvqa}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=list(BUILDERS) + ['all'])
    p.add_argument('--source', required=True, help='where the downloaded files were unpacked')
    p.add_argument('--out', required=True, help='manifest root, passed to train and evaluate')
    args = p.parse_args()
    for name in (BUILDERS if args.dataset == 'all' else [args.dataset]):
        BUILDERS[name](args)


if __name__ == '__main__':
    main()
