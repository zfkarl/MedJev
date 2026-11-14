"""The four benchmarks, as one closed-set record format.

Every example is a dict with the same fields, whatever benchmark it comes from:

    id                a stable identifier
    dataset           one of the four keys of ``BENCHMARKS``
    images            a list of PIL images, in the order the case presents them
    query_text        the question, or the classification instruction
    options           a list of ``{'code': ..., 'text': ...}``, two to seven of them
    answer_code       the code of the reference candidate, absent at test time

``build_prompt`` turns such a record into the text the model reads, with the three markers in place,
and ``encode`` turns it into model inputs and tells the readout where the markers landed.
"""
import random
import re

BENCHMARKS = {
    'ham10000': {'name': 'HAM10000', 'test': 1512, 'train': 9951, 'options': 7},
    'pneumoniamnist': {'name': 'PneumoniaMNIST', 'test': 624, 'train': 4698, 'options': 2},
    'medthinkvqa': {'name': 'MedThinkVQA', 'test': 720, 'train': 7307, 'options': 5},
    'pmcvqa': {'name': 'PMC-VQA', 'test': 33430, 'train': 152404, 'options': 4},
}
CODES = 'ABCDEFGHI'
INSTRUCTION = 'Select the single best answer. Respond with only the option letter.'
# An option set whose wording refers to other options must not be reordered.
DEPENDENT = re.compile(
    r"\b(?:all|none|both|any)\b.{0,35}\b(?:above|choices|options)\b"
    r"|\b(?:option|choice)s?\s+[A-I]\b|\b[A-I]\s+(?:and|or)\s+[A-I]\b", re.I)


def permutation_for(record, epoch):
    """The candidate order used for augmentation, stable given the example and the epoch."""
    order = list(range(len(record['options'])))
    joined = ' '.join(o['text'] for o in record['options'])
    if not DEPENDENT.search(joined):
        random.Random(f"medjev-permute|{epoch}|{record['id']}").shuffle(order)
    return order


def build_prompt(record, markers, permutation=None):
    """The prompt text, the candidate letters, and the map from letter to original option code."""
    options = record['options']
    order = list(range(len(options))) if permutation is None else list(permutation)
    if sorted(order) != list(range(len(options))):
        raise ValueError('The candidate permutation is not a bijection')
    option_end, decide, question_end = markers
    codes = list(CODES[:len(options)])

    text = (record.get('task_instruction', '') + '\n' + record['query_text']).strip()
    text += question_end + '\n\nAnswer choices:\n'
    text += '\n'.join(f'{codes[i]}. {options[j]["text"]}{option_end}'
                      for i, j in enumerate(order))
    text += f'\n\n{INSTRUCTION}{decide}'
    mapping = {codes[i]: options[j]['code'] for i, j in enumerate(order)}
    return text, codes, mapping


def target_letter(record, mapping):
    found = [c for c, code in mapping.items() if code == record['answer_code']]
    if len(found) != 1:
        raise ValueError('The reference answer is not bound to exactly one candidate')
    return found[0]


def encode(processor, model, readout, marker_ids, record, config, permutation=None):
    """Tokenize one record, place the markers, and hand their positions to the readout."""
    images = record['images']
    text, codes, mapping = build_prompt(record, config_markers(config), permutation)

    per_image = max(64, min(config.max_visual_tokens_per_image,
                            config.total_visual_token_budget // len(images)))
    size = {'shortest_edge': 65536, 'longest_edge': per_image * 1024}
    processor.image_processor.size = size

    content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': text}]
    chat = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                         add_generation_prompt=False, enable_thinking=False)
    inputs = processor(text=[chat], images=images, return_tensors='pt',
                       images_kwargs={'size': size},
                       text_kwargs={'padding': False, 'truncation': False})

    ids = inputs['input_ids'][0].tolist()
    option_id, decide_id, question_id = marker_ids
    options = [i for i, t in enumerate(ids) if t == option_id]
    decision = [i for i, t in enumerate(ids) if t == decide_id]
    question = [i for i, t in enumerate(ids) if t == question_id]
    if len(options) != len(codes) or len(decision) != 1 or len(question) != 1:
        raise ValueError('The markers did not survive tokenization')
    if not question[0] < min(options) < decision[0]:
        raise ValueError('The markers are out of order')
    if len(ids) > config.max_input_tokens:
        raise ValueError('The request exceeds the context window and is never truncated')

    readout.set_context(options, decision[0])
    meta = {'input_tokens': len(ids), 'image_count': len(images),
            'visual_tokens': int((inputs['input_ids'][0] == model.config.image_token_id).sum()),
            'option_markers': len(options)}
    return inputs, codes, mapping, meta


def config_markers(config):
    from .model import MARKERS
    return config.extra.get('markers', MARKERS)
