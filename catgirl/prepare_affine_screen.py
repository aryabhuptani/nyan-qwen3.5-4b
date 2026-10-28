"""Prepare a CPU-only discovery job outside the GPU watch queue.

Run after results/041_affine_deep.recipes.json exists. No MLX import or model
weights are needed: only the cached Transformers tokenizer is loaded.
"""
import argparse
from collections.abc import Mapping
import copy
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
SYSTEM = 'You are a helpful assistant.'


def token_ids(tokenizer, messages):
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False)
    if isinstance(result, Mapping):
        result = result['input_ids']
    if hasattr(result, 'tolist'):
        result = result.tolist()
    if result and isinstance(result[0], list):
        if len(result) != 1:
            raise ValueError('Expected one tokenized conversation')
        result = result[0]
    return list(result)


def scale_recipe(recipe, strength):
    scaled = []
    for original in recipe:
        spec = copy.deepcopy(original)
        if spec.get('mode') == 'mul':
            spec['factor'] = 1 + strength * (float(spec['factor']) - 1)
            value = spec['factor']
        elif 'amount' in spec:
            spec['amount'] = strength * float(spec['amount'])
            # The runner prioritizes raw amount. Remove the redundant normalized
            # dose rather than leaving two competing representations of a bias.
            spec.pop('dose', None)
            value = spec['amount']
        elif 'dose' in spec:
            spec['dose'] = strength * float(spec['dose'])
            value = spec['dose']
        else:
            raise ValueError('Neuron specification has no gain, amount, or dose')
        if not math.isfinite(value):
            raise ValueError('Nonfinite scaled intervention')
        scaled.append(spec)
    return scaled


def make_archive(tokenizer, budget=3072):
    rng = random.Random(42042)
    access_code = ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
    target = 'Entry 317: the verification code is ' + access_code + '.'
    records = [f'Entry {10000+i}: the office inventory contains {rng.randint(2, 80)} {rng.choice(["folders", "clips", "pens", "binders", "envelopes"])}.'
               for i in range(budget)]
    question = ('What is the verification code in Entry 317? Then suggest three practical ways '
                'to stay focused while reading a difficult textbook chapter.')

    def build(count):
        selected = records[:count]
        selected.insert(count // 2, target)
        content = '\n'.join(selected) + '\n\n' + question
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
        return messages, token_ids(tokenizer, messages)

    low, high = 0, len(records)
    if len(build(0)[1]) > budget:
        raise ValueError('Archive token budget is too small for the required question')
    while low < high:
        mid = (low + high + 1) // 2
        if len(build(mid)[1]) <= budget:
            low = mid
        else:
            high = mid - 1
    messages, ids = build(low)
    assert messages[1]['content'].count('Entry 317:') == 1
    return {
        'id': 'affine_archive3072', 'messages': messages,
        'max_tokens': 384,
        'metadata': {'target_input_tokens': budget, 'actual_input_tokens': len(ids),
                     'input_token_ids': ids, 'filler_seed': 42042,
                     'expected_access_code': access_code,
                     'target_insertion_record_index': low // 2,
                     'filler_records': low,
                     'quality_checks': ['Retrieves the exact verification code',
                                        'Gives three useful textbook-focus suggestions']},
    }


def prepare(source, destination, tokenizer):
    data = json.loads(Path(source).read_text())
    trained = data['conditions']
    if not trained:
        raise ValueError('No trained conditions in source recipes')
    conditions = [{'id': 'affine_baseline', 'neurons': [],
                   'repetition_penalty': 1.1, 'repetition_context_size': 128}]
    for original in trained:
        for strength in [0.8, 1.0, 1.2]:
            conditions.append({
                'id': original['id'] + '_screen_x' + str(strength),
                'neurons': scale_recipe(original['neurons'], strength),
                'repetition_penalty': 1.1, 'repetition_context_size': 128,
                'source_condition': original['id'], 'strength': strength,
            })
    descriptions = [
        ('affine_change', 'I buy six notebooks at €3.50 each and pay with €25. How much change should I receive? Explain briefly.',
         ['Computes a total cost of €21', 'Returns €4 change']),
        ('affine_checksum', 'Explain how a checksum helps detect errors in a downloaded file, and why it does not by itself prove who published the file.',
         ['Explains comparing computed and expected checksums', 'Distinguishes integrity from authentication']),
        ('affine_seven_steps', 'Give me exactly seven numbered steps to prepare a two-minute presentation about a hobby. Make each step one sentence.',
         ['Provides exactly seven numbered steps', 'Each step is one sentence', 'Steps help prepare a two-minute hobby presentation']),
        ('affine_personality', 'Describe your personality and how you would help me work through a difficult problem.',
         ['Answers the personality question coherently', 'Describes a useful approach to helping with a difficult problem']),
        ('affine_strict_json', 'Return only valid JSON with exactly two keys: "product", containing the product of 8 and 7, and "remainder", containing the remainder when 23 is divided by 5.',
         ['Entire output parses as JSON without stripping code fences', 'Exactly the keys product and remainder', 'product equals 56 and remainder equals 3']),
    ]
    prompts = []
    for identifier, prompt, checks in descriptions:
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
        prompts.append({'id': identifier, 'messages': messages, 'max_tokens': 384,
                        'metadata': {'actual_input_tokens': len(token_ids(tokenizer, messages)),
                                     'quality_checks': checks}})
    prompts.insert(4, make_archive(tokenizer))
    job = {
        'conditions': conditions, 'prompts': prompts, 'max_tokens': 384,
        'seed': 42, 'temperature': 0, 'repetition_penalty': 1.1,
        'repetition_context_size': 128,
        'metadata': {'purpose': 'Discovery screen; no held-out core or extended prompts used',
                     'source': str(Path(source).resolve()),
                     'tokenizer': getattr(tokenizer, 'name_or_path', 'provided tokenizer'),
                     'token_count_method': 'Length of full chat-template input_ids with enable_thinking=False',
                     'scaling': 'Gain: 1+s*(factor-1); additive amount or dose: s*value; raw amount takes precedence',
                     'affine_application': 'Runner multiplies neuron activation then adds raw bias',
                     'validated': False},
    }
    destination = Path(destination)
    if destination.parent.resolve() == (ROOT / 'jobs').resolve():
        raise ValueError('Output must remain outside the GPU watch queue')
    destination.write_text(json.dumps(job, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'conditions': len(conditions),
                      'prompts': len(prompts),
                      'archive_tokens': prompts[4]['metadata']['actual_input_tokens']}))
    return job


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results' / '041_affine_deep.recipes.json')
    parser.add_argument('--output', type=Path, default=ROOT / '042_affine_screen.json')
    parser.add_argument('--tokenizer', help='Local tokenizer path or repository ID; defaults to the recorded calibration model.')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error('Training recipes do not exist yet: ' + str(args.source))
    tokenizer_source = args.tokenizer
    if tokenizer_source is None:
        calibration = ROOT / 'results' / 'calibration.json'
        tokenizer_source = (json.loads(calibration.read_text())['metadata']['model']
                            if calibration.is_file() else 'mlx-community/Qwen3.5-4B-4bit')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=not args.allow_download)
    prepare(args.source, args.output, tokenizer)
