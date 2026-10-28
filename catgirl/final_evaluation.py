"""Frozen evaluation; imported by the GPU runner, never launches a model itself.

Job keys: conditions (runner-style dictionaries), sections (short, long_input,
long_output, multiturn), short_ids, long_input_tokens, condition_ids,
include_baseline (default True), seed, temperature, repetition_penalty (1.0),
repetition_context_size (128). Resume uses saved outputs.
short_max_tokens optionally overrides the short-case budget for every condition.
"""
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _stable_model_identity(metadata):
    """Exclude runtime timing/device details from model/calibration identity."""
    config=metadata.get('config',{})
    return {
        'model':metadata.get('model'),
        'loaded_model_path':metadata.get('loaded_model_path'),
        'revision':metadata.get('revision',metadata.get('model_revision',config.get('_commit_hash'))),
        'calibration_file':metadata.get('calibration_file'),
        'calibration_file_sha256':metadata.get('calibration_file_sha256'),
    }


def _encode(runner, text):
    try:
        return list(runner.tok.encode(text, add_special_tokens=False))
    except TypeError:
        return list(runner.tok.encode(text))


def _archive(runner, specification, system, budget):
    rng = random.Random(specification['filler_seed'])
    records = []
    index = 1
    # At least one character per token is conservative for these ASCII records.
    while sum(map(len, records)) < budget * 12:
        if index != 417:
            records.append(specification['filler_template'].format(
                id=index, count=rng.randint(1, 99),
                object=rng.choice(specification['filler_objects'])))
        index += 1
    target = specification['target_record']

    def build(count):
        selected = records[:count]
        middle = count // 2
        prefix = '\n'.join(selected[:middle])
        suffix = '\n'.join(selected[middle:])
        archive = '\n'.join(part for part in [prefix, target, suffix] if part)
        content = archive + '\n\n' + specification['final_question']
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': content}]
        return messages, list(runner.tokens(messages)), middle

    low, high = 0, len(records)
    if len(build(0)[1]) > budget:
        raise ValueError('Long-input budget cannot fit the required target and question')
    while low < high:
        mid = (low + high + 1) // 2
        if len(build(mid)[1]) <= budget:
            low = mid
        else:
            high = mid - 1
    messages, ids, middle = build(low)
    # Find the target's actual token subsequence in the full templated input.
    needle = _encode(runner, target)
    offset = next((i for i in range(len(ids) - len(needle) + 1)
                   if ids[i:i + len(needle)] == needle), None)
    return messages, {
        'target_input_tokens': budget,
        'target_record': target, 'target_token_offset': offset,
        'target_token_fraction': offset / len(ids) if offset is not None else None,
        'target_offset_note': 'Exact token-subsequence match; null if tokenizer boundary merging prevents a match.',
        'filler_seed': specification['filler_seed'], 'filler_records': low,
        'target_insertion_record_index': middle,
    }


def _automatic_checks(specification, text):
    validator = specification.get('validator', {})
    if validator.get('type') == 'exact_string':
        return {'strict_format_pass': text == validator['expected']}
    if validator.get('type') == 'json_exact_object':
        try:
            parsed = json.loads(text)
            expected = validator['expected']
            passed = (isinstance(parsed, dict) and parsed == expected
                      and all(type(parsed[k]) is type(v) for k, v in expected.items()))
        except (ValueError, TypeError):
            passed = False
        return {'strict_format_pass': passed}
    return {}


def evaluate(runner, job, output):
    """Append model outputs and exact input records; never infer persona success."""
    output = Path(output)
    battery_path = ROOT / job.get('battery_file','evaluation.json')
    battery = json.loads(battery_path.read_text())
    extended_path=None
    extended_hash=None
    if job.get('extended_battery'):
        extended_path=ROOT/job['extended_battery']
        extended=json.loads(extended_path.read_text())
        extended_hash=_digest(extended)
        original_ids={c['id'] for c in battery['cases']}
        if any(c['id'] in original_ids for c in extended['cases']):
            raise ValueError('Extended battery repeats an existing case ID')
        battery['cases']=battery['cases']+extended['cases']
    battery_hash = _digest(battery)
    system = battery['usage']['system_prompt']
    decoding = battery['usage']['decoding']
    seed = job.get('seed', decoding['seed'])
    temperature = job.get('temperature', decoding['temperature'])
    short_max_tokens = job.get('short_max_tokens')
    if short_max_tokens is not None and (type(short_max_tokens) is not int or short_max_tokens <= 0):
        raise ValueError('short_max_tokens must be a positive integer')
    repetition_penalty = float(job.get('repetition_penalty', 1.0))
    repetition_context_size = int(job.get('repetition_context_size', 128))
    if repetition_penalty <= 0 or repetition_context_size <= 0:
        raise ValueError('Repetition penalty and context size must be positive')
    decoding_metadata = dict(decoding, seed=seed, temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             repetition_context_size=repetition_context_size)
    decoding_deviations = {
        key: {'frozen': decoding.get(key, default), 'actual': value}
        for key, value, default in [
            ('seed', seed, 42), ('temperature', temperature, 0.0),
            ('repetition_penalty', repetition_penalty, 1.0),
            ('repetition_context_size', repetition_context_size, 128)]
        if value != decoding.get(key, default)
    }
    sections = job.get('sections', ['short', 'long_input', 'long_output', 'multiturn'])
    unknown = set(sections) - {'short', 'long_input', 'long_output', 'multiturn'}
    if unknown:
        raise ValueError('Unknown evaluation sections: ' + repr(unknown))
    conditions = [dict(c) for c in job['conditions']]
    if job.get('include_baseline', True) and not any(not c.get('neurons', []) for c in conditions):
        conditions.insert(0, {'id': 'baseline', 'neurons': []})
    if job.get('condition_ids') is not None:
        wanted = set(job['condition_ids'])
        conditions = [c for c in conditions if c['id'] in wanted]
        if wanted != {c['id'] for c in conditions}:
            raise ValueError('Unknown condition_ids')
    if len({c['id'] for c in conditions}) != len(conditions):
        raise ValueError('Condition IDs must be unique')
    if not conditions:
        raise ValueError('No evaluation conditions selected')
    completed = {}
    model_identity=_stable_model_identity(getattr(runner,'metadata',{}))
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get('battery_sha256') != battery_hash:
                    raise ValueError('Existing output uses a different evaluation battery')
                completed[(record['condition'], record['prompt_id'])] = record
    output.parent.mkdir(parents=True, exist_ok=True)
    archives = {}
    model_config = getattr(runner, 'config', {})
    model_config = model_config.get('text_config', model_config)
    context_limit = job.get('context_limit', model_config.get('max_position_embeddings'))
    original_early = getattr(runner, 'early_tokens', None)
    original_tail = getattr(runner, 'tail_scale', 0.0)
    original_prefill = getattr(runner, 'prefill_scale', 0.0)
    absent = object()
    original_penalty = getattr(runner, 'repetition_penalty', absent)
    original_context = getattr(runner, 'repetition_context_size', absent)
    try:
        # Apply the same generic decoding processor to baseline and intervention.
        runner.repetition_penalty = repetition_penalty
        runner.repetition_context_size = repetition_context_size
        with output.open('a') as handle:
            for condition in conditions:
                recipe = condition.get('neurons', [])
                runner.steer(recipe)
                runner.early_tokens = condition.get('early_tokens')
                runner.tail_scale = condition.get('tail_scale', 0.0)
                runner.prefill_scale = float(condition.get('prefill_scale',0.0))
                if not math.isfinite(runner.prefill_scale):
                    raise ValueError('prefill_scale must be finite')
                scope = {
                    'early_tokens': runner.early_tokens, 'tail_scale': runner.tail_scale,
                    'prefill_scale': runner.prefill_scale,
                    'prefill': runner.early_tokens is None or runner.prefill_scale!=0,
                    'generation': 'all positions' if runner.early_tokens is None else 'initial positions with configured tail scale',
                    'boundary_note': 'Runner early-token scope includes the final prefill position predicting the first generated token.',
                }
                effective_recipe=recipe
                effective_scope=scope
                bootstrap_first_turn_only=bool(condition.get('bootstrap_first_turn_only',False))
                config_hash = _digest({'condition': condition, 'decoding': decoding_metadata})

                def emit(specification, section, messages, max_tokens, extra=None):
                    key = (condition['id'], specification['id'])
                    if key in completed:
                        prior = completed[key]
                        if (prior.get('condition_sha256') != config_hash
                                or prior.get('model_identity',_stable_model_identity(prior.get('model_metadata',{})))!=model_identity
                                or prior.get('messages') != messages
                                or prior.get('recipe') != effective_recipe
                                or prior.get('intervention_scope',{}).get('prefill_scale',0)!=runner.prefill_scale
                                or prior.get('max_new_tokens') != max_tokens):
                            raise ValueError('Resume configuration or conversation history differs: ' + repr(key))
                        return prior
                    ids = list(runner.tokens(messages))
                    record = dict(
                        condition=condition['id'], prompt_id=specification['id'], section=section,
                        battery_sha256=battery_hash, battery_file=str(battery_path),
                        extended_battery_file=str(extended_path) if extended_path else None,
                        extended_battery_sha256=extended_hash,
                        condition_sha256=config_hash, recipe=effective_recipe, intervention_scope=effective_scope,
                        bootstrap_first_turn_only=bootstrap_first_turn_only,
                        messages=messages, input_token_ids=ids, actual_input_tokens=len(ids),
                        max_new_tokens=max_tokens, seed=seed, temperature=temperature,
                        decoding_metadata=decoding_metadata,
                        decoding_deviations=decoding_deviations,
                        model_identity=model_identity,
                        model_metadata=getattr(runner, 'metadata', {}),
                        persona_applicable=specification.get('persona_applicable', True),
                        quality_checks=specification.get('quality_checks', []), **(extra or {}))
                    if context_limit is not None and len(ids) + max_tokens > context_limit:
                        record.update(status='skipped', skip_reason='Input plus output budget exceeds model context limit', context_limit=context_limit)
                    else:
                        result = runner.generate(messages, max_tokens, seed, temperature)
                        record.update(result, status='complete')
                        output_ids = _encode(runner, result['text'])
                        record['output_token_ids_reencoded'] = output_ids
                        record['output_token_note'] = 'Retokenized decoded text, not raw sampled IDs; generated_tokens is supplied by runner.'
                        record.update(_automatic_checks(specification, result['text']))
                        if section == 'long_input':
                            record['retrieval_pass'] = 'MARBLE' in result['text']
                        if section == 'long_output':
                            n = len(output_ids)
                            record['output_thirds'] = [runner.tok.decode(output_ids[a:b]) for a, b in [(0, n//3), (n//3, 2*n//3), (2*n//3, n)]]
                            record['truncated'] = result.get('finish_reason') == 'length'
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    handle.flush()
                    completed[key] = record
                    print(json.dumps({'event': 'evaluation', 'condition': condition['id'], 'prompt_id': specification['id'], 'status': record['status']}, ensure_ascii=False), flush=True)
                    return record

                def neutral(prompt):
                    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]

                if 'short' in sections:
                    for case in battery['cases']:
                        if job.get('short_ids') is None or case['id'] in job['short_ids']:
                            original_budget = case['max_new_tokens']
                            budget = original_budget if short_max_tokens is None else short_max_tokens
                            emit(case, 'short', neutral(case['prompt']), budget, {
                                'original_case_max_tokens': original_budget,
                                'max_budget_deviation': ({'frozen': original_budget, 'actual': budget}
                                                         if budget != original_budget else None),
                            })
                persistence = battery['persistence']
                if 'long_input' in sections:
                    case = persistence['long_input']
                    for budget in job.get('long_input_tokens', case['target_input_tokens']):
                        if budget not in case['target_input_tokens']:
                            raise ValueError('Requested length is not in the frozen battery')
                        if budget not in archives:
                            archives[budget] = _archive(runner, case, system, budget)
                        messages, details = archives[budget]
                        emit(dict(case, id=case['id'] + '_' + str(budget)), 'long_input', messages, case['max_new_tokens'], details)
                if 'long_output' in sections:
                    case = persistence['long_output']
                    emit(case, 'long_output', neutral(case['prompt']), case['max_new_tokens'])
                if 'multiturn' in sections:
                    case = persistence['multiturn']
                    history = [{'role': 'system', 'content': system}]
                    for turn_number, turn in enumerate(case['turns'], 1):
                        if bootstrap_first_turn_only and turn_number>1:
                            runner.steer([])
                            runner.early_tokens=None
                            runner.tail_scale=0.
                            runner.prefill_scale=0.
                            effective_recipe=[]
                            effective_scope={
                                'early_tokens':None,'tail_scale':0.,'prefill_scale':0.,
                                'prefill':False,'generation':'off',
                                'protocol':'First-turn bootstrap only; later turns retain generated conversation history.',
                            }
                        history = history + [{'role': 'user', 'content': turn['prompt']}]
                        record = emit(turn, 'multiturn', history, case['max_new_tokens_per_turn'], {'conversation_id': case['id'], 'turn_number': turn_number})
                        if record['status'] != 'complete':
                            break
                        history = history + [{'role': 'assistant', 'content': record['text']}]
    finally:
        runner.steer([])
        runner.early_tokens = original_early
        runner.tail_scale = original_tail
        runner.prefill_scale = original_prefill
        for name, value in [('repetition_penalty', original_penalty),
                            ('repetition_context_size', original_context)]:
            if value is absent:
                if hasattr(runner, name):
                    delattr(runner, name)
            else:
                setattr(runner, name, value)
    return {'output': str(output), 'records': len(completed), 'battery_sha256': battery_hash}
