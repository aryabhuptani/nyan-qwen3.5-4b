"""Reproduce frozen native-neuron composition; --check-only never loads MLX."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

CATGIRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CATGIRL))
from composition_worker import configure_public_runner, load_norms, public_job
from layer_schedules import normalize_layer_schedules

FROZEN = Path(__file__).with_name('FROZEN_STYLE_FINALIST.json')
MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configuration(name, finalist=FROZEN):
    finalist = Path(finalist).expanduser().resolve()
    frozen = json.loads(finalist.read_text())
    if (frozen['model'] != MODEL or frozen['revision'] != REVISION
            or frozen.get('no_training') is not True
            or frozen.get('no_runtime_persona_instruction') is not True):
        raise ValueError('Expected the frozen, untrained original-BF16 composition protocol')
    matches = [c for c in frozen['conditions'] if c['id'] == name]
    if len(matches) != 1:
        raise ValueError('Unknown condition; choose: ' + ', '.join(c['id'] for c in frozen['conditions']))
    condition = matches[0]
    normalize_layer_schedules(condition.get('layer_schedules'))
    if len(condition['neurons']) > 8 or condition.get('bootstrap_first_turn_only', False):
        raise ValueError('Expected at most eight native coordinates, applied on every turn')
    early = condition.get('early_tokens')
    if early is not None and (type(early) is not int or early < 1):
        raise ValueError('early_tokens must be null or a positive integer')
    for key in ('prefill_scale', 'tail_scale'):
        if isinstance(condition.get(key), bool) or not isinstance(condition.get(key), (int, float)) or not math.isfinite(condition[key]):
            raise ValueError(key + ' must be a finite number')
    for spec in condition['neurons']:
        if set(spec) != {'layer', 'neuron', 'published_dose'}:
            raise ValueError('Only frozen published-dose additive neurons are permitted')
        if (type(spec['layer']) is not int or not 0 <= spec['layer'] < 32
                or type(spec['neuron']) is not int or not 0 <= spec['neuron'] < 9216
                or isinstance(spec['published_dose'], bool)
                or not math.isfinite(spec['published_dose'])):
            raise ValueError('Invalid native neuron coordinate or dose')
    recipe = public_job({'conditions': [condition]})['conditions'][0]['neurons']
    norm_path, norms, medians = load_norms(finalist.parent / frozen['protocol']['norms'])
    return frozen, condition, recipe, norm_path, norms, medians


def generate_turn(runner, history, prompt, recipe, condition, protocol, max_tokens):
    """One ordinary user turn, preserving only actual generated assistant history."""
    messages = copy.deepcopy(history) + [{'role': 'user', 'content': prompt}]
    runner.early_tokens = condition['early_tokens']
    runner.prefill_scale = condition['prefill_scale']
    runner.tail_scale = condition['tail_scale']
    runner.layer_schedules = normalize_layer_schedules(condition.get('layer_schedules'))
    runner.repetition_penalty = protocol['repetition_penalty']
    runner.repetition_context_size = protocol['repetition_context_size']
    for mlp in runner.mlps:
        mlp.position_mask = None
        mlp.capture = False
        mlp.captured = None
    try:
        runner.steer(recipe)
        result = runner.generate(messages, max_tokens=max_tokens,
                                 seed=protocol['seed'], temperature=protocol['temperature'])
    finally:
        runner.layer_schedules = {}
    history[:] = messages + [{'role': 'assistant', 'content': result['text']}]
    scope = dict(early_tokens=condition['early_tokens'], prefill_scale=condition['prefill_scale'],
                 tail_scale=condition['tail_scale'], every_turn=True,
                 all_positions=condition['early_tokens'] is None)
    layer_schedules = normalize_layer_schedules(condition.get('layer_schedules'))
    if layer_schedules:
        scope['layer_schedules'] = layer_schedules
        scope['all_positions'] = (condition['early_tokens'] is None
                                  and all(s['early_tokens'] is None for s in layer_schedules.values()))
    return dict(messages=messages, effective_recipe=copy.deepcopy(recipe),
                effective_scope=scope, condition=condition['id'],
                max_tokens=max_tokens, result=result)


def model_path(supplied):
    if supplied:
        path = Path(supplied).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('--model must be a local original-BF16 snapshot directory')
        # A different named snapshot is an accidental revision change.
        if path.parent.name == 'snapshots' and path.name != REVISION:
            raise ValueError('Snapshot revision differs from frozen model revision')
        return str(path)
    local = CATGIRL.parents[1] / 'model-cache' / 'hub' / 'models--Qwen--Qwen3.5-4B' / 'snapshots' / REVISION
    if local.is_dir():
        return str(local)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=MODEL, revision=REVISION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalist', type=Path, default=FROZEN, help='Frozen native composition finalist JSON')
    parser.add_argument('--condition', default='cute_hugs')
    parser.add_argument('--prompt', action='append', help='Repeat for a conversation; omit for interactive mode')
    parser.add_argument('--model', help='Local original BF16 model directory; otherwise fetch the pinned snapshot')
    parser.add_argument('--max-tokens', type=int, default=192)
    parser.add_argument('--transcript', type=Path, help='New JSONL file; existing files are refused')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error('--max-tokens must be positive')
    finalist = args.finalist.expanduser().resolve()
    frozen, condition, recipe, norm_path, norms, medians = configuration(args.condition, finalist)
    session = dict(event='session', model=MODEL, revision=REVISION, condition=condition,
                   frozen_file=str(finalist), frozen_sha256=hashlib.sha256(finalist.read_bytes()).hexdigest(),
                   condition_sha256=digest(condition), protocol=frozen['protocol'],
                   norms=str(norm_path), norms_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                   max_tokens=args.max_tokens, model_loaded=False)
    if args.check_only:
        print(json.dumps(session, indent=2))
        return
    # Reserve the transcript before downloading/loading, never overwriting an experiment.
    transcript = args.transcript.open('x') if args.transcript else None
    try:
        resolved_model = model_path(args.model)
        from runner import Runner
        (CATGIRL / 'results').mkdir(exist_ok=True)
        runner = Runner(resolved_model)
        configure_public_runner(runner, norm_path, norms, medians)
        session.update(model_loaded=True, loaded_model_path=resolved_model, runner_metadata=runner.metadata)
        if transcript:
            transcript.write(json.dumps(session) + '\n')
            transcript.flush()
        history = [{'role': 'system', 'content': frozen['protocol']['system']}]
        prompts = iter(args.prompt) if args.prompt else None
        turn = 0
        while True:
            if prompts is not None:
                try:
                    prompt = next(prompts)
                except StopIteration:
                    break
            else:
                try:
                    prompt = input('You (/quit, /reset): ')
                except (EOFError, KeyboardInterrupt):
                    break
                if prompt.strip() == '/quit':
                    break
                if prompt.strip() == '/reset':
                    history = [{'role': 'system', 'content': frozen['protocol']['system']}]
                    if transcript:
                        transcript.write(json.dumps(dict(event='reset', after_turn=turn)) + '\n')
                        transcript.flush()
                    continue
                if not prompt.strip():
                    continue
            record = generate_turn(runner, history, prompt, recipe, condition,
                                   frozen['protocol'], args.max_tokens)
            turn += 1
            record.update(event='turn', turn=turn)
            print(record['result']['text'], flush=True)
            if transcript:
                transcript.write(json.dumps(record) + '\n')
                transcript.flush()
    finally:
        if transcript:
            transcript.close()


if __name__ == '__main__':
    main()
