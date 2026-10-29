"""Freeze outcome-blind Study114 controls, sources, and validation job (no model load)."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('study114_adapter', HERE / 'run_replication.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def make_selection(original):
    excluded = set((layer, neuron) for layer, neuron, _ in original.BASE) | set(original.MODIFIERS.values())
    rng = random.Random(114)
    draws = {}
    for layer in (23, 13):
        eligible = [neuron for neuron in range(9216) if (layer, neuron) not in excluded]
        draws[layer] = rng.sample(eligible, 4)
    return dict(seed=114, rng='Python random.Random',
                algorithm='Create random.Random(114) once. In layer order [23, 13], form ascending eligible neuron IDs from range(9216), excluding every original Study113 BASE and MODIFIERS coordinate, then call rng.sample(eligible, 4). Pair draws by index; no output or semantic screening.',
                layer_order=[23, 13], intermediate_size=9216,
                excluded_coordinates=[dict(layer=l, neuron=n) for l, n in sorted(excluded)],
                draws={str(layer): draws[layer] for layer in (23, 13)},
                pairs=[dict(condition_id=f'random_{i + 1}',
                            paws=dict(layer=23, neuron=draws[23][i]),
                            genki=dict(layer=13, neuron=draws[13][i])) for i in range(4)],
                matching='Each individual additive residual perturbation requests the same signed per-token-per-layer normalized norm as its Paws (+0.4 at L23) or Genki (+0.8 at L13) counterpart. Study113 converts dose using frozen published layer residual norms / actual down-projection column L2. All control coordinates have all-position scope. This is not a native activation quantile control; combined-vector norms and BF16 realized deltas need not match exactly.')


def conditions(selection):
    result = [dict(condition_id=cid, base=True, modifiers=modifiers)
              for cid, modifiers in [('00000', {}), ('10000', {'paws': .5}),
                                     ('00100', {'genki': 1.}), ('10100', {'paws': .5, 'genki': 1.})]]
    for pair in selection['pairs']:
        cid = pair['condition_id']
        result.append(dict(condition_id=cid, base=True,
                           modifiers={cid + '_paws': 1., cid + '_genki': 1.},
                           modifier_scopes={cid + '_paws': 'all', cid + '_genki': 'all'}))
    return result


def self_test(original, sources, selection, rows):
    baseline = adapter.load_original()
    for condition in rows[:4]:
        assert original.recipe(condition, sources['modifiers']) == baseline.recipe(condition, sources['modifiers'])
    base = baseline.recipe(rows[0], sources['modifiers'])
    for condition in rows[4:]:
        recipe = original.recipe(condition, sources['modifiers'])
        assert recipe[:4] == base
        assert [(s['layer'], s['reference_normalized_dose'], s['scope']) for s in recipe[4:]] == [(23, .4, 'all'), (13, .8, 'all')]
    assert len({pair[role]['neuron'] for pair in selection['pairs'] for role in ['paws']}) == 4
    assert len({pair[role]['neuron'] for pair in selection['pairs'] for role in ['genki']}) == 4
    assert original.make_runner.__code__.co_code == baseline.make_runner.__code__.co_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=HERE / 'CASES_validation.json')
    args = parser.parse_args()
    original = adapter.load_original()
    selection = make_selection(original)
    sources = copy.deepcopy(json.loads((adapter.ORIGINAL.parent / 'MODIFIER_SOURCES.json').read_text()))
    sources['study'] = 114
    sources['copied_from_study'] = 113
    sources['original_sources_sha256'] = adapter.sha(adapter.ORIGINAL.parent / 'MODIFIER_SOURCES.json')
    for pair in selection['pairs']:
        for role, dose in [('paws', .4), ('genki', .8)]:
            name = pair['condition_id'] + '_' + role
            coordinate = pair[role]
            sources['modifiers'][name] = dict(coordinate=coordinate, reference_dose=dose,
                                             provenance='Outcome-blind seeded coordinate draw; see CONTROL_SELECTION.json.',
                                             matching=selection['matching'])
            original.MODIFIERS[name] = (coordinate['layer'], coordinate['neuron'])
    rows = conditions(selection)
    self_test(original, sources, selection, rows)
    dump(HERE / 'CONTROL_SELECTION.json', selection)
    dump(HERE / 'SOURCES.json', sources)
    if not args.cases.exists():
        print('Controls and sources frozen; cases absent, so JOB_validation.json was not created.')
        return
    payload = json.loads(args.cases.read_text())
    cases = payload['cases'] if isinstance(payload, dict) else payload
    if len(cases) != 12 or len({case['case_id'] for case in cases}) != 12:
        raise ValueError('Expected exactly 12 unique validation cases')
    if any(case['split'] != 'validation' for case in cases):
        raise ValueError('Validation cases only')
    if any(case.get('max_tokens', 192) != 192 for case in cases):
        raise ValueError('Frozen decoding requires max_tokens=192')
    job = dict(study=114, split='validation', conditions=rows,
               temperature=0, max_tokens=192, seed=42, repetition_penalty=1.1,
               repetition_context_size=128, model_revision=original.REVISION,
               cases_sha256=adapter.sha(args.cases), sources_sha256=adapter.sha(HERE / 'SOURCES.json'),
               replication_frozen_files=dict(wrapper_sha256=adapter.sha(HERE / 'run_replication.py'),
                    helper_sha256=adapter.sha(__file__), original_runner_sha256=adapter.sha(adapter.ORIGINAL),
                    control_selection_sha256=adapter.sha(HERE / 'CONTROL_SELECTION.json'),
                    original_sources_sha256=adapter.sha(adapter.ORIGINAL.parent / 'MODIFIER_SOURCES.json')),
               normalization=selection['matching'])
    dump(HERE / 'JOB_validation.json', job)
    print(json.dumps(dict(valid=True, recipe_self_tests='passed', cases=len(cases), conditions=len(rows), expected_rows=len(cases)*len(rows))))


if __name__ == '__main__':
    main()
