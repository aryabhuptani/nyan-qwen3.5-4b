"""Verify the native composition bundle using only CPU and the standard library."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PINS = {
    'FROZEN_NATIVE_FINALIST.json': '587df82b5041c4e69010135c589c20a0d6c487a88bb11535859fe9c9ec511178',
    'FROZEN_STYLE_FINALIST.json': 'c90ff9df1cf9854aa5daa43e57d7431bc394a2902e27e391a0d6881d915b2472',
    'FROZEN_FOUR_NATIVE_FINALIST.json': 'c1755fb935556945f3d1e4095e5a8357945e32f7f0dc15a0f367015d3b7c1d91',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_rows(job):
    conditions = job['conditions']
    if job.get('action') != 'evaluate':
        return len(conditions) * len(job['prompts'])
    battery = load(ROOT.parent / job['battery_file'])
    cases = 0
    for section in job['sections']:
        if section == 'short':
            cases += sum(job.get('short_ids') is None or c['id'] in job['short_ids']
                         for c in battery['cases'])
        elif section == 'multiturn':
            cases += len(battery['persistence']['multiturn']['turns'])
        elif section == 'long_output':
            cases += 1
        elif section == 'long_input':
            cases += len(job.get('long_input_tokens', battery['persistence']['long_input']['target_input_tokens']))
        else:
            raise ValueError('Unknown evaluation section: ' + section)
    return len(conditions) * cases


def verify():
    frozen = {}
    for name, pin in PINS.items():
        require(sha(ROOT / name) == pin, 'Frozen file identity changed: ' + name)
        frozen[name] = load(ROOT / name)
    compact = load(ROOT / 'native_recipes.json')
    require(compact['frozen_native_sha256'] == PINS['FROZEN_NATIVE_FINALIST.json'], 'Compact native identity mismatch')
    require(compact['frozen_style_sha256'] == PINS['FROZEN_STYLE_FINALIST.json'], 'Compact style identity mismatch')
    require(compact['frozen_four_native_sha256'] == PINS['FROZEN_FOUR_NATIVE_FINALIST.json'], 'Compact final four-coordinate identity mismatch')
    for condition in compact['conditions']:
        sources = [c for f in frozen.values() for c in f['conditions'] if c['id'] == condition['id']]
        require(any(c == condition for c in sources), 'Compact condition differs from frozen source: ' + condition['id'])
    for key in ('model', 'revision', 'protocol'):
        require(compact[key] == frozen['FROZEN_NATIVE_FINALIST.json'][key], 'Compact model/protocol mismatch: ' + key)
    norm_path = (ROOT / compact['protocol']['norms']).resolve()
    require(sha(norm_path) == compact['protocol']['norms_sha256'], 'Published norms identity changed')

    with (ROOT / 'native_coefficients.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    actual = []
    for row in csv_rows:
        actual.append((row['recipe'], int(row['layer_zero_based']), int(row['neuron_zero_based']),
                       float(row['published_dose']), row['early_tokens'], float(row['prefill_scale']),
                       float(row['tail_scale']), row['every_turn']))
        source = compact['sources'][f"L{row['layer_zero_based']}N{row['neuron_zero_based']}"]
        require(row['source_url'] == source['url'] and row['attribution'] == source['attribution']
                and row['dose_note'] == source['dose_note'], 'CSV attribution mismatch')
    expected = []
    for c in compact['conditions']:
        for n in c['neurons']:
            scope = c.get('layer_schedules', {}).get(str(n['layer']), c)
            expected.append((c['id'], n['layer'], n['neuron'], float(n['published_dose']),
                             'all_positions' if scope['early_tokens'] is None else str(scope['early_tokens']),
                             float(scope['prefill_scale']), float(scope['tail_scale']), 'True'))
    require(len(actual) == 15 and sorted(actual) == sorted(expected), 'CSV does not reproduce all fifteen exact coordinate/scope rows')

    # Run the same recipe loader as --check-only for every frozen/compact condition.
    spec = importlib.util.spec_from_file_location('composition_demo_cpu_check', ROOT / 'demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    checked = 0
    for name, payload in list(frozen.items()) + [('native_recipes.json', compact)]:
        for condition in payload['conditions']:
            demo.configuration(condition['id'], ROOT / name)
            checked += 1
    require('mlx.core' not in sys.modules and 'runner' not in sys.modules, 'CPU verifier imported the model runtime')

    jobs = []
    queued = {p.stem for p in (ROOT / 'jobs').iterdir()
              if p.suffix in ('.done', '.running', '.failed', '.json') and p.stem[:3].isdigit()}
    for stem in sorted(queued):
        path = ROOT / (stem + '.json')
        if not path.exists():
            continue
        job = load(path)
        expected_count = expected_rows(job)
        require(job.get('metadata', {}).get('expected_records', expected_count) == expected_count,
                'Preregistered expected row count is inconsistent: ' + stem)
        done = ROOT / 'jobs' / (stem + '.done')
        if not done.exists():
            state = next((suffix for suffix in ('running', 'failed', 'json')
                          if (ROOT / 'jobs' / (stem + '.' + suffix)).exists()), 'not_queued')
            jobs.append(dict(job=stem, state=state, expected_rows=expected_count, raw_count_checked=False))
            continue
        require(load(done) == job, 'Completed queue job differs from committed job: ' + stem)
        output = ROOT / 'results' / (stem + '.jsonl')
        rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        require(len(rows) == expected_count, 'Completed raw output row count differs: ' + stem)
        require(len({(r['condition'], r['prompt_id']) for r in rows}) == len(rows), 'Repeated raw output identity: ' + stem)
        require({r['condition'] for r in rows} == {c['id'] for c in job['conditions']}, 'Raw condition IDs differ: ' + stem)
        jobs.append(dict(job=stem, state='done', expected_rows=expected_count, actual_rows=len(rows), raw_count_checked=True))
    return dict(valid=True, model_loaded=False, frozen_files=PINS, csv_coordinate_rows=15,
                conditions_checked=checked, jobs=jobs,
                note='Artifact identity and row counts only; this does not grade behavior or establish success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Explicit CPU-only check (also the default)')
    parser.parse_args()
    print(json.dumps(verify(), indent=2))


if __name__ == '__main__':
    main()
