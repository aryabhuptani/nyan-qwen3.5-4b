"""Frozen Study115 discovery summary and deterministic protocol selection (stdlib only).

Reads scored CSV, never reply text. Blank feature cells remain unknown.
"""
import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_FEATURES = (
    'coherent', 'assistant_cat_identity', 'assistant_female_identity',
    'performed_feline_vocalization', 'assistant_idol_identity',
    'assistant_mascot_identity', 'multi_character_game_framing',
    'sustained_cute_register',
)
SELECTION_RULE = [
    'maximize clean_catgirl_with_sound', 'maximize clean_catgirl',
    'minimize idol_or_mascot', 'minimize total added normalized dose',
    'smaller Genki', 'smaller Paws',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def conjunction(values):
    """Three-valued AND; only a known 1 contributes to selection counts."""
    return 0 if 0 in values else None if None in values else 1


def disjunction(values):
    return 1 if 1 in values else None if None in values else 0


def parse_feature(value, context):
    require(value in ('', '0', '1'), f'{context}: expected 0, 1 or blank, got {value!r}')
    return None if value == '' else int(value)


def read_design(job_path, cases_path, features_path):
    job, cases, definitions = load(job_path), load(cases_path), load(features_path)
    require(definitions['feature_order'] == list(EXPECTED_FEATURES), 'Feature definition drift')
    require(job['study'] == cases['study'] == 115, 'Expected study 115')
    require(job['split'] == 'dose_discovery', 'Expected discovery job')
    require(digest(cases_path) == job['cases_sha256'], 'Frozen discovery cases hash mismatch')
    conditions = {}
    for condition in job['conditions']:
        name = condition['condition_id']
        require(name not in conditions, f'Duplicate condition {name}')
        doses = condition['normalized_doses']
        require(set(doses) == {'paws', 'genki'}, 'Unexpected dose keys')
        p, g = (Decimal(str(doses[k])) for k in ('paws', 'genki'))
        require(name == f'p{p}_g{g}', f'Condition ID and doses differ: {name}')
        conditions[name] = (p, g)
    expected_doses = {(Decimal(p), Decimal(g)) for p in ('0', '.1', '.2', '.4')
                      for g in ('0', '.1', '.2', '.4', '.8')}
    require(len(conditions) == 20 and set(conditions.values()) == expected_doses,
            'Expected the complete frozen 20-condition dose grid')
    case_ids = [case['case_id'] for case in cases['cases']]
    require(len(case_ids) == len(set(case_ids)) == 3, 'Expected three distinct discovery cases')
    return conditions, set(case_ids)


def read_labels(labels_path, conditions, cases, lock_path=None):
    lock_path = Path(lock_path) if lock_path else labels_path.parent / 'labels_lock.json'
    lock_status = {'status': 'absent', 'path': None, 'sha256': None}
    if lock_path.exists():
        lock = load(lock_path)
        require(labels_path.name == 'labels.csv', 'Selection requires adjudicated labels.csv')
        require(lock.get('labels_sha256', {}).get(labels_path.name) == digest(labels_path),
                'Scoring-label lock mismatch')
        require(lock.get('total_items') == 60, 'Scoring lock does not describe 60 items')
        lock_status = {'status': 'verified', 'path': str(lock_path), 'sha256': digest(lock_path)}
    elif lock_path != labels_path.parent / 'labels_lock.json':
        raise ValueError(f'Explicit scoring-label lock does not exist: {lock_path}')
    with labels_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        require(len(headers) == len(set(headers)), 'Duplicate CSV column names')
        require({'item_id', 'condition_id', 'case_id', *EXPECTED_FEATURES} <= set(headers),
                'Required label columns missing')
        rows, items = {}, set()
        for number, row in enumerate(reader, 2):
            require(None not in row and None not in row.values(), f'Malformed CSV row {number}')
            key = (row['condition_id'], row['case_id'])
            require(key[0] in conditions and key[1] in cases, f'Unexpected condition/case: {key}')
            require(key not in rows, f'Duplicate condition/case: {key}')
            require(row['item_id'] and row['item_id'] not in items, 'Empty or duplicate item_id')
            items.add(row['item_id'])
            rows[key] = {f: parse_feature(row[f], f'row {number}, {f}') for f in EXPECTED_FEATURES}
    require(set(rows) == {(condition, case) for condition in conditions for case in cases},
            'Incomplete discovery condition/case coverage (expected 60 rows)')
    return rows, lock_status


def summarize(conditions, cases, rows):
    summaries = []
    for name, (paws, genki) in conditions.items():
        values = {f: [] for f in (*EXPECTED_FEATURES, 'clean_catgirl',
                                 'clean_catgirl_with_sound', 'idol_or_mascot')}
        for case in sorted(cases):
            ratings = rows[name, case]
            for feature in EXPECTED_FEATURES:
                values[feature].append(ratings[feature])
            clean = conjunction([ratings[f] for f in ('coherent', 'assistant_cat_identity',
                                                        'assistant_female_identity')] +
                                [None if ratings[f] is None else 1 - ratings[f]
                                 for f in ('assistant_idol_identity', 'assistant_mascot_identity')])
            values['clean_catgirl'].append(clean)
            values['clean_catgirl_with_sound'].append(
                conjunction([clean, ratings['performed_feline_vocalization']]))
            values['idol_or_mascot'].append(disjunction(
                [ratings['assistant_idol_identity'], ratings['assistant_mascot_identity']]))
        summary = dict(condition_id=name, cases=len(cases), paws_dose=str(paws),
                       genki_dose=str(genki), total_added_dose=str(paws + genki))
        for feature, cells in values.items():
            summary[feature] = cells.count(1)
            summary[feature + '_zero'] = cells.count(0)
            summary[feature + '_unknown'] = cells.count(None)
        summaries.append(summary)
    return summaries


def selection_key(row):
    return (-row['clean_catgirl_with_sound'], -row['clean_catgirl'], row['idol_or_mascot'],
            Decimal(row['total_added_dose']), Decimal(row['genki_dose']), Decimal(row['paws_dose']))


def select(summaries):
    ranked = sorted(summaries, key=selection_key)
    candidate = ranked[0]
    baseline = next(row for row in summaries if row['condition_id'] == 'p0_g0')
    first = candidate['clean_catgirl_with_sound'] - baseline['clean_catgirl_with_sound']
    second = candidate['clean_catgirl'] - baseline['clean_catgirl']
    improves = first > 0 or first == 0 and second > 0
    return dict(selection_rule=SELECTION_RULE,
                unknown_policy='Unknown cells never satisfy a conjunction; unknown drift is counted separately, not imputed.',
                candidate=candidate, baseline=baseline,
                candidate_improves_baseline=improves,
                improvement_rule='Strict gain in clean_catgirl_with_sound, or strict gain in clean_catgirl with the first tied.',
                confirmation_required=improves,
                disposition='run unchanged selected-dose confirmation' if improves else
                            'stop: dose calibration did not improve the requested target',
                ranked_condition_ids=[row['condition_id'] for row in ranked])


def write_new(path, content):
    path = Path(path)
    require(not path.exists() or path.read_text() == content, f'Refusing changed artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--labels-lock', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    job, cases, features = (HERE / name for name in
                            ('JOB_discovery.json', 'CASES_discovery.json', 'FEATURE_DEFINITIONS.json'))
    conditions, case_ids = read_design(job, cases, features)
    rows, lock = read_labels(args.labels, conditions, case_ids, args.labels_lock)
    summary = summarize(conditions, case_ids, rows)
    decision = select(summary)
    decision['provenance'] = dict(labels_sha256=digest(args.labels), scoring_label_lock=lock,
                                 job_sha256=digest(job), cases_sha256=digest(cases),
                                 features_sha256=digest(features), protocol_sha256=digest(HERE / 'PROTOCOL.md'),
                                 analysis_sha256=digest(Path(__file__)))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(summary)
    write_new(args.out_dir / 'discovery_summary.csv', stream.getvalue())
    write_new(args.out_dir / 'discovery_selection.json', json.dumps(decision, indent=2) + '\n')
    print(json.dumps(dict(summary=str(args.out_dir / 'discovery_summary.csv'),
                          selection=str(args.out_dir / 'discovery_selection.json'))))


if __name__ == '__main__':
    main()
