"""Describe the frozen fresh comparison; never reselect doses from these outcomes."""
import argparse
import csv
import io
import json
from decimal import Decimal
from pathlib import Path
from analyze_discovery import EXPECTED_FEATURES, digest, load, parse_feature, require, summarize, write_new

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    job = load(HERE / 'JOB_confirmation.json')
    cases = load(HERE / 'CASES_confirmation.json')
    require(digest(HERE / 'CASES_confirmation.json') == job['cases_sha256'], 'Case drift')
    require(digest(HERE / 'analysis/discovery/discovery_selection.json') == job['selection_sha256'], 'Selection drift')
    conditions = {c['condition_id']: tuple(Decimal(str(c['normalized_doses'][f]))
                  for f in ('paws', 'genki')) for c in job['conditions']}
    case_ids = {c['case_id'] for c in cases['cases']}
    require(len(conditions) == 5 and len(case_ids) == 6, 'Expected frozen 5 by 6 comparison')
    lock = load(args.labels_dir / 'labels_lock.json')
    require(lock['total_items'] == 30, 'Expected 30 reviewed replies')
    outputs = {}
    for mode in ('adjudicated', 'reviewer_A', 'reviewer_B', 'uncertain_zero', 'uncertain_one'):
        name = 'labels.csv' if mode == 'adjudicated' else f'labels_{mode}.csv'
        path = args.labels_dir / name
        require(digest(path) == lock['labels_sha256'][name], 'Label hash mismatch')
        rows = {}
        for row in csv.DictReader(path.open(newline='')):
            key = row['condition_id'], row['case_id']
            require(key not in rows and key[0] in conditions and key[1] in case_ids, 'Unexpected/duplicate row')
            rows[key] = {f: parse_feature(row[f], str(key)) for f in EXPECTED_FEATURES}
        require(set(rows) == {(c, k) for c in conditions for k in case_ids}, 'Incomplete grid')
        result = summarize(conditions, case_ids, rows)
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(result[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(result)
        output = args.out_dir / ('confirmation_summary.csv' if mode == 'adjudicated' else f'confirmation_{mode}.csv')
        write_new(output, stream.getvalue())
        outputs[output.name] = digest(output)
    provenance = dict(job_sha256=digest(HERE / 'JOB_confirmation.json'),
                      cases_sha256=digest(HERE / 'CASES_confirmation.json'),
                      labels_lock_sha256=digest(args.labels_dir / 'labels_lock.json'),
                      script_sha256=digest(Path(__file__)), outputs_sha256=outputs,
                      policy='Descriptive fresh comparison only; no dose reselection or persistence claim.')
    write_new(args.out_dir / 'confirmation_analysis_lock.json', json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
