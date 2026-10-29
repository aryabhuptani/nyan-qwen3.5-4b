"""Recover public Study115 numeric analyses in a clean temporary tree, on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
LABEL_FILES = ('labels.csv', 'labels_reviewer_A.csv', 'labels_reviewer_B.csv',
               'labels_uncertain_zero.csv', 'labels_uncertain_one.csv')
CONFIRMATION_FILES = ('confirmation_summary.csv', 'confirmation_reviewer_A.csv',
                      'confirmation_reviewer_B.csv', 'confirmation_uncertain_zero.csv',
                      'confirmation_uncertain_one.csv')
DESIGN_FILES = ('PROTOCOL.md', 'FEATURE_DEFINITIONS.json', 'JOB_discovery.json',
                'CASES_discovery.json', 'JOB_confirmation.json', 'CASES_confirmation.json',
                'analyze_discovery.py', 'summarize_confirmation.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def copy_public(relative, tree, copied):
    source = HERE / relative
    require(source.is_file() and not source.is_symlink(), f'Missing or symlinked public input: {relative}')
    target = tree / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied[str(relative)] = sha(target)


def numeric_stage(stage, tree, copied):
    relative = Path('numeric_results') / stage
    lock = load(HERE / relative / 'labels_lock.json')
    expected_rows = 60 if stage == 'discovery' else 30
    require(lock.get('total_items') == expected_rows, f'{stage}: unexpected scoring row count')
    copy_public(relative / 'labels_lock.json', tree, copied)
    for name in LABEL_FILES:
        copy_public(relative / name, tree, copied)
        require(sha(tree / relative / name) == lock.get('labels_sha256', {}).get(name),
                f'{stage}: public numeric labels SHA256 mismatch: {name}')
    # Bind the public scoring lock to its export manifest when provided.
    manifest = HERE / relative / 'export_manifest.json'
    if manifest.exists():
        require(load(manifest)['labels_lock_sha256'] == sha(tree / relative / 'labels_lock.json'),
                f'{stage}: public export manifest/scoring lock mismatch')
    return dict(rows=expected_rows, numeric_csv_sha256={name: copied[str(relative / name)]
                                                     for name in LABEL_FILES},
                labels_lock_sha256=copied[str(relative / 'labels_lock.json')])


def run_cpu(tree, script, arguments):
    # Stdlib entrypoints need no model, GPU library, raw generation or review files.
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONPYCACHEPREFIX'] = str(tree / 'bytecode')
    result = subprocess.run([sys.executable, '-B', '-E', '-s', str(tree / script), *arguments],
                            cwd=tree, env=environment, text=True, capture_output=True)
    # Do not persist subprocess output, which can contain private temporary paths.
    require(result.returncode == 0, f'CPU recovery failed in {script}; exit {result.returncode}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=HERE / 'CPU_RECOVERY.json')
    args = parser.parse_args()
    copied = {}
    with tempfile.TemporaryDirectory(prefix='study115_cpu_recovery_') as temporary:
        tree = Path(temporary)
        for name in DESIGN_FILES:
            copy_public(Path(name), tree, copied)
        # Confirmation job binds the exact already-frozen discovery selection bytes.
        copy_public(Path('analysis/discovery/discovery_selection.json'), tree, copied)
        discovery = numeric_stage('discovery', tree, copied)
        confirmation = numeric_stage('confirmation', tree, copied)
        run_cpu(tree, 'analyze_discovery.py', ['--labels', 'numeric_results/discovery/labels.csv',
                                              '--out-dir', 'recovered/discovery'])
        summary = 'discovery_summary.csv'
        recovered = tree / 'recovered/discovery' / summary
        require(recovered.read_bytes() == (HERE / 'analysis/discovery' / summary).read_bytes(),
                'Discovery summary bytes differ')
        expected = load(tree / 'analysis/discovery/discovery_selection.json')
        actual = load(tree / 'recovered/discovery/discovery_selection.json')
        expected_core = {key: value for key, value in expected.items() if key != 'provenance'}
        actual_core = {key: value for key, value in actual.items() if key != 'provenance'}
        require(actual_core == expected_core, 'Discovery selection core fields differ')
        # All provenance hashes must match as well; only the lock path may change.
        for value in (expected, actual):
            value['provenance']['scoring_label_lock'].pop('path', None)
        require(expected == actual, 'Discovery selection provenance hashes differ')
        discovery.update(summary_byte_identical=True, summary_sha256=sha(recovered),
                         selection_core_identical=True, selection_provenance_hashes_identical=True)
        run_cpu(tree, 'summarize_confirmation.py', ['--labels-dir', 'numeric_results/confirmation',
                                                   '--out-dir', 'recovered/confirmation'])
        matched = {}
        for name in CONFIRMATION_FILES:
            recovered = tree / 'recovered/confirmation' / name
            require(recovered.read_bytes() == (HERE / 'analysis/confirmation' / name).read_bytes(),
                    f'Confirmation summary bytes differ: {name}')
            matched[name] = sha(recovered)
        confirmation.update(all_five_csvs_byte_identical=True, summary_sha256=matched)
        report = dict(study=115, status='passed', method='Clean temporary tree; stdlib CPU numeric recovery.',
                      raw_inference_or_reviews_copied=False, model_or_gpu_required=False,
                      verifier_sha256=sha(Path(__file__)), public_inputs_sha256=copied,
                      discovery=discovery, confirmation=confirmation)
    content = json.dumps(report, indent=2) + '\n'
    require(not args.report.exists() or args.report.read_text() == content,
            'Refusing changed recovery report; select another --report path')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(content)
    print('Public numeric CPU recovery passed; report saved.')


if __name__ == '__main__':
    main()
