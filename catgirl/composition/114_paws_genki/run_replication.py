"""Frozen Study114 adapter; Study113 supplies the unchanged recipe, hook, and run loop."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / '113_catgirl_circuits' / 'run_circuits.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original():
    spec = importlib.util.spec_from_file_location('study114_original113', ORIGINAL)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    return original


def run(args):
    job = json.loads(args.job.read_text())
    frozen = job['replication_frozen_files']
    required = {'wrapper_sha256': HERE / 'run_replication.py',
                'helper_sha256': HERE / 'build_job.py',
                'original_runner_sha256': ORIGINAL,
                'control_selection_sha256': HERE / 'CONTROL_SELECTION.json',
                'original_sources_sha256': ORIGINAL.parent / 'MODIFIER_SOURCES.json'}
    for key, path in required.items():
        if frozen[key] != sha(path):
            raise ValueError('Study114 frozen file hash mismatch: ' + key)
    for key, path in [('cases_sha256', args.cases), ('sources_sha256', args.sources)]:
        if job[key] != sha(path):
            raise ValueError('Study114 frozen input hash mismatch: ' + key)
    original = load_original()
    selection = json.loads((HERE / 'CONTROL_SELECTION.json').read_text())
    for pair in selection['pairs']:
        for role in ('paws', 'genki'):
            coordinate = pair[role]
            original.MODIFIERS[pair['condition_id'] + '_' + role] = (coordinate['layer'], coordinate['neuron'])
    # The original run records its own runner_sha256 plus the final job_sha256.
    # The latter transitively binds this wrapper, helper, and control selection.
    original.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--norms', type=Path, default=HERE.parents[1] / 'public_qwen35_norms.json')
    parser.add_argument('--check-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
