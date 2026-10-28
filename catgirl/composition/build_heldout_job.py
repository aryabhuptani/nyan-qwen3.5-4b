"""CPU-only derivation of an authorized sealed battery after a recipe freeze.

This builder never queues jobs or starts a model. Invoke only after the recipe
file is committed/pushed and the study owner authorizes opening the source.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from composition_worker import validate_job


def sha(data):
    return hashlib.sha256(data).hexdigest()


def derive(source, frozen, freeze_commit, frozen_sha, source_sha, battery_file):
    cases = source['cases']
    long_cases = [c for c in cases if c['category'] == 'long_output']
    if (len(cases) != 8 or len(long_cases) != 1
            or len(source['conversation']['turns']) != 4):
        raise ValueError('Expected eight independent cases, one long guide, and four conversation turns')
    if source['system'] != 'You are a helpful assistant.':
        raise ValueError('Heldout system must remain neutral and helpful')
    if frozen['model'] != 'Qwen/Qwen3.5-4B' or frozen['revision'] != '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a':
        raise ValueError('Expected pinned original Qwen3.5-4B')
    protocol = frozen['protocol']
    for key, expected in dict(system=source['system'], seed=42, temperature=0,
                              repetition_penalty=1.1, repetition_context_size=128).items():
        if protocol.get(key) != expected:
            raise ValueError('Frozen protocol changed: ' + key)
    conditions = copy.deepcopy(frozen['conditions'])
    for c in conditions:
        if len(c['neurons']) > 8:
            raise ValueError('Native finalist exceeds eight coordinates')
        for n in c['neurons']:
            if set(n) != {'layer', 'neuron', 'published_dose'}:
                raise ValueError('Only native published-dose additive coordinates are allowed')
    battery = dict(schema_version=1, name='tertiary_native_prospective_validation',
                   purpose=source['purpose'], source_sha256=source_sha,
                   recipe_freeze_sha256=frozen_sha,
                   usage=dict(freeze_before_final_run=True, use_for_candidate_selection=False,
                              system_prompt=source['system'], persona_cues_in_evaluation_prompts=False,
                              save_all_outputs_including_failures=True,
                              decoding=dict(seed=42, temperature=0, top_p=1.,
                                            repetition_penalty=1.1, repetition_context_size=128)),
                   cases=copy.deepcopy([c for c in cases if c['category'] != 'long_output']),
                   persistence=dict(long_output=copy.deepcopy(long_cases[0]),
                                    multiturn=copy.deepcopy(source['conversation'])),
                   limitations=source.get('limitations', 'Fresh writer-generated set; limited independent sampling.'))
    job = dict(action='evaluate', battery_file=battery_file,
               sections=['short', 'long_output', 'multiturn'], include_baseline=False,
               conditions=conditions, seed=42, temperature=0, repetition_penalty=1.1,
               repetition_context_size=128,
               metadata=dict(stage='prospective_tertiary_native_validation',
                             frozen_recipe_commit=freeze_commit, frozen_recipe_sha256=frozen_sha,
                             tertiary_source_sha256=source_sha, no_retuning=True,
                             independent_cases=8, conversation_turns=4, conditions=len(conditions),
                             expected_records=12 * len(conditions)))
    validate_job(job)
    return battery, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'tertiary_heldout.json')
    parser.add_argument('--finalist', required=True, type=Path)
    parser.add_argument('--freeze-commit', required=True)
    parser.add_argument('--prefix', required=True, help='New job stem, such as 094_tertiary_native')
    parser.add_argument('--check-only', action='store_true', help='Validate without writing artifacts')
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix or not args.prefix:
        parser.error('--prefix must be a plain filename stem')
    repo = ROOT.parents[1]
    finalist = args.finalist.resolve()
    relative = finalist.relative_to(repo)
    committed = subprocess.run(['git', 'show', args.freeze_commit + ':' + relative.as_posix()],
                               cwd=repo, check=True, capture_output=True).stdout
    if committed != finalist.read_bytes():
        raise ValueError('Finalist bytes differ from the specified committed freeze')
    source_bytes = args.source.read_bytes()
    frozen = json.loads(committed)
    battery_name = args.prefix + '_battery.json'
    battery, job = derive(json.loads(source_bytes), frozen, args.freeze_commit,
                          sha(committed), sha(source_bytes), 'composition/' + battery_name)
    outputs = [(ROOT / battery_name, battery), (ROOT / (args.prefix + '.json'), job)]
    if not args.check_only:
        if any(path.exists() for path, _ in outputs):
            raise FileExistsError('Use a new prefix; an output artifact already exists')
        for path, data in outputs:
            with path.open('x') as handle:
                handle.write(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(valid=True, model_loaded=False, queued=False,
                          conditions=len(job['conditions']), expected_records=job['metadata']['expected_records'],
                          source_sha256=sha(source_bytes), frozen_sha256=sha(committed),
                          artifacts=[str(path) for path, _ in outputs], written=not args.check_only), indent=2))


if __name__ == '__main__':
    main()
