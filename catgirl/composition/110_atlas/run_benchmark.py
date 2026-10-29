"""Validate/seal forecasts before starting the native-only GPU benchmark."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
CATGIRL = ROOT.parents[1]
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def validate_forecasts(require_all=True):
    plan = read('110_PLAN.json')
    assert plan['expected_outputs'] == 584
    assert sha(ROOT/'selected30_atlas.json') == plan['source_selected_sha256']
    assert sha(ROOT/'selected_prompts.json') == plan['prompts_sha256']
    assert sha(ROOT/'interventions.json') == plan['interventions_sha256']
    assert sha(ROOT/'110_RUBRIC.md') == plan['rubric_sha256']
    assert sha(ROOT/'projections.json') == plan['projections_sha256']
    conditions = {c['id'] for c in read('interventions.json')['conditions']}
    features = plan['primary_features'] + ['joint_catgirl_coherent']
    sessions, compact, missing = [], [], []
    for session in read('forecast_sessions.json'):
        packet = ROOT / session['packet']
        assert sha(packet) == session['sha256']
        output = ROOT / session['output']
        if not output.exists():
            missing.append(session['output']); continue
        payload = json.loads(output.read_text())
        predictions = payload['predictions']
        assert len(predictions) == len(conditions) == 73
        assert {p['condition_id'] for p in predictions} == conditions
        for p in predictions:
            assert set(p['probabilities']) == set(plan['families'])
            for family, values in p['probabilities'].items():
                assert len(values) == len(features) == 8
                assert all(type(v) in (float,int) and math.isfinite(v) and 0<=v<=1 for v in values)
                assert values[7] <= min(values[0],values[1],values[5]) + 1e-9
                for feature, probability in zip(features, values):
                    compact.append(dict(arm=session['arm'], replicate=session['replicate'],
                        condition_id=p['condition_id'], family=family, feature=feature, probability=probability))
        sessions.append(dict(session, output_sha256=sha(output)))
    if require_all and missing:
        raise ValueError('Missing forecast sessions: '+str(missing))
    return plan, sessions, compact, missing


def validate_runtime_job():
    """Integrity hardening: runtime must implement the already-frozen inputs.

    This check adds no experimental condition or decoding change. The existing
    job was independently verified to satisfy these invariants before adding it.
    """
    interventions = read('interventions.json')
    prompts = read('selected_prompts.json')['prompts']
    job = read('runtime_job.local.json')
    condition_keys = ('id', 'neurons', 'early_tokens', 'prefill_scale', 'tail_scale')
    expected_conditions = [{k: c[k] for k in condition_keys}
                           for c in interventions['conditions']]
    expected_prompts = [{k: p[k] for k in ('id', 'messages', 'max_tokens')}
                        for p in prompts]
    if job['conditions'] != expected_conditions or job['prompts'] != expected_prompts:
        raise ValueError('Runtime job differs from frozen interventions/prompts')
    decoding = dict(action='generate', temperature=0, seed=42,
                    repetition_penalty=1.1, repetition_context_size=128, max_tokens=128)
    if any(job.get(k) != v for k, v in decoding.items()):
        raise ValueError('Runtime decoding differs from frozen benchmark builder')
    return job


def seal():
    plan, sessions, compact, missing = validate_forecasts()
    for path in (ROOT/'results.local').glob('110_validation*.jsonl'):
        assert not path.exists() or path.stat().st_size == 0, 'Cannot seal after validation starts'
    lock = ROOT/'forecast_lock.json'
    assert not lock.exists(), 'Already sealed; do not replace original lock'
    payload = dict(sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   plan_sha256=sha(ROOT/'110_PLAN.json'), sessions=sessions,
                   condition_family_vectors=73*4*8, no_validation_outputs=True)
    with (ROOT/'locked_forecasts.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0])); writer.writeheader(); writer.writerows(compact)
    payload['compact_forecasts_sha256'] = sha(ROOT/'locked_forecasts.csv')
    lock.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(sealed_sessions=len(sessions), forecast_cells=len(compact), lock_sha256=sha(lock))))


def run(model_path):
    plan, sessions, compact, missing = validate_forecasts()
    lock=read('forecast_lock.json')
    assert lock['plan_sha256'] == sha(ROOT/'110_PLAN.json')
    assert lock['compact_forecasts_sha256'] == sha(ROOT/'locked_forecasts.csv')
    assert lock['sessions'] == sessions
    job = validate_runtime_job()
    assert Path(model_path).name == REVISION and Path(model_path).is_dir()
    sys.path.insert(0,str(CATGIRL))
    from composition_worker import configure_public_runner, execute, load_norms
    from runner import Runner
    norms,payload,medians=load_norms(CATGIRL/'public_qwen35_norms.json')
    runner=Runner(model_path)
    runner.metadata['revision']=REVISION
    configure_public_runner(runner,norms,payload,medians)
    job_path=ROOT/'runtime_job.local.json'
    # Keep result basename stable and independent of source descriptions.
    runtime=ROOT/'results.local/110_validation.json'
    if runtime.exists():
        assert json.loads(runtime.read_text()) == job
    else:
        runtime.write_text(json.dumps(job,indent=2)+'\n')
    start=time.time()
    execute(runner,runtime,ROOT/'results.local',norms)
    raw=ROOT/'results.local/110_validation.jsonl'
    rows=[json.loads(x) for x in raw.read_text().splitlines()]
    assert len(rows)==584 and len({(r['condition'],r['prompt_id']) for r in rows})==584
    done=dict(rows=584, model=plan['model'], revision=REVISION, elapsed_seconds=time.time()-start,
              generated_tokens=sum(r['generated_tokens'] for r in rows),
              prompt_tokens=sum(r['prompt_tokens'] for r in rows),
              recorded_seconds=sum(r['seconds'] for r in rows), raw_sha256=sha(raw),
              plan_sha256=sha(ROOT/'110_PLAN.json'), forecast_lock_sha256=sha(ROOT/'forecast_lock.json'))
    (ROOT/'runtime_completion.json').write_text(json.dumps(done,indent=2)+'\n')
    print(json.dumps(dict(event='benchmark_complete',**done)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--seal',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--model',default='/Users/arya/Projects/mechinterp/model-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/'+REVISION)
    args=p.parse_args()
    if args.seal:seal()
    elif args.run:run(args.model)
    else:
        plan,sessions,compact,missing=validate_forecasts(require_all=False)
        print(json.dumps(dict(valid=True,model_loaded=False,complete_sessions=len(sessions),missing=missing)))


if __name__=='__main__':main()
