"""One original-BF16 worker for native-neuron generation/composition jobs only.

Runner/MLX are imported only after an explicit --job or --watch CLI launch.
--check-only validates jobs and public norms without loading a model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

try:
    from .public_control_runner import load_norms, public_job
    from .layer_schedules import normalize_layer_schedules
except ImportError:
    from public_control_runner import load_norms, public_job
    from layer_schedules import normalize_layer_schedules


ROOT=Path(__file__).resolve().parent
ALLOWED_ACTIONS={None,'generate','generation','run','evaluate'}


def validate_job(job):
    if not isinstance(job,dict) or job.get('action') not in ALLOWED_ACTIONS:
        raise ValueError('Composition worker allows generation/run/evaluate only; optimization and capture are prohibited')
    prepared=public_job(job)
    conditions=prepared.get('conditions')
    if not isinstance(conditions,list) or not conditions:
        raise ValueError('Composition job needs a nonempty conditions list')
    names=[]
    for condition in conditions:
        schedules=normalize_layer_schedules(condition.get('layer_schedules'))
        if schedules and prepared.get('action')=='evaluate':
            raise ValueError('Per-layer schedules currently require generation jobs; final_evaluation does not apply them')
        if 'layer_schedules' in condition:
            condition['layer_schedules']=schedules
        name=condition.get('id')
        if not isinstance(name,str) or not name:
            raise ValueError('Every condition needs a nonempty string ID')
        names.append(name)
        specs=condition.get('neurons',[])
        if not isinstance(specs,list):
            raise ValueError('Condition neurons must be a list')
        coordinates=set()
        for spec in specs:
            layer,neuron=spec.get('layer'),spec.get('neuron')
            if type(layer) is not int or type(neuron) is not int or not 0<=layer<32 or not 0<=neuron<9216:
                raise ValueError('Native coordinate must be a zero-based Qwen3.5-4B layer/neuron index')
            coordinates.add((layer,neuron))
            if any(key in spec for key in ('amount','amplitude','seed_gradient','gradient_consistency','mean_delta')):
                raise ValueError('Optimized/raw-amplitude recipes are prohibited; specify native published dose or multiplicative factor')
            mode=spec.get('mode','add')
            if mode not in ('add','mul'):
                raise ValueError('Native intervention mode must be add or mul')
            value=spec.get('factor') if mode=='mul' else spec.get('dose')
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Every native neuron needs a finite dose or multiplicative factor')
        if len(coordinates)>=4096:
            raise ValueError('The learned 4,096-coordinate recipe and recipes of that size are outside this native-composition worker')
    if len(names)!=len(set(names)):
        raise ValueError('Repeated condition IDs would mix output identities')
    if prepared.get('action')!='evaluate':
        prompts=prepared.get('prompts')
        if not isinstance(prompts,list) or not prompts:
            raise ValueError('Generation jobs need a nonempty prompts list')
        ids=[prompt.get('id') for prompt in prompts]
        if any(not isinstance(name,str) or not name for name in ids) or len(ids)!=len(set(ids)):
            raise ValueError('Generation prompt IDs must be unique nonempty strings')
    return prepared


def configure_public_runner(runner,norm_path,payload,medians):
    import mlx.core as mx
    config=runner.config.get('text_config',runner.config)
    if (len(runner.layers)!=32 or config.get('hidden_size')!=2560 or config.get('intermediate_size')!=9216
            or config.get('model_type') not in ('qwen3_5','qwen3_5_text')):
        raise ValueError('Composition worker requires original Qwen3.5-4B architecture')
    for mlp in runner.mlps:
        for name in ('gate_proj','up_proj','down_proj'):
            projection=getattr(mlp,name)
            if hasattr(projection,'scales') or projection.weight.dtype!=mx.bfloat16:
                raise ValueError('Composition worker requires unquantized original BF16 MLP weights')
    original=runner.metadata.get('calibration_file')
    runner.residual={index:value for index,value in enumerate(medians)}
    runner.metadata.update(calibration_file=str(norm_path),
        calibration_source='published layer-output residual medians',
        calibration_file_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        calibration_reference_sha256=payload.get('calibration_sha256'),
        original_local_calibration_file=original,
        experiment_protocol='native single-neuron characterization and intentional composition; no output-trained recipes')


def execute(runner,path,results,norm_path):
    job=validate_job(json.loads(path.read_text()))
    output=results/(path.stem+'.jsonl')
    identity=dict(job=job,model=runner.metadata['model'],
                  revision=runner.metadata.get('revision',runner.config.get('_commit_hash')),
                  norms=str(norm_path),norms_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest())
    manifest=output.with_suffix('.job.json')
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:
            raise ValueError('Existing output identity differs; use a new composition job name')
    else:
        if output.exists() and output.stat().st_size:
            raise ValueError('Existing output has no matching composition identity manifest')
        manifest.write_text(json.dumps(identity,indent=2))
    print(json.dumps(dict(event='composition_job',job=str(path),output=str(output))),flush=True)
    runner.run(job,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='Qwen/Qwen3.5-4B')
    parser.add_argument('--norms',default=str(ROOT/'public_qwen35_norms.json'))
    parser.add_argument('--job',type=Path)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--jobs-dir',type=Path,default=ROOT/'composition'/'jobs')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'composition'/'results')
    parser.add_argument('--poll-seconds',type=float,default=1.)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not args.job and not args.watch and not args.check_only:
        parser.error('Choose --job, --watch, or --check-only')
    if not math.isfinite(args.poll_seconds) or args.poll_seconds<=0 or args.poll_seconds>60:
        parser.error('--poll-seconds must be finite and between 0 and60')
    norm_path,payload,medians=load_norms(args.norms)
    if args.check_only:
        paths=[args.job] if args.job else sorted(args.jobs_dir.glob('*.json'))
        for path in paths:
            job=validate_job(json.loads(path.read_text()))
            print(json.dumps(dict(job=str(path),action=job.get('action','generate'),
                                  conditions=[condition['id'] for condition in job['conditions']],valid=True)))
        print(json.dumps(dict(norms=str(norm_path),medians=len(medians),model=payload['model'],model_loaded=False)))
        return
    # Explicit CLI launch is the only path importing Runner or loading MLX.
    try:
        from .runner import Runner
    except ImportError:
        from runner import Runner
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    args.jobs_dir.mkdir(parents=True,exist_ok=True)
    args.results_dir.mkdir(parents=True,exist_ok=True)
    runner=Runner(args.model)
    configure_public_runner(runner,norm_path,payload,medians)
    print(json.dumps(dict(event='composition_worker_ready',model=args.model,norms=str(norm_path),
                          jobs_dir=str(args.jobs_dir),results_dir=str(args.results_dir))),flush=True)
    if args.job:
        execute(runner,args.job,args.results_dir,norm_path)
    if args.watch:
        while True:
            for path in sorted(args.jobs_dir.glob('*.json')):
                active=path.with_suffix('.running')
                if active.exists():
                    continue
                try:
                    path.rename(active)
                except FileNotFoundError:
                    continue
                try:
                    execute(runner,active,args.results_dir,norm_path)
                    destination=active.with_suffix('.done')
                    if destination.exists():
                        raise ValueError('Existing .done file: composition job IDs must be unique')
                    active.rename(destination)
                    print(json.dumps(dict(event='composition_done',job=path.stem)),flush=True)
                except Exception:
                    traceback.print_exc()
                    destination=active.with_suffix('.failed')
                    if destination.exists():
                        destination=active.with_name(active.stem+'.'+str(time.time_ns())+'.failed')
                    active.rename(destination)
                    print(json.dumps(dict(event='composition_failed',job=path.stem,status_file=str(destination))),flush=True)
            time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
