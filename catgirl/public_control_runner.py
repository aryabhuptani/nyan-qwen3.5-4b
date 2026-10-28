"""Run original BF16 Qwen3.5-4B controls with published residual-dose units."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parent


def load_norms(path):
    path=Path(path).expanduser().resolve()
    payload=json.loads(path.read_text())
    if payload.get('model')!='Qwen/Qwen3.5-4B':
        raise ValueError('Published norms must identify model Qwen/Qwen3.5-4B')
    medians=payload.get('medians')
    if not isinstance(medians,list) or len(medians)!=32:
        raise ValueError('Published norms must contain exactly 32 medians')
    if any(isinstance(value,bool) or not isinstance(value,(int,float))
           or not math.isfinite(value) or value<=0 for value in medians):
        raise ValueError('All 32 published residual medians must be finite and positive')
    return path,payload,[float(value) for value in medians]


def public_job(job):
    """Prefer explicit published doses over earlier locally converted values."""
    job=copy.deepcopy(job)
    for condition in job.get('conditions',[]):
        for spec in condition.get('neurons',[]):
            if 'published_dose' in spec and spec.get('mode')!='mul':
                dose=float(spec['published_dose'])
                if not math.isfinite(dose):
                    raise ValueError('published_dose must be finite')
                spec['dose']=dose
                spec.pop('amount',None)
    return job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='Qwen/Qwen3.5-4B')
    parser.add_argument('--job',required=True)
    parser.add_argument('--norms',default=str(ROOT/'public_qwen35_norms.json'))
    args=parser.parse_args()
    norm_path,norm_payload,medians=load_norms(args.norms)
    job_path=Path(args.job).expanduser().resolve()
    job=public_job(json.loads(job_path.read_text()))
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    try:
        from .runner import Runner
    except ImportError:
        from runner import Runner
    import mlx.core as mx
    runner=Runner(args.model)
    config=runner.config.get('text_config',runner.config)
    if (len(runner.layers)!=32 or config.get('hidden_size')!=2560
            or config.get('intermediate_size')!=9216
            or config.get('model_type') not in ('qwen3_5','qwen3_5_text')):
        raise ValueError('Loaded model architecture is not Qwen3.5-4B')
    for mlp in runner.mlps:
        for name in ('gate_proj','up_proj','down_proj'):
            projection=getattr(mlp,name)
            if hasattr(projection,'scales'):
                raise ValueError('Public original-model control requires unquantized BF16 weights')
            if projection.weight.dtype!=mx.bfloat16:
                raise ValueError('Public original-model control requires BF16 MLP weights')
    original_calibration=runner.metadata.get('calibration_file')
    runner.residual={index:value for index,value in enumerate(medians)}
    runner.metadata.update(calibration_file=str(norm_path),
                           calibration_source='published layer-output residual medians',
                           calibration_file_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                           calibration_reference_sha256=norm_payload.get('calibration_sha256'),
                           original_local_calibration_file=original_calibration)
    output=ROOT/'results'/(job_path.stem+'.jsonl')
    print(json.dumps(dict(event='public_control_calibration',model=args.model,
                          norms=str(norm_path),residual_medians=medians,
                          dose_definition='published median residual norm / original BF16 down-column norm',
                          output=str(output))),flush=True)
    runner.run(job,output)


if __name__=='__main__':
    main()
