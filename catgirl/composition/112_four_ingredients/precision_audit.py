"""Prospective BF16-weight/float32-computation lexical sensitivity audit.

No GPU or validation outcomes are read without --run. See PRECISION_PLAN.md.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parent
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def readrows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def preference(r): return r['choices']['target']['mean_logprob']-r['choices']['control']['mean_logprob']


def contrasts(rows,cases):
    table={(r['condition_id'],r['case_id']):r for r in rows}
    result={}
    for case in cases:
        cid=case['case_id']; name=case['ingredient']
        base=preference(table[('baseline',cid)])
        for prefix in ('canonical','attenuate'):
            cond=prefix+'_'+name
            result[(cid,cond)]=preference(table[(cond,cid)])-base
    return result


def selftest():
    from gpu_integrity import extract_probe
    fake=type('NN',(),{'Module':object})
    extract_probe(ROOT/'measure.py',{'nn':fake,'mx':None})
    case={'case_id':'toy','ingredient':'genki'}
    rows=[dict(case_id='toy',condition_id=c,choices={'target':{'mean_logprob':p},'control':{'mean_logprob':0.}})
          for c,p in [('baseline',.5),('canonical_genki',.7),('attenuate_genki',.4)]]
    values=contrasts(rows,[case]); assert abs(values[('toy','canonical_genki')]-.2)<1e-12
    assert abs(values[('toy','attenuate_genki')]+.1)<1e-12


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cases',type=Path,default=ROOT/'CASES_lexical_validation.json')
    p.add_argument('--job',type=Path,default=ROOT/'jobs/JOB_validation_lexical_generation.json')
    p.add_argument('--bf16-results',type=Path)
    p.add_argument('--forecast-lock',type=Path)
    p.add_argument('--model',default='/Users/arya/Projects/mechinterp/model-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/'+REVISION)
    p.add_argument('--run',action='store_true')
    args=p.parse_args(); selftest()
    if not args.run:
        print(json.dumps(dict(valid=True,model_loaded=False,validation_outcomes_read=False,ast_extraction=True,contrast_math=True)))
        return
    assert args.bf16_results and args.forecast_lock and args.forecast_lock.exists()
    run(args)


def run(args):
    import importlib.metadata
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load
    from mlx.utils import tree_flatten
    from measure import INGREDIENTS,segmented_tokens
    from gpu_integrity import extract_probe
    assert Path(args.model).name==REVISION
    plan=read(ROOT/'112_PLAN.json')
    assert sha(ROOT/'measure.py')==plan['input_sha256']['measure.py']
    assert not args.output.exists(), 'Fresh output directory required'
    args.output.mkdir(parents=True)
    cases=read(args.cases)['cases']; assert len(cases)==12 and all(c['split']=='validation' for c in cases)
    assert len({c['case_id'] for c in cases})==12
    job=read(args.job); assert job['revision']==REVISION
    condition_map={c['condition_id']:c for c in job['conditions']}
    mx.set_default_device(mx.gpu)
    model,tok,config=load(args.model,return_config=True)
    before={}; after={}
    for name,param in tree_flatten(model.parameters()):
        before[str(param.dtype)]=before.get(str(param.dtype),0)+param.size
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    for name,param in tree_flatten(model.parameters()):
        after[str(param.dtype)]=after.get(str(param.dtype),0)+param.size
        if mx.issubdtype(param.dtype,mx.floating): assert param.dtype==mx.float32
    assert any('bfloat16' in name for name in before)
    Probe=extract_probe(ROOT/'measure.py',{'nn':nn,'mx':mx})
    probes={}
    for name,(layer,neuron,_) in INGREDIENTS.items():
        probe=Probe(model.layers[layer].mlp,neuron); model.layers[layer].mlp=probe; probes[name]=probe
        assert probe.gate_proj.weight.dtype==probe.up_proj.weight.dtype==probe.down_proj.weight.dtype==mx.float32
    metadata=dict(revision=REVISION,weight_source='OriginalBF16valuescasttofloat32; noquantizationorweightchange',
                  before_parameter_dtype_counts=before,after_parameter_dtype_counts=after,
                  cases_sha256=sha(args.cases),job_sha256=sha(args.job),forecast_lock_sha256=sha(args.forecast_lock),
                  bf16_results_sha256=sha(args.bf16_results),runner_sha256=sha(__file__),probe_source_sha256=sha(ROOT/'measure.py'),
                  precision_plan_sha256=sha(ROOT/'PRECISION_PLAN.md'),
                  versions={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers')},hardware=mx.device_info())
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def reset(condition,length):
        operations={o['ingredient']:o for o in condition.get('operations',[])}
        for name,probe in probes.items():
            op=operations.get(name)
            assert op is None or op['scope']=='generation'
            probe.reset(op,(length-1,10**9))
    def capture():
        result={}
        for name,probe in probes.items():
            mx.eval([r[1:] for r in probe.records])
            result[name]=dict(positions=[i for offset,pre,post in probe.records for i in range(offset,offset+len(pre))],
                              pre=[float(v) for _,pre,_ in probe.records for v in np.asarray(pre)],
                              post=[float(v) for _,_,post in probe.records for v in np.asarray(post)])
        return result
    started=time.monotonic(); output=[]
    with (args.output/'float32.jsonl').open('x') as f:
        for case in cases:
            cid=case['case_id']; ingredient=case['ingredient']
            ids,spans,rendered,audit=segmented_tokens(tok,case,probe=True)
            for condid in ('baseline','canonical_'+ingredient,'attenuate_'+ingredient):
                condition=condition_map[condid]; row=dict(case_id=cid,split='validation',ingredient=ingredient,condition_id=condid,
                       token_ids=ids,spans=spans,tokenization_audit=audit,operations=condition.get('operations',[]),choices={})
                for label,continuation in case['choices'].items():
                    cids=tok.encode(continuation,add_special_tokens=False); assert cids
                    reset(condition,len(ids))
                    logits=model(mx.array(ids+cids[:-1])[None],cache=None)[0].astype(mx.float32)
                    mx.eval(logits)
                    logits=np.asarray(logits)[len(ids)-1:len(ids)+len(cids)-1]
                    peak=logits.max(axis=1); logz=peak+np.log(np.exp(logits-peak[:,None]).sum(axis=1))
                    logps=logits[np.arange(len(cids)),cids]-logz
                    row['choices'][label]=dict(continuation=continuation,token_ids=cids,token_logprobs=logps.tolist(),
                         sum_logprob=float(logps.sum()),mean_logprob=float(logps.mean()),activations=capture())
                sums=np.array([v['sum_logprob'] for v in row['choices'].values()]); peak=float(sums.max())
                logmass=peak+float(np.log(np.exp(sums-peak).sum()))
                row.update(candidate_logmass=logmass,candidate_mass=float(np.exp(logmass)),
                           conditional_choice_probabilities={k:float(np.exp(v['sum_logprob']-logmass)) for k,v in row['choices'].items()})
                output.append(row); f.write(json.dumps(row)+'\n'); f.flush()
            print(json.dumps(dict(event='precision_case_complete',case_id=cid,rows=len(output))),flush=True)
    # Comparison is deliberately delayed until all float32 rows are collected.
    bf16=readrows(args.bf16_results); bf16table={(r['condition_id'],r['case_id']):r for r in bf16}
    for r in output:
        other=bf16table[(r['condition_id'],r['case_id'])]
        assert r['token_ids']==other['token_ids'] and r['operations']==other['operations']
        assert all(v['token_ids']==other['choices'][label]['token_ids'] for label,v in r['choices'].items())
    bf=contrasts(bf16,cases); fp=contrasts(output,cases)
    compared=[]
    for case in cases:
        for prefix in ('canonical','attenuate'):
            cid=case['case_id']; condid=prefix+'_'+case['ingredient']; b=bf[(cid,condid)]; v=fp[(cid,condid)]
            compared.append(dict(case_id=cid,ingredient=case['ingredient'],condition_id=condid,bf16_change=b,float32_change=v,
                                 difference=v-b,absolute_difference=abs(v-b),same_nonzero_sign=(b*v>0),bf16_zero=b==0,float32_zero=v==0))
    with (args.output/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(compared[0])); w.writeheader(); w.writerows(compared)
    def summarize(rs):
        return dict(contrasts=len(rs),mean_absolute_discrepancy=statistics.mean(r['absolute_difference'] for r in rs),
                    maximum_absolute_discrepancy=max(r['absolute_difference'] for r in rs),same_nonzero_sign=sum(r['same_nonzero_sign'] for r in rs),
                    bf16_exact_zero=sum(r['bf16_zero'] for r in rs),float32_exact_zero=sum(r['float32_zero'] for r in rs),
                    mean_bf16_change=statistics.mean(r['bf16_change'] for r in rs),mean_float32_change=statistics.mean(r['float32_change'] for r in rs))
    summary=dict(all=summarize(compared),by_ingredient={n:summarize([r for r in compared if r['ingredient']==n]) for n in INGREDIENTS},
                 by_operation={p:summarize([r for r in compared if r['condition_id'].startswith(p)]) for p in ('canonical','attenuate')},
                 metadata=metadata,rows=len(output),seconds=time.monotonic()-started,
                 output_sha256=sha(args.output/'float32.jsonl'),comparison_sha256=sha(args.output/'comparison.csv'))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(event='precision_complete',rows=len(output),contrasts=len(compared),output=str(args.output))),flush=True)

if __name__=='__main__': main()
