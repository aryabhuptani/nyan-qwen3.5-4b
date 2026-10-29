"""Audit the actual frozen measure.py Probe class; GPU only with --run.

AST extraction executes the original hook implementation, not a rewritten copy.
No intervention outcomes or validation cases are used to choose this audit.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_probe(source, environment):
    tree=ast.parse(Path(source).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    klass=next(n for n in run.body if isinstance(n,ast.ClassDef) and n.name=='Probe')
    module=ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[]))
    exec(compile(module,str(source),'exec'),environment)
    return environment['Probe']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model',default='/Users/arya/Projects/mechinterp/model-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/'+REVISION)
    p.add_argument('--cases',type=Path,default=ROOT/'CASES_natural_discovery.json')
    p.add_argument('--calibration',type=Path,default=ROOT/'results.local/discovery_capture/calibration.json')
    p.add_argument('--max-logit-error',type=float,default=.5)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    source=ROOT/'measure.py'
    # Extraction is also checked in dry mode, without importing a GPU library.
    import numpy as np
    NN=type('NN',(),{'Module':object,'silu':staticmethod(lambda x:x)})
    ToyProbe=extract_probe(source,{'nn':NN,'mx':np})
    original=type('Original',(),{'gate_proj':staticmethod(lambda x:x),'up_proj':staticmethod(np.ones_like),'down_proj':staticmethod(lambda x:x)})()
    toy=ToyProbe(original,2); x=np.zeros((1,5,8),dtype=np.float32); x[0,:,2]=[-1.,0.,1.,2.,3.]
    toy.reset(dict(mode='toward',value=2.,fraction=.5,direction=1),(1,4))
    out=toy(x); assert np.array_equal(out[0,:,2],[-1.,1.,1.5,2.,3.])
    toy.reset(dict(mode='toward',value=2.,fraction=.5,direction=1),(1,4))
    chunks=np.concatenate([toy(x[:,:2]),toy(x[:,2:])],axis=1)
    assert np.array_equal(out,chunks), 'Actual extracted hook global-mask toy check'
    toy.reset(); assert np.array_equal(toy(x),x)
    if not args.run:
        print(json.dumps(dict(valid=True,source_sha256=sha(source),ast_extraction=True,actual_hook_cpu_checks=True,model_loaded=False)))
        return
    run(args,source)


def run(args,source):
    import importlib.metadata
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from measure import INGREDIENTS,segmented_tokens
    assert Path(args.model).name==REVISION
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    corpus=json.loads(args.cases.read_text())
    cases=[c for c in corpus['cases'] if c['split']=='discovery'][:2]
    assert len(cases)==2
    cal=json.loads(args.calibration.read_text())
    mx.set_default_device(mx.gpu)
    model,tok,config=load(args.model,return_config=True)
    originals={name:model.layers[layer].mlp for name,(layer,_,_) in INGREDIENTS.items()}
    for mlp in originals.values():
        for attr in ('gate_proj','up_proj','down_proj'):
            p=getattr(mlp,attr); assert not hasattr(p,'scales') and p.weight.dtype==mx.bfloat16
    Probe=extract_probe(source,{'nn':nn,'mx':mx})
    probes={name:Probe(originals[name],neuron) for name,(_,neuron,_) in INGREDIENTS.items()}
    def install(wrapped):
        for name,(layer,_,_) in INGREDIENTS.items():
            model.layers[layer].mlp=probes[name] if wrapped else originals[name]
    def reset(ops,bounds):
        for name,probe in probes.items(): probe.reset(ops.get(name),bounds)
    def forward(ids,cache=None):
        values=model(mx.array(ids)[None],cache=cache)[0,-1].astype(mx.float32)
        mx.eval(values)
        return np.asarray(values)
    def capture():
        result={}
        for name,probe in probes.items():
            mx.eval([r[1:] for r in probe.records])
            positions=np.array([i for offset,pre,post in probe.records for i in range(offset,offset+len(pre))])
            pre=np.concatenate([np.asarray(r[1]) for r in probe.records])
            post=np.concatenate([np.asarray(r[2]) for r in probe.records])
            result[name]=(positions,pre,post)
        return result
    yes,no=[tok.encode(s,add_special_tokens=False) for s in ('Yes','No')]
    assert len(yes)==len(no)==1
    def contrast(logits): return float(logits[yes[0]]-logits[no[0]])
    def compare(a,b):
        return dict(max_abs_logit_difference=float(np.max(np.abs(a-b))),top1_matches=int(a.argmax())==int(b.argmax()),
                    yes_no_logodds_difference=contrast(b)-contrast(a))
    endpoint={}
    for name,(_,_,direction) in INGREDIENTS.items():
        values=cal['values'][name]
        scope='readout_text' if values.get('readout_text',{}).get('count') else 'readout'
        endpoint[name]=values[scope]['quantiles']['95' if direction>0 else '5']
    records=[]; failures=[]
    started=time.monotonic()
    for case in cases:
        ids,spans,rendered,audit=segmented_tokens(tok,case)
        install(False); original=forward(ids)
        install(True); reset({},(0,0)); wrapped=forward(ids); captured=capture()
        exact=np.array_equal(original,wrapped)
        untouched=all(np.array_equal(pre,post) for _,pre,post in captured.values())
        if not exact or not untouched: failures.append(case['case_id']+':unwrapped_identity')
        reset({name:dict(mode='restore') for name in probes},(0,len(ids)))
        restored=forward(ids); noop=np.array_equal(wrapped,restored)
        if not noop: failures.append(case['case_id']+':restore_identity')
        record=dict(case_id=case['case_id'],tokenization_audit=audit,prompt_tokens=len(ids),
                    empty_wrapper_exact=exact,empty_capture_exact=untouched,restore_noop_exact=noop,comparisons=[])
        # A fixed teacher-forced tail tests global positioning beyond the prompt.
        full_ids=ids+tok.encode(' Thank you.',add_special_tokens=False)
        conditions=[('baseline',{},(0,0))]
        for count in (1,4):
            selected=list(INGREDIENTS)[:count]
            ops={name:dict(mode='toward',value=endpoint[name],fraction=.5,direction=INGREDIENTS[name][2]) for name in selected}
            conditions.extend([(f'toward_{count}_lastprompt',ops,(len(ids)-1,len(ids))),
                               (f'toward_{count}_generation',ops,(len(ids)-1,10**9))])
        baseline_full=None; baseline_cached=None
        for condition,ops,bounds in conditions:
            reset(ops,bounds); full=forward(full_ids); full_capture=capture()
            reset(ops,bounds); cache=make_prompt_cache(model)
            # Split immediately before the final original prompt position, so
            # intervention starts in the second call and continues into tail.
            cut=max(1,len(ids)-1)
            forward(full_ids[:cut],cache=cache)
            mx.eval([c.state for c in cache])
            chunked=forward(full_ids[cut:],cache=cache); chunk_capture=capture()
            checks={}
            for name,(positions,pre,post) in full_capture.items():
                active=(positions>=bounds[0])&(positions<bounds[1]) if name in ops else np.zeros_like(positions,dtype=bool)
                outside_exact=np.array_equal(pre[~active],post[~active])
                direction=INGREDIENTS[name][2]
                changed=post-pre
                direction_ok=bool(np.all(changed[active]*direction>=0))
                finite=bool(np.isfinite(post).all())
                checks[name]=dict(outside_mask_exact=outside_exact,directional_change=direction_ok,finite=finite,
                                  changed_positions=int(np.count_nonzero(changed)),max_abs_delta=float(np.max(np.abs(changed))),
                                  activation_dtype='BF16 native, saved as float32')
                if not outside_exact or not direction_ok or not finite: failures.append(case['case_id']+':'+condition+':'+name+':hook')
            comparison=compare(full,chunked)
            cache_pass=comparison['max_abs_logit_difference']<=args.max_logit_error and comparison['top1_matches']
            if not cache_pass: failures.append(case['case_id']+':'+condition+':cache')
            if condition=='baseline': baseline_full,baseline_cached=full,chunked
            effect_full=contrast(full)-contrast(baseline_full)
            effect_cached=contrast(chunked)-contrast(baseline_cached)
            record['comparisons'].append(dict(condition=condition,bounds=list(bounds),operations=ops,
                 full_chunked=comparison,cache_diagnostic_pass=cache_pass,full_intervention_yes_no_effect=effect_full,
                 chunked_intervention_yes_no_effect=effect_cached,effect_difference=effect_cached-effect_full,
                 checks=checks,full_positions={k:v[0].tolist() for k,v in full_capture.items()},
                 chunked_positions={k:v[0].tolist() for k,v in chunk_capture.items()}))
        records.append(record)
    result=dict(passed=not failures,failures=failures,checks=records,source_sha256=sha(source),
                audit_sha256=sha(__file__),cases_sha256=sha(args.cases),calibration_sha256=sha(args.calibration),revision=REVISION,
                versions={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers')},
                maximum_logit_error_threshold=args.max_logit_error,require_top1_match=True,seconds=time.monotonic()-started,
                limitation='Two fixed discovery cases, numeric implementation audit, not behavioral generalization. Cache arithmetic need not be bitwise identical.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='integrity_complete',passed=result['passed'],failures=failures,output=str(args.output))),flush=True)
    if failures: raise RuntimeError('GPU integrity audit failed; inspect saved checks before validation')

if __name__=='__main__': main()
