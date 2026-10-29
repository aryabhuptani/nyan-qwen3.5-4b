"""Fixed persistence extension; root-only GPU execution after masked-label lock.

Uses actual frozen Probe via AST. Each condition/conversation starts independently.
No persistence outcome selects or tunes a condition.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def selected_conditions(selection,job):
    table={c['condition_id']:c for c in job['conditions']}
    selected=selection['selected_condition_id']; single=selection['best_single_condition_id']
    assert selected in table and len(selected)==4 and set(selected)<={'0','1'}
    assert single in table and single.count('1')==1 and len(single)==4
    assert not table['0000'].get('operations'), 'Baseline must be unsteered'
    roles=[('selected',selected),('baseline','0000'),('best_single',single)]
    for _,cid in roles:
        for op in table[cid].get('operations',[]): assert op['scope']=='generation'
    return [(role,table[cid]) for role,cid in roles]



STRICT_FEATURES=['sustained_cute_register','explicit_virtual_affection','own_animal_body','female_assistant_selfrole']
SELECTION_RULE='Main: maximize count of coherent all-four-strict-trait replies across20cases; tie fewer active bits then lexical ID. If all jointcounts zero choose1111 diagnostic. Singleton: maximize sum of four strict trait positives in coherent replies; tie lexical ID. Uncertainty counts zero for selection only.'


def choose_persistence(rows,case_ids,condition_ids):
    table={(r['condition_id'],r['case_id']):r for r in rows}
    assert len(table)==len(rows)==len(case_ids)*len(condition_ids)
    assert set(table)=={(cond,cid) for cond in condition_ids for cid in case_ids}
    def positive(value): return str(value).strip() in ('1','1.0')
    scores={}
    for cond in condition_ids:
        rs=[table[(cond,cid)] for cid in case_ids]
        coherent=lambda r:positive(r.get('coherent'))
        scores[cond]=dict(joint_count=sum(coherent(r) and all(positive(r.get(k)) for k in STRICT_FEATURES) for r in rs),
                         coherent_strict_trait_sum=sum(sum(positive(r.get(k)) for k in STRICT_FEATURES) for r in rs if coherent(r)),
                         female_feline_coherent_count=sum(coherent(r) and positive(r.get('female_assistant_selfrole')) and positive(r.get('feline_assistant_selfrole')) for r in rs),
                         active_bits=cond.count('1'),cases=len(rs))
    diagnostic=not any(s['joint_count'] for s in scores.values())
    selected='1111' if diagnostic else min(condition_ids,key=lambda c:(-scores[c]['joint_count'],c.count('1'),c))
    singles=[c for c in condition_ids if c.count('1')==1]
    bestsingle=min(singles,key=lambda c:(-scores[c]['coherent_strict_trait_sum'],c))
    return dict(selected_condition_id=selected,best_single_condition_id=bestsingle,
                selection_rule=SELECTION_RULE,strict_features=STRICT_FEATURES,scores=scores,
                selected_diagnostic_no_joint=diagnostic,singleton_diagnostic_no_strict_features=scores[bestsingle]['coherent_strict_trait_sum']==0,
                interpretation='Strict four-trait composed-role selection does not require feline identity; female+feline+coherent counts retained separately.')


def build_selection(args):
    assert args.labels and args.labels_lock and args.labels_lock.exists()
    assert not args.output.exists()
    cases=read(ROOT/'CASES_generation_validation.json')['cases']
    job=read(args.job)
    with args.labels.open() as f: labels=list(csv.DictReader(f))
    labels=[r for r in labels if r.get('reviewer','final')==args.reviewer]
    condition_ids=[c['condition_id'] for c in job['conditions']]
    labels=[r for r in labels if r['condition_id'] in condition_ids and r['case_id'] in {c['case_id'] for c in cases}]
    selected=choose_persistence(labels,[c['case_id'] for c in cases],condition_ids)
    selected.update(labels_sha256=sha(args.labels),labels_lock_sha256=sha(args.labels_lock),
                    job_sha256=sha(args.job),reviewer=args.reviewer,selector_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps(dict(event='persistence_selection_frozen',selected=selected['selected_condition_id'],best_single=selected['best_single_condition_id'],diagnostic=selected['selected_diagnostic_no_joint'],path=str(args.output))))


def selftest():
    from gpu_integrity import extract_probe
    extract_probe(ROOT/'measure.py',{'nn':type('NN',(),{'Module':object}),'mx':None})
    job={'conditions':[{'condition_id':'0000'},{'condition_id':'1111','operations':[{'scope':'generation'}]},
                       {'condition_id':'0001','operations':[{'scope':'generation'}]}]}
    selected=selected_conditions(dict(selected_condition_id='1111',best_single_condition_id='0001'),job)
    assert [r for r,_ in selected]==['selected','baseline','best_single']
    history=[{'role':'system','content':'neutral'}]
    for i in range(4):
        history.append({'role':'user','content':str(i)}); before=[dict(m) for m in history]
        history.append({'role':'assistant','content':'actual'+str(i)})
        assert before[-1]['role']=='user' and len(history)==1+2*(i+1)
    assert history[-1]['content']=='actual3'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--selection',type=Path)
    p.add_argument('--labels-lock',type=Path)
    p.add_argument('--forecast-lock',type=Path,default=ROOT/'forecast_lock.json')
    p.add_argument('--job',type=Path,default=ROOT/'jobs/JOB_validation_factorial16.json')
    p.add_argument('--cases',type=Path,default=ROOT/'CASES_persistence.json')
    p.add_argument('--model',default='/Users/arya/Projects/mechinterp/model-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/'+REVISION)
    p.add_argument('--build-selection',action='store_true',help='CPU-only, root after labels lock; --output is the new selection JSON file')
    p.add_argument('--labels',type=Path)
    p.add_argument('--reviewer',default='final')
    p.add_argument('--run',action='store_true')
    args=p.parse_args(); selftest()
    if args.build_selection:
        assert not args.run
        build_selection(args); return
    corpus=read(args.cases)
    assert len(corpus['multiturn_cases'])==3 and all(len(c['turns'])==4 for c in corpus['multiturn_cases'])
    assert len(corpus['late_prompt_cases'])==3
    if not args.run:
        print(json.dumps(dict(valid=True,ast_extraction=True,history_checks=True,model_loaded=False,validation_outcomes_read=False,
                              expected_rows=45)))
        return
    assert args.selection and args.labels_lock and args.labels_lock.exists() and args.forecast_lock.exists()
    selection=read(args.selection)
    assert selection['labels_lock_sha256']==sha(args.labels_lock), 'Selection must bind to frozen masked/adjudicated label lock'
    assert selection.get('selection_rule'), 'Explicit frozen selection rule required'
    run(args,corpus,selection)


def run(args,corpus,selection):
    import importlib.metadata
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler,make_logits_processors
    from measure import INGREDIENTS,segmented_tokens
    from gpu_integrity import extract_probe
    assert Path(args.model).name==REVISION
    assert sha(ROOT/'measure.py')==read(ROOT/'112_PLAN.json')['input_sha256']['measure.py']
    assert not args.output.exists(), 'Fresh output directory required'
    args.output.mkdir(parents=True)
    job=read(args.job); assert job['revision']==REVISION
    assert selection['job_sha256']==sha(args.job), 'Selection must bind to exact frozen factorial operations'
    comparisons=selected_conditions(selection,job)
    mx.set_default_device(mx.gpu)
    model,tok,config=load(args.model,return_config=True)
    for layer in model.layers:
        for attr in ('gate_proj','up_proj','down_proj'):
            p=getattr(layer.mlp,attr); assert not hasattr(p,'scales') and p.weight.dtype==mx.bfloat16
    Probe=extract_probe(ROOT/'measure.py',{'mx':mx,'nn':nn})
    probes={}
    for name,(layer,neuron,_) in INGREDIENTS.items():
        probe=Probe(model.layers[layer].mlp,neuron); model.layers[layer].mlp=probe; probes[name]=probe
    review_job=dict(revision=REVISION,metadata_only=True,
                    case_ids=[c['case_id'] for c in corpus['multiturn_cases']+corpus['late_prompt_cases']],
                    conditions=[dict(condition_id=role,comparison_role=role,intervention_condition_id=c['condition_id'],operations=c.get('operations',[])) for role,c in comparisons],
                    source_factorial_job_sha256=sha(args.job),selection_sha256=sha(args.selection),
                    identity_policy='UniquecomparisonroleconditionIDs; zero-basedturn_index; actualfrozeninterventionIDinseparatefield')
    (args.output/'review_job.json').write_text(json.dumps(review_job,indent=2)+'\n')
    metadata=dict(revision=REVISION,selection_sha256=sha(args.selection),labels_lock_sha256=sha(args.labels_lock),
                  forecast_lock_sha256=sha(args.forecast_lock),job_sha256=sha(args.job),cases_sha256=sha(args.cases),
                  probe_source_sha256=sha(ROOT/'measure.py'),runner_sha256=sha(__file__),selection=selection,
                  comparison_roles={role:c['condition_id'] for role,c in comparisons},
                  duplicate_condition_roles=len({c['condition_id'] for _,c in comparisons})<3,
                  repetition_penalty=job.get('repetition_penalty',1.1),repetition_context_size=128,
                  temperature=job.get('temperature',0.),seed=job.get('seed',42),
                  scope='Every answer prediction from last current prompt position onward; original history is recaptured unsteered',
                  conversation_policy='Each comparison role and conversation starts independently; actual generated assistant text appended, full history re-prefilled each turn',
                  versions={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers')},hardware=mx.device_info())
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def generate(messages,condition,max_tokens):
        ids,spans,rendered,audit=segmented_tokens(tok,dict(messages=messages))
        operations={o['ingredient']:o for o in condition.get('operations',[])}
        for name,probe in probes.items(): probe.reset(operations.get(name),(len(ids)-1,10**9))
        mx.random.seed(job.get('seed',42)); chunks=[]; last=None; started=time.monotonic()
        for part in stream_generate(model,tok,ids,max_tokens=max_tokens,sampler=make_sampler(job.get('temperature',0.)),
                     logits_processors=make_logits_processors(repetition_penalty=job.get('repetition_penalty',1.1),repetition_context_size=128)):
            chunks.append(part.text); last=part
        activations={}
        for name,probe in probes.items():
            mx.eval([r[1:] for r in probe.records])
            activations[name]=dict(positions=[i for offset,pre,post in probe.records for i in range(offset,offset+len(pre))],
                 pre=[float(v) for _,pre,_ in probe.records for v in np.asarray(pre)],
                 post=[float(v) for _,_,post in probe.records for v in np.asarray(post)])
        return dict(messages=[dict(m) for m in messages],text=''.join(chunks),token_ids=ids,rendered_input=rendered,
                    prompt_tokens=len(ids),generation_prediction_boundary=len(ids)-1,generated_tokens=last.generation_tokens,
                    finish_reason=last.finish_reason,activations=activations,spans=spans,tokenization_audit=audit,
                    seconds=time.monotonic()-started,operations=condition.get('operations',[]))
    count=0; started=time.monotonic()
    with (args.output/'persistence.jsonl').open('x') as f:
        for role,condition in comparisons:
            for case in corpus['multiturn_cases']:
                history=[{'role':'system','content':case['system_message']}]
                for turn,prompt in enumerate(case['turns'],1):
                    history.append({'role':'user','content':prompt})
                    row=generate(history,condition,case['max_tokens_per_turn'])
                    history.append({'role':'assistant','content':row['text']})
                    row.update(case_id=case['case_id'],split=case['split'],family=case['family'],comparison_role=role,
                               condition_id=role,intervention_condition_id=condition['condition_id'],turn=turn,turn_index=turn-1,history_after=[dict(m) for m in history],
                               quality_expected=case['quality_expected'][turn-1],mode='multiturn')
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();count+=1
                    print(json.dumps(dict(event='persistence_reply',role=role,case_id=case['case_id'],turn=turn,rows=count)),flush=True)
            for case in corpus['late_prompt_cases']:
                row=generate([dict(m) for m in case['messages']],condition,case['max_tokens'])
                row.update(case_id=case['case_id'],split=case['split'],family=case['family'],comparison_role=role,
                           condition_id=role,intervention_condition_id=condition['condition_id'],turn=1,turn_index=0,mode='late_input',quality_expected=case['quality_expected'],
                           history_after=row['messages']+[{'role':'assistant','content':row['text']}],placement_note=case['placement_note'])
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();count+=1
                print(json.dumps(dict(event='persistence_reply',role=role,case_id=case['case_id'],turn=1,rows=count)),flush=True)
    assert count==45
    completion=dict(rows=count,raw_sha256=sha(args.output/'persistence.jsonl'),seconds=time.monotonic()-started,metadata_sha256=sha(args.output/'metadata.json'),
                    limit='Steering applied every answer. This does not test persistence after steering is removed. Late input is fixed short filler, not arbitrary long-context robustness.')
    (args.output/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(dict(event='persistence_complete',**completion)),flush=True)

if __name__=='__main__':main()
