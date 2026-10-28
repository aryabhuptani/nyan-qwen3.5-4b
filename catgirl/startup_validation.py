"""Separate startup protocol: genuine neural introduction, then unsteered task."""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .demo import Conversation, digest, load_candidate, resolve_model
    from .final_evaluation import _archive, _automatic_checks, _stable_model_identity
except ImportError:
    from demo import Conversation, digest, load_candidate, resolve_model
    from final_evaluation import _archive, _automatic_checks, _stable_model_identity


ROOT=Path(__file__).resolve().parent
STARTUP='Hello! Please introduce yourself.'
IDENTITY='What is your name, and what kind of assistant are you?'


def run_case(runner,candidate,spec,startup_max_tokens,identity_probe=False,baseline=False):
    conversation=Conversation(runner,candidate,baseline)
    startup=conversation.reply(STARTUP,startup_max_tokens)
    details={}
    if spec['section']=='long_input':
        system=candidate['protocol']['system_prompt']
        bare=[{'role':'system','content':system},{'role':'user','content':''}]
        with_history=conversation.messages+[{'role':'user','content':''}]
        overhead=len(runner.tokens(with_history))-len(runner.tokens(bare))
        builder_budget=spec['input_tokens']-overhead
        messages,details=_archive(runner,spec,system,builder_budget)
        prompt=messages[-1]['content']
        details.update(requested_total_input_tokens=spec['input_tokens'],
                       archive_builder_budget=builder_budget,
                       startup_history_token_overhead=overhead)
    else:
        prompt=spec['prompt']
    target=conversation.reply(prompt,spec['max_new_tokens'])
    ids=list(runner.tokens(target['messages']))
    if spec['section']=='long_input':
        needle=list(runner.tok.encode(spec['target_record'],add_special_tokens=False))
        offset=next((i for i in range(len(ids)-len(needle)+1) if ids[i:i+len(needle)]==needle),None)
        details.update(target_token_offset=offset,target_token_fraction=offset/len(ids) if offset is not None else None,
                       actual_total_input_tokens=len(ids))
    probe=conversation.reply(IDENTITY,spec.get('identity_probe_max_tokens',128)) if identity_probe else None
    record=dict(prompt_id=spec['id'],section=spec['section'],condition='baseline_startup' if baseline else 'catgirl_v1_startup',
                startup=startup,messages=target['messages'],input_token_ids=ids,
                actual_input_tokens=len(ids),text=target['text'],target=target,
                intervention_scope=target['intervention_scope'],identity_probe=probe,
                persona_applicable=spec.get('persona_applicable',True),quality_checks=spec.get('quality_checks',[]),
                **details)
    record.update(_automatic_checks(spec,target['text']))
    if spec['section']=='long_input':
        record['retrieval_pass']='MARBLE' in target['text']
    if spec['section']=='long_output':
        output_ids=list(runner.tok.encode(target['text'],add_special_tokens=False))
        n=len(output_ids)
        record['output_thirds']=[runner.tok.decode(output_ids[a:b]) for a,b in ((0,n//3),(n//3,2*n//3),(2*n//3,n))]
        record['truncated']=target.get('finish_reason')=='length'
    return record


def specifications(extended,core,case_ids=None,include_extended=True,long_input_tokens=None,long_output=False,max_tokens=None):
    available={case['id']:case for case in extended['cases']}
    if case_ids is not None:
        unknown=set(case_ids)-set(available)
        if unknown:
            raise ValueError('Unknown extended case IDs: '+repr(sorted(unknown)))
        cases=[available[name] for name in case_ids]
    else:
        cases=extended['cases'] if include_extended else []
    result=[dict(case,section='extended',max_new_tokens=max_tokens or case['max_new_tokens']) for case in cases]
    for budget in long_input_tokens or []:
        if budget<256:
            raise ValueError('Long-input budget must be at least 256 tokens')
        case=core['persistence']['long_input']
        result.append(dict(case,id='startup_archive_'+str(budget),section='long_input',
                           input_tokens=budget,max_new_tokens=max_tokens or case['max_new_tokens']))
    if long_output:
        case=core['persistence']['long_output']
        result.append(dict(case,section='long_output',max_new_tokens=max_tokens or case['max_new_tokens']))
    if not result:
        raise ValueError('No startup validation cases selected')
    if len({case['id'] for case in result})!=len(result):
        raise ValueError('Repeated startup case IDs')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',help='Optional model path/ID override; default resolves the exact frozen4-bit revision')
    parser.add_argument('--candidate',default=str(ROOT/'candidates'/'catgirl_v1.json'))
    parser.add_argument('--jobfile','--job',dest='jobfile',type=Path)
    parser.add_argument('--case-ids',nargs='+',help='Extended case IDs, separated by spaces or commas')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--identity-probe',action='store_true')
    parser.add_argument('--long-input-tokens',type=int,nargs='+')
    parser.add_argument('--long-output',action='store_true')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--max-tokens',type=int)
    parser.add_argument('--startup-max-tokens',type=int)
    parser.add_argument('--check-only',action='store_true',help='List and validate selected cases without importing MLX or loading a model')
    args=parser.parse_args()
    job=json.loads(args.jobfile.read_text()) if args.jobfile else {}
    candidate_path,candidate=load_candidate(args.candidate)
    extended_path=ROOT/job.get('extended_battery','evaluation_extended.json')
    core_path=ROOT/job.get('battery_file','evaluation.json')
    extended=json.loads(extended_path.read_text());core=json.loads(core_path.read_text())
    case_ids=[part for name in args.case_ids for part in name.split(',')] if args.case_ids is not None else job.get('case_ids')
    budgets=args.long_input_tokens if args.long_input_tokens is not None else job.get('long_input_tokens',[])
    max_tokens=args.max_tokens if args.max_tokens is not None else job.get('max_tokens')
    if max_tokens is not None and max_tokens<1:
        parser.error('--max-tokens must be positive')
    startup_max=args.startup_max_tokens if args.startup_max_tokens is not None else int(job.get('startup_max_tokens',128))
    if startup_max<1:
        parser.error('startup_max_tokens must be positive')
    identity_probe=args.identity_probe or bool(job.get('identity_probe',False))
    baseline=args.baseline or bool(job.get('baseline',False))
    specs=specifications(extended,core,case_ids,job.get('include_extended',True),budgets,
                         args.long_output or bool(job.get('long_output',False)),max_tokens)
    settings=dict(protocol='neural startup followed by unsteered task and optional unsteered identity probe',
                  blinded_held_out_claim=False,
                  protocol_note='Separate post-exposure startup protocol; no recipe retuning or persona text injection.',
                  candidate_condition_sha256=candidate['condition_sha256'],startup_prompt=STARTUP,
                  startup_max_tokens=startup_max,identity_probe=identity_probe,
                  identity_probe_prompt=IDENTITY if identity_probe else None,baseline=baseline,
                  decoding=candidate['decoding'],specifications=specs,
                  extended_file=str(extended_path),extended_sha256=hashlib.sha256(extended_path.read_bytes()).hexdigest(),
                  core_file=str(core_path),core_sha256=hashlib.sha256(core_path.read_bytes()).hexdigest())
    if args.check_only:
        print(json.dumps(dict(cases=[dict(id=s['id'],section=s['section'],max_new_tokens=s['max_new_tokens']) for s in specs],
                              startup_max_tokens=startup_max,identity_probe=identity_probe,baseline=baseline,
                              protocol_note=settings['protocol_note']),indent=2));return
    model=args.model or job.get('model') or resolve_model(candidate,job.get('cache_dir'))
    settings['loaded_model_path']=str(model)
    protocol_hash=digest(settings)
    output=args.output or Path(job.get('output',ROOT/'results'/((args.jobfile.stem if args.jobfile else 'startup_validation')+'.jsonl')))
    output.parent.mkdir(parents=True,exist_ok=True)
    completed={}
    if output.exists():
        for line in output.read_text().splitlines():
            record=json.loads(line)
            if record.get('protocol_sha256')!=protocol_hash:
                raise ValueError('Startup resume settings differ; use a new output path')
            completed[record['prompt_id']]=record
    if all(spec['id'] in completed for spec in specs):
        print(json.dumps(dict(event='startup_already_complete',output=str(output))));return
    try:
        from .runner import Runner
    except ImportError:
        from runner import Runner
    runner=Runner(str(model))
    identity=_stable_model_identity(runner.metadata)
    for record in completed.values():
        if record.get('model_identity')!=identity:
            raise ValueError('Startup resume model/calibration identity differs')
    try:
        with output.open('a') as handle:
            for spec in specs:
                if spec['id'] in completed:
                    continue
                record=run_case(runner,candidate,spec,startup_max,identity_probe,baseline)
                record.update(protocol_sha256=protocol_hash,protocol_settings=settings,
                              candidate_file=str(candidate_path),neuron_recipe_sha256=candidate['neuron_recipe_sha256'],
                              model_identity=identity,model_metadata=runner.metadata)
                handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush()
                print(json.dumps(dict(event='startup_case',prompt_id=record['prompt_id'],
                                      target_input_tokens=record['actual_input_tokens'],
                                      startup_text=record['startup']['text'],text=record['text'],
                                      identity_probe_text=record['identity_probe']['text'] if record['identity_probe'] else None),
                                 ensure_ascii=False),flush=True)
    finally:
        runner.steer([])


if __name__=='__main__':
    main()
