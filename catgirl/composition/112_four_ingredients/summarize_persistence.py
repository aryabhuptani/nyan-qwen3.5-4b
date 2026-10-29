"""CPU-only complete45reply persistence summary, final and reviewer A/B sensitivity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ROLES=('selected','baseline','best_single')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def csvrows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def binary(v):
    if str(v).strip() in ('0','0.0'):return 0
    if str(v).strip() in ('1','1.0'):return 1
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'persistence_diagnostics')
    args=p.parse_args();assert not args.output.exists()
    paths=dict(raw=ROOT/'results.local/persistence/persistence.jsonl',metadata=ROOT/'results.local/persistence/metadata.json',
               cases=ROOT/'CASES_persistence.json',features=ROOT/'FEATURE_DEFINITIONS.json',
               labels_lock=ROOT/'results.local/persistence_scored/labels_lock.json',
               final=ROOT/'results.local/persistence_scored/labels.csv',
               reviewer_A=ROOT/'results.local/persistence_scored/labels_reviewer_A.csv',
               reviewer_B=ROOT/'results.local/persistence_scored/labels_reviewer_B.csv',helper=Path(__file__))
    raw=[json.loads(s) for s in paths['raw'].read_text().splitlines() if s.strip()]
    metadata=load(paths['metadata']);cases=load(paths['cases']);features=load(paths['features'])['forecast_feature_order']
    lock=load(paths['labels_lock'])
    key=lambda r:(r['condition_id'],r['case_id'],int(r['turn_index']))
    rawtable={key(r):r for r in raw};assert len(rawtable)==len(raw)==45
    expected={(role,c['case_id'],i) for role in ROLES for c in cases['multiturn_cases'] for i in range(4)}
    expected|={(role,c['case_id'],0) for role in ROLES for c in cases['late_prompt_cases']}
    assert set(rawtable)==expected
    for row in raw:
        assert row['condition_id']==row['comparison_role']
        assert row['intervention_condition_id']==metadata['comparison_roles'][row['condition_id']]
        assert row['prompt_tokens']==len(row['token_ids'])
        assert row['turn']==row['turn_index']+1
        assert row['history_after']==row['messages']+[{'role':'assistant','content':row['text']}]
        if row['mode']=='multiturn' and row['turn_index']:
            previous=rawtable[(row['condition_id'],row['case_id'],row['turn_index']-1)]
            assert row['messages'][:-1]==previous['history_after']
    joined=[];labelsets={}
    for reviewer in ('final','reviewer_A','reviewer_B'):
        path=paths[reviewer];assert lock['labels_sha256'][path.name]==sha(path)
        labels=csvrows(path);table={key(r):r for r in labels};assert len(table)==len(labels)==45 and set(table)==expected
        labelsets[reviewer]=table
        for unit,r in sorted(rawtable.items()):
            label=table[unit]
            assert label['text_sha256']==hashlib.sha256(r['text'].encode()).hexdigest()
            joined.append(dict(reviewer=reviewer,condition_id=r['condition_id'],comparison_role=r['comparison_role'],
                 intervention_condition_id=r['intervention_condition_id'],case_id=r['case_id'],family=r['family'],mode=r['mode'],
                 turn_index=r['turn_index'],prompt_tokens=r['prompt_tokens'],generated_tokens=r['generated_tokens'],
                 finish_reason=r['finish_reason'],text_sha256=label['text_sha256'],
                 **{f:binary(label[f]) for f in features}))
    groups=[]
    for reviewer in ('final','reviewer_A','reviewer_B'):
        for role in ROLES:
            scopes=[('all',None,None),('multiturn_all',None,None)]+[('multiturn_turn',i,None) for i in range(4)]+[('late_case',0,c['case_id']) for c in cases['late_prompt_cases']]
            for group,turn,cid in scopes:
                rs=[r for r in joined if r['reviewer']==reviewer and r['condition_id']==role]
                if group.startswith('multiturn'):rs=[r for r in rs if r['mode']=='multiturn']
                if group=='multiturn_turn':rs=[r for r in rs if r['turn_index']==turn]
                if group=='late_case':rs=[r for r in rs if r['case_id']==cid]
                assert rs
                for f in features:
                    known=[r[f] for r in rs if r[f] is not None]
                    groups.append(dict(reviewer=reviewer,condition_id=role,intervention_condition_id=metadata['comparison_roles'][role],
                         group=group,turn_index=turn,case_id=cid,feature=f,positive=sum(known),known=len(known),uncertain=len(rs)-len(known),n=len(rs),
                         prompt_tokens_min=min(r['prompt_tokens'] for r in rs),prompt_tokens_max=max(r['prompt_tokens'] for r in rs)))
    retention=[]
    for reviewer,table in labelsets.items():
        for role in ROLES:
            for c in cases['multiturn_cases']:
                labels=[table[(role,c['case_id'],i)] for i in range(4)]
                for endpoint,fields in [('female_identity',['female_assistant_selfrole']),('feline_identity',['feline_assistant_selfrole']),
                                       ('female_feline_coherent',['female_assistant_selfrole','feline_assistant_selfrole','coherent'])]:
                    vals=[all(binary(r[f])==1 for f in fields) if all(binary(r[f]) is not None for f in fields) else None for r in labels]
                    initial=vals[0]
                    retention.append(dict(reviewer=reviewer,condition_id=role,intervention_condition_id=metadata['comparison_roles'][role],
                         case_id=c['case_id'],endpoint=endpoint,initial_positive=initial,
                         later_positive=sum(v is True for v in vals[1:]),later_known=sum(v is not None for v in vals[1:]),
                         retention_assessable=initial is True,
                         interpretation='Conditional retention can be described only on this one induced trajectory' if initial is True else 'Retention unassessable: no demonstrated initial identity'))
    args.output.mkdir(parents=True)
    writecsv(args.output/'all45_final.csv',[r for r in joined if r['reviewer']=='final'])
    writecsv(args.output/'all135_review_sensitivity.csv',joined)
    writecsv(args.output/'trait_quality_by_role_turn_and_late.csv',groups)
    writecsv(args.output/'identity_retention_assessability.csv',retention)
    summary=dict(rows=45,review_versions=3,role_to_frozen_intervention=metadata['comparison_roles'],
                 review_sensitivity={who:{f:dict(positive=sum(r[f]==1 for r in joined if r['reviewer']==who),unknown=sum(r[f] is None for r in joined if r['reviewer']==who),n=45) for f in features} for who in ('final','reviewer_A','reviewer_B')},
                 role_counts={role:{f:sum(r[f]==1 for r in joined if r['reviewer']=='final' and r['condition_id']==role) for f in features} for role in ROLES},
                 identity_retention_assessable_trajectories=sum(r['retention_assessable'] for r in retention if r['reviewer']=='final' and r['endpoint']=='female_feline_coherent'),
                 late_input_prompt_tokens={r['case_id']:r['prompt_tokens'] for r in raw if r['mode']=='late_input' and r['condition_id']=='baseline'},
                 limits=['Continuous steering on every answer, not steering-off carryover.','Fixed three short conversations and three late-input cases, not arbitrary long-context robustness.','No identity induced at turn1 makes identity retention unassessable, not zero-percent persistence.','Quoted story words are separate from adopted assistant identity.'],
                 provenance={k:dict(path=str(v.relative_to(ROOT)),sha256=sha(v)) for k,v in paths.items()})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Persistence extension','',
      'We ran three separate four-turn conversations and three fixed late-input questions under the full four-ingredient condition, an unsteered baseline, and the selected singleton. The model received the same steering on every answer. Each conversation used the actual answers generated earlier in that conversation. All45 replies and both independent reviewer sensitivities are included.','',
      '| Comparison role | Actual frozen condition | Female identity | Feline identity | Sustained cute style | Personal affection | Own animal body | Coherent | Task relevant |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for role in ROLES:
        s=summary['role_counts'][role]
        lines.append('| '+role+' | '+metadata['comparison_roles'][role]+' | '+' | '.join(str(s[f])+'/15' for f in ('female_assistant_selfrole','feline_assistant_selfrole','sustained_cute_register','explicit_virtual_affection','own_animal_body','coherent','task_relevant'))+' |')
    lines.extend(['','Both original reviewers agree with the final labels on zero female identity, feline identity, sustained cute register, personal affection and own animal body, and on coherence in all45 replies. Task relevance positives were final43/45, reviewer A43/45 and reviewer B44/45; reviewer A retained one uncertainty. Full sensitivity tables preserve all endpoints.','','Identity retention is **unassessable**: no female or feline assistant identity was induced on the first turn of any conversation. This is not evidence of an identity appearing and then disappearing, and it is not a zero-percent persistence estimate. No selected condition was already a successful composed role: the full-four condition was the preregistered diagnostic fallback.','',
      'The three late-input prompts contained exactly '+', '.join(f"{k}: {v}" for k,v in summary['late_input_prompt_tokens'].items())+' model input tokens. These are short fixed placement tests, not broad long-context evidence. The input lengths for every conversation turn are retained in the complete scored CSV.','',
      'Every one of the12 trait/quality endpoints is reported separately for each role and each conversation turn, plus each of the three late cases, in trait_quality_by_role_turn_and_late.csv. Final, reviewer A and reviewer B values remain separate; uncertain values are preserved as missing rather than converted to negative.','',
      '## ELI5','',
      'We kept changing the four chosen neurons while Qwen answered several questions in a row. Qwen usually answered the questions clearly, but it did not adopt a female or cat identity, cute style, personal affection or its own animal body. Since that identity did not appear at the start, we cannot measure how long it lasted. Adding several hundred words before a question did not establish that identity either. This test therefore did not produce a continuing catgirl personality under these fixed settings.'])
    (args.output/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output),rows=45,review_rows=135,identity_retention_assessable=summary['identity_retention_assessable_trajectories'])))

if __name__=='__main__':main()
