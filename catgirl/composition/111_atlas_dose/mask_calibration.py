"""CPU-only fixed masking and global selection for111 calibration.

The reviewer receives only calibration_review.local.jsonl, the frozen feature
rubric, and CALIBRATION_REVIEW.md; never the mapping, job, plan or source records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value,jsonl=False):
    content=(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in value) if jsonl else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    if path.exists():
        if path.read_text()!=content:raise ValueError('Refusing to overwrite '+str(path))
    else:path.write_text(content)
def inputs():
    plan=json.loads((ROOT/'111_CALIBRATION_PLAN.json').read_text())
    assert sha(ROOT/'job.local.json')==plan['job_sha256']
    job=json.loads((ROOT/'job.local.json').read_text())
    return plan,job

def mask(raw):
    plan,job=inputs()
    expected=[(c['id'],p['id']) for c in job['conditions'] for p in job['prompts']]
    rows={}
    for line in raw.read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line);key=(r['condition'],r['prompt_id'])
        assert key in expected and key not in rows
        source=next(p for p in job['prompts'] if p['id']==key[1])
        assert r['messages']==source['messages']
        if 'max_new_tokens' in r:assert r['max_new_tokens']==source['max_tokens']
        rows[key]=r
    assert len(rows)==len(expected)==120,'Complete120outputs required before masking'
    random.Random(11121).shuffle(expected)
    review=[];mapping={}
    for cond,prompt in expected:
        r=rows[cond,prompt]
        rid=hashlib.sha256(('111calibration/'+cond+'/'+prompt).encode()).hexdigest()[:20]
        review.append(dict(review_id=rid,prompt_id=prompt,messages=r['messages'],text=r['text'],finish_reason=r['finish_reason']))
        mapping[rid]=dict(condition_id=cond,prompt_id=prompt,**plan['condition_map'][cond])
    save(ROOT/'calibration_review.local.jsonl',review,jsonl=True)
    save(ROOT/'calibration_mapping.local.json',dict(plan_sha256=sha(ROOT/'111_CALIBRATION_PLAN.json'),raw_sha256=sha(raw),review_sha256=sha(ROOT/'calibration_review.local.jsonl'),identities=mapping))
    print(json.dumps(dict(masked_rows=120,review_sha256=sha(ROOT/'calibration_review.local.jsonl'))))

def choose(ratings_path):
    plan,job=inputs();mapping=json.loads((ROOT/'calibration_mapping.local.json').read_text())
    assert mapping['plan_sha256']==sha(ROOT/'111_CALIBRATION_PLAN.json')
    assert mapping['review_sha256']==sha(ROOT/'calibration_review.local.jsonl')
    value=json.loads(ratings_path.read_text());ratings=value['ratings'];ids=mapping['identities']
    assert len(ratings)==len({r['review_id'] for r in ratings})==120
    assert {r['review_id'] for r in ratings}==set(ids)
    table={m:dict(multiplier=m,coherent_replies=0,uncertain_feature_cells=0,persona_units=set(),persona_types=set(),replies=0) for m in plan['multipliers']}
    for r in ratings:
        vals=r['values'];assert len(vals)==7
        assert all((type(v) is int and v in [0,1]) or v=='uncertain' for v in vals)
        assert isinstance(r['evidence'],str) and r['evidence'].strip()
        assert isinstance(r['hard_loop'],bool)
        assert not r['hard_loop'] or vals[5]==0
        v=[0 if x=='uncertain' else x for x in vals]
        ident=ids[r['review_id']];t=table[ident['multiplier']];t['replies']+=1;t['coherent_replies']+=v[5]
        t['uncertain_feature_cells']+=sum(x=='uncertain' for x in vals)
        if any(v[:5]):t['persona_units'].add(ident['unit_id'])
        for n,x in enumerate(v[:5]):
            if x:t['persona_types'].add(plan['persona_features'][n])
    for t in table.values():
        assert t['replies']==40
        t['persona_units']=sorted(t['persona_units']);t['persona_types']=sorted(t['persona_types'])
        t['qualifying_units']=len(t['persona_units']);t['distinct_persona_types']=len(t['persona_types'])
        t['coherence_eligible']=t['coherent_replies']>=36
        t['primary_gate']=t['coherence_eligible'] and t['qualifying_units']>=4 and t['distinct_persona_types']>=2
    primary=[m for m in plan['multipliers'] if table[m]['primary_gate']]
    coherent=[m for m in plan['multipliers'] if table[m]['coherence_eligible']]
    if primary:selection=min(primary);mode='qualified'
    elif coherent:selection=min(coherent,key=lambda m:(-table[m]['qualifying_units'],m));mode='fallback_no_domain_feasibility'
    else:selection=None;mode='stop_no_coherent_dose'
    result=dict(rule_status=mode,selected_multiplier=selection,confirmatory_allowed=mode=='qualified',selection_counts=list(table.values()),calibration_plan_sha256=sha(ROOT/'111_CALIBRATION_PLAN.json'),review_sha256=mapping['review_sha256'],calibration_labels_sha256=sha(ratings_path),calibration_outputs_sha256=mapping['raw_sha256'],interpretation='Exploratory calibration only. Proceed to fresh111forecasts/validation ONLY if rule_status is qualified. Fallback is diagnostic and blocks confirmation. No per-unit tuning.')
    save(ROOT/'dose_selection.json',result)
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--raw',type=Path,help='Completed120-row calibration output JSONL; writes masked review/mapping')
    g.add_argument('--ratings',type=Path,help='Independent complete masked review; writes fixed global selection')
    a=p.parse_args()
    if a.raw:mask(a.raw)
    else:choose(a.ratings)
if __name__=='__main__':main()
