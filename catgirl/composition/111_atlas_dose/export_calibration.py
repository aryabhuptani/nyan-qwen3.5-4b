"""Export all exploratory calibration evidence; never alter the dose decision."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FEATURES=['feline_assistant_selfrole','female_assistant_selfrole','performed_feline_vocalization',
          'sustained_cute_register','explicit_virtual_affection','coherent','task_relevant']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path,rows):
    assert not path.exists(), 'Refusing to replace existing export'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

decision=json.loads((ROOT/'dose_selection.json').read_text())
mapping=json.loads((ROOT/'calibration_mapping.local.json').read_text())
raw=ROOT/'results.local/job.local.jsonl'
assert sha(raw)==decision['calibration_outputs_sha256']==mapping['raw_sha256']
outputs={(r['condition'],r['prompt_id']):r for r in map(json.loads,raw.read_text().splitlines())}
assert len(outputs)==120
summaries=[]
for name in ('primary','sensitivity'):
    source=ROOT/('calibration_ratings.local.json' if name=='primary' else 'calibration_sensitivity_ratings.local.json')
    if not source.exists():continue
    if name=='primary':assert sha(source)==decision['calibration_labels_sha256']
    public=ROOT/f'calibration_{name}_ratings.json'
    if public.exists():assert public.read_bytes()==source.read_bytes()
    else:public.write_bytes(source.read_bytes())
    ratings=json.loads(source.read_text())['ratings']
    assert len(ratings)==len({r['review_id'] for r in ratings})==120
    assert {r['review_id'] for r in ratings}==set(mapping['identities'])
    rows=[];tables={m:dict(multiplier=m,coherent=0,hard_loops=0,task_relevant=0,
                          uncertain_cells=0,persona_units=set(),persona_types=set()) for m in (1.,1.5,2.)}
    for r in ratings:
        ident=mapping['identities'][r['review_id']];v=r['values']
        assert len(v)==7 and all(x in (0,1,'uncertain') for x in v)
        assert not r['hard_loop'] or v[5]==0
        t=tables[ident['multiplier']];t['coherent']+=int(v[5]==1);t['task_relevant']+=int(v[6]==1)
        t['hard_loops']+=int(r['hard_loop']);t['uncertain_cells']+=v.count('uncertain')
        if any(x==1 for x in v[:5]):t['persona_units'].add(ident['unit_id'])
        t['persona_types'].update(FEATURES[i] for i in range(5) if v[i]==1)
        output=outputs[(ident['condition_id'],ident['prompt_id'])]
        rows.append(dict(review_id=r['review_id'],**ident,**dict(zip(FEATURES,v)),
                         hard_loop=r['hard_loop'],evidence=r['evidence'],text=output['text'],
                         finish_reason=output['finish_reason']))
    for t in tables.values():
        t['qualifying_units']=len(t['persona_units']);t['distinct_persona_types']=len(t['persona_types'])
        t['persona_units']=sorted(t['persona_units']);t['persona_types']=sorted(t['persona_types'])
        t['would_meet_frozen_gate']=t['coherent']>=36 and t['qualifying_units']>=4 and t['distinct_persona_types']>=2
    export=ROOT/f'calibration_{name}_observations.csv'
    if not export.exists():write_csv(export,rows)
    summaries.append(dict(review=name,ratings_sha256=sha(source),rows=120,counts=list(tables.values()),
                          observations_sha256=sha(export)))
summary=dict(primary_decision_sha256=sha(ROOT/'dose_selection.json'),reviews=summaries,
             interpretation='Primary single-review rule determines stop. Second review is post-selection sensitivity only and cannot unblock confirmation.')
(ROOT/'calibration_review_sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
