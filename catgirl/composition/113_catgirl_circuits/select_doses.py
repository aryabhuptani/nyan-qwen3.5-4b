"""Deterministic discovery-only calibration, never reads heldout outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
NAMES=['paws','hugs','genki','sounds','pride']
TARGETS=['own_animal_body','own_affection','excited_expression','performed_feline_vocalization','self_pride']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def positive(x): return str(x) in ('1','1.0','True')
def select(rows):
    conditions={}
    for row in rows:
        if row.get('split') not in ('discovery',''):
            raise ValueError('Only discovery labels are permitted')
        key=row['condition_id'],row['case_id']
        if key in conditions: raise ValueError('Duplicate row')
        conditions[key]=row
    cases={case for cond,case in conditions if cond=='base'}
    if len(cases)!=4: raise ValueError('Exactly four base discovery cases required')
    fractions={}; audit={}
    for name,target in zip(NAMES,TARGETS):
        base=[conditions['base',case] for case in sorted(cases)]
        base_count=sum(positive(r['coherent']) and positive(r[target]) for r in base)
        candidates=[]
        for fraction in (.25,.5,1.):
            cond=f'{name}_{fraction:g}'
            if {case for condition,case in conditions if condition==cond}!=cases:
                raise ValueError('Incomplete discovery condition '+cond)
            rr=[conditions[cond,case] for case in sorted(cases)]
            coherent=sum(positive(r['coherent']) for r in rr)
            count=sum(positive(r['coherent']) and positive(r[target]) for r in rr)
            candidates.append({'condition_id':cond,'fraction':fraction,'coherent_count':coherent,
                               'coherent_target_positive_count':count,'gain_vs_base':count-base_count,
                               'unknown_target_count':sum(r[target] in ('',None,'null') for r in rr),
                               'eligible':coherent>=2})
        eligible=[c for c in candidates if c['eligible']]
        best=max(eligible,key=lambda c:(c['coherent_target_positive_count'],c['coherent_count'],-c['fraction'])) if eligible else None
        responsive=bool(best and best['gain_vs_base']>0 and base_count<4)
        selected=best['fraction'] if responsive else 1.
        fractions[name]=selected
        audit[name]={'target':target,'base_coherent_target_positive_count':base_count,
                     'candidates':candidates,'selected_fraction':selected,'responsive':responsive,
                     'saturated':base_count==4,'no_eligible_dose':not bool(eligible),
                     'fallback_reference_dose':not responsive}
    return fractions,audit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--labels-lock',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    # The immutable review lock is retained as provenance; review tool verifies its own member hashes.
    lock=json.loads(args.labels_lock.read_text())
    with args.labels.open(newline='') as f: rows=list(csv.DictReader(f))
    fractions,audit=select(rows)
    args.out.mkdir(parents=True,exist_ok=True)
    p=args.out/'SELECTED_FRACTIONS.json'; a=args.out/'DOSE_SELECTION_AUDIT.json'
    if p.exists() or a.exists(): raise FileExistsError('Selection output already exists')
    p.write_text(json.dumps(fractions,indent=2)+'\n')
    a.write_text(json.dumps({'fractions':fractions,'ingredients':audit,'labels_sha256':sha(args.labels),
                            'labels_lock_sha256':sha(args.labels_lock),'labels_lock_content':lock,
                            'selector_sha256':sha(__file__),'rule':'max coherent target positives among doses with>=2 coherent replies; then coherence; then smallest. Require positive gain, otherwise reference1.'},indent=2)+'\n')
    print(json.dumps(fractions))

if __name__=='__main__': main()
