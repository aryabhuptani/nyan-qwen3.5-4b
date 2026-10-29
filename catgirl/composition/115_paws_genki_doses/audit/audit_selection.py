"""Independent CPU selector reconstruction from numeric labels only; no primary selector import."""
import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--primary-dir',type=Path)
    parser.add_argument('--out-dir',type=Path)
    args=parser.parse_args()
    job=json.loads((HERE/'JOB_discovery.json').read_text())
    cases={c['case_id'] for c in json.loads((HERE/'CASES_discovery.json').read_text())['cases']}
    features=json.loads((HERE/'FEATURE_DEFINITIONS.json').read_text())['feature_order']
    grid={c['condition_id']:c['normalized_doses'] for c in job['conditions']}
    lock=args.labels.parent/'labels_lock.json'
    check(lock.exists(),'Numeric labels require a lock')
    lock_data=json.loads(lock.read_text())
    check(lock_data['labels_sha256'][args.labels.name]==sha(args.labels) and lock_data['total_items']==60,'Label lock')
    with args.labels.open(newline='') as handle:
        records=list(csv.DictReader(handle))
    expected={(condition,case) for condition in grid for case in cases}
    check(len(records)==60 and len({r['item_id'] for r in records})==60,'60 unique item IDs')
    check({(r['condition_id'],r['case_id']) for r in records}==expected,'Full design')
    by_condition={c:[] for c in grid}
    for r in records:
        check(all(r[f] in ('0','1','') for f in features),'binary/unknown')
        v={f:None if r[f]=='' else int(r[f]) for f in features}
        # Explicitly evaluate known-false, unknown, known-true states, independently.
        requirements=[v[f] for f in features[:3]]+[None if v[f] is None else int(v[f]==0) for f in ('assistant_idol_identity','assistant_mascot_identity')]
        clean=0 if any(x==0 for x in requirements) else (1 if all(x==1 for x in requirements) else None)
        sound=v['performed_feline_vocalization']
        clean_sound=0 if clean==0 or sound==0 else (1 if clean==1 and sound==1 else None)
        roles=[v['assistant_idol_identity'],v['assistant_mascot_identity']]
        drift=1 if any(x==1 for x in roles) else (0 if all(x==0 for x in roles) else None)
        v.update(clean_catgirl=clean,clean_catgirl_with_sound=clean_sound,idol_or_mascot=drift)
        by_condition[r['condition_id']].append(v)
    summaries=[]
    for c,dose in grid.items():
        p,g=(Decimal(str(dose[k])) for k in ('paws','genki'))
        s=dict(condition_id=c,cases=3,paws_dose=str(p),genki_dose=str(g),total_added_dose=str(p+g))
        for f in [*features,'clean_catgirl','clean_catgirl_with_sound','idol_or_mascot']:
            values=[v[f] for v in by_condition[c]]
            s[f]=sum(v==1 for v in values)
            s[f+'_zero']=sum(v==0 for v in values)
            s[f+'_unknown']=sum(v is None for v in values)
        summaries.append(s)
    ranking=sorted(summaries,key=lambda s:(-s['clean_catgirl_with_sound'],-s['clean_catgirl'],s['idol_or_mascot'],Decimal(s['total_added_dose']),Decimal(s['genki_dose']),Decimal(s['paws_dose'])))
    candidate=ranking[0]
    baseline=next(s for s in summaries if s['condition_id']=='p0_g0')
    improvement=(candidate['clean_catgirl_with_sound'],candidate['clean_catgirl'])>(baseline['clean_catgirl_with_sound'],baseline['clean_catgirl'])
    result=dict(study=115,independent_reconstruction=True,labels_sha256=sha(args.labels),labels_lock_sha256=sha(lock),script_sha256=sha(Path(__file__)),candidate=candidate,baseline=baseline,confirmation_required=improvement,ranked_condition_ids=[s['condition_id'] for s in ranking])
    if args.primary_dir:
        with (args.primary_dir/'discovery_summary.csv').open(newline='') as handle:
            actual=list(csv.DictReader(handle))
        check(actual==[{k:str(v) for k,v in s.items()} for s in summaries],'Every summary cell must match')
        primary=json.loads((args.primary_dir/'discovery_selection.json').read_text())
        for key in ('candidate','baseline','confirmation_required','ranked_condition_ids'):
            check(primary[key]==result[key],'Selection mismatch: '+key)
        result['all_primary_summary_and_selection_cells_match']=True
    out=args.out_dir or HERE/'audit'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'independent_discovery_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    (out/'independent_discovery_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
