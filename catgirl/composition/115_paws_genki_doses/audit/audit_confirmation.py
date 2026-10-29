"""Independently audit all frozen confirmation summary cells. No dose selection or text reads."""
import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
FEATURES=('coherent','assistant_cat_identity','assistant_female_identity','performed_feline_vocalization','assistant_idol_identity','assistant_mascot_identity','multi_character_game_framing','sustained_cute_register')
MODES=('adjudicated','reviewer_A','reviewer_B','uncertain_zero','uncertain_one')

def check(ok,why):
    if not ok:
        raise ValueError(why)

def load(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def frozen_path(name):
    """Only this missing local lock may resolve to its byte-identical public copy."""
    path=HERE/name
    if not path.exists() and name=='results.local/labels_discovery/labels_lock.json':
        return HERE/'numeric_results/discovery/labels_lock.json'
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-dir',type=Path,required=True)
    parser.add_argument('--primary-dir',type=Path,required=True)
    args=parser.parse_args()
    job=load(HERE/'JOB_confirmation.json')
    case_ids={c['case_id'] for c in load(HERE/'CASES_confirmation.json')['cases']}
    check(sha(HERE/'CASES_confirmation.json')==job['cases_sha256'],'Case lock')
    check(sha(HERE/'analysis/discovery/discovery_selection.json')==job['selection_sha256'],'Selection lock')
    check(len(job['conditions'])==5 and len(case_ids)==6,'5 x 6 comparison')
    expected_conditions={'p0_g0','p0.4_g0.8','p0.1_g0.4','p0.1_g0','p0_g0.4'}
    check({c['condition_id'] for c in job['conditions']}==expected_conditions,'Unchanged five states')
    expected={(c,k) for c in expected_conditions for k in case_ids}
    for name in ('FROZEN_CONFIRMATION.json','CONFIRMATION_ANALYSIS_FREEZE.json'):
        for file,digest in load(HERE/name)['sha256'].items():
            check(sha(frozen_path(file))==digest,'Frozen hash '+file)
    lock=load(args.labels_dir/'labels_lock.json')
    check(lock['total_items']==30,'Label count lock')
    report=dict(study=115,policy='Descriptive confirmation only; no dose reselection.',labels_lock_sha256=sha(args.labels_dir/'labels_lock.json'),job_sha256=sha(HERE/'JOB_confirmation.json'),script_sha256=sha(Path(__file__)),modes={})
    out=HERE/'audit/confirmation'
    out.mkdir(parents=True,exist_ok=True)
    for mode in MODES:
        label_name='labels.csv' if mode=='adjudicated' else 'labels_'+mode+'.csv'
        path=args.labels_dir/label_name
        check(sha(path)==lock['labels_sha256'][label_name],'Label hash '+mode)
        with path.open(newline='') as f:
            rows=list(csv.DictReader(f))
        check(len(rows)==30 and len({r['item_id'] for r in rows})==30,'30 unique item IDs '+mode)
        check({(r['condition_id'],r['case_id']) for r in rows}==expected,'Complete grid '+mode)
        by_condition={c:[] for c in expected_conditions}
        for row in rows:
            check(all(row[f] in ('0','1','') for f in FEATURES),'Invalid rating')
            values={f:None if row[f]=='' else int(row[f]) for f in FEATURES}
            requirements=[values[f] for f in ('coherent','assistant_cat_identity','assistant_female_identity')]
            for f in ('assistant_idol_identity','assistant_mascot_identity'):
                requirements.append(None if values[f] is None else int(values[f]==0))
            clean=0 if 0 in requirements else (None if None in requirements else 1)
            sound=values['performed_feline_vocalization']
            clean_sound=0 if clean==0 or sound==0 else (1 if clean==1 and sound==1 else None)
            roles=[values['assistant_idol_identity'],values['assistant_mascot_identity']]
            drift=1 if 1 in roles else (None if None in roles else 0)
            values.update(clean_catgirl=clean,clean_catgirl_with_sound=clean_sound,idol_or_mascot=drift)
            by_condition[row['condition_id']].append(values)
        summaries=[]
        for condition in job['conditions']:
            name=condition['condition_id']
            paws,genki=(Decimal(str(condition['normalized_doses'][f])) for f in ('paws','genki'))
            summary=dict(condition_id=name,cases=6,paws_dose=str(paws),genki_dose=str(genki),total_added_dose=str(paws+genki))
            for f in (*FEATURES,'clean_catgirl','clean_catgirl_with_sound','idol_or_mascot'):
                cells=[r[f] for r in by_condition[name]]
                summary[f]=cells.count(1)
                summary[f+'_zero']=cells.count(0)
                summary[f+'_unknown']=cells.count(None)
            summaries.append(summary)
        summary_name='confirmation_summary.csv' if mode=='adjudicated' else f'confirmation_{mode}.csv'
        with (args.primary_dir/summary_name).open(newline='') as f:
            primary=list(csv.DictReader(f))
        check(primary==[{k:str(v) for k,v in s.items()} for s in summaries],'Every primary summary cell '+mode)
        with (out/('independent_'+summary_name)).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        report['modes'][mode]=dict(rows=30,every_primary_summary_cell_matches=True,labels_sha256=sha(path),summary_sha256=sha(args.primary_dir/summary_name))
    analysis_lock=load(args.primary_dir/'confirmation_analysis_lock.json')
    check(analysis_lock['labels_lock_sha256']==report['labels_lock_sha256'],'Primary label lock provenance')
    check(analysis_lock['job_sha256']==report['job_sha256'],'Primary job provenance')
    for file,digest in analysis_lock['outputs_sha256'].items():
        check(sha(args.primary_dir/file)==digest,'Primary output lock '+file)
    (out/'independent_confirmation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
