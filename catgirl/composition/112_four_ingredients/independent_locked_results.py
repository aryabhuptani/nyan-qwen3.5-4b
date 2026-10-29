"""Independent post-lock point results; no primary analyzer results are read."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from independent_recompute import NAMES,SIGN,sha,jsonfile,jsonlines,csvrows,final,make_objective_truth,scores,mean

ROOT=Path(__file__).resolve().parent


def writecsv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def label_summary(rows,features):
    answer=[]
    for condition in sorted({r['condition_id'] for r in rows}):
        selected=[r for r in rows if r['condition_id']==condition]
        record=dict(condition_id=condition,n=len(selected))
        for feature in features:
            record[feature]=sum(r.get(feature)=='1' for r in selected)
            record[feature+'_unknown']=sum(r.get(feature) in ('','uncertain',None) for r in selected)
        strict=('sustained_cute_register','explicit_virtual_affection','own_animal_body','female_assistant_selfrole','coherent')
        record['strict_composed_role_coherent']=sum(all(r[f]=='1' for f in strict) for r in selected)
        record['female_feline_coherent']=sum(all(r[f]=='1' for f in ('female_assistant_selfrole','feline_assistant_selfrole','coherent')) for r in selected)
        answer.append(record)
    return answer


def independent_association(discovery,natural,cases):
    details=[]
    for name in NAMES:
        values=[final(r,name) for r in discovery if r['condition_id']=='baseline'];mu=mean(values)
        sd=math.sqrt(mean([(v-mu)**2 for v in values]))
        matching={cid:c for cid,c in cases.items() if c['ingredient']==name}
        raw={r['case_id']:r for r in natural if r['condition_id']=='baseline' and r['case_id'] in matching}
        families=sorted({c['family'] for c in matching.values()})
        effects=[]
        for family in families:
            cells={}
            for s,k in itertools.product((0,1),repeat=2):
                relevant=[cid for cid,c in matching.items() if c['family']==family and int(c['semantic_present'])==s and int(c['keyword_present'])==k]
                cells[s,k]=mean([final(raw[cid],name) for cid in relevant])
            semantic=SIGN[name]*(cells[1,0]-cells[0,0]);words=SIGN[name]*(cells[0,1]-cells[0,0])
            effects.append((semantic,words))
            details.append(dict(ingredient=name,family=family,semantic_signed=semantic,word_signed=words,discovery_final_sd=sd,semantic_discovery_sd=semantic/sd if sd else None))
        semantic=mean([x[0] for x in effects]);word=mean([x[1] for x in effects]);agree=sum(x[0]>0 for x in effects)
        details.append(dict(ingredient=name,family='ALL',semantic_signed=semantic,word_signed=word,discovery_final_sd=sd,
                            semantic_discovery_sd=semantic/sd if sd else None,positive_families=agree,
                            practical_marker=bool(sd and semantic/sd>=.25 and agree>=2)))
    return details


def arm_comparisons(unit_losses):
    """Separate session losses, then equal-case paired differences."""
    result=[]
    domains={key[2:6] for key in unit_losses}
    for domain in sorted(domains):
        arms={}
        for key,loss in unit_losses.items():
            if key[2:6]!=domain:continue
            arms.setdefault(key[0],{}).setdefault(key[6:],[]).append(loss)
        for a,b in itertools.combinations(sorted(arms),2):
            if set(arms[a])!=set(arms[b]):raise ValueError('Independent arm units differ')
            bycase={}
            for unit in arms[a]:
                delta=mean(arms[a][unit])-mean(arms[b][unit]);bycase.setdefault(unit[0],[]).append(delta)
            result.append(dict(task=domain[0],endpoint=domain[1],reviewer=domain[2],metric=domain[3],arm_a=a,arm_b=b,
                               n_cases=len(bycase),estimate=mean([mean(v) for v in bycase.values()])))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'independent_results')
    parser.add_argument('--labels-dir',type=Path,default=ROOT/'results.local/scored')
    args=parser.parse_args();output=args.output
    if output.exists():raise FileExistsError(output)
    lockpath=args.labels_dir/'labels_lock.json';lock=jsonfile(lockpath)
    labels=[];original=None;input_hashes={str(lockpath):sha(lockpath)}
    for mode,file in [('final','labels.csv'),('reviewer_A','labels_reviewer_A.csv'),('reviewer_B','labels_reviewer_B.csv'),('uncertain_zero','labels_uncertain_zero.csv'),('uncertain_one','labels_uncertain_one.csv')]:
        path=args.labels_dir/file
        if sha(path)!=lock['labels_sha256'][file]:raise ValueError('Unlocked behavior labels')
        rows=csvrows(path);input_hashes[str(path)]=sha(path)
        if original is None:original=rows
        for row in rows:row=dict(row);row['reviewer']=mode;labels.append(row)
    paths={key:ROOT/value for key,value in dict(discovery='results.local/discovery_capture/natural.jsonl',
        natural='results.local/validation_natural/natural.jsonl',lexical='results.local/validation_lexical_generation/probe.jsonl',
        natural_cases='CASES_natural_validation.json',lexical_cases='CASES_lexical_validation.json',forecasts='forecasts_normalized.csv').items()}
    for path in paths.values():input_hashes[str(path)]=sha(path)
    discovery=jsonlines(paths['discovery']);natural=jsonlines(paths['natural']);lexical=jsonlines(paths['lexical'])
    cases={c['case_id']:c for c in jsonfile(paths['natural_cases'])['cases']}
    lex_cases={c['case_id']:c for c in jsonfile(paths['lexical_cases'])['cases']}
    thresholds,ntruth,ltruth=make_objective_truth(discovery,natural,lexical,set(cases))
    forecast_rows=csvrows(paths['forecasts'])
    point_scores,unit_losses=scores(forecast_rows,ntruth,ltruth,labels,with_units=True)
    score_rows=[dict(zip(('arm','replicate','task','endpoint','reviewer','metric'),key),**value) for key,value in sorted(point_scores.items())]
    objective=[]
    for task,truth in [('natural',ntruth),('lexical',ltruth)]:
        for (cid,name),actual in sorted(truth.items()):
            objective.append(dict(task=task,case_id=cid,ingredient=name,actual=actual,
                                  intended_ingredient=(cases if task=='natural' else lex_cases)[cid]['ingredient'],
                                  threshold=thresholds[name] if task=='natural' else None))
    features=jsonfile(ROOT/'FEATURE_DEFINITIONS.json')['forecast_feature_order']
    summaries=label_summary(original,features)
    associations=independent_association(discovery,natural,cases)
    comparisons=arm_comparisons(unit_losses)
    # Save independent results before inspecting/reporting their numerical values.
    output.mkdir(parents=True)
    writecsv(output/'forecast_scores.csv',score_rows)
    writecsv(output/'forecast_comparisons.csv',comparisons)
    writecsv(output/'objective_outcomes.csv',objective)
    writecsv(output/'generation_counts.csv',summaries)
    writecsv(output/'natural_associations.csv',associations)
    metadata=dict(study=112,independent=True,labels_lock_verified=True,primary_analysis_results_read=False,
                  objective_units={'natural':len(ntruth),'lexical':len(ltruth)},forecast_score_groups=len(score_rows),
                  input_sha256=input_hashes,runner_sha256=sha(__file__),helper_sha256=sha(ROOT/'independent_recompute.py'),
                  output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'saved':str(output),'groups':len(score_rows),'natural_units':len(ntruth),'lexical_units':len(ltruth),'primary_analyzer_results_read':False}))


if __name__=='__main__':main()
