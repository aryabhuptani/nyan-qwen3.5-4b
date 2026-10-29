"""Post-lock comparison against the immutable independently saved results.

Additional causal CSV fields are recomputed directly from the explicit raw
files, without importing analyze.py. No GPU or study-setting writes.
"""
import argparse
import csv
import json
from pathlib import Path

from independent_recompute import sha,jsonfile,jsonlines,csvrows

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'INDEPENDENT_PRIMARY_FULL_AUDIT.json')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    tolerance=1e-12;errors=[];checked=0
    def equal(a,b,label):
        nonlocal checked
        error=abs(float(a)-float(b));errors.append(error);checked+=1
        if error>tolerance:raise ValueError(f'Independent-primary mismatch {label}: {error}')
    first=ROOT/'independent_results';primary=ROOT/'analysis'
    original_scores={tuple(r[k] for k in ('arm','replicate','task','endpoint','reviewer','metric')):r for r in csvrows(first/'forecast_scores.csv')}
    primary_scores=csvrows(primary/'forecast_scores.csv')
    if len(original_scores)!=len(primary_scores):raise ValueError('Score group count mismatch')
    for row in primary_scores:
        key=tuple(row[k] for k in ('arm','replicate','task','endpoint','reviewer','metric'))
        old=original_scores[key]
        if old['n']!=row['n']:raise ValueError('Forecast sample count mismatch')
        equal(old['score'],row['score'],key)
    truth={(r['task'],r['case_id'],r['ingredient']):float(r['actual']) for r in csvrows(first/'objective_outcomes.csv')}
    seen=set()
    for row in csvrows(primary/'forecast_unit_losses.csv'):
        if row['task']=='natural' or (row['task']=='lexical' and row['metric']=='mae'):
            key=(row['task'],row['case_id'],row['ingredient'])
            equal(truth[key],row['actual'],key);seen.add(key)
    if seen!=set(truth):raise ValueError('Not every independent objective outcome matched')
    original_natural={r['ingredient']:r for r in csvrows(first/'natural_associations.csv') if r['family']=='ALL'}
    for row in csvrows(primary/'natural_effects.csv'):
        if row['scope']=='readout' and row['family']=='ALL':
            original=original_natural[row['ingredient']]
            for old,new in [('semantic_signed','semantic_signed'),('word_signed','keyword_signed'),('discovery_final_sd','discovery_final_sd'),('semantic_discovery_sd','semantic_discovery_sd')]:
                equal(original[old],row[new],('natural',row['ingredient'],new))
    manifest=jsonfile(ROOT/'analysis_inputs.local.json')
    causal_counts={}
    for category in ('lexical','semantic'):
        raw_index={}
        for section in manifest[category]:
            rows=jsonlines(ROOT/section['raw']);bycase={(r['case_id'],r['condition_id']):r for r in rows}
            for r in rows:
                key=(section['scope'],r['case_id'],r['condition_id'])
                raw_index.setdefault(key,[]).append((r,bycase[r['case_id'],'baseline']))
        report=csvrows(primary/(category+'_cases.csv'));causal_counts[category]=len(report)
        if sum(map(len,raw_index.values()))!=len(report):raise ValueError('Raw/report causal grid mismatch')
        for row in report:
            raw,base=raw_index[row['scope'],row['case_id'],row['condition_id']].pop(0)
            if category=='lexical':
                def pref(r,field):return r['choices']['target'][field]-r['choices']['control'][field]
                equal(row['preference'],pref(raw,'mean_logprob'),'lexical preference')
                equal(row['baseline_preference'],pref(base,'mean_logprob'),'baseline lexical preference')
                equal(row['change'],pref(raw,'mean_logprob')-pref(base,'mean_logprob'),'lexical change')
                equal(row['sum_preference_change'],pref(raw,'sum_logprob')-pref(base,'sum_logprob'),'summed lexical change')
            else:
                def odds(r):return r['choices']['yes']['sum_logprob']-r['choices']['no']['sum_logprob']
                equal(row['yes_log_odds'],odds(raw),'semantic odds')
                equal(row['baseline_yes_log_odds'],odds(base),'baseline semantic odds')
                equal(row['change'],odds(raw)-odds(base),'semantic odds change')
                expected=.5 if odds(raw)==0 else int((odds(raw)>0)==bool(int(row['semantic_present'])))
                equal(row['accuracy'],expected,'semantic accuracy')
    references=[first/'provenance.json',first/'forecast_scores.csv',first/'objective_outcomes.csv',first/'natural_associations.csv',
                primary/'forecast_scores.csv',primary/'forecast_unit_losses.csv',primary/'natural_effects.csv',
                primary/'lexical_cases.csv',primary/'semantic_cases.csv',ROOT/'analysis_inputs.local.json']
    result=dict(study=112,passed=True,tolerance=tolerance,maximum_absolute_error=max(errors),numerical_values_checked=checked,
                score_groups=len(primary_scores),unique_primary_objective_outcomes=len(seen),causal_rows_checked=causal_counts,
                independently_saved_results_unchanged=True,settings_or_metrics_changed=False,
                auditor_sha256=sha(__file__),reference_sha256={str(path):sha(path) for path in references})
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reference_sha256'}))


if __name__=='__main__':main()
