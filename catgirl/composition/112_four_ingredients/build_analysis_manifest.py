"""Prepare study112 analysis inputs using SOURCE/JOB metadata only.

This script does not open model outputs, behavior labels, forecast probabilities
or prior analysis results. It writes future file paths and a condition index.
Root must lock labels before executing analyze.py or independent_recompute.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    text=json.dumps(obj,indent=2,ensure_ascii=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=text:raise FileExistsError(f'Refuse differing replacement: {path}')
    else:path.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-dir',type=Path,required=True,help='Future joined label directory; no files inside are opened')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis_inputs.local.json')
    parser.add_argument('--conditions-output',type=Path,default=ROOT/'analysis_conditions.json')
    parser.add_argument('--primary-only',action='store_true',help='Omit original-review and uncertainty sensitivity files')
    args=parser.parse_args();output=args.output.resolve();base=output.parent
    def relative(path):
        import os
        return os.path.relpath(Path(path).resolve(),base)
    conditions=[];source_hashes={}
    for name in ('JOB_validation_factorial16.json','JOB_validation_strong_singletons.json','JOB_validation_random_four.json'):
        file=ROOT/'jobs'/name;job=load(file);source_hashes[relative(file)]=sha(file)
        for c in job['conditions']:
            item={'condition_id':c['condition_id'],'operations':c['operations']}
            cid=c['condition_id']
            if len(cid)==4 and set(cid)<={'0','1'}:item.update(mask=[int(bit) for bit in cid],kind='factorial')
            else:item['kind']='control'
            conditions.append(item)
    if len({c['condition_id'] for c in conditions})!=len(conditions):raise ValueError('Duplicate analysis condition ID')
    if len(conditions)!=22:raise ValueError('Expected16 factorial+4strong+2random states')
    write(args.conditions_output,{'study':112,'bit_order':['genki','hugs','paws','female'],'conditions':conditions,
                                  'source_job_sha256':source_hashes})
    features=load(ROOT/'FEATURE_DEFINITIONS.json')['forecast_feature_order']
    primaries=['bubbly_excited_expression','lexical_hugs_cuddles','lexical_paws_tail_fur','female_assistant_selfrole']
    labels=args.labels_dir.resolve()
    label_files={'final':relative(labels/'labels.csv')}
    if not args.primary_only:
        for mode in ('reviewer_A','reviewer_B','uncertain_zero','uncertain_one'):
            label_files[mode]=relative(labels/f'labels_{mode}.csv')
    manifest=dict(study=112,created_without_opening_outcomes=True,
        execution_precondition='Root confirms all behavioral labels locked before outcome analysis',
        labels_lock=relative(labels/'labels_lock.json'),
        discovery={'raw':relative(ROOT/'results.local/discovery_capture/natural.jsonl')},
        natural={'raw':relative(ROOT/'results.local/validation_natural/natural.jsonl'),'cases':relative(ROOT/'CASES_natural_validation.json')},
        lexical=[],semantic=[],
        factorial={'labels_by_reviewer':label_files,'conditions':relative(args.conditions_output),
                   'endpoint_columns':{name:name for name in primaries},'extra_endpoints':[name for name in features if name not in primaries]},
        forecasts_csv=relative(ROOT/'forecasts_normalized.csv'),perturbations=[])
    for category,scopes in [('lexical',('generation','readout')),('semantic',('carrier','readout'))]:
        for scope in scopes:
            for random in (False,True):
                suffix='_random' if random else ''
                directory=ROOT/'results.local'/f'validation_{category}_{scope}{suffix}'
                cases=ROOT/('CASES_lexical_validation.json' if category=='lexical' else 'CASES_natural_validation.json')
                manifest[category].append({'raw':relative(directory/'probe.jsonl'),'cases':relative(cases),'scope':scope})
                section={'raw':relative(directory/'probe.jsonl'),'metadata':relative(directory/'probe_metadata.json'),
                         'label':f'{category}_{scope}{suffix}','coordinate_kind':'random' if random else 'target',
                         'selection':relative(ROOT/'jobs/SELECTED_DOSES.json')}
                if random:section['calibration']=relative(ROOT/'results.local/control_capture/calibration.json')
                manifest['perturbations'].append(section)
    for name,random in [('factorial',False),('strong_singletons',False),('random_four',True)]:
        directory=ROOT/'results.local'/('validation_'+name)
        section={'raw':relative(directory/'generation.jsonl'),'metadata':relative(directory/'generation_metadata.json'),
                 'label':name,'coordinate_kind':'random' if random else 'target','selection':relative(ROOT/'jobs/SELECTED_DOSES.json')}
        if random:section['calibration']=relative(ROOT/'results.local/control_capture/calibration.json')
        manifest['perturbations'].append(section)
    manifest['metadata_only_builder']={'sha256':sha(__file__),'source_job_sha256':source_hashes,
        'note':'No raw output, label, forecast prediction or analysis file was opened. Natural random-channel association and the float32 sensitivity audit are reported separately; main natural analysis uses the four published coordinates.'}
    write(output,manifest)
    print(json.dumps({'manifest':str(output),'conditions':str(args.conditions_output),'condition_count':len(conditions),
                      'review_modes':list(label_files),'validation_outputs_opened':False,'forecast_predictions_opened':False}))


if __name__=='__main__':main()
