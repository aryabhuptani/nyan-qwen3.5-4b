"""Compact discovery-only export for independently checking ranges and doses.

No validation outputs, forecasts, reviews or model inference are read.

From the study112 directory, after the coordinator authorizes export:
 python3 export_discovery.py \
 --target-capture results.local/discovery_capture/natural.jsonl \
 --control-capture results.local/control_capture/natural.jsonl \
 --writing results.local/discovery_writing/probe.jsonl \
 --job jobs/JOB_discovery_writing.json --selection jobs/SELECTED_DOSES.json \
 --output public_discovery

Each raw file must have its adjacent natural_metadata.json or probe_metadata.json.
Optional calibration.json files adjacent to captures are exported when present.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
ROOT=Path(__file__).resolve().parent
PRIVATE=re.compile(r'(?:/Users/|/home/|/private/|/tmp/|/var/folders/|[A-Za-z]:\\)[^\s\"\'<>]*')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def jsonrows(path):
 with Path(path).open() as f:
  return [(i,json.loads(s)) for i,s in enumerate(f,1) if s.strip()]
def safe(value):
 if isinstance(value,dict):return {safe(str(k)):safe(v) for k,v in value.items()}
 if isinstance(value,list):return [safe(x) for x in value]
 if isinstance(value,str):return PRIVATE.sub(lambda m:'source_'+hashlib.sha256(m.group().encode()).hexdigest()[:10],value)
 return value
def write_json(path,value):Path(path).write_text(json.dumps(safe(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows):
 assert rows
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def metadata(path,phase):
 p=path.with_name(phase+'_metadata.json');data=load(p)
 assert 'ingredients' in data and 'runner_sha256' in data
 return p,data

def capture(kind,path,out,manifest):
 meta_path,meta=metadata(path,'natural');coordinates=meta['ingredients'];rows=[];cases=set();coordinate_cases={}
 for line,r in jsonrows(path):
  assert r['split'] in ('discovery','calibration'),'Discovery-only capture required'
  assert r['case_id'] not in cases;cases.add(r['case_id'])
  assert r['condition_id']=='baseline' and not r.get('operations'), 'Natural capture must be unmodified'
  a,b=r['spans']['readout_text'];fa,fb=r['spans']['readout'];assert fb-fa==1
  positions=set(range(a,b))|{fa}
  for name,values in r['activations'].items():
   layer,neuron,direction=coordinates[name]
   assert len(values['positions'])==len(values['pre'])==len(values['post'])
   lookup={int(p):(float(pre),float(post)) for p,pre,post in zip(values['positions'],values['pre'],values['post'])}
   assert len(lookup)==len(values['positions'])
   assert positions<=lookup.keys()
   coordinate_cases.setdefault(name,set()).add(r['case_id'])
   for pos in sorted(positions):
    pre,post=lookup[pos];assert math.isfinite(pre) and pre==post
    rows.append(dict(dataset=kind,case_id=r['case_id'],condition_id=r['condition_id'],coordinate_name=name,
     layer=layer,neuron=neuron,canonical_direction=direction,position=pos,
     token_id=r['token_ids'][pos],is_readout_text=int(a<=pos<b),is_final_readout=int(pos==fa),
     readout_text_start=a,readout_text_end=b,scalar=pre,post_scalar=post,
     source_line_number=line))
 assert len(cases)==60,'Expected all60 discovery capture cases'
 assert len(coordinates)==4 and all(coordinate_cases.get(n)==cases for n in coordinates)
 write_csv(out/f'{kind}_readout_scalars.csv',rows)
 summaries={}
 for n in coordinates:
  x=[r['scalar'] for r in rows if r['coordinate_name']==n and r['is_readout_text']]
  final=[r['scalar'] for r in rows if r['coordinate_name']==n and r['is_final_readout']]
  summaries[n]=dict(readout_text_count=len(x),final_readout_count=len(final),
   readout_text_mean=statistics.mean(x) if x else None,
   final_readout_mean=statistics.mean(final),
   readout_scope_fallback=not bool(x),
   quantile_instruction='Filter is_readout_text=1 (or is_final_readout=1 if empty); use NumPy percentile default linear method. Threshold quartiles use final readout only. Pooled token values, not case means.')
 write_json(out/f'{kind}_metadata.json',dict(dataset=kind,coordinates=coordinates,revision=meta.get('revision'),
  down_column_l2=meta.get('down_column_l2'),cases_sha256=meta.get('cases_sha256'),runner_sha256=meta['runner_sha256'],
  tokenization=meta.get('tokenization'),summary=summaries))
 manifest['source_sha256'][kind+'_raw']=sha(path);manifest['source_sha256'][kind+'_metadata']=sha(meta_path)
 calibration=path.with_name('calibration.json')
 if calibration.exists():
  cal=load(calibration);assert cal['raw_sha256']==sha(path)
  write_json(out/f'{kind}_original_calibration.json',cal)
  manifest['source_sha256'][kind+'_calibration']=sha(calibration)
 manifest['datasets'][kind]=dict(cases=len(cases),coordinates=len(coordinates),scoped_scalar_rows=len(rows),all_cases_retained=True)


def writing(path,job_path,selection_path,out,manifest):
 meta_path,meta=metadata(path,'probe');job=load(job_path);selection=load(selection_path)
 assert meta.get('job_sha256')==sha(job_path)
 expected={(c['condition_id'],case) for c in job['conditions'] for case in job['case_ids']}
 assert len(expected)==156
 index={};rows=[]
 for line,r in jsonrows(path):
  assert r['split']=='discovery'
  key=r['condition_id'],r['case_id'];assert key in expected and key not in index;index[key]=r
  assert set(r['choices'])=={'target','control'}
  operations=next(c.get('operations',[]) for c in job['conditions'] if c['condition_id']==r['condition_id'])
  assert r.get('operations',[])==operations
  for choice,c in r['choices'].items():
   token_ids=c['token_ids'];logps=c['token_logprobs'];assert len(token_ids)==len(logps)>0
   assert all(math.isfinite(float(v)) for v in logps)
   assert abs(sum(logps)-c['sum_logprob'])<=1e-4
   assert abs(statistics.mean(logps)-c['mean_logprob'])<=1e-5
   rows.append(dict(case_id=r['case_id'],condition_id=r['condition_id'],choice=choice,
    continuation=c['continuation'],token_count=len(token_ids),sum_logprob=c['sum_logprob'],mean_logprob=c['mean_logprob'],
    unconditional_candidate_mass=math.exp(c['sum_logprob']),joint_candidate_mass=r.get('candidate_mass'),
    joint_candidate_logmass=r.get('candidate_logmass'),conditional_choice_probability=r['conditional_choice_probabilities'][choice],
    token_ids_json=json.dumps(token_ids,separators=(',',':')),token_logprobs_json=json.dumps(logps,separators=(',',':')),
    operations_json=json.dumps(operations,separators=(',',':')),source_line_number=line))
 assert set(index)==expected,'Incomplete discovery writing grid'
 assert selection['discovery_writing_sha256']==sha(path)
 write_csv(out/'discovery_writing_choices.csv',rows)
 write_json(out/'discovery_writing_job.json',job)
 write_json(out/'selected_doses.json',selection)
 # All conditions and both candidates remain, including failed calibration doses.
 gains=[]
 for name,details in selection['ingredients'].items():
  for curve in details['curves']:
   for case_id,gain in curve['case_gains'].items():
    gains.append(dict(coordinate_name=name,fraction=curve['fraction'],case_id=case_id,
      target_control_mean_logprob_change=gain,mean_gain=curve['mean_gain'],positive_cases=curve['positive_cases'],
      eligible=curve['eligible'],selected_fraction=details['selected_fraction'],nonresponsive=details['nonresponsive']))
 write_csv(out/'dose_selection_case_gains.csv',gains)
 manifest['source_sha256'].update(discovery_writing_raw=sha(path),discovery_writing_metadata=sha(meta_path),
  discovery_writing_job=sha(job_path),selected_doses=sha(selection_path))
 manifest['datasets']['writing']=dict(cases=len(job['case_ids']),conditions=len(job['conditions']),condition_case_rows=len(index),candidate_rows=len(rows),all_conditions_retained=True,all_failed_doses_retained=True)

def export(args):
 out=Path(args.output);assert not out.exists(),'Fresh output directory required'
 staging=out.with_name(out.name+'.building');assert not staging.exists();staging.mkdir(parents=True)
 manifest=dict(study=112,schema_version=1,source_sha256={},datasets={},exporter_sha256=sha(__file__),
  reproducibility='Capture CSVs retain every discovery case and every canonical token overlapping readout_text plus final-readout token. Writing retains all156condition-case rows and312candidates. No validation outcomes, forecasts or reviews are read. Large full-position/branch activation arrays omitted.',
  method='Endpoint quantiles: NumPy percentile default linear method on pooled readout_text scalars, fallback final readout when empty. Final-readout threshold quartiles use one final token per case. Dose selection uses matched target-control mean-token-logp gains and prespecified fractions; all candidate masses and failed conditions retained.')
 capture('target',args.target_capture,staging,manifest);capture('control',args.control_capture,staging,manifest)
 writing(args.writing,args.job,args.selection,staging,manifest)
 for name in ['measure.py','build_jobs.py','112_PLAN.json']:
  p=ROOT/name
  if p.exists():manifest['source_sha256'][name]=sha(p)
 manifest['export_sha256']={str(p.relative_to(staging)):sha(p) for p in sorted(staging.iterdir()) if p.is_file()}
 for p in staging.iterdir():assert not PRIVATE.search(p.read_text()),'Private path escaped export'
 write_json(staging/'manifest.json',manifest);staging.rename(out)
 print(json.dumps(dict(export_complete=True,output=str(out),datasets=manifest['datasets'])))

def parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 for name in ['target-capture','control-capture','writing','job','selection','output']:p.add_argument('--'+name,type=Path,required=True)
 return p
if __name__=='__main__':export(parser().parse_args())
