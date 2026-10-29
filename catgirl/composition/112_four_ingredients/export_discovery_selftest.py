"""Synthetic-only CPU checks for compact discovery exports."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('export112_discovery',HERE/'export_discovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
NAMES=['genki','hugs','paws','female']

def jsonlines(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
def main():
 with tempfile.TemporaryDirectory(prefix='study112_discovery_synthetic_') as temporary:
  root=Path(temporary);m.ROOT=root
  captures={}
  for kind in ['target','control']:
   d=root/kind;d.mkdir();rows=[]
   for i in range(60):
    activations={n:dict(positions=list(range(4)),pre=[0.,.1,.2,.3],post=[0.,.1,.2,.3]) for n in NAMES}
    rows.append(dict(case_id=f'capture_{i:02}',condition_id='baseline',split='discovery',operations=[],token_ids=[1,2,3,4],spans={'readout_text':[2,4],'readout':[3,4]},activations=activations))
   raw=d/'natural.jsonl';jsonlines(raw,rows)
   m.write_json(d/'natural_metadata.json',dict(ingredients={n:[j+1,j+10,1] for j,n in enumerate(NAMES)},runner_sha256='a'*64,revision='synthetic',down_column_l2={n:1. for n in NAMES}))
   m.write_json(d/'calibration.json',dict(raw_sha256=m.sha(raw),values={'synthetic':True},percentiles=[5,25,50,75,95]))
   captures[kind]=raw
  conditions=[dict(condition_id='baseline',operations=[])]+[dict(condition_id=f'{n}_{f}',operations=[dict(ingredient=n,mode='toward',value=.3,fraction=f,scope='generation',direction=1)]) for n in NAMES for f in [.25,.5,1.]]
  cases=[f'{n}_{i}' for n in NAMES for i in range(3)]
  job=root/'job.json';m.write_json(job,dict(case_ids=cases,conditions=conditions))
  writing=root/'writing';writing.mkdir();raw=writing/'probe.jsonl';rows=[]
  for cond in conditions:
   for case in cases:
    choices={k:dict(continuation=k,token_ids=[1,2],token_logprobs=[-1.,-2.],sum_logprob=-3.,mean_logprob=-1.5,activations={'large_array_not_exported':[0]*100}) for k in ['target','control']}
    rows.append(dict(case_id=case,condition_id=cond['condition_id'],split='discovery',operations=cond['operations'],choices=choices,candidate_mass=2*__import__('math').exp(-3),candidate_logmass=__import__('math').log(2)-3,conditional_choice_probabilities={'target':.5,'control':.5}))
  jsonlines(raw,rows);m.write_json(writing/'probe_metadata.json',dict(job_sha256=m.sha(job),ingredients={n:[j,j,1] for j,n in enumerate(NAMES)},runner_sha256='b'*64))
  selection=root/'selection.json';m.write_json(selection,dict(discovery_writing_sha256=m.sha(raw),ingredients={n:dict(selected_fraction=1.,nonresponsive=True,curves=[dict(fraction=f,case_gains={f'{n}_{i}':0. for i in range(3)},mean_gain=0.,positive_cases=0,eligible=False) for f in [.25,.5,1.]]) for n in NAMES}))
  out=root/'public';args=argparse.Namespace(target_capture=captures['target'],control_capture=captures['control'],writing=raw,job=job,selection=selection,output=out)
  m.export(args)
  def csvrows(path):
   with path.open() as f:return list(csv.DictReader(f))
  assert len(csvrows(out/'target_readout_scalars.csv'))==480
  assert sum(int(r['is_final_readout']) for r in csvrows(out/'target_readout_scalars.csv'))==240
  assert len(csvrows(out/'discovery_writing_choices.csv'))==312
  assert len(csvrows(out/'dose_selection_case_gains.csv'))==36
  assert all(r['eligible']=='False' for r in csvrows(out/'dose_selection_case_gains.csv'))
  assert 'large_array_not_exported' not in (out/'discovery_writing_choices.csv').read_text()
  manifest=m.load(out/'manifest.json')
  for path,h in manifest['export_sha256'].items():assert m.sha(out/path)==h
  for path in out.iterdir():assert '/Users/' not in path.read_text()
  try:m.export(args)
  except AssertionError:pass
  else:raise AssertionError('Existing export overwritten')
 print('Synthetic discovery export passed:120capture cases, scoped scalar positions, all156writing conditions/cases,312candidates,36failed-dose case gains, valid hashes, no raw arrays, overwrite protection. No real outcomes read.')
if __name__=='__main__':main()
