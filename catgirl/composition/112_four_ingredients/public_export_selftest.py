"""CPU-only public-export fixtures; never reads actual experimental outcomes."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('study112_public_export',HERE/'public_export.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture(directory,kind):
 directory.mkdir();raw=[]
 if kind=='generation':
  conditions=[f'{i:04b}' for i in range(16)]+['strong_genki','strong_hugs','strong_paws','strong_female','random_1111']
  for cond in conditions:
   for i in range(20):raw.append(dict(condition_id=cond,case_id=f'case_{i:02}',messages=[{'role':'user','content':'Synthetic question.'}],text='Synthetic neutral reply.',finish_reason='stop',generated_tokens=4,activations_reference=dict(path='/Users/private-person/secret/run.jsonl',sha256='a'*64,line_number=len(raw)+1,condition_id=cond,case_id=f'case_{i:02}')))
 else:
  for role in ['selected','baseline','best_single']:
   for c in range(3):
    for turn in range(4):raw.append(dict(condition_id=role,comparison_role=role,intervention_condition_id='1000' if role!='baseline' else '0000',case_id=f'conversation_{c}',turn_index=turn,turn=turn+1,mode='multiturn',messages=[{'role':'user','content':'Synthetic turn question.'}],text='Synthetic neutral turn reply.',finish_reason='stop',generated_tokens=5))
   for c in range(3):raw.append(dict(condition_id=role,comparison_role=role,intervention_condition_id='1000' if role!='baseline' else '0000',case_id=f'late_{c}',turn_index=0,turn=1,mode='late_input',messages=[{'role':'user','content':'Synthetic late question.'}],text='Synthetic neutral late reply.',finish_reason='stop',generated_tokens=5))
 rawpath=directory/'raw.jsonl';module.write_lines(rawpath,raw)
 labelrows=[];evidence=[]
 for i,row in enumerate(raw):
  rid=f'{kind}_{i:04}';textsha=__import__('hashlib').sha256(row['text'].encode()).hexdigest()
  labelrows.append(dict(review_id=rid,condition_id=row['condition_id'],case_id=row['case_id'],family='synthetic',split='validation',turn_index=row.get('turn_index',''),text_sha256=textsha,coherent=1,task_relevant=1,female_assistant_selfrole=0,lexical_hugs_cuddles=0))
  rating=dict(review_id=rid,values=[0,0,0,0,0,0,0,1,1,0],vocalization_components=[0,0,0,0],hard_loop=False,truncated=False,evidence='Neutral synthetic answer; no target traits.',notes='')
  evidence.append(dict(review_id=rid,identity=dict(text_sha256=textsha,mechanical_hits={'hug':[]}),reviewer_A=rating,reviewer_B=rating,adjudication=None))
 labels=directory/'labels.csv'
 with labels.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(labelrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(labelrows)
 evidencepath=directory/'label_evidence.json';module.write_json(evidencepath,{'rows':evidence})
 module.write_json(directory/'labels_lock.json',dict(labels_sha256={'labels.csv':module.sha(labels)},evidence_sha256=module.sha(evidencepath)))
 return rawpath,labels,evidencepath

def main():
 with tempfile.TemporaryDirectory(prefix='study112_export_synthetic_') as temporary:
  root=Path(temporary);generation=fixture(root/'generation','generation');persistence=fixture(root/'persistence','persistence')
  analysis=root/'analysis';analysis.mkdir()
  module.write_json(analysis/'summary.json',{'input_hashes':{'/Users/private-person/secret/raw.jsonl':'b'*64},'actual':0})
  (analysis/'natural_cases.csv').write_text('case_id,path,value\nexample,/Users/private-person/secret/raw.jsonl,0\n')
  output=root/'public'
  args=argparse.Namespace(generation_raw=generation[0],generation_labels=generation[1],generation_evidence=generation[2],persistence_raw=persistence[0],persistence_labels=persistence[1],persistence_evidence=persistence[2],analysis_dir=analysis,output=output)
  module.export(args)
  assert len(module.lines(output/'generation_replies.jsonl'))==420
  assert len(module.lines(output/'persistence_replies.jsonl'))==45
  persisted=module.lines(output/'persistence_replies.jsonl')
  assert len({(r['condition_id'],r['case_id'],r['turn_index']) for r in persisted})==45
  assert all(r['labels']['female_assistant_selfrole']==0 for r in module.lines(output/'generation_replies.jsonl'))
  assert all('activations' not in r for r in module.lines(output/'generation_replies.jsonl'))
  for p in output.rglob('*'):
   if p.is_file():assert '/Users/' not in p.read_text()
  manifest=module.load(output/'manifest.json')
  for name,h in manifest['export_sha256'].items():assert module.sha(output/name)==h
  assert manifest['source_sha256']['generation_raw']==module.sha(generation[0])
  assert all(r['text']=='Synthetic neutral reply.' for r in module.lines(output/'generation_replies.jsonl'))
  try:module.export(args)
  except AssertionError:pass
  else:raise AssertionError('Existing export overwrite was allowed')
 print('Synthetic export passed: all420+45 replies retained, turn identities distinct, negative labels retained, hashes valid, private source paths removed, overwrite rejected. No real outputs read.')
if __name__=='__main__':main()
