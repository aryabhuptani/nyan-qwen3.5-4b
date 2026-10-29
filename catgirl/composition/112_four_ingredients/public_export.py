"""Export complete locked study112 results without raw arrays or private paths.

Example (from study112 directory):
 python3 public_export.py --generation-raw results.local/review_input.jsonl \
 --generation-labels results.local/scored/labels.csv \
 --generation-evidence results.local/scored/label_evidence.json \
 --persistence-raw results.local/persistence/persistence.jsonl \
 --persistence-labels results.local/persistence_scored/labels.csv \
 --persistence-evidence results.local/persistence_scored/label_evidence.json \
 --analysis-dir analysis --output public_results

Labels and evidence must be bound by their adjacent labels_lock.json. No model
inference, scoring, selective examples, or outcome filtering occurs here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
PRIVATE_PATH=re.compile(r'(?:/Users/|/home/|/private/|/tmp/|/var/folders/|[A-Za-z]:\\)[^\s\"\'<>]*')
OBJECTIVE_FILES={'natural_cases.csv','natural_effects.csv','lexical_cases.csv','lexical_effects.csv',
 'semantic_cases.csv','semantic_effects.csv','factorial_details.csv','factorial_effects.csv',
 'factorial_counts.csv','forecast_scores.csv','forecast_comparisons.csv','forecast_unit_losses.csv',
 'perturbation_cases.csv','perturbation_summary.csv','summary.json'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_lines(path,rows):Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
def safe_metadata(value):
 if isinstance(value,dict):return {safe_metadata(str(k)):safe_metadata(v) for k,v in value.items()}
 if isinstance(value,list):return [safe_metadata(x) for x in value]
 if isinstance(value,str):
  def clean(m):
   full=m.group();return 'source_'+hashlib.sha256(full.encode()).hexdigest()[:10]+'/'+full.replace('\\','/').rstrip('/').split('/')[-1]
  return PRIVATE_PATH.sub(clean,value)
 return value

def ensure_no_private_text(value):
 # Preserve model text exactly. If it contains a private-host path, stop for an
 # explicit, documented redaction decision rather than silently rewriting it.
 if isinstance(value,str):
  assert not PRIVATE_PATH.search(value),'Private absolute path in verbatim text/messages; explicit redaction review required'
 elif isinstance(value,dict):
  for key,item in value.items():
   ensure_no_private_text(key)
   ensure_no_private_text(item)
 elif isinstance(value,(list,tuple)):
  for item in value:ensure_no_private_text(item)

def turn_index(row):
 if row.get('turn_index') not in ('',None):return int(row['turn_index'])
 if row.get('turn') not in ('',None):return int(row['turn'])-1
 return None

def rowkey(row):return row['condition_id'],row['case_id'],turn_index(row)
def binary(value):
 if value=='':return None
 if value=='uncertain':return 'uncertain'
 if value in ('0','1'):return int(value)
 try:return float(value)
 except ValueError:return value

def check_lock(labels,evidence):
 lockpath=labels.parent/'labels_lock.json';lock=load(lockpath)
 assert lock.get('labels_sha256',{}).get(labels.name)==sha(labels),'Labels are not bound by their lock'
 assert lock.get('evidence_sha256')==sha(evidence),'Evidence changed after label lock'
 return lockpath

def export_dataset(kind,raw_path,labels_path,evidence_path,out,manifest):
 raw=lines(raw_path)
 expected=420 if kind=='generation' else 45
 assert len(raw)==expected,f'{kind}: expected {expected} raw replies, got {len(raw)}'
 raw_index={rowkey(r):r for r in raw};assert len(raw_index)==len(raw),'Raw output identity collision'
 lockpath=check_lock(labels_path,evidence_path)
 with labels_path.open() as f:labels=list(csv.DictReader(f))
 assert len(labels)==expected,f'{kind}: incomplete label coverage'
 label_index={rowkey(r):r for r in labels};assert len(label_index)==len(labels)
 assert set(label_index)==set(raw_index),f'{kind}: labels/raw identity mismatch'
 evidence_obj=load(evidence_path);evidence_rows=evidence_obj.get('rows',evidence_obj.get('replies'))
 assert isinstance(evidence_rows,list)
 evidence={r['review_id']:r for r in evidence_rows};assert len(evidence)==len(evidence_rows)
 assert {r['review_id'] for r in labels}==set(evidence),'Evidence coverage mismatch'
 if kind=='generation':
  assert len({r['condition_id'] for r in raw})==21 and len({r['case_id'] for r in raw})==20
 else:
  roles={r['condition_id'] for r in raw};assert roles=={'selected','baseline','best_single'}
  for role in roles:
   subset=[r for r in raw if r['condition_id']==role]
   assert len(subset)==15
   assert sum(r.get('mode')=='multiturn' for r in subset)==12
   assert sum(r.get('mode')=='late_input' for r in subset)==3
 exported=[];public_evidence=[];source_refs={}
 identity_fields={'review_id','condition_id','case_id','family','split','turn_index','turn','comparison_role','intervention_condition_id','text_sha256','adjudicated'}
 for key in sorted(raw_index,key=lambda k:(k[0],k[1],str(k[2]))):
  row=raw_index[key];label=label_index[key];rid=label['review_id']
  messages=row.get('messages');assert isinstance(messages,list)
  ensure_no_private_text({'messages':messages,'text':row['text']})
  text_hash=hashlib.sha256(row['text'].encode()).hexdigest()
  if label.get('text_sha256'):assert label['text_sha256']==text_hash
  e=evidence[rid]
  recorded=e.get('identity',{}).get('text_sha256',e.get('reply_sha256'))
  if recorded:assert recorded==text_hash
  labels_out={k:binary(v) for k,v in label.items() if k not in identity_fields}
  result=dict(review_id=rid,condition_id=row['condition_id'],case_id=row['case_id'],
   family=label.get('family',row.get('family')),split=label.get('split',row.get('split')),
   messages=messages,text=row['text'],text_sha256=text_hash,finish_reason=row.get('finish_reason'),
   generated_tokens=row.get('generated_tokens'),labels=labels_out,
   evidence_ref=dict(file=f'{kind}_evidence.jsonl',review_id=rid))
  if kind=='persistence':
   result.update(turn_index=turn_index(row),turn=turn_index(row)+1,
     comparison_role=row.get('comparison_role',row['condition_id']),
     intervention_condition_id=row.get('intervention_condition_id'),mode=row.get('mode'),
     prompt_tokens=row.get('prompt_tokens'),generation_prediction_boundary=row.get('generation_prediction_boundary'))
  ref=row.get('source_reference',row.get('activations_reference'))
  if ref:
   h=ref['sha256'];alias='raw_'+h[:16];source_refs[alias]=h
   result['activation_source_ref']=dict(source_id=alias,sha256=h,line_number=ref['line_number'],condition_id=ref.get('condition_id'),case_id=ref.get('case_id'))
  else:
   result['activation_source_ref']=dict(source_id=f'{kind}_raw',sha256=sha(raw_path),identity=dict(condition_id=row['condition_id'],case_id=row['case_id'],turn_index=turn_index(row)))
  exported.append(result)
  # Preserve original reviewer judgments and adjudication, omitting redundant
  # hidden identity/source paths; lexical hits and uncertainty remain available.
  public_evidence.append(safe_metadata(dict(review_id=rid,reviewer_A=e.get('reviewer_A'),reviewer_B=e.get('reviewer_B'),
   adjudication=e.get('adjudication'),mechanical_hits=e.get('identity',{}).get('mechanical_hits'),
   text_sha256=text_hash)))
 write_lines(out/f'{kind}_replies.jsonl',exported);write_lines(out/f'{kind}_evidence.jsonl',public_evidence)
 for alias,path in [('raw',raw_path),('labels',labels_path),('evidence',evidence_path),('labels_lock',lockpath)]:
  manifest['source_sha256'][f'{kind}_{alias}']=sha(path)
 manifest['activation_source_sha256'].update(source_refs)
 manifest['datasets'][kind]=dict(replies=len(exported),all_raw_replies_retained=True,all_labels_retained=True,
  full_actual_histories_retained=True,raw_arrays_published=False)

def export(args):
 output=Path(args.output);assert not output.exists(),'Fresh output directory required'
 optional=[args.persistence_raw,args.persistence_labels,args.persistence_evidence]
 assert not any(optional) or all(optional),'Supply all three persistence inputs together'
 # Validate and build in a sibling staging directory; incomplete exports are
 # never mistaken for a completed published bundle.
 staging=output.with_name(output.name+'.building');assert not staging.exists();staging.mkdir(parents=True)
 manifest=dict(schema_version=1,study=112,datasets={},source_sha256={},activation_source_sha256={},
  limitation='Raw activation arrays omitted; immutable source hashes and row references preserve provenance. All replies, negative outcomes, original reviews and uncertainty are retained.',
  exporter_sha256=sha(__file__))
 export_dataset('generation',args.generation_raw,args.generation_labels,args.generation_evidence,staging,manifest)
 if all(optional):export_dataset('persistence',*optional,staging,manifest)
 else:manifest['datasets']['persistence']=dict(executed_export=False,reason='No locked persistence inputs supplied')
 if args.analysis_dir:
  objective=staging/'objective';objective.mkdir()
  for p in sorted(Path(args.analysis_dir).iterdir()):
   if p.name not in OBJECTIVE_FILES:continue
   manifest['source_sha256']['objective/'+p.name]=sha(p)
   if p.suffix=='.json':write_json(objective/p.name,safe_metadata(load(p)))
   else:
    with p.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    assert all(len(str(v))<50000 for row in rows for v in row.values()),'Unexpected large objective field'
    with (objective/p.name).open('w',newline='') as f:
     writer=csv.DictWriter(f,fieldnames=[safe_metadata(x) for x in fields],lineterminator='\n');writer.writeheader();writer.writerows(safe_metadata(rows))
 for p in staging.rglob('*'):
  if not p.is_file():continue
  # Inspect decoded values, never JSON escape sequences such as a colon
  # followed by an encoded newline, which are not Windows drive paths.
  if p.suffix=='.jsonl':ensure_no_private_text(lines(p))
  elif p.suffix=='.json':ensure_no_private_text(load(p))
  elif p.suffix=='.csv':
   with p.open() as handle:ensure_no_private_text(list(csv.DictReader(handle)))
  else:ensure_no_private_text(p.read_text())
 manifest['export_sha256']={str(p.relative_to(staging)):sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
 write_json(staging/'manifest.json',manifest);staging.rename(output)
 print(json.dumps(dict(export_complete=True,output=str(output),datasets=manifest['datasets'])))

def parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 for kind in ['generation','persistence']:
  for field in ['raw','labels','evidence']:p.add_argument(f'--{kind}-{field}',type=Path,required=kind=='generation')
 p.add_argument('--analysis-dir',type=Path);p.add_argument('--output',type=Path,required=True);return p
if __name__=='__main__':export(parser().parse_args())
