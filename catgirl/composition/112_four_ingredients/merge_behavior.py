"""Merge420 masked-review replies and verify20 duplicate baselines.

Reads outcomes only when explicitly executed after all three jobs complete.
No model inference or scoring. Massive activation arrays remain in raw sources;
compact merged rows retain immutable file/line/hash references.
"""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
JOBS={'factorial':'JOB_validation_factorial16.json','strong':'JOB_validation_strong_singletons.json','random':'JOB_validation_random_four.json'}

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):write_text(p,json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def write_text(p,text):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==text,f'Refusing to overwrite differing file:{p}'
 else:p.write_text(text)
def metadata():
 jobs={key:load(ROOT/'jobs'/name) for key,name in JOBS.items()}
 cases=load(ROOT/'CASES_generation_validation.json')['cases'];ids={c['case_id'] for c in cases}
 for key,job in jobs.items():assert set(job['case_ids'])==ids
 assert len(ids)==20
 assert len(jobs['factorial']['conditions'])==16 and len(jobs['strong']['conditions'])==4
 randomconditions={c['condition_id']:c for c in jobs['random']['conditions']}
 assert set(randomconditions)=={'random_0000','random_1111'}
 assert not randomconditions['random_0000'].get('operations')
 conditions=[dict(condition_id=c['condition_id'],source_job=JOBS[key]) for key in ['factorial','strong'] for c in jobs[key]['conditions']]
 conditions.append(dict(condition_id='random_1111',source_job=JOBS['random']))
 assert len(conditions)==21 and len({c['condition_id'] for c in conditions})==21
 obj=dict(study=112,metadata_only=True,gpu_execution_forbidden=True,
          purpose='Condition IDs solely for prepare_review.py mask; never execute as GPU job.',
          conditions=conditions,case_ids=sorted(ids),expected_review_replies=420,
          duplicate_baseline_check=dict(source_condition='random_0000',reference_condition='0000',expected_pairs=20,comparison='Exact generated text equality per case; excluded from review'),
          source_job_sha256={name:sha(ROOT/'jobs'/name) for name in JOBS.values()},cases_sha256=sha(ROOT/'CASES_generation_validation.json'))
 write_json(ROOT/'BEHAVIOR_REVIEW_JOB.json',obj)
 return jobs,{c['case_id']:c for c in cases}
def read_grid(path,job):
 expected={(c['condition_id'],cid) for c in job['conditions'] for cid in job['case_ids']}
 rows={}
 with Path(path).open() as f:
  for line_number,line in enumerate(f,1):
   if not line.strip():continue
   row=json.loads(line);key=(row['condition_id'],row['case_id'])
   assert key in expected and key not in rows,f'Duplicate/unexpected row in{path}:{line_number}'
   assert isinstance(row.get('text'),str) and 'finish_reason' in row
   rows[key]=(row,line_number)
 assert set(rows)==expected,f'Incomplete grid:{path}: {len(rows)}/{len(expected)}'
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata-only',action='store_true')
 p.add_argument('--factorial',type=Path);p.add_argument('--strong',type=Path);p.add_argument('--random',type=Path)
 p.add_argument('--output',type=Path,default=ROOT/'results.local/review_input.jsonl');args=p.parse_args()
 jobs,cases=metadata()
 if args.metadata_only:
  print(json.dumps(dict(metadata_ready=True,conditions=21,cases=20,expected_replies=420)));return
 assert args.factorial and args.strong and args.random,'All three explicit raw paths required'
 paths={'factorial':args.factorial,'strong':args.strong,'random':args.random}
 grids={key:read_grid(path,jobs[key]) for key,path in paths.items()}
 for cid in cases:
  original=grids['factorial'][('0000',cid)][0]
  duplicate=grids['random'][('random_0000',cid)][0]
  assert original['text']==duplicate['text'],f'Duplicate baseline text mismatch atcase{cid}'
 hashes={key:sha(path) for key,path in paths.items()};merged=[]
 for kind in ['factorial','strong','random']:
  for (condition,cid),(row,line) in sorted(grids[kind].items()):
   if kind=='random' and condition=='random_0000':continue
   case=cases[cid]
   reference=dict(path=str(paths[kind].resolve()),sha256=hashes[kind],line_number=line,
                  condition_id=condition,case_id=cid)
   messages=row.get('messages',case['messages'])
   assert messages==case['messages'],'Unexpected independent-prompt history'
   merged.append(dict(condition_id=condition,case_id=cid,messages=messages,text=row['text'],
                      finish_reason=row['finish_reason'],generated_tokens=row.get('generated_tokens'),
                      activations_reference=reference,source_reference=reference))
 assert len(merged)==420
 write_text(args.output,''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in merged))
 manifest=dict(study=112,review_rows=420,duplicate_baseline_pairs=20,duplicate_baseline_texts_equal=True,
               excluded_condition='random_0000',raw_sha256={str(paths[k].resolve()):v for k,v in hashes.items()},
               merged_sha256=sha(args.output),metadata_sha256=sha(ROOT/'BEHAVIOR_REVIEW_JOB.json'),cases_sha256=sha(ROOT/'CASES_generation_validation.json'))
 write_json(args.output.with_suffix('.manifest.json'),manifest)
 print(json.dumps(dict(review_rows=420,baseline_integrity_verified=20,merged=str(args.output),sha256=sha(args.output))))
if __name__=='__main__':main()
