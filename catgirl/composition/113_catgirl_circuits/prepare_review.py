"""Immutable condition-masked review, independent reviews, adjudication and public export.

Only supplied generation, case, rubric, and review files are read. No forecasts.
"""
import argparse,csv,hashlib,io,json,random,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
FEATURES=json.loads((HERE/'FEATURE_DEFINITIONS.json').read_text())['feature_order']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(p,text):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==text,f'Refusing changed artifact {p}'
 else:p.write_text(text)
def save(p,o):write(p,json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def csvsave(p,rows):
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);write(p,s.getvalue())
def lexical(text):
 pats={'lexical_body':r'\b(?:paws?|tails?|fur|furry)\b','lexical_affection':r'\b(?:hugs?|hugged|hugging|cuddles?|cuddled|cuddling|snuggles?|snuggled|snuggling|nuzzles?|nuzzled|nuzzling)\b','lexical_feline_sound':r'\b(?:meows?|meowed|meowing|mews?|mewed|mewing|ny+a+|purrs?|purred|purring)\b','lexical_pride':r'\b(?:pride|proud|proudly)\b'}
 return {k:int(bool(re.search(p,text,re.I))) for k,p in pats.items()}
def mask(a):
 cases={c['case_id']:c for c in load(a.cases)['cases']};conditions={c['condition_id'] for c in load(a.job)['conditions']};rows={}
 for p in a.raw:
  for r in lines(p):
   key=(r['condition_id'],r['case_id']);assert key not in rows and key[0] in conditions and key[1] in cases
   assert r['messages']==cases[key[1]]['messages'],'Raw context differs from frozen case';assert isinstance(r['text'],str);rows[key]=r
 assert set(rows)=={(c,k) for c in conditions for k in cases},'Incomplete or extra output grid'
 keys=sorted(rows);random.Random(a.seed).shuffle(keys);mapping={};batches=[];out=Path(a.out_dir)
 for start in range(0,len(keys),a.batch_size):
  batch=[]
  for key in keys[start:start+a.batch_size]:
   r=rows[key];c=cases[key[1]];item=hashlib.sha256(f'113/{a.tag}/{key[0]}/{key[1]}'.encode()).hexdigest()[:20];assert item not in mapping
   batch.append(dict(item_id=item,messages=r['messages'],text=r['text'],quality_criteria=c['quality_criteria'],finish_reason=r.get('finish_reason')))
   mapping[item]=dict(condition_id=key[0],case_id=key[1],family_id=c['family_id'],split=c['split'],text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),lexical=lexical(r['text']))
  p=out/f'batch_{start//a.batch_size+1:02}.jsonl';write(p,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in batch));batches.append(p)
 save(out/'mapping.local.json',dict(study=113,tag=a.tag,feature_order=FEATURES,identities=mapping,raw_sha256={str(p):sha(p) for p in a.raw},cases_sha256=sha(a.cases),job_sha256=sha(a.job),rubric_sha256=sha(HERE/'REVIEW_RUBRIC.md'),features_sha256=sha(HERE/'FEATURE_DEFINITIONS.json'),batch_sha256={p.name:sha(p) for p in batches}))
 print(json.dumps(dict(rows=len(rows),batches=len(batches))))
def inputs(a):
 d=Path(a.out_dir);m=load(d/'mapping.local.json');assert m['feature_order']==FEATURES;examples={}
 assert m['rubric_sha256']==sha(HERE/'REVIEW_RUBRIC.md') and m['features_sha256']==sha(HERE/'FEATURE_DEFINITIONS.json'),'Rubric changed'
 for name,h in m['batch_sha256'].items():
  assert sha(d/name)==h
  for r in lines(d/name):assert r['item_id'] not in examples;examples[r['item_id']]=r
 assert set(examples)==set(m['identities']);reviews=[];files={}
 for who,paths in [('A',a.review_a),('B',a.review_b)]:
  ratings={}
  for p in paths:
   o=load(p);assert isinstance(o.get('reviewer_id'),str)
   for r in o['items']:
    i=r['item_id'];assert i in examples and i not in ratings
    validate(r);ratings[i]=r
   files[f'{who}/{p}']=sha(p)
  assert set(ratings)==set(examples),f'Incomplete review {who}';reviews.append(ratings)
 return d,m,examples,reviews,files
def validate(r):
 assert set(r['ratings'])==set(FEATURES)
 assert all(v is None or type(v) is int and v in (0,1) for v in r['ratings'].values())
 assert isinstance(r['evidence'],dict)
 for f,v in r['ratings'].items():
  if v in (1,None):assert isinstance(r['evidence'].get(f),str) and r['evidence'][f].strip(),f'Missing evidence {f}'
def adjudicate(a):
 d,m,ex,reviews,files=inputs(a);disputes=[]
 for i,e in ex.items():
  rr=[r[i] for r in reviews];fields=[f for f in FEATURES if rr[0]['ratings'][f]!=rr[1]['ratings'][f] or rr[0]['ratings'][f] is None]
  if fields:disputes.append(dict(**e,disputed_features=fields,review_1=rr[0],review_2=rr[1]))
 lock=dict(mapping_sha256=sha(d/'mapping.local.json'),review_sha256=files,total_items=len(ex),disputed_items=len(disputes),disputed_cells=sum(len(r['disputed_features']) for r in disputes));save(d/'reviews_lock.json',lock)
 for j in range(0,len(disputes),a.batch_size):save(d/f'adjudication_{j//a.batch_size+1:02}.json',dict(feature_order=FEATURES,items=disputes[j:j+a.batch_size]))
 print(json.dumps(lock))
 if a.command!='join':return
 resolved={};byid={r['item_id']:r for r in disputes}
 for p in a.adjudications:
  for r in load(p)['items']:
   i=r['item_id'];assert i in byid and i not in resolved;validate(r)
   for f in FEATURES:
    if f not in byid[i]['disputed_features']:assert r['ratings'][f]==reviews[0][i]['ratings'][f]
   resolved[i]=r
 assert set(resolved)==set(byid),'Incomplete adjudication'
 target=Path(a.labels_dir);outputs={};evidence=[]
 for mode in ['adjudicated','reviewer_A','reviewer_B','uncertain_zero','uncertain_one']:
  rows=[]
  for i in sorted(ex):
   pair=[r[i] for r in reviews];final=resolved.get(i,pair[0]);ratings=dict(final['ratings'])
   if mode=='reviewer_A':ratings=dict(pair[0]['ratings'])
   elif mode=='reviewer_B':ratings=dict(pair[1]['ratings'])
   elif mode.startswith('uncertain_'):
    bound=int(mode.endswith('one'));ratings={f:bound if pair[0]['ratings'][f] is None or pair[1]['ratings'][f] is None or ratings[f] is None else ratings[f] for f in FEATURES}
   identity=m['identities'][i]
   rows.append(dict(item_id=i,**{k:v for k,v in identity.items() if k!='lexical'},**identity['lexical'],**{f:'' if v is None else v for f,v in ratings.items()},adjudicated=int(i in resolved)))
  name='labels.csv' if mode=='adjudicated' else f'labels_{mode}.csv';p=target/name;csvsave(p,rows);outputs[name]=sha(p)
 for i in sorted(ex):evidence.append(dict(item_id=i,identity=m['identities'][i],reviewer_A=reviews[0][i],reviewer_B=reviews[1][i],adjudication=resolved.get(i)))
 save(target/'label_evidence.json',dict(feature_order=FEATURES,items=evidence));save(target/'labels_lock.json',dict(**lock,adjudication_sha256={str(p):sha(p) for p in a.adjudications},labels_sha256=outputs,evidence_sha256=sha(target/'label_evidence.json')))
 # Compact readable public output keeps all replies, complete contexts, and ratings.
 public=[]
 for i in sorted(ex):
  r=dict(ex[i],**m['identities'][i]);r['ratings']=resolved.get(i,reviews[0][i])['ratings'];public.append(r)
 write(target/'replies_and_labels.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in public))
 csvsave(target/'replies_and_labels.csv',[dict(item_id=r['item_id'],condition_id=r['condition_id'],case_id=r['case_id'],family_id=r['family_id'],messages=json.dumps(r['messages'],ensure_ascii=False),text=r['text'],**{f:'' if v is None else v for f,v in r['ratings'].items()}) for r in public])
 save(target/'export_manifest.json',dict(labels_lock_sha256=sha(target/'labels_lock.json'),replies_jsonl_sha256=sha(target/'replies_and_labels.jsonl'),replies_csv_sha256=sha(target/'replies_and_labels.csv'),rows=len(public)))
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('mask');q.add_argument('--raw',type=Path,action='append',required=True);q.add_argument('--cases',type=Path,required=True);q.add_argument('--job',type=Path,required=True);q.add_argument('--tag',default='factorial');q.add_argument('--seed',type=int,default=11321);q.add_argument('--out-dir',required=True);q.add_argument('--batch-size',type=int,default=66)
 for stage in ['adjudicate','join']:
  q=sub.add_parser(stage);q.add_argument('--out-dir',required=True);q.add_argument('--review-a',type=Path,nargs='+',required=True);q.add_argument('--review-b',type=Path,nargs='+',required=True);q.add_argument('--batch-size',type=int,default=66)
  if stage=='join':q.add_argument('--adjudications',type=Path,nargs='*',default=[]);q.add_argument('--labels-dir',required=True)
 a=p.parse_args();mask(a) if a.command=='mask' else adjudicate(a)
if __name__=='__main__':main()
