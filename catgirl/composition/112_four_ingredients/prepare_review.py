"""Condition-masked review packaging, lexical measurement, adjudication and joining.

No forecasts, source descriptions, or GPU inference are read. Commands refuse to
replace existing differing artifacts. Run --help for the three explicit stages.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re

FEATURES = ['bubbly_excited_expression', 'female_assistant_selfrole',
            'sustained_cute_register', 'explicit_virtual_affection', 'own_animal_body',
            'feline_assistant_selfrole', 'performed_feline_vocalization', 'coherent',
            'task_relevant', 'emoji_in_assistant_voice']
VOCAL_COMPONENTS = ['self_english_meow', 'self_nya', 'self_chinese_meow', 'self_meow_action']
PATTERNS = {
    'hug': r'\b(?:hug|hugs|hugged|hugging)\b',
    'cuddle': r'\b(?:cuddle|cuddles|cuddled|cuddling)\b',
    'paw': r'\b(?:paw|paws|pawed|pawing)\b',
    'tail': r'\b(?:tail|tails|tailed|tailing)\b',
    'fur': r'\b(?:fur|furs|furry|furred)\b',
    'female': r'\b(?:female|females|woman|women|girl|girls|catgirl|catgirls)\b',
}
# Codepoint diagnostic, not a full Unicode grapheme/emoji implementation.
EMOJI = re.compile('[\U0001F300-\U0001FAFF\u2600-\u27BF]')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def save(p, value):
    p=Path(p); text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if p.exists():
        assert p.read_text()==text, f'Refusing to overwrite differing artifact: {p}'
    else:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
def jsonlines(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
def mechanical(text):
    hits={name:[m.group() for m in re.finditer(pattern,text,re.I)] for name,pattern in PATTERNS.items()}
    result={name+'_count':len(v) for name,v in hits.items()}
    result.update(lexical_hugs_cuddles=int(bool(hits['hug'] or hits['cuddle'])),
                  lexical_paws_tail_fur=int(bool(hits['paw'] or hits['tail'] or hits['fur'])),
                  emoji_codepoint_count=len(EMOJI.findall(text)),
                  emoji_codepoint_present=int(bool(EMOJI.search(text))))
    return result,hits

def mask(args):
    cases={}
    for p in args.cases:
        obj=load(p)
        for c in obj.get('cases',obj.get('multiturn_cases',[])+obj.get('late_prompt_cases',[])):
            key=c.get('case_id',c.get('id'));assert key not in cases;cases[key]=c
    conditions={c['condition_id'] for c in load(args.job)['conditions']}
    rows={}
    for p in args.raw:
        for row in jsonlines(p):
            cid=row['case_id'];condition=row['condition_id'];turn=row.get('turn_index',row.get('turn'))
            assert cid in cases and condition in conditions
            key=(condition,cid,turn);assert key not in rows, f'Duplicate output {key}'
            assert isinstance(row['text'],str);rows[key]=row
    if all(k[2] is None for k in rows):
        expected={(c,i,None) for c in conditions for i in cases}
        assert set(rows)==expected, f'Incomplete or extra grid: got {len(rows)}, expected {len(expected)}'
    else:
        for condition in conditions:
            for cid,c in cases.items():
                if 'turns' not in c: continue
                actual={k[2] for k in rows if k[:2]==(condition,cid)}
                assert len(actual)==len(c['turns']), f'Incomplete conversation {condition}/{cid}'
    keys=sorted(rows,key=lambda k:(k[0],k[1],str(k[2])))
    random.Random(args.seed).shuffle(keys)
    identities={}; batches=[]; out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(keys),args.batch_size):
        package=[]
        for condition,cid,turn in keys[start:start+args.batch_size]:
            row=rows[(condition,cid,turn)];case=cases[cid]
            rid=hashlib.sha256(f'112/{args.tag}/{condition}/{cid}/{turn}'.encode()).hexdigest()[:20]
            assert rid not in identities
            messages=row.get('messages',case.get('messages'))
            if 'turns' in case:
                assert isinstance(row.get('messages'),list), 'Persistence requires actual full conversation history in each raw row'
            assert messages and isinstance(messages,list), 'Missing prompt messages'
            quality=case.get('quality_expected')
            if isinstance(quality,list):
                assert isinstance(turn,int) and 0<=turn<len(quality), 'Persistence turn_index must be zero based'
                quality=quality[turn]
            package.append(dict(review_id=rid,messages=messages,text=row['text'],
                                finish_reason=row.get('finish_reason'),quality_expected=quality))
            measured,hits=mechanical(row['text'])
            identities[rid]=dict(condition_id=condition,case_id=cid,family=case['family'],
                                 split=case.get('split'),turn_index=turn,mechanical=measured,
                                 mechanical_hits=hits,text_sha256=hashlib.sha256(row['text'].encode()).hexdigest())
        path=out/f'batch_{start//args.batch_size+1:02}.jsonl'
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in package)
        if path.exists(): assert path.read_text()==content, 'Masked batch changed'
        else: path.write_text(content)
        batches.append(path)
    save(out/'mapping.local.json',dict(study=112,tag=args.tag,feature_order=FEATURES,
        raw_sha256={str(p):digest(p) for p in args.raw},cases_sha256={str(p):digest(p) for p in args.cases},
        job_sha256=digest(args.job),identities=identities,
        batch_sha256={p.name:digest(p) for p in batches}))
    print(json.dumps(dict(rows=len(rows),batches=len(batches),out_dir=str(out))))

def review_inputs(args):
    directory=Path(args.out_dir);mapping=load(directory/'mapping.local.json');examples={}
    for name,expected in mapping['batch_sha256'].items():
        path=directory/name;assert digest(path)==expected
        for row in jsonlines(path):
            assert row['review_id'] not in examples;examples[row['review_id']]=row
    assert set(examples)==set(mapping['identities'])
    reviews=[];files=[]
    for who,paths in [('A',args.review_a),('B',args.review_b)]:
        ratings={}
        for path in paths:
            review_obj=load(path)
            assert review_obj.get('feature_order')==FEATURES, 'Review feature order missing or changed'
            for row in review_obj['ratings']:
                rid=row['review_id'];assert rid in examples and rid not in ratings
                v=row['values'];assert len(v)==len(FEATURES) and all(x in (0,1,'uncertain') for x in v)
                components=row['vocalization_components'];assert len(components)==4
                assert all(x in (0,1,'uncertain') for x in components)
                union=1 if 1 in components else ('uncertain' if 'uncertain' in components else 0)
                assert v[6]==union, f'Meow union mismatch: {rid}'
                assert isinstance(row['hard_loop'],bool) and isinstance(row['truncated'],bool)
                assert not row['hard_loop'] or v[7]==0
                assert isinstance(row.get('evidence'),str) and row['evidence'].strip()
                ratings[rid]=row
            files.append((who,Path(path)))
        assert set(ratings)==set(examples), f'Reviewer {who} incomplete'
        reviews.append(ratings)
    return directory,mapping,examples,reviews,files

def adjudicate(args):
    directory,mapping,examples,reviews,files=review_inputs(args)
    disputed=[]
    for rid,example in examples.items():
        a,b=[r[rid] for r in reviews]
        indices=[i for i,(x,y) in enumerate(zip(a['values'],b['values'])) if x!=y or x=='uncertain']
        if indices:
            disputed.append(dict(**example,disputed_features=[FEATURES[i] for i in indices],rating_1=a,rating_2=b))
    lock=dict(mapping_sha256=digest(directory/'mapping.local.json'),
              review_sha256={f'{who}/{p.name}':digest(p) for who,p in files},
              rows=len(examples),disputed_rows=len(disputed),
              disputed_feature_cells=sum(len(r['disputed_features']) for r in disputed))
    save(directory/'reviews_lock.json',lock)
    for i in range(0,len(disputed),args.batch_size):
        save(directory/f'adjudication_{i//args.batch_size+1:02}.json',dict(feature_order=FEATURES,cases=disputed[i:i+args.batch_size]))
    print(json.dumps(lock))
    if args.command!='join': return
    resolved={};disputed_by_id={r['review_id']:r for r in disputed}
    for p in args.adjudications:
        for row in load(p)['ratings']:
            rid=row['review_id'];assert rid in disputed_by_id and rid not in resolved
            assert len(row['values'])==len(FEATURES) and all(x in (0,1) for x in row['values'])
            assert isinstance(row.get('evidence'),str) and row['evidence'].strip()
            disputed_fields=disputed_by_id[rid]['disputed_features']
            for i,f in enumerate(FEATURES):
                if f not in disputed_fields: assert row['values'][i]==reviews[0][rid]['values'][i]
            resolved[rid]=row
    assert set(resolved)==set(disputed_by_id), 'Incomplete adjudication'
    target=Path(args.labels_dir);target.mkdir(parents=True,exist_ok=True);outputs=[]
    for mode in ['adjudicated','reviewer_A','reviewer_B','uncertain_zero','uncertain_one']:
        rows=[]
        for rid in sorted(examples):
            a,b=[r[rid] for r in reviews];final=resolved.get(rid,a)['values']
            if mode=='reviewer_A': values=a['values']
            elif mode=='reviewer_B': values=b['values']
            elif mode.startswith('uncertain_'):
                bound=int(mode.endswith('one'))
                values=[bound if 'uncertain' in (a['values'][i],b['values'][i]) else final[i] for i in range(len(FEATURES))]
            else: values=final
            identity=mapping['identities'][rid]
            rows.append(dict(review_id=rid,**{k:v for k,v in identity.items() if k not in ('mechanical','mechanical_hits')},
                **identity['mechanical'],**dict(zip(FEATURES,values)),adjudicated=int(rid in resolved)))
        name='labels.csv' if mode=='adjudicated' else f'labels_{mode}.csv';path=target/name
        import io
        buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        if path.exists():assert path.read_text()==buffer.getvalue()
        else:path.write_text(buffer.getvalue())
        outputs.append(path)
    evidence=[]
    for rid in sorted(examples):
        evidence.append(dict(review_id=rid,identity=mapping['identities'][rid],reviewer_A=reviews[0][rid],reviewer_B=reviews[1][rid],adjudication=resolved.get(rid)))
    save(target/'label_evidence.json',dict(feature_order=FEATURES,rows=evidence))
    save(target/'labels_lock.json',dict(**lock,adjudication_sha256={str(p):digest(p) for p in args.adjudications},
         labels_sha256={p.name:digest(p) for p in outputs},evidence_sha256=digest(target/'label_evidence.json')))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    m=sub.add_parser('mask');m.add_argument('--raw',type=Path,action='append',required=True);m.add_argument('--cases',type=Path,action='append',required=True)
    m.add_argument('--job',type=Path,required=True);m.add_argument('--tag',default='factorial');m.add_argument('--seed',type=int,default=11221)
    for q in [m]:q.add_argument('--out-dir',required=True);q.add_argument('--batch-size',type=int,default=70)
    for command in ['adjudicate','join']:
        q=sub.add_parser(command);q.add_argument('--out-dir',required=True);q.add_argument('--review-a',type=Path,nargs='+',required=True);q.add_argument('--review-b',type=Path,nargs='+',required=True);q.add_argument('--batch-size',type=int,default=70)
        if command=='join':q.add_argument('--adjudications',type=Path,nargs='*',default=[]);q.add_argument('--labels-dir',required=True)
    args=p.parse_args()
    if args.command=='mask':mask(args)
    else:adjudicate(args)

if __name__=='__main__':main()
