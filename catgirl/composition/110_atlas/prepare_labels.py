"""Validate two masked reviews, prepare masked adjudication, then join locked labels.

No forecasts are read. All evidence and original uncertainty survive the join.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = json.loads((ROOT/'110_PLAN.json').read_text())['primary_features']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    content = json.dumps(value, indent=2, ensure_ascii=False)+'\n'
    if p.exists():
        assert p.read_text() == content, f'Refusing to overwrite {p}'
    else: p.write_text(content)

def load_reviews():
    directory = ROOT/'results.local/review'
    mapping=json.loads((ROOT/'results.local/review_mapping.json').read_text())
    examples = {}
    for p in sorted(directory.glob('batch_*.jsonl')):
        assert sha(p)==mapping['batch_sha256'][p.name], 'Masked review batch changed'
        for line in p.read_text().splitlines():
            r=json.loads(line)
            assert r['review_id'] not in examples
            examples[r['review_id']]=r
    assert len(examples)==584
    assert set(examples)==set(mapping['identities']), 'Masked ID coverage changed'
    reviews=[]
    files=[]
    for reviewer in ('A','B'):
        ratings={}
        for group in range(1,5):
            p=directory/f'ratings_{reviewer}{group}.json'
            value=json.loads(p.read_text())
            expected={r['review_id'] for batch in (2*group-1,2*group)
                      for r in map(json.loads,(directory/f'batch_{batch:02d}.jsonl').read_text().splitlines())}
            actual=set()
            for r in value['ratings']:
                rid=r['review_id']; assert rid in expected and rid not in actual and rid not in ratings
                actual.add(rid)
                assert len(r['values'])==7 and all(v in (0,1,'uncertain') for v in r['values'])
                assert len(r['vocalization_components'])==4 and all(v in (0,1,'uncertain') for v in r['vocalization_components'])
                components=r['vocalization_components']
                union=1 if 1 in components else ('uncertain' if 'uncertain' in components else 0)
                assert r['values'][2]==union, (rid,'vocalization union mismatch')
                assert isinstance(r['hard_loop'],bool) and isinstance(r['truncated'],bool)
                assert not r['hard_loop'] or r['values'][5]==0
                assert isinstance(r['evidence'],str) and r['evidence'].strip()
                ratings[rid]=r
            assert actual==expected, (p,'incomplete review')
            files.append(p)
        assert set(ratings)==set(examples)
        reviews.append(ratings)
    return examples,reviews,files

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--join',action='store_true'); args=parser.parse_args()
    examples,reviews,files=load_reviews()
    directory=ROOT/'results.local/review'
    disagreements=[]
    for rid,r in examples.items():
        a,b=(rv[rid] for rv in reviews)
        indices=[i for i,(x,y) in enumerate(zip(a['values'],b['values'])) if x!=y or x=='uncertain']
        if indices:
            disagreements.append(dict(**r, disputed_features=[FEATURES[i] for i in indices],
                                      rating_1=a,rating_2=b))
    review_lock=dict(review_sha256={p.name:sha(p) for p in files},
                     raw_mapping_sha256=sha(ROOT/'results.local/review_mapping.json'),
                     replies=584,disputed_replies=len(disagreements),
                     disputed_feature_cells=sum(len(r['disputed_features']) for r in disagreements))
    save(ROOT/'reviews_lock.json',review_lock)
    # Adjudicator receives only masked examples and anonymized ratings.
    for i in range(0,len(disagreements),73):
        save(directory/f'adjudication_{i//73+1:02d}.json',dict(cases=disagreements[i:i+73]))
    print(json.dumps(review_lock))
    if not args.join:return
    adjudicated={}
    adjudication_files=sorted(directory.glob('adjudicated_*.json'))
    expected={r['review_id'] for r in disagreements}
    for p in adjudication_files:
        for r in json.loads(p.read_text())['ratings']:
            rid=r['review_id'];assert rid in expected and rid not in adjudicated
            assert len(r['values'])==7 and all(v in (0,1) for v in r['values'])
            assert isinstance(r['evidence'],str) and r['evidence'].strip()
            original=reviews[0][rid]['values']
            disputed=next(x['disputed_features'] for x in disagreements if x['review_id']==rid)
            assert all(r['values'][i]==original[i] for i,f in enumerate(FEATURES) if f not in disputed)
            adjudicated[rid]=r
    assert set(adjudicated)==expected, 'Adjudication incomplete'
    identities=json.loads((ROOT/'results.local/review_mapping.json').read_text())['identities']
    for mode in ('adjudicated','reviewer_A','reviewer_B','uncertain_zero','uncertain_one'):
        rows=[]
        for rid in sorted(examples):
            a,b=(rv[rid] for rv in reviews)
            final=adjudicated.get(rid,a)['values']
            if mode.startswith('reviewer_'):
                original=(a if mode.endswith('A') else b)['values']
                values=original
            elif mode.startswith('uncertain_'):
                extreme=int(mode.endswith('one'))
                values=[extreme if 'uncertain' in (a['values'][i],b['values'][i]) else final[i] for i in range(7)]
            else: values=final
            joint_parts=[values[i] for i in (0,1,5)]
            joint=0 if 0 in joint_parts else ('uncertain' if 'uncertain' in joint_parts else 1)
            rows.append(dict(review_id=rid,**identities[rid],**dict(zip(FEATURES,values)),
                             joint_catgirl_coherent=joint,
                             label_uncertflag=int(rid in adjudicated)))
        path=ROOT/('labels.csv' if mode=='adjudicated' else f'labels_{mode}.csv')
        assert not path.exists(), f'Refusing to overwrite {path}'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    evidence=[]
    for rid in sorted(examples):
        a,b=(rv[rid] for rv in reviews)
        evidence.append(dict(review_id=rid,**identities[rid],reviewer_A=a,reviewer_B=b,
                             adjudication=adjudicated.get(rid),reply_sha256=hashlib.sha256(examples[rid]['text'].encode()).hexdigest()))
    save(ROOT/'label_evidence.json',dict(replies=evidence))
    save(ROOT/'labels_lock.json',dict(**review_lock,adjudication_sha256={p.name:sha(p) for p in adjudication_files},
                                    labels_sha256={p.name:sha(p) for p in ROOT.glob('labels*.csv')},
                                    evidence_sha256=sha(ROOT/'label_evidence.json')))

if __name__=='__main__':main()
