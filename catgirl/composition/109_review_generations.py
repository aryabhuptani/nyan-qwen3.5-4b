"""Join locked masked readable-confirmation ratings after review is complete."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'results'
rating_file = ROOT / '109_generation_masked_ratings.json'
assert hashlib.sha256(rating_file.read_bytes()).hexdigest() == '94faa06eea33d2757b54d826613f50e1a546e302f2bb7b908484e8f2b476ffaa'
rating = json.loads(rating_file.read_text())
mapping = json.loads((ROOT / '109_generation_mapping.json').read_text())
masked_file = ROOT / '109_masked_generations.jsonl'
assert hashlib.sha256(masked_file.read_bytes()).hexdigest() == mapping['masked_sha256'] == rating['metadata']['source_sha256']
source_file = ROOT / '109_generations.jsonl'
assert hashlib.sha256(source_file.read_bytes()).hexdigest() == mapping['source_sha256']
masked = {r['review_id']: r for r in map(json.loads, masked_file.read_text().splitlines())}
rows = rating['ratings']
assert len(rows) == 36 and len({r['review_id'] for r in rows}) == 36
assert {r['review_id'] for r in rows} == set(masked) == set(mapping['identities'])
joined, groups = [], defaultdict(list)
for r in rows:
    source = masked[r['review_id']]
    assert all(r[k] == source[k] for k in ['arm', 'case_id', 'language'])
    identity = mapping['identities'][r['review_id']]
    assert identity['case_id'] == r['case_id']
    for field in ['own_accomplishment_attribution', 'external_praise', 'coherence', 'relevance']:
        assert type(r[field]) is int and 0 <= r[field] <= 2
    for field in ['literal_pride_word', 'unnatural_pride', 'hard_loop']:
        assert type(r[field]) is bool
    row = dict(r, condition=identity['condition'], text=source['text'])
    joined.append(row)
    groups[row['condition']].append(row)
summary = dict(review='One condition-label-masked synthetic agent reviewer, locked before joining; not a human/native-speaker or full-blinding claim',
               rating_sha256=hashlib.sha256(rating_file.read_bytes()).hexdigest(), conditions={})
for condition, rs in groups.items():
    summary['conditions'][condition] = dict(n=len(rs), clear_attribution=sum(r['own_accomplishment_attribution'] == 2 for r in rs),
        ambiguous_attribution=sum(r['own_accomplishment_attribution'] == 1 for r in rs),
        coherent=sum(r['coherence'] == 2 for r in rs), fully_relevant=sum(r['relevance'] == 2 for r in rs),
        literal_pride=sum(r['literal_pride_word'] for r in rs), unnatural_pride=sum(r['unnatural_pride'] for r in rs),
        hard_loops=sum(r['hard_loop'] for r in rs))
with (ROOT / '109_generation_unmasked.csv').open('w') as f:
    fields = sorted({k for r in joined for k in r})
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(joined)
(ROOT / '109_generation_review.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
