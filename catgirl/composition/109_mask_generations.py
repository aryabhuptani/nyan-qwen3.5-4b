"""Make a condition-label-masked copy of the fixed 36 readable confirmations."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent / 'results'
source = ROOT / '109_generations.jsonl'
rows = [json.loads(line) for line in source.read_text().splitlines()]
assert len(rows) == 36
rng = random.Random(109)
conditions = sorted({r['condition'] for r in rows})
rng.shuffle(conditions)
mapping = {c: 'arm_' + chr(65 + i) for i, c in enumerate(conditions)}
rng.shuffle(rows)
masked, identities = [], {}
for row in rows:
    rid = hashlib.sha256((row['case_id'] + '/' + row['condition']).encode()).hexdigest()[:16]
    masked.append(dict(review_id=rid, arm=mapping[row['condition']], case_id=row['case_id'],
                       language=row['language'], messages=row['messages'], text=row['text'],
                       generated_tokens=row['generated_tokens'], finish_reason=row['finish_reason']))
    identities[rid] = dict(case_id=row['case_id'], condition=row['condition'])
target = ROOT / '109_masked_generations.jsonl'
with target.open('x') as f:
    for row in masked:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
with (ROOT / '109_generation_mapping.json').open('x') as f:
    json.dump(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   masked_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   identities=identities), f, indent=2)
print(json.dumps(dict(rows=len(masked), masked_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
