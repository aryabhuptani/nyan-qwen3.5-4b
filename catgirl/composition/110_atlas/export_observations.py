"""Export the complete scored reply set without bulky runtime/agent transcripts."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
completion=json.loads((ROOT/'runtime_completion.json').read_text())
raw=ROOT/'results.local/110_validation.jsonl'
assert hashlib.sha256(raw.read_bytes()).hexdigest()==completion['raw_sha256']
lock=json.loads((ROOT/'labels_lock.json').read_text())
assert hashlib.sha256((ROOT/'labels.csv').read_bytes()).hexdigest()==lock['labels_sha256']['labels.csv']
outputs={}
for line in raw.read_text().splitlines():
    r=json.loads(line);key=(r['condition'],r['prompt_id']);assert key not in outputs;outputs[key]=r
labels=list(csv.DictReader((ROOT/'labels.csv').open()))
assert len(labels)==len(outputs)==584
rows=[]
for label in labels:
    r=outputs[(label['condition_id'],label['prompt_id'])]
    rows.append(dict(**label,text=r['text'],finish_reason=r['finish_reason'],
                     prompt_tokens=r['prompt_tokens'],generated_tokens=r['generated_tokens']))
path=ROOT/'observations.csv'
assert not path.exists(), 'Refusing to replace public observation export'
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
