"""Create anonymous prompt/reply review batches without atlas or forecasts.

Supports complete batches during generation, after all forecasts are locked.
Once written, a batch is immutable; no output-dependent batch assignment.
"""
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert (ROOT/'forecast_lock.json').exists()
    prompts=json.loads((ROOT/'selected_prompts.json').read_text())['prompts']
    cases={p['id']:p for p in prompts}
    conditions=json.loads((ROOT/'interventions.json').read_text())['conditions']
    allkeys=[(c['id'],p['id']) for c in conditions for p in prompts]
    # Fixed random grouping independent of replies, committed before generation.
    random.Random(11021).shuffle(allkeys)
    grouping={k:i//73 for i,k in enumerate(allkeys)}
    assert len(allkeys)==584 and len(set(grouping.values()))==8
    raw=ROOT/'results.local/110_validation.jsonl'
    if not raw.exists():
        print('No replies yet');return
    rows={}
    for line in raw.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue  # a currently flushing final line
        key=(r['condition'],r['prompt_id']);assert key in grouping
        assert key not in rows;rows[key]=r
    out=ROOT/'results.local/review'
    out.mkdir(exist_ok=True)
    mapping={}
    count=0
    for batch in range(8):
        keys=[k for k in allkeys if grouping[k]==batch]
        if not all(k in rows for k in keys):continue
        package=[]
        for k in keys:
            r=rows[k]
            rid=hashlib.sha256(('110review/'+k[0]+'/'+k[1]).encode()).hexdigest()[:18]
            package.append(dict(review_id=rid,prompt_id=k[1],messages=r['messages'],text=r['text'],finish_reason=r['finish_reason']))
            mapping[rid]=dict(condition_id=k[0],prompt_id=k[1],family=cases[k[1]]['family'])
        path=out/f'batch_{batch+1:02d}.jsonl'
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in package)
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
        count+=1
    if len(rows)==584:
        mapping_path=ROOT/'results.local/review_mapping.json'
        value=dict(raw_sha256=sha(raw),identities=mapping,batch_sha256={p.name:sha(p) for p in out.glob('batch_*.jsonl')})
        if mapping_path.exists():assert json.loads(mapping_path.read_text())==value
        else:mapping_path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(available_rows=len(rows),immutable_complete_batches=count,batch_size=73)))


if __name__=='__main__':main()
