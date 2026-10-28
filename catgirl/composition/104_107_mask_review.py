"""Create a condition-label-masked review copy after both fixed jobs complete."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
JOBS = [('104', '104_happy_general_validation'), ('107', '107_happy_mood_validation')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mapping_path = RESULTS / '104_107_condition_mapping.json'
    mapping = json.loads(mapping_path.read_text())['mapping']
    rows, identities, sources = [], {}, {}
    for dataset, stem in JOBS:
        job = json.loads((ROOT / (stem + '.json')).read_text())
        path = RESULTS / (stem + '.jsonl')
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        expected = {(c['id'], p['id']) for c in job['conditions'] for p in job['prompts']}
        assert len(raw) == len(expected), 'Wait until the full fixed job completes'
        assert {(r['condition'], r['prompt_id']) for r in raw} == expected
        prompts = {p['id']: p for p in job['prompts']}
        for r in raw:
            assert r['messages'] == prompts[r['prompt_id']]['messages']
            key = dataset + '|' + r['condition'] + '|' + r['prompt_id']
            rid = 'review_' + hashlib.sha256(key.encode()).hexdigest()[:16]
            assert rid not in identities
            identities[rid] = dict(dataset=dataset, condition=r['condition'], prompt_id=r['prompt_id'])
            rows.append(dict(review_id=rid, dataset=dataset,
                             condition=mapping[r['condition']], prompt_id=r['prompt_id'],
                             section=prompts[r['prompt_id']].get('section', 'mood_probe' if dataset == '107' else 'unspecified'),
                             messages=r['messages'], text=r['text'],
                             generated_tokens=r['generated_tokens'], max_new_tokens=r['max_new_tokens'],
                             finish_reason=r['finish_reason']))
        sources[str(path.relative_to(ROOT))] = sha(path)
    assert len(rows) == 66
    random.Random(107104).shuffle(rows)
    output = RESULTS / '104_107_masked_review.jsonl'
    if output.exists():
        raise FileExistsError('Refusing to change the review copy after creation')
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (RESULTS / '104_107_review_identities.json').write_text(json.dumps(dict(
        identities=identities, raw_sha256=sources, masked_sha256=sha(output),
        protocol='Graders must not read this identity file, condition mapping, raw jobs or outputs until their ratings are saved. Condition labels are masked; familiar styles may still identify interventions.'), indent=2) + '\n')
    print(json.dumps(dict(rows=len(rows), output=str(output), sha256=sha(output),
                          masking='Condition labels and order; behavior itself remains visible.')))


if __name__ == '__main__':
    main()
