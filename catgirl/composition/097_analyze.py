"""Aggregate the prespecified meow study after semantic review; no model imports."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIELDS = ['self_meow', 'self_nya', 'self_chinese_meow', 'catgirl_identity',
          'female_animal_only', 'coherent', 'hard_loop', 'task_hijack']


def main():
    job = json.loads((ROOT / '097_meow_composition.json').read_text())
    raw_path = RESULTS / '097_meow_composition.jsonl'
    ratings_path = RESULTS / '097_semantic_ratings.json'
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    ratings = json.loads(ratings_path.read_text())
    if isinstance(ratings, dict):
        ratings = ratings['ratings']
    expected = {(c['id'], p['id']) for c in job['conditions'] for p in job['prompts']}
    key = lambda row: (row['condition'], row['prompt_id'])
    raw_index = {key(row): row for row in raw}
    review_index = {key(row): row for row in ratings}
    assert len(raw_index) == len(raw) == len(expected), 'Raw rows incomplete or duplicated'
    assert len(review_index) == len(ratings) == len(expected), 'Semantic review incomplete or duplicated'
    assert set(raw_index) == set(review_index) == expected
    for row in ratings:
        assert all(type(row[field]) is bool for field in FIELDS), 'Semantic flags must be boolean'
    metrics = []
    for condition in job['conditions']:
        name = condition['id']
        rows = [r for r in ratings if r['condition'] == name]
        originals = [raw_index[key(r)] for r in rows]
        item = dict(condition=name, n=len(rows), unique_neurons=len(condition['neurons']))
        item.update({field: sum(r[field] for r in rows) for field in FIELDS})
        item.update(
            coherent_meow=sum(r['self_meow'] and r['coherent'] and not r['hard_loop'] for r in rows),
            joint_catgirl_meow=sum(r['self_meow'] and r['catgirl_identity'] for r in rows),
            coherent_joint_catgirl_meow=sum(r['self_meow'] and r['catgirl_identity'] and r['coherent'] and not r['hard_loop'] for r in rows),
            raw_english_meow_substring=sum(bool(re.search('meow', r['text'], re.I)) for r in originals),
            truncated=sum(r.get('finish_reason') == 'length' or r['generated_tokens'] >= r['max_new_tokens'] for r in originals))
        metrics.append(item)
    paired = []
    for name in ['catgirl4_catboost', 'catgirl5_animal', 'catgirl5_animal_catboost']:
        gains = losses = both = neither = 0
        for prompt in job['prompts']:
            old = review_index['catgirl4', prompt['id']]['self_meow']
            new = review_index[name, prompt['id']]['self_meow']
            gains += new and not old
            losses += old and not new
            both += old and new
            neither += not old and not new
        paired.append(dict(condition=name, comparator='catgirl4', gained_meow=gains,
                           lost_meow=losses, both_meow=both, neither_meow=neither))
    summary = dict(raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                   semantic_sha256=hashlib.sha256(ratings_path.read_bytes()).hexdigest(),
                   records=len(raw), metrics=metrics, paired=paired,
                   limitations='Fixed manual prompt set, one greedy output per prompt, visible-label agent review. Counts are behavioral observations, not proof of statistical interaction or natural neuron functions.')
    (RESULTS / '097_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (RESULTS / '097_ratings.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['condition', 'prompt_id'] + FIELDS + ['notes'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ratings)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
