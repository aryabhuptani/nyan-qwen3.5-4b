"""Apply the preregistered personality-first gate to complete 099 ratings."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    job = json.loads((ROOT / '099_personality_meow_search.json').read_text())
    ratings = json.loads((ROOT / 'results/099_semantic_ratings.json').read_text())
    if isinstance(ratings, dict):
        ratings = ratings['ratings']
    expected = {(c['id'], p['id']) for c in job['conditions'] for p in job['prompts']}
    assert len(ratings) == len(expected)
    assert {(r['condition'], r['prompt_id']) for r in ratings} == expected
    groups = []
    for condition in job['conditions']:
        rows = [r for r in ratings if r['condition'] == condition['id']]
        good = lambda r: r['coherent'] and not r['hard_loop']
        girl = sum(r['catgirl_identity'] and good(r) for r in rows)
        joint = sum(r['catgirl_identity'] and r['self_meow'] and good(r) for r in rows)
        animal = next((n for n in condition['neurons'] if (n['layer'], n['neuron']) == (7, 7848)), None)
        cat = next(n for n in condition['neurons'] if (n['layer'], n['neuron']) == (26, 6173))
        integrated_a = abs(animal['published_dose']) * condition['layer_schedules']['7']['early_tokens'] if animal else 0
        groups.append(dict(condition=condition['id'], n=len(rows), coherent_catgirl=girl,
                           coherent_joint_meow=joint, self_meow=sum(r['self_meow'] for r in rows),
                           hard_loops=sum(r['hard_loop'] for r in rows), animal=bool(animal),
                           cat_dose=cat['published_dose'], integrated_a=integrated_a))
    mild = next(r for r in groups if r['condition'] == 'B0.525_noA')
    for row in groups:
        row['eligible'] = row['coherent_catgirl'] >= mild['coherent_catgirl'] and row['hard_loops'] <= mild['hard_loops']
        row['strict_joint_improvement'] = row['coherent_joint_meow'] > mild['coherent_joint_meow']
    contenders = [r for r in groups if r['eligible'] and r['strict_joint_improvement']]
    contenders.sort(key=lambda r: (-r['coherent_joint_meow'], r['animal'], r['cat_dose'], r['integrated_a']))
    selected = contenders[0]['condition'] if contenders else mild['condition']
    result = dict(reference=mild, selected=selected, new_improvement=bool(contenders),
                  metrics=groups, rule='099_PERSONALITY_MEOW_HYPOTHESES.json; no post-output tie-break changes')
    (ROOT / 'results/099_selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
