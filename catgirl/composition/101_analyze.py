"""CPU-only aggregation of frozen101 semantic ratings; never launches a model."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def main():
    raw_path = RESULTS / '101_weak_composition.jsonl'
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    ratings = json.loads((RESULTS / '101_semantic_ratings.json').read_text())
    job = json.loads((ROOT / '101_weak_composition.json').read_text())
    expected = {(c['id'], p['id']) for c in job['conditions'] for p in job['prompts']}
    assert len(ratings) == len(raw) == len(expected) == 80
    assert {(r['condition'], r['prompt_id']) for r in ratings} == expected
    keys = ['strict_coherent_catgirl', 'lenient_coherent_catgirl',
            'strict_joint_english_meow', 'lenient_joint_english_meow',
            'strict_joint_any_vocalization', 'lenient_joint_any_vocalization',
            'strict_joint_inclusive_vocalization_sensitivity',
            'lenient_joint_inclusive_vocalization_sensitivity',
            'self_meow', 'self_chinese_meow', 'self_nya', 'self_meow_action',
            'any_vocalization', 'catgirl_identity', 'hard_loop', 'task_hijack']
    metrics = {}
    indexed = {}
    csv_rows = []
    for row, source in zip(ratings, raw):
        assert (row['condition'], row['prompt_id']) == (source['condition'], source['prompt_id'])
        assert row['coherence'] in (0, 1, 2)
        for mode, threshold in [('strict', 2), ('lenient', 1)]:
            girl = row['catgirl_identity'] and row['coherence'] >= threshold and not row['hard_loop']
            assert row[mode + '_coherent_catgirl'] == girl
            assert row[mode + '_joint_english_meow'] == (girl and row['self_meow'])
        indexed[(row['condition'], row['prompt_id'])] = row
        for section in ('all', row['section']):
            group = metrics.setdefault(row['condition'], {}).setdefault(section, dict(n=0, truncated=0, **{k: 0 for k in keys}))
            group['n'] += 1
            group['truncated'] += int(source['finish_reason'] == 'length')
            for key in keys:
                group[key] += int(row[key])
        csv_rows.append(dict(**row, raw_text=source['text'], messages=json.dumps(source['messages'], ensure_ascii=False), recipe=json.dumps(source['recipe']), layer_schedules=json.dumps(source['layer_schedules']), temperature=source['temperature'], seed=source['seed'], repetition_penalty=source['repetition_penalty'], repetition_context_size=source['repetition_context_size'], prompt_tokens=source['prompt_tokens'], max_new_tokens=source['max_new_tokens']))
    comparisons = {}
    for pulse, base in [('original_weak_animal', 'original_control'), ('mild_weak_animal', 'mild_control')]:
        comparison = dict(matched_base=base, original_reference='original_control', modes={}, paired=[])
        for mode in ('strict', 'lenient'):
            a, z, original = (metrics[c]['all'] for c in (pulse, base, 'original_control'))
            girl_key, joint_key = mode + '_coherent_catgirl', mode + '_joint_english_meow'
            flags = dict(joint_increase=a[joint_key] - z[joint_key], matched_girl_change=a[girl_key] - z[girl_key], matched_extra_loops=a['hard_loop'] - z['hard_loop'], original_girl_change=a[girl_key] - original[girl_key], original_extra_loops=a['hard_loop'] - original['hard_loop'])
            flags['passes_matched_gate'] = flags['joint_increase'] >= 1 and flags['matched_girl_change'] >= -1 and flags['matched_extra_loops'] <= 0
            flags['passes_original_guard'] = flags['original_girl_change'] >= -1 and flags['original_extra_loops'] <= 0
            flags['passes_full_gate'] = flags['passes_matched_gate'] and flags['passes_original_guard']
            comparison['modes'][mode] = flags
        for p in job['prompts']:
            a, z = indexed[(pulse, p['id'])], indexed[(base, p['id'])]
            comparison['paired'].append(dict(prompt_id=p['id'], section=a['section'], **{k: dict(base=z[k], pulse=a[k], delta=int(a[k]) - int(z[k])) for k in keys}))
        comparisons[pulse] = comparison
    review = dict(name='101 frozen weak-pulse composition comparison', complete=True, n=80, conditions=4, prompts_per_condition=20, sections=dict(identity_role=6, ordinary_task=14), blinded=False, source=raw_path.name, source_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(), frozen_plan='../101_WEAK_COMPOSITION_PLAN.json', numeric_coherence='Frozen101 persona coherence:0 includes no assistant catgirl even when generic prose is readable;2 is strict coherent role;1 is ambiguous/corrupt role. Natural budget truncation alone does not invalidate completed role.', metrics=metrics, comparisons=comparisons, recommendation='Neither pulse passes frozen strict gate. Keep original as default; do not promote weak animal pulse as personality-preserving improvement.', critical_positive=dict(condition='mild_weak_animal', prompt_id='101_role_05', source_index=64, evidence='Meow! (喵！)', note='Explicit catgirl and performed English/Chinese, primary coherence2. Earlier my meow-cats is an odd but complete pet-like noun phrase; treating it as corruption would remove this clean joint. Sound and girl attribution remain clear.'), sensitivity=dict(primary_ratings_finalized_before_independent_review=True, original_independent=dict(strict_girl=dict(original_control=5, original_weak_animal=6), lenient_girl=dict(original_control=9, original_weak_animal=10), strict_english_joint=dict(original_control=0, original_weak_animal=0), loops=dict(original_control=0, original_weak_animal=1), explanation='Primary strict7/5 versus independent5/6 differs over age fantasy, draft-message self-attribution and Me-profile adoption. Plural girls headings are generic in primary but lenient girl in guard. Identity drop is not an agreed measurement; strict joint and extra-loop rejection agree.'), chinese_heading=dict(source_indices=[3,43], primary='Profile-label attribution ambiguous, Chinese performance excluded conservatively.', inclusive='Counting heading 喵 as performed adds one strict any-vocalization joint to original_control and mild_control. It never changes English counts.'), own_sound_profile=dict(source_indices=sorted([10,35,40,54,55,60,73,74,75]), note='Possible performance retained separately; no automatic conversion of profile, prospective greeting, or described sound to English primary.'), mild_pulse_positive='If my meow-cats is judged corrupt, clean joint1 becomes0; full gate already fails from extra loops.'), limits=['Small separately authored fresh20 neutral prompts, one greedy trajectory, labels visible; not a representative untouched benchmark.', '099 already tested weak pulse at mild/higher B on familiar prompts; original+B.3 weak pulse is new in101.', 'At most1/20 role loss is a descriptive tolerance, not statistical equivalence.', 'No settings retuned or candidate selected from these outputs. No long-conversation persistence measured here.', 'Capabilities and task displacement descriptive, distinct from readable self-role. Body anatomy quota not imposed.'])
    review['sensitivity']['mild_independent'] = dict(strict_girl=dict(mild_control=2, mild_weak_animal=2), lenient_girl=dict(mild_control=7, mild_weak_animal=6), strict_joint=dict(mild_control=0, mild_weak_animal=0), lenient_joint=dict(mild_control=1, mild_weak_animal=2), raw_english=dict(mild_control=2, mild_weak_animal=3), loops=dict(mild_control=1, mild_weak_animal=3), explanation='Independent guard is stricter about meow-cats and cat-cat compounds and broader about own sound profiles. Actual final English/Chinese performance in pulse role05 agrees; strict clean joint primary1 versus guard0 is coherence sensitivity. Both reject gate from extra loops. Guard explicitly allows alternate strict joint1 if meow-cats is tolerated.')
    (RESULTS / '101_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    with (RESULTS / '101_ratings.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({c: m['all'] for c, m in metrics.items()}, indent=2))
    print(json.dumps({c: x['modes'] for c, x in comparisons.items()}, indent=2))


if __name__ == '__main__':
    main()
