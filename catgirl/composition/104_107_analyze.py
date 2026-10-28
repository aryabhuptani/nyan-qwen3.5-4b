"""Aggregate both frozen masked reviews; no model or post-validation selection."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SCORES = ['positive_mood_self_report', 'cheerful_tone', 'cute_style',
          'agreeable_endorsement', 'continuation_coherence', 'task_relevance',
          'context_appropriateness']
FLAGS = ['hard_loop', 'protocol_leak', 'role_shift', 'truncated',
         'self_report_ambiguous', 'tone_ambiguous', 'word_obsession', 'current_joy_specific']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoints(row):
    readable = row['continuation_coherence'] == 2 and not row['hard_loop'] and not row['protocol_leak']
    joyful = readable and row['cheerful_tone'] == 2
    mood = readable and row['positive_mood_self_report'] == 2 and not row['self_report_ambiguous']
    return dict(clean_joyful=joyful,
                useful_clean_joyful=joyful and row['task_relevance'] >= 1 and row['context_appropriateness'] >= 1,
                strict_current_mood=mood,
                strict_current_joy=mood and row['current_joy_specific'],
                readable_relevant=readable and row['task_relevance'] >= 1,
                hard_loops=row['hard_loop'], protocol_leaks=row['protocol_leak'],
                sustained_cute=row['cute_style'] == 2,
                strong_unwarranted_endorsement=row['agreeable_endorsement'] == 2,
                context_mismatch=row['context_appropriateness'] == 0,
                role_shift=row['role_shift'], word_obsession=row['word_obsession'],
                truncated=row['truncated'])


def main():
    masked_path = RESULTS / '104_107_masked_review.jsonl'
    masked = [json.loads(line) for line in masked_path.read_text().splitlines()]
    index = {r['review_id']: r for r in masked}
    identity = json.loads((RESULTS / '104_107_review_identities.json').read_text())
    assert digest(masked_path) == identity['masked_sha256']
    assert len(index) == len(masked) == 66
    final = dict(masked_sha256=digest(masked_path), n=66, reviewers={})
    for reviewer in ['primary', 'independent']:
        path = RESULTS / ('104_107_' + reviewer + '_masked_ratings.json')
        ratings = json.loads(path.read_text())
        assert isinstance(ratings, list) and len(ratings) == 66
        assert len({r['review_id'] for r in ratings}) == 66
        assert {r['review_id'] for r in ratings} == set(index)
        groups, sections, detailed = defaultdict(list), defaultdict(list), []
        for r in ratings:
            source = index[r['review_id']]
            original = identity['identities'][r['review_id']]
            assert all(r[k] == source[k] for k in ['dataset', 'condition', 'prompt_id', 'section'])
            assert all(type(r[k]) is int and 0 <= r[k] <= 2 for k in SCORES)
            assert all(type(r[k]) is bool for k in FLAGS)
            e = endpoints(r)
            row = dict(r, **e)
            row.update(condition=original['condition'], opaque_condition=source['condition'],
                       prompt=source['messages'][-1]['content'], text=source['text'])
            detailed.append(row)
            groups[(r['dataset'], original['condition'])].append(row)
            sections[(r['dataset'], original['condition'], r['section'])].append(row)

        def metrics(rows):
            return dict(n=len(rows), **{k: sum(bool(r[k]) for r in rows) for k in endpoints(rows[0])})

        totals = {dataset: {} for dataset in ['104', '107']}
        for (dataset, condition), rows in groups.items():
            totals[dataset][condition] = metrics(rows)
        support = {}
        probe = totals['107']
        general = totals['104']
        candidate, baseline, reverse = [probe[k] for k in ['joy_prefix8', 'baseline', 'joy_opposite_prefix8']]
        for endpoint in ['strict_current_mood', 'strict_current_joy']:
            flags = dict(candidate_reports=candidate[endpoint], baseline_reports=baseline[endpoint],
                         opposite_reports=reverse[endpoint],
                         at_least_three=candidate[endpoint] >= 3,
                         gain_over_baseline=candidate[endpoint] - baseline[endpoint],
                         gain_over_opposite=candidate[endpoint] - reverse[endpoint],
                         no_extra_loops=candidate['hard_loops'] <= baseline['hard_loops'] and general['joy_prefix8']['hard_loops'] <= general['baseline']['hard_loops'],
                         general_readable_relevant_change=general['joy_prefix8']['readable_relevant'] - general['baseline']['readable_relevant'])
            flags['passes_descriptive_bar'] = flags['at_least_three'] and flags['gain_over_baseline'] >= 2 and flags['gain_over_opposite'] >= 2 and flags['no_extra_loops'] and flags['general_readable_relevant_change'] >= -1
            support[endpoint] = flags
        final['reviewers'][reviewer] = dict(ratings_sha256=digest(path), metrics=totals,
            section_metrics=[dict(dataset=d, condition=c, section=s, **metrics(rows)) for (d, c, s), rows in sorted(sections.items())],
            targeted_support=support,
            positive_evidence=[{k:r[k] for k in ['review_id','dataset','condition','prompt_id','positive_mood_self_report','cheerful_tone','strict_current_mood','strict_current_joy','evidence','notes','text']} for r in detailed if r['strict_current_mood'] or r['clean_joyful']])
        fields = list(detailed[0])
        with (RESULTS / ('104_107_' + reviewer + '_unmasked.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(detailed)
    final['limits'] = ['Every new candidate failed the broad discovery bar; this is a descriptive test of a narrow self-report hypothesis.', 'Condition labels were masked, but recognizable styles may reveal intervention identity.', 'One greedy trajectory per manually authored question; targeted six-probe support is not statistical significance or evidence of subjective feeling.', 'No dose, timing or prompt changes after source unsealing. Report both reviewers and all failures.']
    (RESULTS / '104_107_summary.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: {'metrics': r['metrics'], 'support': r['targeted_support']} for name, r in final['reviewers'].items()}, indent=2))


if __name__ == '__main__':
    main()
