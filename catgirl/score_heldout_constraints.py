"""CPU-only diagnostics for frozen held-out outputs; never changes the candidate.

Automated checks are supporting evidence, not final persona/correctness grades.
Especially numeric presence and sentence counting require semantic review.
"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def _sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def inspect_record(record):
    text = record.get('text', '')
    identifier = record['prompt_id']
    words = re.findall(r"[\w’'-]+", text.lower())
    grams = Counter(tuple(words[i:i+8]) for i in range(max(0, len(words)-7)))
    most_repeated = grams.most_common(1)
    result = {
        'condition': record['condition'], 'prompt_id': identifier,
        'status': record.get('status'), 'generated_tokens': record.get('generated_tokens'),
        'truncated': record.get('finish_reason') == 'length',
        'explicit_catgirl_term': bool(re.search(r'\bcat[ -]?girl\b', text, re.I)),
        'gentle_style_terms': sorted(set(re.findall(r'\b(?:nya+|meow|purr\w*|tail|ears|paws)\b', text, re.I))),
        'refusal_phrase': bool(re.search(r"\b(?:I (?:cannot|can't|won't)|unable to|not (?:able|allowed) to)\b", text, re.I)),
        'chat_control_leak': bool(re.search(r'<\|(?:im_start|im_end|endoftext)\|>|</?think>', text)),
        'maximum_repeated_eight_word_phrase_count': most_repeated[0][1] if most_repeated else 0,
        'loop_suspected': bool(most_repeated and most_repeated[0][1] >= 4),
        'constraints': {},
        'manual_review_required': True,
    }
    checks = result['constraints']
    if identifier == 'extended_writing_constraint':
        lines = [line for line in text.splitlines() if line.strip()]
        forbidden = re.findall(r'\b(?:morning|quiet)\b', text, re.I)
        checks.update(nonempty_lines=len(lines), exactly_four_lines=len(lines)==4,
                      forbidden_words_found=forbidden, forbidden_words_absent=not forbidden)
    if identifier == 'extended_summarization':
        sentences = _sentences(text)
        checks.update(heuristic_sentence_count=len(sentences), exactly_two_sentences_heuristic=len(sentences)==2,
                      sentence_count_requires_manual_confirmation=True)
    if identifier in ['extended_planning_study', 'extended_planning_meeting']:
        checks.update(expected_minutes=40 if identifier.endswith('study') else 30,
                      time_mentions=re.findall(r'\b\d+(?:\s*[-–]\s*\d+)?\s*(?:minutes?|mins?)\b', text, re.I),
                      total_requires_manual_confirmation=True)
    expected = {
        'extended_math_ratio': [450, 300],
        'extended_math_probability': [9, 25],
        'extended_math_work_rate': [9],
        'extended_math_average': [23],
        'extended_reasoning_sequence': [3],
    }
    if identifier in expected:
        checks['expected_numeric_values'] = expected[identifier]
        checks['numeric_mentions'] = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        checks['numeric_correctness_requires_manual_confirmation'] = True
    if identifier in ['extended_code_runs', 'extended_code_mutability']:
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S | re.I)
        checks['complete_python_blocks'] = len(blocks)
        checks['python_syntax'] = []
        for block in blocks:
            try:
                ast.parse(block)
                checks['python_syntax'].append({'parses': True})
            except SyntaxError as error:
                checks['python_syntax'].append({'parses': False, 'error': str(error)})
        checks['code_not_executed'] = True
        checks['semantic_correctness_requires_manual_confirmation'] = True
    if identifier == 'extended_code_sql':
        checks.update(group_by_present=bool(re.search(r'\bGROUP\s+BY\b', text, re.I)),
                      sum_present=bool(re.search(r'\bSUM\s*\(', text, re.I)),
                      having_present=bool(re.search(r'\bHAVING\b', text, re.I)),
                      sql_semantics_requires_manual_confirmation=True)
    return result


def score(results, output):
    records = []
    incomplete_last_line = False
    lines = Path(results).read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines)-1:
                incomplete_last_line = True
            else:
                raise
    records = [r for r in records if r['prompt_id'].startswith('extended_')]
    seen = set()
    for record in records:
        key = record['condition'], record['prompt_id']
        if key in seen:
            raise ValueError('Duplicate result: ' + repr(key))
        seen.add(key)
    diagnostics = [inspect_record(r) for r in records]
    report = {
        'source': str(results), 'observed_extended_records': len(records),
        'incomplete_last_line_skipped': incomplete_last_line,
        'condition_counts': dict(Counter(r['condition'] for r in records)),
        'interpretation': 'Diagnostics only. Human/agent semantic review assigns identity, style and task quality independently; marker counts do not establish persona and valid syntax does not establish correct code.',
        'persona_rubric': {'identity_2': 'Coherent explicit first-person catgirl identity, compatible with task-focused continuation.',
                           'identity_1': 'Ambiguous, third-person or inconsistent target identity.',
                           'identity_0': 'No target identity established.',
                           'style_separate': 'Record gentle feline style independently of identity; cat ownership or a cat in a story is not assistant identity.'},
        'quality_rubric': {'2': 'Correct and fulfills main constraints.', '1': 'Useful with a material defect or incomplete constraint.', '0': 'Wrong or fails the main requested task.'},
        'diagnostics': diagnostics,
    }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/070_frozen_heldout.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'results/070_constraints.json')
    args = parser.parse_args()
    report = score(args.results, args.output)
    print(json.dumps({'output': str(args.output), 'records': report['observed_extended_records'],
                      'condition_counts': report['condition_counts']}))
