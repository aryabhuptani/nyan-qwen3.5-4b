"""Create fixed discovery or selected-dose full modifier factorial jobs."""
import argparse
import itertools
import json
from pathlib import Path
from run_circuits import HERE, MODIFIERS, sha

ORDER = ['paws', 'hugs', 'genki', 'sounds', 'pride']


def build(phase, selected=None):
    conditions = []
    if phase == 'discovery':
        conditions.append(dict(condition_id='base', base=True, modifiers={}))
        for name in ORDER:
            for fraction in [.25, .5, 1.]:
                conditions.append(dict(condition_id=f'{name}_{fraction:g}', base=True,
                                       modifiers={name: fraction}))
    else:
        if set(selected) != set(ORDER) or any(selected[name] not in [.25, .5, 1.] for name in ORDER):
            raise ValueError('Exactly five frozen discovery-grid fractions required')
        for bits in itertools.product('01', repeat=5):
            conditions.append(dict(condition_id=''.join(bits), base=True,
                                   modifiers={name: selected[name] for name, bit in zip(ORDER, bits) if bit == '1'}))
    conditions.extend([dict(condition_id='plain', base=False, modifiers={}),
                       dict(condition_id='positive_control_genki_hugs', base=False,
                            modifiers={'genki': 1., 'hugs': 1.})])
    cases = HERE / f'CASES_{phase}.json'
    result = dict(study=113, phase=phase, bit_order=ORDER, max_tokens=192, seed=42,
                  temperature=0, repetition_penalty=1.1, repetition_context_size=128,
                  conditions=conditions, cases_sha256=sha(cases),
                  sources_sha256=sha(HERE / 'MODIFIER_SOURCES.json'))
    if selected is not None:
        result['selected_reference_fractions'] = selected
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['discovery', 'validation'], required=True)
    p.add_argument('--selected', type=Path, help='JSON object with five ingredient fractions')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    selected = json.loads(a.selected.read_text()) if a.selected else None
    if a.phase == 'validation' and selected is None:
        p.error('Validation requires frozen selected fractions')
    result = build(a.phase, selected)
    if a.selected:
        result['selection_sha256'] = sha(a.selected)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(output=str(a.output), conditions=len(result['conditions']))))


if __name__ == '__main__':
    main()
