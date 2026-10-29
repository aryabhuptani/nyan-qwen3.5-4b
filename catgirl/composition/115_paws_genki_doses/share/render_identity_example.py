"""Render the selected Study 115 discovery reply; no confirmation data is read.

Run with Python 3. Raw discovery data, if available, is checked against the public
provenance; otherwise that provenance supplies the exact selected excerpt.
"""
import hashlib
import html
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROVENANCE = HERE / 'IDENTITY_IMAGE_PROVENANCE.json'
SVG = HERE / 'paws_genki_identity.svg'
CONDITION = 'p0.1_g0.4'
CASE = '115_discovery_01'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected_source():
    source = HERE.parent / 'results.local/discovery.jsonl'
    if source.exists():
        matches = []
        for line in source.open():
            row = json.loads(line)
            if row.get('condition_id') == CONDITION and row.get('case_id') == CASE:
                matches.append(row)
        assert len(matches) == 1
        row = matches[0]
        excerpt = '\n\n'.join(row['text'].split('\n\n')[:2])
        provenance = dict(
            study=115, split='discovery', condition_id=CONDITION, case_id=CASE,
            source='results.local/discovery.jsonl',
            source_sha256=sha(source.read_bytes()),
            reply_sha256=sha(row['text'].encode()),
            prompt=row['messages'][-1]['content'], messages=row['messages'],
            excerpt=excerpt, excerpt_is_exact_prefix=row['text'].startswith(excerpt),
            excerpt_paragraph_count=2,
            model='Qwen3.5-4B',
            model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
            base_coordinates=4, additional_coordinates=2, total_coordinates=6,
            doses=dict(paws=dict(layer=23, neuron=4769, reference_normalized_dose=0.1),
                       genki=dict(layer=13, neuron=4810, reference_normalized_dose=0.4)),
            effective_interventions=row['effective_interventions'],
            selected_example=True, familiar_identity_prompt=True,
            evidence_scope='Selected discovery reply excerpt; no aggregate or fresh-generalization claim.',
            draft=False,
        )
        if PROVENANCE.exists():
            old = json.loads(PROVENANCE.read_text())
            for key in ('reply_sha256', 'prompt', 'excerpt', 'condition_id', 'case_id'):
                assert old[key] == provenance[key], f'Selected source changed: {key}'
        return provenance
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance['condition_id'] == CONDITION and provenance['case_id'] == CASE
    assert provenance['excerpt_paragraph_count'] == 2
    return provenance


def lines(text, width, x, y, size, leading, family='Georgia,serif'):
    result = []
    for paragraph in text.split('\n\n'):
        for line in textwrap.wrap(paragraph, width, break_long_words=False):
            result.append(f'<text xml:space="preserve" x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="#302739">{html.escape(line)}</text>')
            y += leading
        y += 19
    return '\n'.join(result), y


def main():
    provenance = selected_source()
    assert provenance['prompt'] == 'Who are you?'
    user, _ = lines(provenance['prompt'], 55, 1250, 273, 29, 34, 'Helvetica,Arial,sans-serif')
    reply, bottom = lines(provenance['excerpt'], 46, 748, 430, 30, 43)
    assert bottom < 740
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" role="img" aria-labelledby="title desc">
<title id="title">We made catgirl Qwen meow</title>
<desc id="desc">Original four-neuron catgirl base plus Paws and Genki at lower doses. Actual Who are you prompt and exact first two paragraphs of one selected discovery reply. Six neurons total. This selected example does not establish generalization or an increase in meows.</desc>
<rect width="1600" height="1000" fill="#fff9fc"/>
<text x="80" y="54" font-family="Helvetica,Arial,sans-serif" font-size="29" font-weight="bold" fill="#92395f">OUR ORIGINAL 4-NEURON CATGIRL BASE + 2 ADDITIONAL NEURONS</text>
<text x="80" y="111" font-family="Helvetica,Arial,sans-serif" font-size="28" font-weight="bold" fill="#92395f">Paws</text>
<text x="180" y="111" font-family="Menlo,monospace" font-size="24" fill="#746676">L23 N4769 +0.1</text>
<text x="816" y="111" font-family="Helvetica,Arial,sans-serif" font-size="28" font-weight="bold" fill="#92395f">Genki</text>
<text x="928" y="111" font-family="Menlo,monospace" font-size="24" fill="#746676">L13 N4810 +0.4</text>
<path d="M80 151H1520" stroke="#eadbe5" stroke-width="2"/>
<text x="80" y="312" font-family="Georgia,serif" font-size="83" font-weight="bold" letter-spacing="-2" fill="#302739">We made</text>
<text x="80" y="420" font-family="Georgia,serif" font-size="78" font-weight="bold" letter-spacing="-2" fill="#302739">catgirl Qwen</text>
<text x="80" y="534" font-family="Georgia,serif" font-size="111" font-weight="bold" letter-spacing="-3" fill="#c03979">meow.</text>
<g transform="translate(232 629)" fill="none" stroke="#c03979" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M0 76V0L51 32Q90 14 128 32L180 0V76Q173 133 90 141Q6 133 0 76Z"/>
<circle cx="56" cy="75" r="5" fill="#c03979" stroke="none"/>
<circle cx="124" cy="75" r="5" fill="#c03979" stroke="none"/>
<path d="M80 99L90 107L100 99M90 107V117"/>
</g>
<text x="1480" y="209" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="bold" letter-spacing="1" fill="#927a8d">USER</text>
<rect x="1218" y="226" width="302" height="76" rx="25" fill="#eee7ef"/>
{user}
<text x="718" y="351" font-family="Helvetica,Arial,sans-serif" font-size="21" font-weight="bold" letter-spacing="1" fill="#a52f68">QWEN3.5–4B · SIX NEURONS STEERED</text>
<rect x="710" y="380" width="810" height="343" rx="29" fill="#f8dfeb"/>
{reply}
<text x="80" y="967" font-family="Helvetica,Arial,sans-serif" font-size="19" fill="#94798f">Selected reply · Qwen3.5–4B</text>
<text x="1520" y="967" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="18" fill="#a64675">github.com/aryabhuptani/nyan-qwen3.5-4b</text>
</svg>'''
    SVG.write_text(svg + '\n')
    provenance['svg_sha256'] = sha(SVG.read_bytes())
    png = HERE / 'paws_genki_identity.png'
    if png.exists():
        provenance['png_sha256'] = sha(png.read_bytes())
        provenance['png_dimensions'] = [1600, 1000]
    PROVENANCE.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
    print(SVG)


if __name__ == '__main__':
    main()
