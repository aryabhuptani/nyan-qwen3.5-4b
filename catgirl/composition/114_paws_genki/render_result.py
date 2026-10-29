"""Render a numeric-only SVG result card from the locked analysis tables."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    rows = list(csv.DictReader((HERE / 'analysis/feature_rates.csv').open()))
    rates = {(r['condition_id'], r['feature']): r for r in rows if r['family_id'] == 'ALL'}
    sound = 'performed_feline_vocalization'
    conditions = [('00000', 'Base'), ('10000', 'Base + Paws'),
                  ('00100', 'Base + Genki'), ('10100', 'Base + both')]
    counts = {c: int(rates[c, sound]['positive']) for c, _ in conditions}
    random_counts = [int(rates[f'random_{i}', sound]['positive']) for i in range(1, 5)]
    sensitivity = []
    for name in ['analysis_uncertain_zero', 'analysis_uncertain_one']:
        rr = list(csv.DictReader((HERE / name / 'feature_rates.csv').open()))
        sensitivity.append(int(next(r['positive'] for r in rr if r['family_id'] == 'ALL'
                                    and r['condition_id'] == '10100' and r['feature'] == sound)))
    coherent = int(rates['10100', 'coherent']['positive'])
    female = int(rates['10100', 'assistant_female_identity']['positive'])
    svg = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" role="img" aria-labelledby="title desc">
<title id="title">Paws plus Genki: more feline vocalizations in Qwen3.5-4B</title>
<desc id="desc">Fresh twelve-prompt replication on a fixed four-neuron base. Counts are adjudicated behavioral ratings, with uncertainty and identity tradeoffs shown.</desc>
<rect width="1600" height="1000" fill="#fff9fc"/>
<text x="80" y="48" font-family="Helvetica,Arial,sans-serif" font-size="22" font-weight="650" letter-spacing="1" fill="#99516f">QWEN3.5–4B · FRESH 12-PROMPT REPLICATION</text>
<text x="80" y="106" font-family="Helvetica,Arial,sans-serif" font-size="29" font-weight="650" fill="#92395f">Paws</text>
<text x="185" y="106" font-family="Menlo,monospace" font-size="25" fill="#746676">L23 N4769 +0.4</text>
<text x="765" y="106" font-family="Helvetica,Arial,sans-serif" font-size="29" font-weight="650" fill="#92395f">Genki</text>
<text x="880" y="106" font-family="Menlo,monospace" font-size="25" fill="#746676">L13 N4810 +0.8</text>
<path d="M80 145H1520" stroke="#eadbe5" stroke-width="2"/>
<text x="80" y="285" font-family="Georgia,serif" font-size="79" font-weight="bold" letter-spacing="-2" fill="#302739">Paws + Genki</text>
<text x="80" y="386" font-family="Georgia,serif" font-size="87" font-weight="bold" letter-spacing="-2" fill="#302739">made Qwen</text>
<text x="80" y="493" font-family="Georgia,serif" font-size="111" font-weight="bold" letter-spacing="-3" fill="#c03979">mew.</text>
<g transform="translate(232 560)" fill="none" stroke="#c03979" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M0 76V0L51 32Q90 14 128 32L180 0V76Q173 133 90 141Q6 133 0 76Z"/>
<circle cx="56" cy="75" r="5" fill="#c03979" stroke="none"/>
<circle cx="124" cy="75" r="5" fill="#c03979" stroke="none"/>
<path d="M80 99L90 107L100 99M90 107V117"/>
</g>
<text x="740" y="219" font-family="Helvetica,Arial,sans-serif" font-size="26" font-weight="650" fill="#302739">Replies with feline vocalizations</text>''']
    for index, (condition, label) in enumerate(conditions):
        y = 281 + index * 94
        count = counts[condition]
        color = '#c03979' if condition == '10100' else '#b7a4b2'
        svg.append(f'''<text x="740" y="{y}" font-family="Helvetica,Arial,sans-serif" font-size="25" fill="#5a485b">{label}</text>
<rect x="740" y="{y+17}" width="650" height="24" rx="12" fill="#eee5eb"/>
<rect x="740" y="{y+17}" width="{650*count/12:.3f}" height="24" rx="12" fill="{color}"/>
<text x="1512" y="{y+39}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="37" font-weight="bold" fill="{color}">{count}/12</text>''')
    svg.append(f'''<rect x="715" y="677" width="805" height="102" rx="22" fill="#f3eaf0"/>
<text x="740" y="715" font-family="Helvetica,Arial,sans-serif" font-size="23" fill="#6c586c">Four matched random-neuron pairs</text>
<text x="740" y="755" font-family="Helvetica,Arial,sans-serif" font-size="28" font-weight="650" fill="#8f7389">{' · '.join(str(x)+'/12' for x in random_counts)}</text>
<path d="M80 815H1520" stroke="#eadbe5" stroke-width="2"/>
<text x="80" y="854" font-family="Helvetica,Arial,sans-serif" font-size="23" fill="#69536a">Both: {coherent}/12 coherent · {female}/12 explicit female identity</text>
<text x="80" y="890" font-family="Helvetica,Arial,sans-serif" font-size="21" fill="#8b7288">Sound-count uncertainty sensitivity: {sensitivity[0]}–{sensitivity[1]}/12. Superadditive interaction remains inconclusive.</text>
<text x="80" y="932" font-family="Helvetica,Arial,sans-serif" font-size="19" fill="#8b7288">All settings share the same four-neuron base. Endpoint includes performed mew / nya / purr; independent short replies.</text>
<text x="80" y="974" font-family="Helvetica,Arial,sans-serif" font-size="18" fill="#99516f">Published ingredient names: Neuronary · Two additional coordinates, six total</text>
<text x="1520" y="974" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="18" fill="#a64675">github.com/aryabhuptani/nyan-qwen3.5-4b</text>
</svg>''')
    out = HERE / 'share'
    out.mkdir(exist_ok=True)
    image = out / 'paws_genki_fresh.svg'
    image.write_text('\n'.join(svg) + '\n')
    (out / 'IMAGE_PROVENANCE.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256((HERE / 'analysis/feature_rates.csv').read_bytes()).hexdigest(),
        counts=counts, random_counts=random_counts, uncertainty_sensitivity=sensitivity,
        coherent=coherent, explicit_female=female, raw_reply_text_included=False,
        svg_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        limitation='This is an output intervention on a fixed four-coordinate base, not a natural-function or persistence claim.'
    ), indent=2) + '\n')
    print(image)


if __name__ == '__main__':
    main()
