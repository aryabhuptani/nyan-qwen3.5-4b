"""Render an exact, selected reply excerpt; preserve the separate numeric card."""
import hashlib
import html
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent


def lines(text, width, x, y, size, leading, family='Helvetica,Arial,sans-serif', color='#302739'):
    result = []
    for paragraph in text.split('\n\n'):
        for line in textwrap.wrap(paragraph, width, break_long_words=False):
            result.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{color}">{html.escape(line)}</text>')
            y += leading
        y += 13
    return '\n'.join(result), y


def main():
    source = HERE / 'results.local/validation_scored/replies_and_labels.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    row = next(r for r in rows if r['condition_id'] == '10100' and r['case_id'] == '114_validation_10')
    prompt = row['messages'][-1]['content']
    excerpt = '\n\n'.join(row['text'].split('\n\n')[:3])
    assert row['text'].startswith(excerpt)
    user, bottom_user = lines(prompt, 55, 765, 260, 25, 34)
    reply, bottom_reply = lines(excerpt, 46, 748, 520, 30, 40, family='Georgia,serif')
    assert bottom_user < 410 and bottom_reply < 900
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" role="img" aria-labelledby="title desc">
<title id="title">Paws and Genki added to our original four-neuron catgirl base</title>
<desc id="desc">Two additional neurons, six total. Actual prompt and verbatim beginning of a fresh Qwen3.5-4B reply containing Mew and Meow. Selected example, not an aggregate result.</desc>
<rect width="1600" height="1000" fill="#fff9fc"/>
<text x="80" y="54" font-family="Helvetica,Arial,sans-serif" font-size="29" font-weight="bold" fill="#92395f">OUR ORIGINAL 4-NEURON CATGIRL BASE + 2 ADDITIONAL NEURONS</text>
<text x="80" y="111" font-family="Helvetica,Arial,sans-serif" font-size="28" font-weight="bold" fill="#92395f">Paws</text>
<text x="180" y="111" font-family="Menlo,monospace" font-size="24" fill="#746676">L23 N4769 +0.4</text>
<text x="816" y="111" font-family="Helvetica,Arial,sans-serif" font-size="28" font-weight="bold" fill="#92395f">Genki</text>
<text x="928" y="111" font-family="Menlo,monospace" font-size="24" fill="#746676">L13 N4810 +0.8</text>
<path d="M80 151H1520" stroke="#eadbe5" stroke-width="2"/>
<text x="80" y="312" font-family="Georgia,serif" font-size="79" font-weight="bold" letter-spacing="-2" fill="#302739">Paws + Genki</text>
<text x="80" y="420" font-family="Georgia,serif" font-size="87" font-weight="bold" letter-spacing="-2" fill="#302739">made Qwen</text>
<text x="80" y="534" font-family="Georgia,serif" font-size="111" font-weight="bold" letter-spacing="-3" fill="#c03979">mew.</text>
<g transform="translate(232 629)" fill="none" stroke="#c03979" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M0 76V0L51 32Q90 14 128 32L180 0V76Q173 133 90 141Q6 133 0 76Z"/>
<circle cx="56" cy="75" r="5" fill="#c03979" stroke="none"/>
<circle cx="124" cy="75" r="5" fill="#c03979" stroke="none"/>
<path d="M80 99L90 107L100 99M90 107V117"/>
</g>
<text x="1480" y="209" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="18" font-weight="bold" letter-spacing="1" fill="#927a8d">USER</text>
<rect x="732" y="226" width="788" height="175" rx="25" fill="#eee7ef"/>
{user}
<text x="718" y="459" font-family="Helvetica,Arial,sans-serif" font-size="21" font-weight="bold" letter-spacing="1" fill="#a52f68">QWEN3.5–4B · SIX NEURONS STEERED</text>
<rect x="710" y="482" width="810" height="412" rx="29" fill="#f8dfeb"/>
{reply}
<text x="80" y="967" font-family="Helvetica,Arial,sans-serif" font-size="19" fill="#94798f">Actual reply excerpt · Published ingredient names: Neuronary</text>
<text x="1520" y="967" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="18" fill="#a64675">github.com/aryabhuptani/nyan-qwen3.5-4b</text>
</svg>'''
    out = HERE / 'share'
    path = out / 'paws_genki_chat.svg'
    path.write_text(svg + '\n')
    (out / 'CHAT_IMAGE_PROVENANCE.json').write_text(json.dumps(dict(
        study=114, condition_id=row['condition_id'], case_id=row['case_id'],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reply_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
        prompt=prompt, excerpt=excerpt, excerpt_is_exact_prefix=True,
        base_coordinates=4, additional_coordinates=2, total_coordinates=6,
        actual_sound_spellings=['Mew', 'Meow'], selected_example=True,
        svg_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    ), indent=2, ensure_ascii=False) + '\n')
    print(path)


if __name__ == '__main__':
    main()
