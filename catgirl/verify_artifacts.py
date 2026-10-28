"""CPU-only integrity verification of the committed reproduction bundle."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / 'catgirl/reproducibility_manifest.json').read_text())
    errors = []
    for item in manifest['files']:
        path = root / item['path']
        if not path.is_file():
            errors.append(f"Missing: {item['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f"SHA256 mismatch: {item['path']}")
    from demo import load_candidate
    _, candidate = load_candidate(root / 'catgirl/candidates/catgirl_v1.json')
    expected = {}
    for spec in candidate['condition']['neurons']:
        key = (spec['layer'], spec['neuron'])
        row = expected.setdefault(key, [1.0, 0.0])
        if spec.get('mode') == 'mul':
            row[0] = spec['factor']
        else:
            row[1] += spec['amount']
    with (root / 'catgirl/candidates/catgirl_v1_coordinates.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    actual = {(int(r['layer']), int(r['neuron'])): [float(r['gain']), float(r['bias'])] for r in rows}
    if actual != expected or len(rows) != 4096:
        errors.append('CSV does not exactly match frozen recipe')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"PASS: {len(manifest['files'])} files match SHA-256; all 4096 CSV coordinates exactly match frozen JSON.")
    print('Model IDs and revisions are pinned in the manifest; model weights are downloaded separately.')


if __name__ == '__main__':
    main()
