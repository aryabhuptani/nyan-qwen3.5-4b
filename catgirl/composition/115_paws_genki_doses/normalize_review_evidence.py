"""Lossless evidence-format adapter; leaves every reviewer rating unchanged."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    original = json.loads(args.source.read_text())
    adapted = json.loads(args.source.read_text())
    for item in adapted['items']:
        item['evidence'] = {
            feature: value if isinstance(value, str) else
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            for feature, value in item['evidence'].items()
        }
    assert [r['ratings'] for r in original['items']] == [r['ratings'] for r in adapted['items']]
    content = json.dumps(adapted, ensure_ascii=False, indent=2) + '\n'
    if args.destination.exists():
        assert args.destination.read_text() == content
    else:
        args.destination.write_text(content)


if __name__ == '__main__':
    main()
