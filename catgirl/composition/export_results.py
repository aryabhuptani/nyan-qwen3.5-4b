"""Export native-study generations to an ordinary CSV; no model imports."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('native_generations.csv')
FIELDS = ['source', 'source_line', 'condition', 'prompt_id', 'section', 'turn',
          'prompt_tokens', 'generated_tokens', 'finish_reason', 'early_tokens',
          'prefill_scale', 'tail_scale', 'layer_schedules', 'text']


def export():
    paths = [ROOT / 'catgirl/results/075_single_characterization.jsonl']
    paths += sorted(Path(__file__).with_name('results').glob('*.jsonl'))
    count = 0
    with OUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for path in paths:
            if not path.exists():
                continue
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                row = json.loads(line)
                result = row.get('result', row)
                if 'text' not in result:
                    continue
                record = {key: row.get(key, result.get(key, '')) for key in FIELDS}
                record.update(source=str(path.relative_to(ROOT)), source_line=line_number,
                              text=result['text'])
                record['layer_schedules'] = json.dumps(row.get('layer_schedules', row.get('effective_scope', {}).get('layer_schedules', {})), sort_keys=True)
                writer.writerow(record)
                count += 1
    print(json.dumps({'path': str(OUT.relative_to(ROOT)), 'generations': count}))


if __name__ == '__main__':
    export()
