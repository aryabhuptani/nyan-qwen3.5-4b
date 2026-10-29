"""Original BF16 additive neuron compositions; coordinate-specific temporal scope.

No model imports occur during --check-only. The historical Runner supplies token
formatting and generation; its calibration method is overridden to prevent writes
to existing experiment files. All doses use frozen published layer norms.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CATGIRL = HERE.parents[1]
sys.path.insert(0, str(CATGIRL))
from public_control_runner import load_norms

REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BASE = [(9, 9009, .7), (26, 6173, .3), (26, 8930, .7), (22, 8295, -.4)]
MODIFIERS = {'paws': (23, 4769), 'hugs': (4, 8016), 'genki': (13, 4810),
             'sounds': (7, 7848), 'pride': (22, 1911)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe(condition, sources):
    result = []
    if condition.get('base', True):
        for layer, neuron, dose in BASE:
            result.append(dict(layer=layer, neuron=neuron, reference_normalized_dose=dose,
                               scope='prefill_and_first16' if (layer, neuron) == (22, 8295) else 'all'))
    for name, fraction in condition.get('modifiers', {}).items():
        if name not in MODIFIERS or isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Invalid modifier fraction')
        source = sources[name]
        layer, neuron = MODIFIERS[name]
        coordinate = source.get('coordinate', source)
        if (coordinate['layer'], coordinate['neuron']) != (layer, neuron):
            raise ValueError('Source coordinate mismatch')
        reference = next((source.get(key) for key in ('published_reference_dose', 'project_reference_dose', 'reference_dose')
                          if source.get(key) is not None), coordinate.get('dose'))
        dose = reference * fraction
        if not math.isfinite(dose):
            raise ValueError('Nonfinite dose')
        if dose:
            result.append(dict(layer=layer, neuron=neuron, reference_normalized_dose=dose,
                               scope=condition.get('modifier_scopes', {}).get(name, 'all'),
                               ingredient=name, reference_fraction=fraction))
    for spec in result:
        scales(spec['scope'], [], 0)
    return result


def scales(scope, positions, boundary):
    if scope == 'all':
        return [1.] * len(positions)
    if scope not in ('prefill_and_first16', 'prefill_and_first8'):
        raise ValueError('Unknown coordinate scope')
    early = 8 if scope.endswith('8') else 16
    return [float(position < boundary + early) for position in positions]


def load_inputs(args):
    job = json.loads(args.job.read_text())
    case_payload = json.loads(args.cases.read_text())
    cases = case_payload['cases'] if isinstance(case_payload, dict) else case_payload
    source_payload = json.loads(args.sources.read_text())
    for key, path in [('cases_sha256', args.cases), ('sources_sha256', args.sources)]:
        if key in job and job[key] != sha(path):
            raise ValueError('Job frozen input hash mismatch: ' + key)
    sources = source_payload.get('modifiers', source_payload.get('ingredients', source_payload))
    if isinstance(sources, list):
        sources = {item['id']: item for item in sources}
    names = [c['condition_id'] for c in job['conditions']]
    ids = [c['case_id'] for c in cases]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate condition or case IDs')
    for condition in job['conditions']:
        recipe(condition, sources)
    for case in cases:
        if not isinstance(case['messages'], list) or not case['messages']:
            raise ValueError('Explicit complete messages required')
    if job.get('temperature', 0) != 0:
        raise ValueError('Frozen greedy generation required')
    return job, cases, sources


def make_runner(model, norms_path):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from runner import Runner
    from composition_worker import configure_public_runner

    class CoordinateMLP(nn.Module):
        def __init__(self, original, layer):
            super().__init__()
            self.gate_proj = original.gate_proj
            self.up_proj = original.up_proj
            self.down_proj = original.down_proj
            self.layer = layer
            self.specs = []
            self.delta_all = None
            self.delta_early = None
            self.delta_first8 = None
            self.seen = 0
            self.boundary = 0
            self.trace = []

        def __call__(self, x):
            h = nn.silu(self.gate_proj(x)) * self.up_proj(x)
            old = h
            if self.delta_all is not None:
                h = h + self.delta_all.astype(h.dtype)
            if self.delta_early is not None:
                positions = mx.arange(self.seen, self.seen + h.shape[1])
                mask = (positions < self.boundary + 16)[None, :, None].astype(h.dtype)
                h = h + self.delta_early.astype(h.dtype) * mask
            if self.delta_first8 is not None:
                positions = mx.arange(self.seen, self.seen + h.shape[1])
                mask = (positions < self.boundary + 8)[None, :, None].astype(h.dtype)
                h = h + self.delta_first8.astype(h.dtype) * mask
            if self.specs:
                indices = mx.array([s['neuron'] for s in self.specs])
                before = mx.take(old, indices, axis=-1).astype(mx.float32)
                after = mx.take(h, indices, axis=-1).astype(mx.float32)
                pair = mx.stack([before, after], axis=-1)
                mx.async_eval(pair)
                self.trace.append((self.seen, pair))
            self.seen += h.shape[1]
            return self.down_proj(h)

    class CircuitRunner(Runner):
        def calibrate(self):
            pass

        def set_recipe(self, specs):
            effective = []
            for mlp in self.mlps:
                mlp.specs = []
                mlp.delta_all = None
                mlp.delta_early = None
                mlp.delta_first8 = None
                mlp.trace = []
            for spec in specs:
                item = copy.deepcopy(spec)
                layer, neuron = item['layer'], item['neuron']
                column_l2 = float(mx.linalg.norm(self.column(layer, neuron)))
                amount = item['reference_normalized_dose'] * self.residual[layer] / column_l2
                item.update(amount_float32=amount, column_l2=column_l2,
                            residual_norm=self.residual[layer],
                            amount_bfloat16=float(mx.array(amount).astype(mx.bfloat16)))
                self.mlps[layer].specs.append(item)
                effective.append(item)
            for mlp in self.mlps:
                vectors = {'all': np.zeros(9216, dtype=np.float32),
                           'prefill_and_first16': np.zeros(9216, dtype=np.float32),
                           'prefill_and_first8': np.zeros(9216, dtype=np.float32)}
                for item in mlp.specs:
                    vectors[item['scope']][item['neuron']] += item['amount_float32']
                if any(s['scope'] == 'all' for s in mlp.specs):
                    mlp.delta_all = mx.array(vectors['all'])
                if any(s['scope'] == 'prefill_and_first16' for s in mlp.specs):
                    mlp.delta_early = mx.array(vectors['prefill_and_first16'])
                if any(s['scope'] == 'prefill_and_first8' for s in mlp.specs):
                    mlp.delta_first8 = mx.array(vectors['prefill_and_first8'])
            mx.eval([m.delta_all for m in self.mlps if m.delta_all is not None],
                    [m.delta_early for m in self.mlps if m.delta_early is not None],
                    [m.delta_first8 for m in self.mlps if m.delta_first8 is not None])
            return effective

        def trace_summary(self):
            result = []
            for mlp in self.mlps:
                if not mlp.specs or not mlp.trace:
                    continue
                blocks = [np.array(block).reshape(-1, len(mlp.specs), 2)
                          for _, block in mlp.trace]
                data = np.concatenate(blocks, axis=0)
                positions = np.concatenate([np.arange(start, start + block.shape[1])
                                            for start, block in mlp.trace])
                for i, spec in enumerate(mlp.specs):
                    delta = data[:, i, 1] - data[:, i, 0]
                    expected = np.array(scales(spec['scope'], positions.tolist(), mlp.boundary))
                    active = expected != 0
                    result.append(dict(layer=mlp.layer, neuron=spec['neuron'], scope=spec['scope'],
                        positions=len(delta), active_positions=int(active.sum()),
                        unchanged_active_positions=int(np.sum((delta == 0) & active)),
                        inactive_nonzero_positions=int(np.sum((delta != 0) & ~active)),
                        actual_delta_min=float(delta.min()), actual_delta_max=float(delta.max()),
                        actual_delta_rms=float(np.sqrt(np.mean(delta ** 2))),
                        actual_active_delta_mean=float(delta[active].mean()) if active.any() else 0.,
                        pre_min=float(data[:, i, 0].min()), pre_max=float(data[:, i, 0].max()),
                        post_min=float(data[:, i, 1].min()), post_max=float(data[:, i, 1].max())))
                mlp.trace = []
            return result

    runner = CircuitRunner(model)
    runner.mlps = [CoordinateMLP(mlp, i) for i, mlp in enumerate(runner.mlps)]
    for layer, mlp in zip(runner.layers, runner.mlps):
        layer.mlp = mlp
    path, payload, medians = load_norms(norms_path)
    configure_public_runner(runner, path, payload, medians)
    return runner


def run(args):
    job, cases, sources = load_inputs(args)
    norm_path, _, _ = load_norms(args.norms)
    frozen = {'job_sha256': sha(args.job), 'cases_sha256': sha(args.cases),
              'sources_sha256': sha(args.sources), 'norms_sha256': sha(norm_path),
              'runner_sha256': sha(__file__)}
    if args.check_only:
        print(json.dumps(dict(valid=True, rows=len(cases) * len(job['conditions']), **frozen)))
        return
    if not args.model or Path(args.model).name != REVISION:
        raise ValueError('Supply pinned original BF16 snapshot directory')
    identity = dict(**frozen, model=str(Path(args.model).resolve()), revision=REVISION)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.output.with_suffix('.identity.json')
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Output identity differs; use a new output file')
    else:
        if args.output.exists() and args.output.stat().st_size:
            raise ValueError('Existing output lacks identity')
        manifest.write_text(json.dumps(identity, indent=2))
    completed = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            key = row['condition_id'], row['case_id']
            if key in completed:
                raise ValueError('Duplicate output identity')
            completed[key] = row
    expected = {(c['condition_id'], p['case_id']) for c in job['conditions'] for p in cases}
    if not set(completed) <= expected:
        raise ValueError('Unexpected existing output rows')
    if set(completed) == expected:
        print(json.dumps(dict(event='already_complete', rows=len(completed))))
        return
    runner = make_runner(args.model, norm_path)
    runner.repetition_penalty = job.get('repetition_penalty', 1.1)
    runner.repetition_context_size = job.get('repetition_context_size', 128)
    runner.layer_schedules = {}
    with args.output.open('a') as output:
        for condition in job['conditions']:
            for case in cases:
                key = condition['condition_id'], case['case_id']
                if key in completed:
                    continue
                effective = runner.set_recipe(recipe(condition, sources))
                result = runner.generate(case['messages'], case.get('max_tokens', job.get('max_tokens', 192)),
                                         seed=job.get('seed', 42), temperature=0)
                traces = runner.trace_summary()
                if any(t['inactive_nonzero_positions'] for t in traces):
                    raise RuntimeError('Coordinate schedule changed an inactive position')
                row = dict(condition_id=condition['condition_id'], case_id=case['case_id'],
                           family_id=case['family_id'], split=case['split'], messages=case['messages'],
                           effective_interventions=effective, intervention_trace_summary=traces,
                           input_hashes=frozen, max_tokens=case.get('max_tokens', job.get('max_tokens', 192)),
                           **result)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                completed[key] = row
                print(json.dumps(dict(event='progress', done=len(completed), total=len(expected),
                                      condition_id=key[0], case_id=key[1])), flush=True)
    args.output.with_suffix('.completion.json').write_text(json.dumps(
        dict(rows=len(completed), raw_sha256=sha(args.output), **identity), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--norms', type=Path, default=CATGIRL / 'public_qwen35_norms.json')
    parser.add_argument('--check-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
