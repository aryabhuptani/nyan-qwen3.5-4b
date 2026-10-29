"""CPU schedule assertions plus explicit GPU comparisons to historical hooks."""
import argparse
import json
from pathlib import Path
from run_circuits import HERE, CATGIRL, BASE, recipe, scales, make_runner, sha


def cpu_checks():
    assert scales('prefill_and_first16', [0, 98, 99, 114, 115], 99) == [1., 1., 1., 1., 0.]
    assert scales('prefill_and_first8', [98, 99, 106, 107], 99) == [1., 1., 1., 0.]
    assert scales('all', [0, 99, 115, 10000], 99) == [1.] * 4
    source = json.loads((HERE / 'MODIFIER_SOURCES.json').read_text())['modifiers']
    full = recipe(dict(base=True, modifiers={name: 1. for name in source}), source)
    assert len(full) == 9
    assert len([s for s in full if s['layer'] == 26]) == 2
    assert len([s for s in full if s['layer'] == 22]) == 2
    assert [s['scope'] for s in full if s['neuron'] == 8295] == ['prefill_and_first16']
    assert [s['scope'] for s in full if s['neuron'] == 1911] == ['all']
    return dict(cpu_pass=True, unique_coordinates=len(full))


def gpu_checks(model):
    import mlx.core as mx
    import numpy as np
    from runner import SteeredMLP
    runner = make_runner(model, CATGIRL / 'public_qwen35_norms.json')
    new = runner.mlps
    old = [SteeredMLP(mlp, i) for i, mlp in enumerate(new)]
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello. How are you today?'}]
    ids = mx.array(runner.tokens(messages))[None]

    def forward(wrappers):
        for layer, wrapper in zip(runner.layers, wrappers):
            layer.mlp = wrapper
            wrapper.seen = 0
            wrapper.boundary = ids.shape[1] - 1
        result = runner.model(ids)
        mx.eval(result)
        return np.array(result.astype(mx.float32))

    baseline_new = forward(new)
    baseline_old = forward(old)
    noop_error = float(np.max(np.abs(baseline_new - baseline_old)))
    assert noop_error == 0., 'Empty wrapper mismatch'
    sources = json.loads((HERE / 'MODIFIER_SOURCES.json').read_text())['modifiers']
    effective = runner.set_recipe(recipe(dict(base=True, modifiers={}), sources))
    for item in effective:
        layer, neuron = item['layer'], item['neuron']
        if old[layer].delta is None:
            old[layer].delta = mx.zeros(9216)
        old[layer].delta = old[layer].delta.at[neuron].add(item['amount_float32'])
    old[22].early_tokens = 16
    old[22].prefill_scale = 1.
    old[22].tail_scale = 0.
    base_new = forward(new)
    base_old = forward(old)
    base_error = float(np.max(np.abs(base_new - base_old)))
    assert base_error == 0., 'Original base steering mismatch'
    runner.trace_summary()

    # Directly invoke the actual layer22 wrapper beyond the female cutoff.
    runner.set_recipe(recipe(dict(base=True, modifiers={'pride': 1.}), sources))
    layer = new[22]
    layer.boundary = 10
    layer.seen = 30
    x = mx.ones((1, 2, 2560), dtype=mx.bfloat16)
    mx.eval(layer(x))
    summaries = runner.trace_summary()
    female = next(t for t in summaries if t['neuron'] == 8295)
    pride = next(t for t in summaries if t['neuron'] == 1911)
    assert female['active_positions'] == 0 and female['actual_delta_rms'] == 0
    assert pride['active_positions'] == 2 and pride['actual_delta_rms'] > 0
    for l, wrapper in zip(runner.layers, new):
        l.mlp = wrapper
    return dict(gpu_pass=True, noop_max_logit_error=noop_error,
                original_base_max_logit_error=base_error,
                late_female_summary=female, late_pride_summary=pride,
                runner_metadata=runner.metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = cpu_checks()
    if a.model:
        result.update(gpu_checks(a.model))
    result.update(integrity_sha256=sha(__file__), runner_sha256=sha(HERE / 'run_circuits.py'))
    if a.output:
        with a.output.open('x') as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
