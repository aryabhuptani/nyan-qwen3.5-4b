"""Causal optimization of sparse additive neuron offsets, never model weights.

Runner.run can dispatch ``return optimize(self, job, output)``. Suggested job:
{"action":"optimize", "layers":[20,21,22,23,24,25,26,27], "iterations":18}.
The initial six gradient probes and at most 24 optimization updates use one
matched pair at a time. Exported doses use Runner.steer's exact convention.
"""
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def optimize(runner, job, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(job.get('discovery', Path(__file__).with_name('discovery.json')))
    data = json.loads(source.read_text())
    pairs = data['pairs'][:max(1, min(6, int(job.get('pairs', 6))))]
    layers = [int(x) for x in job.get('layers', list(range(20, 28)))]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError('layers must be a nonempty list of distinct indices')
    width = runner.config.get('text_config', runner.config)['intermediate_size']
    target_limit = max(4, int(job.get('target_tokens', 40)))
    iterations = max(0, min(30-len(pairs), int(job.get('iterations', 18))))
    max_size = min(128, int(job.get('neurons', 128)))
    if max_size < 1:
        raise ValueError('neurons must be positive')
    penalty = float(job.get('l2', 0.08))
    neutral_weight = float(job.get('neutral_weight', 0.15))
    learning_rate = float(job.get('learning_rate', 0.008))
    layer_budget = float(job.get('layer_dose_norm', 0.18))
    total_budget = float(job.get('total_dose_norm', 0.4))
    previous = [m.delta for m in runner.mlps]
    was_training = bool(getattr(runner.model, 'training', False))
    started = time.time()
    log = []

    def emit(record):
        record = dict(record, elapsed_seconds=time.time()-started)
        log.append(record)
        with output.open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    # A coordinate of one means exactly one upstream residual-relative dose.
    # Dequantize once per layer, rather than once for every candidate column.
    units = []
    column_norms = []
    for layer in layers:
        proj = runner.mlps[layer].down_proj
        if hasattr(proj, 'scales'):
            weight = mx.dequantize(proj.weight, proj.scales,
                                   getattr(proj, 'biases', None),
                                   group_size=proj.group_size, bits=proj.bits)
        else:
            weight = proj.weight
        norm = mx.linalg.norm(weight.astype(mx.float32), axis=0)
        mx.eval(norm)
        norms = np.maximum(np.array(norm), 1e-8)
        column_norms.append(norms)
        units.append(float(runner.residual[layer])/norms)
    units_np = np.stack(units).astype(np.float32)
    units_mx = mx.array(units_np)

    examples = []
    for pair in pairs:
        messages = [{'role':'system', 'content':data['neutral_system']},
                    {'role':'user', 'content':pair['prompt']}]
        prefix = list(runner.tokens(messages))
        poles = []
        for pole in ('catgirl', 'neutral'):
            target = runner.tok.encode(pair[pole], add_special_tokens=False)[:target_limit]
            if not target:
                raise ValueError('Empty continuation for '+pair['id'])
            # Position prefix_len-1 predicts target[0]; the final target token
            # is a label and must not be fed as an unshifted prediction input.
            ids = mx.array(prefix+target[:-1], dtype=mx.int32)[None]
            labels = mx.array(target, dtype=mx.int32)[None]
            poles.append((ids, labels, len(prefix)-1))
        examples.append((pair['id'], poles))

    def nll(theta, example):
        for m in runner.mlps:
            m.delta = None
        amounts = theta*units_mx
        for row, layer in enumerate(layers):
            runner.mlps[layer].delta = amounts[row]
        ids, labels, start = example
        logits = runner.model(ids, cache=None)[:, start:start+labels.shape[1], :]
        return nn.losses.cross_entropy(logits.astype(mx.float32), labels, reduction='mean')

    # Cap the reward for suppressing neutral answers. This prevents maximizing
    # contrast by destroying ordinary task answers instead of raising catgirl
    # likelihood. Each likelihood is length normalized independently.
    def objective(theta, poles, baseline_neutral):
        cat = nll(theta, poles[0])
        neutral = nll(theta, poles[1])
        suppression = mx.minimum(neutral-baseline_neutral, mx.array(0.4))
        return cat-neutral_weight*suppression+penalty*mx.sum(theta*theta)

    try:
        for m in runner.mlps:
            m.capture = False
            m.captured = None
            m.delta = None
        # Select the differentiable reference recurrence, not its inference-only
        # custom Metal kernel. Qwen3.5 has no dropout in this path.
        runner.model.train()
        # Before the first intervention all activations are constant with
        # respect to theta, so keep those layers on fast inference kernels.
        for layer in runner.layers[:min(layers)]:
            layer.eval()
        zeros = mx.zeros((len(layers), width), dtype=mx.float32)
        gradients = []
        baselines = []
        for name, poles in examples:
            cat0, neutral0 = nll(zeros, poles[0]), nll(zeros, poles[1])
            mx.eval(cat0, neutral0)
            baselines.append((float(cat0), float(neutral0)))
            fn = mx.value_and_grad(lambda theta: objective(theta, poles, float(neutral0)))
            value, grad = fn(zeros)
            mx.eval(value, grad)
            g = np.array(grad)
            if not np.isfinite(g).all():
                raise FloatingPointError('Nonfinite seed gradient for '+name)
            gradients.append(g)
            emit(dict(event='gradient_probe', pair=name, catgirl_nll=float(cat0),
                      neutral_nll=float(neutral0), objective=float(value),
                      gradient_norm=float(np.linalg.norm(g))))
        probes = np.stack(gradients)
        mean_gradient = probes.mean(axis=0)
        consistency = (np.sign(probes)==np.sign(mean_gradient)[None]).mean(axis=0)
        score = np.abs(mean_gradient)*consistency**2
        selected = np.argsort(score.ravel())[-max_size:]
        mask_np = np.zeros_like(score, dtype=np.float32)
        mask_np.ravel()[selected] = 1
        mask = mx.array(mask_np)
        theta_np = np.zeros_like(mask_np)
        momentum = np.zeros_like(mask_np)
        variance = np.zeros_like(mask_np)
        emit(dict(event='gradient_selected', count=len(selected), layers=layers))
        for step in range(iterations):
            index = step % len(examples)
            name, poles = examples[index]
            neutral0 = baselines[index][1]
            fn = mx.value_and_grad(lambda theta: objective(theta*mask, poles, neutral0))
            value, grad = fn(mx.array(theta_np))
            mx.eval(value, grad)
            g = np.array(grad)*mask_np
            if not np.isfinite(g).all() or not np.isfinite(float(value)):
                raise FloatingPointError('Nonfinite optimization step')
            t = step+1
            momentum = .9*momentum+.1*g
            variance = .999*variance+.001*g*g
            theta_np -= learning_rate*(momentum/(1-.9**t))/(np.sqrt(variance/(1-.999**t))+1e-8)
            theta_np *= mask_np
            # Triangle bounds on column-normalized coordinates control dose
            # energy; actual summed residual-vector norms are reported below.
            per_layer = np.linalg.norm(theta_np, axis=1, keepdims=True)
            theta_np *= np.minimum(1, layer_budget/np.maximum(per_layer, 1e-8))
            theta_np *= min(1., total_budget/max(float(np.linalg.norm(theta_np)), 1e-8))
            emit(dict(event='gradient_update', step=t, pair=name,
                      objective=float(value), total_dose_norm=float(np.linalg.norm(theta_np))))

        # A zero-update run still yields a causal first-order candidate.
        if not iterations:
            theta_np = -np.sign(mean_gradient)*mask_np*learning_rate
        ordered = np.argsort(np.abs(theta_np).ravel())[::-1]
        ordered = [int(i) for i in ordered if theta_np.ravel()[i] != 0]
        recipes = []
        for size in sorted(set((min(32, max_size), max_size))):
            recipe = []
            for flat in ordered[:size]:
                row, neuron = np.unravel_index(flat, theta_np.shape)
                layer = layers[row]
                dose = float(theta_np[row, neuron])
                recipe.append(dict(layer=layer, neuron=int(neuron), dose=dose,
                                   amplitude=float(dose*units_np[row, neuron]),
                                   down_column_norm=float(column_norms[row][neuron]),
                                   residual_norm=float(runner.residual[layer]),
                                   seed_gradient=float(mean_gradient[row, neuron]),
                                   gradient_consistency=float(consistency[row, neuron])))
            write_norms = {}
            for layer in layers:
                members = [r for r in recipe if r['layer']==layer]
                if members:
                    write = sum((runner.column(layer, r['neuron'])*r['amplitude'] for r in members))
                    mx.eval(write)
                    write_norms[str(layer)] = float(mx.linalg.norm(write))/float(runner.residual[layer])
            recipes.append(dict(id='gradient_'+str(size), neurons=recipe,
                                actual_residual_write_fractions=write_norms))
        result = dict(method='sparse additive MLP offset gradient optimization',
                      model=runner.metadata.get('model'), layers=layers,
                      training_pairs=[p['id'] for p in pairs], target_tokens=target_limit,
                      updates=iterations, baseline_nll=[dict(pair=e[0], catgirl=c, neutral=n)
                                                       for e,(c,n) in zip(examples,baselines)],
                      conditions=recipes, validated=False,
                      note='Candidates require held-out generation evaluation; original weights are unchanged.')
        destination = output.with_suffix('.recipes.json')
        destination.write_text(json.dumps(result, indent=2))
        emit(dict(event='gradient_complete', recipes=str(destination), validated=False))
        return result
    finally:
        for m, delta in zip(runner.mlps, previous):
            m.delta = delta
        runner.model.train(was_training)
