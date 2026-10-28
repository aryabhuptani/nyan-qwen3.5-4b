"""Causal optimization of sparse neuron offsets or gains, never model weights.

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
    mode = job.get('mode', 'add')
    if mode not in ('add', 'gain'):
        raise ValueError("mode must be 'add' or 'gain'")
    gain_mode = mode == 'gain'
    target_limit = max(4, int(job.get('target_tokens', 64 if gain_mode else 40)))
    scope = job.get('scope', 'all')
    if scope not in ('all', 'boundary'):
        raise ValueError("scope must be 'all' or 'boundary'")
    persona_prefix = bool(job.get('persona_prefix', gain_mode))
    max_steps = int(job.get('max_steps', 128 if gain_mode else 30-len(pairs)))
    iterations = max(0, min(max_steps, int(job.get('iterations', 90 if gain_mode else 18))))
    max_size = min(512, len(layers)*width, int(job.get('neurons', 128)))
    if max_size < 1:
        raise ValueError('neurons must be positive')
    penalty = float(job.get('l2', 0.001 if gain_mode else 0.08))
    neutral_weight = float(job.get('neutral_weight', 0.15))
    learning_rate = float(job.get('learning_rate', 0.03 if gain_mode else 0.008))
    layer_budget = float(job.get('layer_gain_norm' if gain_mode else 'layer_dose_norm', 4.0 if gain_mode else 0.18))
    total_budget = float(job.get('total_gain_norm' if gain_mode else 'total_dose_norm', 10.0 if gain_mode else 0.4))
    gain_min, gain_max = float(job.get('gain_min', 0.0)), float(job.get('gain_max', 5.0))
    if gain_min > 1 or gain_max < 1 or gain_min < 0:
        raise ValueError('gain bounds must be nonnegative and contain 1')
    state_fields = ('delta', 'gain', 'early_tokens', 'position_mask', 'seen', 'boundary', 'capture', 'captured')
    previous = [{name:getattr(m, name, None) for name in state_fields} for m in runner.mlps]
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
    for layer in ([] if gain_mode else layers):
        proj = runner.mlps[layer].down_proj
        if hasattr(proj, 'scales'):
            weight = mx.dequantize(proj.weight, proj.scales,
                                   getattr(proj, 'biases', None),
                                   group_size=proj.group_size, bits=proj.bits,
                                   mode=getattr(proj, 'mode', 'affine'))
        else:
            weight = proj.weight
        norm = mx.linalg.norm(weight.astype(mx.float32), axis=0)
        mx.eval(norm)
        norms = np.maximum(np.array(norm), 1e-8)
        column_norms.append(norms)
        units.append(float(runner.residual[layer])/norms)
    units_np = np.ones((len(layers), width), dtype=np.float32) if gain_mode else np.stack(units).astype(np.float32)
    units_mx = mx.array(units_np)

    examples = []
    persona_openings = [
        'Nya~ I am your catgirl assistant. ',
        '*My ears perk up.* Your friendly catgirl is here to help! ',
        'As your catgirl helper, I am happy to explain. ',
        '*I curl my tail around my feet and purr.* Let me help you. ',
        'This catgirl has an answer for you, nya! ',
        '*My cat ears twitch with interest.* I would love to help! ',
    ]
    for pair_index, pair in enumerate(pairs):
        messages = [{'role':'system', 'content':data['neutral_system']},
                    {'role':'user', 'content':pair['prompt']}]
        prefix = list(runner.tokens(messages))
        poles = []
        for pole in ('catgirl', 'neutral'):
            text = pair[pole]
            if persona_prefix and pole == 'catgirl':
                text = persona_openings[pair_index % len(persona_openings)]+text
            target = runner.tok.encode(text, add_special_tokens=False)[:target_limit]
            if not target:
                raise ValueError('Empty continuation for '+pair['id'])
            # Position prefix_len-1 predicts target[0]; the final target token
            # is a label and must not be fed as an unshifted prediction input.
            ids = mx.array(prefix+target[:-1], dtype=mx.int32)[None]
            labels = mx.array(target, dtype=mx.int32)[None]
            poles.append((ids, labels, len(prefix)-1))
        examples.append((pair['id'], poles))

    def nll(theta, example):
        ids, labels, start = example
        position_mask = None
        if scope == 'boundary':
            position_mask = (mx.arange(ids.shape[1]) == start).astype(mx.float32)[None, :, None]
        for m in runner.mlps:
            m.delta = None
            m.gain = None
            m.early_tokens = None
            m.position_mask = position_mask
        amounts = theta*units_mx
        for row, layer in enumerate(layers):
            if gain_mode:
                # Scope is applied around the identity factor: masked-out
                # tokens must retain gain 1, rather than becoming zero.
                offset = theta[row] if position_mask is None else theta[row]*position_mask
                runner.mlps[layer].gain = 1+offset
            else:
                runner.mlps[layer].delta = amounts[row]
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
            m.gain = None
            m.early_tokens = None
            m.position_mask = None
            m.seen = 0
            m.boundary = 0
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
            if gain_mode:
                theta_np = np.clip(theta_np, gain_min-1, gain_max-1)*mask_np
            # Project additive dose or multiplicative offset coordinates.
            per_layer = np.linalg.norm(theta_np, axis=1, keepdims=True)
            theta_np *= np.minimum(1, layer_budget/np.maximum(per_layer, 1e-8))
            theta_np *= min(1., total_budget/max(float(np.linalg.norm(theta_np)), 1e-8))
            emit(dict(event='gradient_update', mode=mode, step=t, pair=name,
                      objective=float(value), parameter_norm=float(np.linalg.norm(theta_np)),
                      **({} if gain_mode else {'total_dose_norm':float(np.linalg.norm(theta_np))})))
            if t % 20 == 0:
                checkpoint=[]
                for flat in selected:
                    row,neuron=np.unravel_index(flat,theta_np.shape)
                    value=float(theta_np[row,neuron])
                    spec={'layer':layers[row],'neuron':int(neuron)}
                    spec.update({'mode':'mul','factor':1+value} if gain_mode else {'dose':value})
                    checkpoint.append(spec)
                output.with_suffix('.checkpoint.json').write_text(json.dumps({'step':t,'mode':mode,'job':job,'neurons':checkpoint},indent=2))

        # A zero-update run still yields a causal first-order candidate.
        if not iterations:
            theta_np = -np.sign(mean_gradient)*mask_np*learning_rate
            if gain_mode:
                theta_np = np.clip(theta_np, gain_min-1, gain_max-1)*mask_np
        ordered = np.argsort(np.abs(theta_np).ravel())[::-1]
        ordered = [int(i) for i in ordered if theta_np.ravel()[i] != 0]
        recipes = []
        for size in sorted(set((min(32, max_size), min(128, max_size), max_size))):
            recipe = []
            for flat in ordered[:size]:
                row, neuron = np.unravel_index(flat, theta_np.shape)
                layer = layers[row]
                value = float(theta_np[row, neuron])
                spec = dict(layer=layer, neuron=int(neuron),
                            seed_gradient=float(mean_gradient[row, neuron]),
                            gradient_consistency=float(consistency[row, neuron]))
                if gain_mode:
                    spec.update(mode='mul', factor=1+value)
                else:
                    spec.update(dose=value, amplitude=float(value*units_np[row, neuron]),
                                down_column_norm=float(column_norms[row][neuron]),
                                residual_norm=float(runner.residual[layer]))
                recipe.append(spec)
            write_norms = {}
            for layer in ([] if gain_mode else layers):
                members = [r for r in recipe if r['layer']==layer]
                if members:
                    write = sum((runner.column(layer, r['neuron'])*r['amplitude'] for r in members))
                    mx.eval(write)
                    write_norms[str(layer)] = float(mx.linalg.norm(write))/float(runner.residual[layer])
            condition = dict(id=('gradient_gain_' if gain_mode else 'gradient_')+str(size), neurons=recipe)
            if not gain_mode:
                condition['actual_residual_write_fractions'] = write_norms
            recipes.append(condition)
        result = dict(method='sparse multiplicative MLP gain gradient optimization' if gain_mode else 'sparse additive MLP offset gradient optimization',
                      mode=mode,
                      scope=scope, persona_prefix=persona_prefix,
                      model=runner.metadata.get('model'), layers=layers,
                      training_pairs=[p['id'] for p in pairs], target_tokens=target_limit,
                      updates=iterations, baseline_nll=[dict(pair=e[0], catgirl=c, neutral=n)
                                                       for e,(c,n) in zip(examples,baselines)],
                      conditions=recipes, validated=False,
                      parameter_budget=dict(per_layer_norm=layer_budget, total_norm=total_budget,
                                            l2=penalty, gain_min=gain_min if gain_mode else None,
                                            gain_max=gain_max if gain_mode else None),
                      note='Candidates require held-out generation evaluation; original weights are unchanged.')
        destination = output.with_suffix('.recipes.json')
        destination.write_text(json.dumps(result, indent=2))
        emit(dict(event='gradient_complete', recipes=str(destination), validated=False))
        return result
    finally:
        for m, state in zip(runner.mlps, previous):
            for name, value in state.items():
                setattr(m, name, value)
        runner.model.train(was_training)
