"""Approximate paired persona residual directions with sparse MLP columns.

Only activation offsets are constructed; model weights are never changed.
Call search(runner, job, output) in the existing GPU worker.
"""
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np


def search(runner, job, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(job.get('discovery', Path(__file__).with_name('discovery.json'))).read_text())
    pairs = data['pairs'][:max(1, int(job.get('pairs', 8)))]
    layers = [int(i) for i in job.get('layers', [12, 16, 20, 24])]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError('layers must be nonempty and distinct')
    if min(layers) < 0 or max(layers) >= len(runner.layers):
        raise ValueError('layer index outside model')
    width = runner.config.get('text_config', runner.config)['intermediate_size']
    sizes = sorted(set(int(k) for k in job.get('sizes', [32, 128])))
    if not sizes or min(sizes) < 1 or max(sizes) > min(512, width):
        raise ValueError('sizes must lie between 1 and min(512, MLP width)')
    strengths = [float(s) for s in job.get('strengths', [.5, 1, 2, 4])]
    refine = bool(job.get('lstsq', True))
    started = time.time()

    def emit(record):
        record = dict(record, elapsed_seconds=time.time()-started)
        with output.open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    # Preserve all intervention and capture state, including generation counters.
    fields = ('delta', 'early_tokens', 'position_mask', 'capture', 'captured', 'seen', 'boundary')
    previous = [{name:getattr(m, name, None) for name in fields} for m in runner.mlps]
    was_training = bool(getattr(runner.model, 'training', False))
    classes = {type(layer) for layer in runner.layers}
    originals = {klass:klass.__call__ for klass in classes}
    identities = {id(layer):i for i,layer in enumerate(runner.layers)}
    captured = {}

    def make_call(original):
        def call(layer, *args, **kwargs):
            result = original(layer, *args, **kwargs)
            index = identities.get(id(layer))
            if index in layers:
                hidden = result[0] if isinstance(result, tuple) else result
                captured[index] = hidden[0, -1, :].astype(mx.float32)
            return result
        return call

    try:
        for m in runner.mlps:
            m.delta = None
            m.early_tokens = None
            m.position_mask = None
            m.capture = False
            m.captured = None
        runner.model.eval()
        for klass, original in originals.items():
            klass.__call__ = make_call(original)
        observations = []
        for pair in pairs:
            poles = []
            for pole in ('neutral', 'catgirl'):
                captured.clear()
                ids = runner.tokens([{'role':'system', 'content':data[pole+'_system']},
                                     {'role':'user', 'content':pair['prompt']}])
                logits = runner.model(mx.array(ids, dtype=mx.int32)[None], cache=None)
                mx.eval(logits, list(captured.values()))
                if set(captured) != set(layers):
                    raise RuntimeError('Some requested decoder layer outputs were not captured')
                poles.append(np.stack([np.array(captured[layer]) for layer in layers]))
            observations.append(poles)
            emit(dict(event='residual_capture', pair=pair['id'], layers=layers))
        # Remove class patches immediately after capture, before decomposition.
        for klass, original in originals.items():
            klass.__call__ = original
        observations = np.asarray(observations, dtype=np.float32)
        differences = observations[:, 1]-observations[:, 0]
        directions = differences.mean(axis=0)
        np.savez_compressed(output.with_suffix('.residuals.npz'),
                            observations=observations, differences=differences,
                            directions=directions, layers=np.asarray(layers))
        conditions = []
        decompositions = []
        for row, layer in enumerate(layers):
            target_np = directions[row]
            target_norm = float(np.linalg.norm(target_np))
            if target_norm < 1e-8:
                emit(dict(event='residual_skip', layer=layer, reason='zero paired mean direction'))
                continue
            projection = runner.mlps[layer].down_proj
            if hasattr(projection, 'scales'):
                weight = mx.dequantize(projection.weight, projection.scales,
                                       getattr(projection, 'biases', None),
                                       group_size=projection.group_size,
                                       bits=projection.bits,
                                       mode=getattr(projection, 'mode', 'affine'))
            else:
                weight = projection.weight
            weight = weight.astype(mx.float32)
            norms = mx.maximum(mx.linalg.norm(weight, axis=0), mx.array(1e-8))
            dictionary = weight/norms[None, :]
            residual = mx.array(target_np)
            allowed = mx.ones((width,), dtype=mx.bool_)
            mx.eval(dictionary, norms, residual)
            chosen = []
            greedy_coefficients = []
            for step in range(max(sizes)):
                correlations = dictionary.T@residual
                scores = mx.where(allowed, mx.abs(correlations), mx.array(-float('inf')))
                index = int(mx.argmax(scores))
                coefficient = correlations[index]
                mx.eval(coefficient)
                chosen.append(index)
                greedy_coefficients.append(float(coefficient))
                residual = residual-coefficient*dictionary[:, index]
                allowed = allowed & (mx.arange(width) != index)
                mx.eval(residual, allowed)
                count = step+1
                if count not in sizes:
                    continue
                # Least squares corrects correlated-column interactions in one
                # small solve; the greedy loop itself uses GPU matrix vectors.
                columns = np.array(weight[:, mx.array(chosen, dtype=mx.int32)])
                if refine:
                    amounts = np.linalg.lstsq(columns, target_np,
                                             rcond=float(job.get('rcond', 1e-4)))[0]
                else:
                    selected_norms = np.array(norms[mx.array(chosen, dtype=mx.int32)])
                    amounts = np.asarray(greedy_coefficients)/selected_norms
                reconstructed = columns@amounts
                if refine:
                    # At each requested checkpoint continue pursuit from the
                    # refined residual rather than stale greedy coefficients.
                    residual = mx.array(target_np-reconstructed)
                    mx.eval(residual)
                write_norm = float(np.linalg.norm(reconstructed))
                cosine = float(np.dot(target_np, reconstructed)/max(target_norm*write_norm, 1e-12))
                relative_error = float(np.linalg.norm(target_np-reconstructed)/target_norm)
                pair_alignment = [float(np.dot(d, target_np)/(max(float(np.linalg.norm(d))*target_norm, 1e-12)))
                                  for d in differences[:, row]]
                decomposition = dict(layer=layer, neurons=count, cosine=cosine,
                                     relative_error=relative_error,
                                     target_norm=target_norm, write_norm=write_norm,
                                     write_fraction=write_norm/float(runner.residual[layer]),
                                     paired_direction_cosines=pair_alignment,
                                     lstsq=refine)
                decompositions.append(decomposition)
                emit(dict(event='residual_decomposition', **decomposition))
                column_norms = np.linalg.norm(columns, axis=0)
                for strength in strengths:
                    recipe = [dict(layer=layer, neuron=int(neuron),
                                   amount=float(amount*strength),
                                   dose=float(amount*strength*norm/float(runner.residual[layer])))
                              for neuron,amount,norm in zip(chosen, amounts, column_norms)]
                    condition = dict(id=f'residual_L{layer}_k{count}_x{strength:g}',
                                     neurons=recipe,
                                     reconstruction=decomposition,
                                     strength=strength,
                                     actual_residual_write_fraction=strength*write_norm/float(runner.residual[layer]))
                    if 'early_tokens' in job:
                        condition['early_tokens'] = job['early_tokens']
                    conditions.append(condition)
            del weight, dictionary, norms, residual
        result = dict(method='paired system-persona layer-output residual difference, sparse same-layer MLP reconstruction',
                      model=runner.metadata.get('model'),
                      layers=layers, pairs=[p['id'] for p in pairs],
                      capture_position='last prompt token',
                      neutral_system=data['neutral_system'], catgirl_system=data['catgirl_system'],
                      conditions=conditions, decompositions=decompositions,
                      validated=False,
                      note='Sparse CAA approximations require causal held-out generation tests; original weights are unchanged.')
        destination = output.with_suffix('.recipes.json')
        destination.write_text(json.dumps(result, indent=2))
        emit(dict(event='residual_complete', recipes=str(destination), conditions=len(conditions)))
        return result
    finally:
        for klass, original in originals.items():
            klass.__call__ = original
        for m, state in zip(runner.mlps, previous):
            for name, value in state.items():
                setattr(m, name, value)
        runner.model.train(was_training)
