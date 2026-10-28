"""Sparse affine neuron optimization with detached long-prefix caches and KL.

Only selected neuron gains and activation biases are differentiated. Prefix
caches are recomputed with current interventions before every update and are
constants for the differentiable final-prompt-token/assistant tail.
"""
import copy
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def optimize(runner, job, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    source = Path(job.get('warm_start', job.get('source', root/'results'/'034_broad_gain.recipes.json')))
    warm = json.loads(source.read_text())
    condition = max(warm['conditions'], key=lambda c:len(c['neurons'])) if 'conditions' in warm else warm
    data = json.loads(Path(job.get('discovery', root/'discovery_broad.json')).read_text())
    pairs = data['pairs'][:max(1, int(job.get('pairs', 18)))]
    if not pairs:
        raise ValueError('No affine training pairs')
    width = runner.config.get('text_config', runner.config)['intermediate_size']
    merged = {}
    for spec in condition['neurons']:
        key = (int(spec['layer']), int(spec['neuron']))
        if not (0 <= key[0] < len(runner.mlps) and 0 <= key[1] < width):
            raise ValueError('Warm-start coordinate outside model')
        value = merged.setdefault(key, dict(factor=1., amount=0., dose=0.))
        if spec.get('mode') == 'mul':
            value['factor'] *= float(spec['factor'])
        elif 'amount' in spec:
            value['amount'] += float(spec['amount'])
        elif 'amplitude' in spec:
            value['amount'] += float(spec['amplitude'])
        else:
            value['dose'] += float(spec.get('dose', 0.))
    limit = min(4096, max(1, int(job.get('neurons', 4096))))
    keys = sorted(merged, key=lambda k:abs(merged[k]['factor']-1)+abs(merged[k]['dose']), reverse=True)[:limit]
    if not keys:
        raise ValueError('Empty affine coordinate set')
    layers = sorted({layer for layer,_ in keys})
    groups = {layer:np.array([i for i,key in enumerate(keys) if key[0]==layer], dtype=np.int32) for layer in layers}
    neuron_indices = {layer:mx.array([keys[i][1] for i in group], dtype=mx.int32) for layer,group in groups.items()}
    group_indices = {layer:mx.array(group) for layer,group in groups.items()}
    units_np = np.zeros(len(keys), dtype=np.float32)
    column_norms = np.zeros(len(keys), dtype=np.float32)
    for layer in layers:
        projection = runner.mlps[layer].down_proj
        weight = mx.dequantize(projection.weight, projection.scales, getattr(projection,'biases',None),
                               group_size=projection.group_size, bits=projection.bits,
                               mode=getattr(projection,'mode','affine')) if hasattr(projection,'scales') else projection.weight
        norms = mx.linalg.norm(weight.astype(mx.float32), axis=0)[neuron_indices[layer]]
        mx.eval(norms)
        column_norms[groups[layer]] = np.maximum(np.array(norms), 1e-8)
        units_np[groups[layer]] = float(runner.residual[layer])/column_norms[groups[layer]]
    units = mx.array(units_np)
    parameters = dict(gain=np.array([merged[k]['factor']-1 for k in keys], dtype=np.float32),
                      bias=np.array([merged[k]['dose']+merged[k]['amount']/units_np[i] for i,k in enumerate(keys)], dtype=np.float32))
    gain_min, gain_max = float(job.get('gain_min', 0)), float(job.get('gain_max', 8))
    if not np.isfinite([gain_min,gain_max]).all() or gain_min > 1 or gain_max < 1:
        raise ValueError('Gain bounds must be finite and include 1')
    learning_rate = float(job.get('learning_rate', .015))
    gain_lr = float(job.get('gain_learning_rate', learning_rate))
    bias_lr = float(job.get('bias_learning_rate', learning_rate*.1))
    budgets = dict(gain=(float(job.get('layer_gain_norm', 10)), float(job.get('total_gain_norm', 30))),
                   bias=(float(job.get('layer_bias_norm', .05)), float(job.get('total_bias_norm', .15))))
    iterations = max(0, int(job.get('iterations', 120)))
    target_limit = max(1, int(job.get('target_tokens', 128)))
    top_k = max(1, min(1024, int(job.get('kl_top_k', 64))))
    kl_weight = float(job.get('kl_weight', 1.))
    persona_weight, persona_tokens = float(job.get('persona_weight', 1.)), int(job.get('persona_tokens', 20))
    eos_weight = float(job.get('eos_weight', 4.))
    append_eos = bool(job.get('append_eos', True))
    eos = runner.tok.eos_token_id
    if isinstance(eos, (list,tuple)):
        eos = next((i for i in eos if i is not None), None)
    if append_eos and not isinstance(eos, (int,np.integer)):
        raise ValueError('A tokenizer EOS token ID is required')
    fields = ('delta','gain','early_tokens','position_mask','capture','captured','seen','boundary','tail_scale')
    previous = [{field:getattr(m,field,None) for field in fields} for m in runner.mlps]
    previous_modes = [(module,bool(module.training)) for _,module in runner.model.named_modules()]
    started = time.time()

    def emit(record):
        record = dict(record, elapsed_seconds=time.time()-started)
        with output.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    def project(values):
        values['gain'] = np.clip(values['gain'], gain_min-1, gain_max-1)
        for kind, (local,total) in budgets.items():
            for group in groups.values():
                norm = float(np.linalg.norm(values[kind][group]))
                values[kind][group] *= min(1., local/max(norm,1e-8))
            values[kind] *= min(1., total/max(float(np.linalg.norm(values[kind])),1e-8))
        return values

    def apply(theta):
        for m in runner.mlps:
            m.delta = None; m.gain = None; m.early_tokens = None
            m.position_mask = None; m.capture = False; m.captured = None
            m.seen = 0; m.boundary = 0; m.tail_scale = 0.
        if theta is None:
            return
        for layer in layers:
            group = group_indices[layer]
            gains = mx.zeros((width,), dtype=mx.float32).at[neuron_indices[layer]].add(theta['gain'][group])
            biases = mx.zeros((width,), dtype=mx.float32).at[neuron_indices[layer]].add(theta['bias'][group]*units[group])
            runner.mlps[layer].gain = 1+gains
            runner.mlps[layer].delta = biases

    def prefill(prefix, theta):
        apply(theta)
        runner.model.eval()
        cache = runner.model.make_cache()
        # Calling the text backbone avoids the enormous vocabulary projection
        # for tokens whose logits are discarded. Use chunks to bound memory.
        backbone = getattr(runner.model, 'model', runner.model)
        chunk = max(1, int(job.get('prefill_chunk', 512)))
        for start in range(0,len(prefix)-1,chunk):
            ids = mx.array(prefix[start:min(start+chunk,len(prefix)-1)], dtype=mx.int32)[None]
            result = backbone(ids, cache=cache)
            mx.eval(result, [c.state for c in cache])
        return cache

    def tail_logits(theta, example, target, cache):
        apply(theta)
        ids = mx.array([example['prefix'][-1]]+target[:-1], dtype=mx.int32)[None]
        # deepcopy is implemented for mx.array; both branch caches own their
        # mutable KV buffers and offsets. The evaluated prefix is closed over.
        branch = copy.deepcopy(cache)
        return runner.model(ids, cache=branch).astype(mx.float32)

    def export(theta):
        specs=[]
        for i,(layer,neuron) in enumerate(keys):
            specs.append(dict(layer=layer,neuron=neuron,mode='mul',factor=1+float(theta['gain'][i])))
            if theta['bias'][i] != 0:
                specs.append(dict(layer=layer,neuron=neuron,amount=float(theta['bias'][i]*units_np[i]),
                                  dose=float(theta['bias'][i]),down_column_norm=float(column_norms[i]),
                                  residual_norm=float(runner.residual[layer])))
        return specs

    try:
        parameters = project(parameters)
        examples=[]
        for pair in pairs:
            messages = pair.get('messages', [{'role':'system','content':data['neutral_system']},
                                             {'role':'user','content':pair['prompt']}])
            prefix = list(runner.tokens(messages))
            targets=[]
            for pole in ('catgirl','neutral'):
                target = list(runner.tok.encode(pair[pole], add_special_tokens=False))[:target_limit]
                if append_eos:
                    target.append(int(eos))
                if not target:
                    raise ValueError('Empty target for '+pair['id'])
                targets.append(target)
            example=dict(id=pair['id'],prefix=prefix,positive=targets[0],neutral=targets[1],
                         preserve_format=bool(pair.get('preserve_format',False)))
            teacher_cache = prefill(prefix,None)
            logits = tail_logits(None,example,targets[1],teacher_cache)
            logp = logits-mx.logsumexp(logits,axis=-1,keepdims=True)
            # Top-k plus one complement bucket is a true KL between the
            # coarsened teacher/student distributions, not target-label CE.
            indices = mx.argpartition(logp, logp.shape[-1]-top_k, axis=-1)[..., -top_k:]
            teacher_logp = mx.take_along_axis(logp,indices,axis=-1)
            teacher_p = mx.exp(teacher_logp)
            tail_p = mx.maximum(1-mx.sum(teacher_p,axis=-1),mx.array(1e-8))
            mx.eval(indices,teacher_logp,teacher_p,tail_p)
            example.update(teacher_ids=mx.stop_gradient(indices),teacher_logp=mx.stop_gradient(teacher_logp),
                           teacher_p=mx.stop_gradient(teacher_p),teacher_tail=mx.stop_gradient(tail_p))
            examples.append(example)
            emit(dict(event='affine_teacher',pair=pair['id'],prefix_tokens=len(prefix),
                      positive_tokens=len(targets[0]),neutral_tokens=len(targets[1]),
                      preserve_format=example['preserve_format']))
            del teacher_cache,logits,logp

        def objective(theta, example, cache):
            positive = tail_logits(theta,example,example['positive'],cache)
            labels = mx.array(example['positive'],dtype=mx.int32)[None]
            ce = nn.losses.cross_entropy(positive,labels,reduction='none')[0]
            weights = mx.where(mx.arange(len(example['positive']))<persona_tokens,persona_weight,1.)
            if append_eos:
                weights = mx.where(mx.arange(len(example['positive']))==len(example['positive'])-1,eos_weight,weights)
            ce = mx.sum(ce*weights)/mx.sum(weights)
            neutral = tail_logits(theta,example,example['neutral'],cache)
            logq = neutral-mx.logsumexp(neutral,axis=-1,keepdims=True)
            selected = mx.take_along_axis(logq,example['teacher_ids'],axis=-1)
            tail_q = mx.maximum(1-mx.sum(mx.exp(selected),axis=-1),mx.array(1e-8))
            kl = mx.sum(example['teacher_p']*(example['teacher_logp']-selected),axis=-1)
            kl += example['teacher_tail']*(mx.log(example['teacher_tail'])-mx.log(tail_q))
            start = 0 if example['preserve_format'] else min(4,kl.shape[1]-1)
            kl = mx.mean(kl[:,start:])
            gain_reg = float(job.get('gain_l2', .0001))*mx.sum(theta['gain']**2)
            bias_reg = float(job.get('bias_l2', .08))*mx.sum(theta['bias']**2)
            return ce+kl_weight*kl+gain_reg+bias_reg, (ce,kl)

        momentum={k:np.zeros_like(v) for k,v in parameters.items()}
        variance={k:np.zeros_like(v) for k,v in parameters.items()}
        for step in range(iterations):
            example=examples[step%len(examples)]
            theta={k:mx.array(v) for k,v in parameters.items()}
            # Materialize current prefix state before tracing gradients.
            cache=prefill(example['prefix'],theta)
            runner.model.train()
            for layer in runner.layers[:min(layers)]:
                layer.eval()
            # MLX preserves auxiliary tuple outputs automatically; unlike
            # JAX its value_and_grad has no has_aux keyword.
            fn=mx.value_and_grad(lambda values:objective(values,example,cache))
            (loss,(ce,kl)),gradient=fn(theta)
            mx.eval(loss,ce,kl,gradient)
            if not np.isfinite([float(loss),float(ce),float(kl)]).all():
                raise FloatingPointError('Nonfinite affine loss')
            for kind in parameters:
                g=np.array(gradient[kind])
                if not np.isfinite(g).all():
                    raise FloatingPointError('Nonfinite affine gradient')
                momentum[kind]=.9*momentum[kind]+.1*g
                variance[kind]=.999*variance[kind]+.001*g*g
                rate=gain_lr if kind=='gain' else bias_lr
                parameters[kind]-=rate*(momentum[kind]/(1-.9**(step+1)))/(np.sqrt(variance[kind]/(1-.999**(step+1)))+1e-8)
            parameters=project(parameters)
            emit(dict(event='affine_update',step=step+1,pair=example['id'],loss=float(loss),
                      positive_ce=float(ce),neutral_kl=float(kl),prefix_tokens=len(example['prefix']),
                      gain_offset_norm=float(np.linalg.norm(parameters['gain'])),
                      bias_dose_norm=float(np.linalg.norm(parameters['bias']))))
            if (step+1)%20==0:
                output.with_suffix('.checkpoint.json').write_text(json.dumps(dict(step=step+1,mode='affine',
                    neurons=export(parameters),job=job),indent=2))
            del cache,gradient,theta,fn
        result=dict(method='sparse affine neuron optimization with detached current-prefix caches and coarsened original-model KL',
                    mode='affine',source=str(source),unique_neurons=len(keys),layers=layers,
                    training_pairs=[p['id'] for p in pairs],append_eos=append_eos,eos_weight=eos_weight,
                    kl_weight=kl_weight,kl_top_k=top_k,updates=iterations,
                    conditions=[dict(id='affine_'+str(len(keys)),neurons=export(parameters))],
                    validated=False,note='Prefix state is recomputed at current parameters and detached; only tail gradients are optimized. Native weights remain unchanged.')
        destination=output.with_suffix('.recipes.json')
        destination.write_text(json.dumps(result,indent=2))
        emit(dict(event='affine_complete',recipes=str(destination)))
        return result
    finally:
        for module,training in previous_modes:
            module.train(training)
        for m,state in zip(runner.mlps,previous):
            for field,value in state.items():
                setattr(m,field,value)
