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
from mlx.utils import tree_map
import numpy as np


def _reference_attention(queries, keys, values, scale, mask):
    """Differentiable GQA attention, including offset-aware cached causality."""
    repeats=queries.shape[1]//keys.shape[1]
    if repeats>1:
        keys=mx.repeat(keys,repeats,axis=1)
        values=mx.repeat(values,repeats,axis=1)
    scores=(queries.astype(mx.float32)*scale)@mx.swapaxes(keys.astype(mx.float32),-1,-2)
    if isinstance(mask,str):
        if mask!='causal':
            raise ValueError('Unsupported reference attention mask: '+mask)
        query_length,key_length=queries.shape[-2],keys.shape[-2]
        mask=mx.arange(key_length)[None,:]<=mx.arange(query_length)[:,None]+key_length-query_length
    if mask is not None:
        if mask.dtype==mx.bool_:
            scores=mx.where(mask,scores,-float('inf'))
        else:
            scores=scores+mask
    probabilities=mx.softmax(scores,axis=-1,precise=True)
    return (probabilities@values.astype(mx.float32)).astype(queries.dtype)


def _select_condition(value, condition_id=None):
    """Accept raw recipes, checkpoints, recipe files, and nested candidates."""
    found=[]
    def visit(node):
        if isinstance(node,list):
            if node and all(isinstance(spec,dict) and 'layer' in spec and 'neuron' in spec for spec in node):
                found.append({'neurons':node})
            else:
                for item in node:
                    visit(item)
        elif isinstance(node,dict):
            if isinstance(node.get('neurons'),list):
                found.append(node)
            else:
                for key in ('condition','candidate','conditions','candidates','recipe','recipes'):
                    if key in node:
                        visit(node[key])
    visit(value)
    if condition_id is not None:
        found=[condition for condition in found if condition.get('id')==condition_id]
    if not found:
        raise ValueError('No neuron recipe found in warm-start candidate')
    return max(found,key=lambda condition:len({(spec['layer'],spec['neuron']) for spec in condition['neurons']}))


def _positive_span_weights(tokenizer, text, encoded, retained, spans):
    """Map exact character spans onto overlapping tokenizer positions."""
    weights=np.ones(retained,dtype=np.float32)
    if not spans:
        return weights
    raw=getattr(tokenizer,'_tokenizer',tokenizer)
    tokens=raw(text,add_special_tokens=False,return_offsets_mapping=True)
    if list(tokens['input_ids']) != list(encoded):
        raise ValueError('Span tokenizer IDs differ from training tokenizer IDs')
    offsets=tokens['offset_mapping'][:retained]
    for span in spans:
        fragment=span['text']
        weight=float(span['weight'])
        if not fragment or not np.isfinite(weight) or weight<=0:
            raise ValueError('Positive spans need nonempty text and finite positive weights')
        first=text.find(fragment)
        if first<0:
            raise ValueError('Positive span text is absent from catgirl target: '+fragment)
        while first>=0:
            last=first+len(fragment)
            for index,(start,end) in enumerate(offsets):
                if end>first and start<last:
                    weights[index]=max(weights[index],weight)
            first=text.find(fragment,first+len(fragment))
    return weights


def optimize(runner, job, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    source = Path(job.get('warm_start', job.get('source', root/'results'/'034_broad_gain.recipes.json')))
    warm = json.loads(source.read_text())
    condition = _select_condition(warm,job.get('condition_id'))
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
    kl_weight_persona = float(job.get('kl_weight_persona',kl_weight))
    kl_weight_preserve = float(job.get('kl_weight_preserve',kl_weight))
    kl_weight_identity = float(job.get('kl_weight_identity',kl_weight_persona))
    disable_identity_kl = bool(job.get('disable_identity_kl',False))
    if not np.isfinite([kl_weight_persona,kl_weight_preserve,kl_weight_identity]).all() or min(kl_weight_persona,kl_weight_preserve,kl_weight_identity)<0:
        raise ValueError('KL weights must be finite and nonnegative')
    persona_weight, persona_tokens = float(job.get('persona_weight', 1.)), int(job.get('persona_tokens', 20))
    eos_weight = float(job.get('eos_weight', 4.))
    append_eos = bool(job.get('append_eos', True))
    training_early_tokens=job.get('training_early_tokens')
    if training_early_tokens is not None:
        if isinstance(training_early_tokens,bool) or not isinstance(training_early_tokens,int) or training_early_tokens<0:
            raise ValueError('training_early_tokens must be a nonnegative integer or null')
    training_tail_scale=float(job.get('training_tail_scale',0.))
    if not np.isfinite(training_tail_scale):
        raise ValueError('training_tail_scale must be finite')
    scope=job.get('scope','assistant' if training_early_tokens is not None else 'all')
    if scope not in ('all','assistant'):
        raise ValueError("Affine scope must be 'all' or 'assistant'")
    if training_early_tokens is not None and scope!='assistant':
        raise ValueError('Bootstrap training requires assistant scope and neutral prefix caches')
    eos = runner.tok.eos_token_id
    if isinstance(eos, (list,tuple)):
        eos = next((i for i in eos if i is not None), None)
    if append_eos and not isinstance(eos, (int,np.integer)):
        raise ValueError('A tokenizer EOS token ID is required')
    fields = ('delta','gain','early_tokens','position_mask','capture','captured','seen','boundary','tail_scale','prefill_scale')
    previous = [{field:getattr(m,field,None) for field in fields} for m in runner.mlps]
    previous_modes = [(module,bool(module.training)) for _,module in runner.model.named_modules()]
    attention_namespaces={}
    for layer in runner.layers:
        if hasattr(layer,'self_attn'):
            namespace=type(layer.self_attn).__call__.__globals__
            if 'scaled_dot_product_attention' in namespace:
                attention_namespaces[id(namespace)]=(namespace,namespace['scaled_dot_product_attention'])
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
            m.seen = 0; m.boundary = 0; m.tail_scale = 0.; m.prefill_scale = 0.
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
        # Keep prefix computation outside the derivative even if MLX retains
        # evaluated graph nodes, and even when theta arrays are reused as the
        # tail function's explicit differentiation inputs.
        for entry in cache:
            entry.state = tree_map(lambda value:mx.stop_gradient(value) if isinstance(value,mx.array) else value,
                                   entry.state)
        mx.eval([entry.state for entry in cache])
        return cache

    def tail_logits(theta, example, target, cache):
        apply(theta)
        ids = mx.array([example['prefix'][-1]]+target[:-1], dtype=mx.int32)[None]
        if theta is not None and training_early_tokens is not None:
            position_mask=mx.where(mx.arange(ids.shape[1])<training_early_tokens,
                                   1.,training_tail_scale)[None,:,None]
            for mlp in runner.mlps:
                mlp.position_mask=position_mask
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
        def wrap_attention(original):
            def attention(queries,keys,values,cache,scale,mask,sinks=None):
                # MLX's short-query vector kernel does not emit the logsumexp
                # needed by its VJP. Keep fused attention for longer queries
                # and for every inference/prefill call.
                if (runner.model.training and queries.shape[-2]<=int(job.get('reference_attention_max_queries',32))
                        and not hasattr(cache,'bits')):
                    if sinks is not None:
                        raise ValueError('Reference short-query attention does not support sinks')
                    return _reference_attention(queries,keys,values,scale,mask)
                return original(queries,keys,values,cache,scale,mask,sinks=sinks)
            return attention
        for namespace,original in attention_namespaces.values():
            namespace['scaled_dot_product_attention']=wrap_attention(original)
        parameters = project(parameters)
        examples=[]
        for pair in pairs:
            messages = pair['messages'] if 'messages' in pair else [
                {'role':'system','content':data['neutral_system']},
                {'role':'user','content':pair['prompt']}]
            prefix = list(runner.tokens(messages))
            assistant_prefix=list(runner.tok.encode(pair.get('assistant_prefix',''),add_special_tokens=False))
            prefix.extend(assistant_prefix)
            targets=[]
            eos_appended=[]
            positive_spans=None
            for pole in ('catgirl','neutral'):
                complete = list(runner.tok.encode(pair[pole], add_special_tokens=False))
                target = complete[:target_limit]
                if pole=='catgirl':
                    spans=[] if pair.get('preserve_format',False) else pair.get('positive_span_weights',[])
                    positive_spans=_positive_span_weights(runner.tok,pair[pole],complete,len(target),spans)
                add_end = append_eos and len(complete)<=target_limit and (not target or target[-1]!=int(eos))
                if add_end:
                    target.append(int(eos))
                    if pole=='catgirl':
                        positive_spans=np.append(positive_spans,np.float32(1.))
                if not target:
                    raise ValueError('Empty target for '+pair['id'])
                targets.append(target)
                eos_appended.append(add_end)
            example=dict(id=pair['id'],prefix=prefix,positive=targets[0],neutral=targets[1],
                         assistant_prefix_tokens=len(assistant_prefix),
                         positive_span_weights=mx.array(positive_spans),
                         positive_eos_appended=eos_appended[0],neutral_eos_appended=eos_appended[1],
                         preserve_format=bool(pair.get('preserve_format',False)),
                         persona_identity=bool(pair.get('persona_identity',False)))
            example['kl_weight'] = (kl_weight_preserve if example['preserve_format'] else
                                    (0. if disable_identity_kl else kl_weight_identity) if example['persona_identity'] else kl_weight_persona)
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
                      assistant_prefix_tokens=len(assistant_prefix),
                      positive_tokens=len(targets[0]),neutral_tokens=len(targets[1]),
                      positive_eos_appended=eos_appended[0],neutral_eos_appended=eos_appended[1],
                      preserve_format=example['preserve_format'],persona_identity=example['persona_identity'],
                      kl_weight=example['kl_weight']))
            del teacher_cache,logits,logp

        def objective(theta, example, cache):
            positive = tail_logits(theta,example,example['positive'],cache)
            labels = mx.array(example['positive'],dtype=mx.int32)[None]
            ce = nn.losses.cross_entropy(positive,labels,reduction='none')[0]
            first_weight = 1. if example['preserve_format'] else persona_weight
            weights = mx.where(mx.arange(len(example['positive']))<persona_tokens,first_weight,1.)
            weights=mx.maximum(weights,example['positive_span_weights'])
            if example['positive_eos_appended']:
                weights = mx.where(mx.arange(len(example['positive']))==len(example['positive'])-1,eos_weight,weights)
            ce = mx.sum(ce*weights)/mx.sum(weights)
            neutral = tail_logits(theta,example,example['neutral'],cache)
            logq = neutral-mx.logsumexp(neutral,axis=-1,keepdims=True)
            selected = mx.take_along_axis(logq,example['teacher_ids'],axis=-1)
            tail_q = mx.maximum(1-mx.sum(mx.exp(selected),axis=-1),mx.array(1e-8))
            kl = mx.sum(example['teacher_p']*(example['teacher_logp']-selected),axis=-1)
            kl += example['teacher_tail']*(mx.log(example['teacher_tail'])-mx.log(tail_q))
            start = 0 if example['preserve_format'] else 4
            kl = mx.mean(kl[:,start:]) if kl.shape[1]>start else mx.array(0.,dtype=mx.float32)
            gain_reg = float(job.get('gain_l2', .0001))*mx.sum(theta['gain']**2)
            bias_reg = float(job.get('bias_l2', .08))*mx.sum(theta['bias']**2)
            return ce+example['kl_weight']*kl+gain_reg+bias_reg, (ce,kl)

        momentum={k:np.zeros_like(v) for k,v in parameters.items()}
        variance={k:np.zeros_like(v) for k,v in parameters.items()}
        seed=int(job.get('seed',0))
        shuffle_pairs=bool(job.get('shuffle_pairs',True))
        random=np.random.default_rng(seed)
        order=np.arange(len(examples))
        for step in range(iterations):
            if step%len(examples)==0:
                order=random.permutation(len(examples)) if shuffle_pairs else np.arange(len(examples))
            example=examples[int(order[step%len(examples)])]
            theta={k:mx.array(v) for k,v in parameters.items()}
            # Materialize current prefix state before tracing gradients.
            cache=prefill(example['prefix'],None if scope=='assistant' else theta)
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
                      assistant_prefix_tokens=example['assistant_prefix_tokens'],
                      kl_weight=example['kl_weight'],
                      gain_offset_norm=float(np.linalg.norm(parameters['gain'])),
                      bias_dose_norm=float(np.linalg.norm(parameters['bias']))))
            if (step+1)%20==0:
                output.with_suffix('.checkpoint.json').write_text(json.dumps(dict(step=step+1,mode='affine',
                    neurons=export(parameters),job=job),indent=2))
            del cache,gradient,theta,fn
        exported_condition=dict(id='affine_'+str(len(keys)),neurons=export(parameters))
        if training_early_tokens is not None:
            exported_condition.update(early_tokens=training_early_tokens,
                                      tail_scale=training_tail_scale,prefill_scale=0.)
        elif scope=='assistant':
            exported_condition.update(early_tokens=1000000,tail_scale=1.,prefill_scale=0.)
        result=dict(method='sparse affine neuron optimization with detached current-prefix caches and coarsened original-model KL',
                    scope=scope,
                    training_early_tokens=training_early_tokens,training_tail_scale=training_tail_scale,
                    mode='affine',source=str(source),unique_neurons=len(keys),layers=layers,
                    source_condition=condition.get('id'),job=job,
                    parameter_budget=dict(gain_min=gain_min,gain_max=gain_max,
                                          gain_layer_norm=budgets['gain'][0],gain_total_norm=budgets['gain'][1],
                                          bias_layer_norm=budgets['bias'][0],bias_total_norm=budgets['bias'][1],
                                          gain_learning_rate=gain_lr,bias_learning_rate=bias_lr),
                    training_pairs=[p['id'] for p in pairs],append_eos=append_eos,eos_weight=eos_weight,
                    seed=seed,shuffle_pairs=shuffle_pairs,
                    kl_weight=kl_weight,kl_weight_persona=kl_weight_persona,
                    kl_weight_preserve=kl_weight_preserve,kl_weight_identity=kl_weight_identity,
                    disable_identity_kl=disable_identity_kl,kl_top_k=top_k,updates=iterations,
                    conditions=[exported_condition],
                    validated=False,note='Prefix state is recomputed at current parameters and detached; only tail gradients are optimized. Native weights remain unchanged.')
        destination=output.with_suffix('.recipes.json')
        destination.write_text(json.dumps(result,indent=2))
        emit(dict(event='affine_complete',recipes=str(destination)))
        return result
    finally:
        for namespace,original in attention_namespaces.values():
            namespace['scaled_dot_product_attention']=original
        for module,training in previous_modes:
            module.train(training)
        for m,state in zip(runner.mlps,previous):
            for field,value in state.items():
                setattr(m,field,value)
