"""Find fresh additive neuron coordinates using detached-prefix CE gradients."""
import copy
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
import numpy as np

try:
    from .affine_optimizer import _select_condition, _reference_attention
except ImportError:
    from affine_optimizer import _select_condition, _reference_attention


def select(runner, job, output):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent
    data=json.loads(Path(job.get('discovery',root/'discovery_broad.json')).read_text())
    available=[p for p in data['pairs'] if not p.get('preserve_format',False)]
    if job.get('pair_ids'):
        lookup={p['id']:p for p in data['pairs']}
        pairs=[lookup[name] for name in job['pair_ids']]
    else:
        count=min(len(available),max(1,int(job.get('pairs',8))))
        indices=np.linspace(0,len(available)-1,count).round().astype(int)
        pairs=[available[i] for i in indices]
    if not pairs:
        raise ValueError('No bias selection examples')
    layers=[int(layer) for layer in job.get('layers',list(range(8,32)))]
    if not layers or len(set(layers))!=len(layers) or min(layers)<0 or max(layers)>=len(runner.mlps):
        raise ValueError('Invalid bias selection layers')
    width=runner.config.get('text_config',runner.config)['intermediate_size']
    count=min(len(layers)*width,max(1,int(job.get('neurons',4096))))
    target_limit=max(1,int(job.get('target_tokens',64)))
    persona_weight=float(job.get('persona_weight',3.))
    persona_tokens=max(0,int(job.get('persona_tokens',20)))
    warm_specs=[]
    if job.get('warm_start'):
        warm_specs=_select_condition(json.loads(Path(job['warm_start']).read_text()),job.get('condition_id'))['neurons']
    fields=('delta','gain','early_tokens','position_mask','capture','captured','seen','boundary','tail_scale','prefill_scale')
    previous=[{key:getattr(m,key,None) for key in fields} for m in runner.mlps]
    previous_modes=[(module,bool(module.training)) for _,module in runner.model.named_modules()]
    attention_globals={}
    for layer in runner.layers:
        if hasattr(layer,'self_attn'):
            namespace=type(layer.self_attn).__call__.__globals__
            if 'scaled_dot_product_attention' in namespace:
                attention_globals[id(namespace)]=(namespace,namespace['scaled_dot_product_attention'])
    started=time.time()

    def emit(record):
        record=dict(record,elapsed_seconds=time.time()-started)
        with output.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    def reset():
        for m in runner.mlps:
            m.delta=None; m.gain=None; m.position_mask=None; m.early_tokens=None
            m.capture=False; m.captured=None; m.seen=0; m.boundary=0; m.tail_scale=0.; m.prefill_scale=0.
        runner.steer(warm_specs)

    def wrap(original):
        def attention(queries,keys,values,cache,scale,mask,sinks=None):
            if runner.model.training and queries.shape[-2]<=32 and not hasattr(cache,'bits'):
                if sinks is not None:
                    raise ValueError('Short reference attention does not support sinks')
                return _reference_attention(queries,keys,values,scale,mask)
            return original(queries,keys,values,cache,scale,mask,sinks=sinks)
        return attention

    try:
        for namespace,original in attention_globals.values():
            namespace['scaled_dot_product_attention']=wrap(original)
        units_np=[]
        for layer in layers:
            proj=runner.mlps[layer].down_proj
            weight=mx.dequantize(proj.weight,proj.scales,getattr(proj,'biases',None),
                                 group_size=proj.group_size,bits=proj.bits,
                                 mode=getattr(proj,'mode','affine')) if hasattr(proj,'scales') else proj.weight
            norms=mx.linalg.norm(weight.astype(mx.float32),axis=0)
            mx.eval(norms)
            units_np.append(float(runner.residual[layer])/np.maximum(np.array(norms),1e-8))
        units_np=np.stack(units_np).astype(np.float32)
        units=mx.array(units_np)
        normalized=[]; raw=[]
        for pair in pairs:
            reset(); runner.model.eval()
            messages=pair['messages'] if 'messages' in pair else [
                {'role':'system','content':data['neutral_system']}, {'role':'user','content':pair['prompt']}]
            prefix=list(runner.tokens(messages))
            target=list(runner.tok.encode(pair['catgirl'],add_special_tokens=False))[:target_limit]
            if not prefix or not target:
                raise ValueError('Empty prefix/target for '+pair['id'])
            cache=runner.model.make_cache()
            backbone=getattr(runner.model,'model',runner.model)
            chunk=max(1,int(job.get('prefill_chunk',512)))
            for start in range(0,len(prefix)-1,chunk):
                value=backbone(mx.array(prefix[start:min(start+chunk,len(prefix)-1)],dtype=mx.int32)[None],cache=cache)
                mx.eval(value,[entry.state for entry in cache])
            for entry in cache:
                entry.state=tree_map(lambda value:mx.stop_gradient(value) if isinstance(value,mx.array) else value,entry.state)
            mx.eval([entry.state for entry in cache])
            # Capture warm biases once; only newly proposed offsets vary.
            base_bias={layer:runner.mlps[layer].delta for layer in layers}
            runner.model.train()
            for layer in runner.layers[:min(layers)]:
                layer.eval()
            ids=mx.array([prefix[-1]]+target[:-1],dtype=mx.int32)[None]
            labels=mx.array(target,dtype=mx.int32)[None]
            weights=mx.where(mx.arange(len(target))<persona_tokens,persona_weight,1.)
            def loss(theta):
                for row,layer in enumerate(layers):
                    added=theta[row]*units[row]
                    runner.mlps[layer].delta=added if base_bias[layer] is None else base_bias[layer]+added
                branch=copy.deepcopy(cache)
                logits=runner.model(ids,cache=branch).astype(mx.float32)
                losses=nn.losses.cross_entropy(logits,labels,reduction='none')[0]
                return mx.sum(losses*weights)/mx.sum(weights)
            value,gradient=mx.value_and_grad(loss)(mx.zeros((len(layers),width),dtype=mx.float32))
            mx.eval(value,gradient)
            g=np.array(gradient)
            if not np.isfinite(g).all() or not np.isfinite(float(value)):
                raise FloatingPointError('Nonfinite bias probe for '+pair['id'])
            raw.append(g)
            normalized.append(g/max(float(np.linalg.norm(g)),1e-8))
            emit(dict(event='bias_probe',pair=pair['id'],catgirl_ce=float(value),
                      gradient_norm=float(np.linalg.norm(g)),prefix_tokens=len(prefix),target_tokens=len(target)))
            del cache,gradient
        gradients=np.stack(normalized)
        mean=gradients.mean(axis=0)
        consistency=(np.sign(gradients)==np.sign(mean)[None]).mean(axis=0)
        score=np.abs(mean)*consistency**float(job.get('consistency_power',2.))
        # Avoid assigning the entire search to one early layer solely because
        # it amplifies every target loss. Actual score ordering is retained.
        layer_cap=max(1,int(job.get('max_layer_share',.25)*count))
        selected=[]; per_layer={layer:0 for layer in layers}
        for flat in np.argsort(score.ravel())[::-1]:
            row,neuron=np.unravel_index(flat,score.shape)
            layer=layers[row]
            if per_layer[layer]>=layer_cap:
                continue
            selected.append((int(row),int(neuron)))
            per_layer[layer]+=1
            if len(selected)==count:
                break
        if len(selected)<count:
            raise ValueError('Layer-share cap is too small to select requested neuron count')
        norm=float(job.get('initial_dose_norm',0.))
        amount=norm/max(np.sqrt(len(selected)),1.)
        specs=[]
        for row,neuron in selected:
            dose=-float(np.sign(mean[row,neuron]))*amount
            specs.append(dict(layer=layers[row],neuron=neuron,dose=dose,
                              amount=float(dose*units_np[row,neuron]),
                              selection_score=float(score[row,neuron]),
                              mean_normalized_gradient=float(mean[row,neuron]),
                              gradient_consistency=float(consistency[row,neuron])))
        conditions=[dict(id='fresh_bias_'+str(len(specs)),neurons=specs)]
        if job.get('include_warm',False) and warm_specs:
            conditions.append(dict(id='warm_plus_fresh_bias',neurons=warm_specs+specs))
        np.savez_compressed(output.with_suffix('.gradients.npz'),normalized_gradients=gradients,
                            raw_gradients=np.stack(raw),layers=np.array(layers),score=score)
        result=dict(method='fresh additive dose-coordinate CE gradient selection with detached prefix caches',
                    model=runner.metadata.get('model'),
                    warm_start=job.get('warm_start'),layers=layers,pairs=[p['id'] for p in pairs],
                    target_tokens=target_limit,unique_selected_neurons=len(specs),
                    initial_dose_norm=norm,per_layer_selected=per_layer,conditions=conditions,
                    validated=False,
                    note='Default zero-dose recipe is a coordinate shortlist. Run affine optimizer with gain_learning_rate=0 for bias-only training.')
        destination=output.with_suffix('.recipes.json'); destination.write_text(json.dumps(result,indent=2))
        emit(dict(event='bias_selection_complete',recipes=str(destination),neurons=len(specs)))
        return result
    finally:
        for namespace,original in attention_globals.values():
            namespace['scaled_dot_product_attention']=original
        for module,training in previous_modes:
            module.train(training)
        for m,state in zip(runner.mlps,previous):
            for key,value in state.items():
                setattr(m,key,value)
