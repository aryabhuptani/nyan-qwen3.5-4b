"""Compare reference, fused, and cached likelihoods with identical steering."""
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def diagnose(runner, job, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    source = Path(job.get('source', root/'results'/'028_wide_gain.recipes.json'))
    discovery = Path(job.get('discovery', root/'discovery_identity.json'))
    learned = json.loads(source.read_text())
    data = json.loads(discovery.read_text())
    condition = max(learned['conditions'], key=lambda c:len(c['neurons'])) if 'conditions' in learned else learned
    recipe = condition['neurons']
    if not recipe:
        raise ValueError('Learned diagnostic recipe is empty')
    pair = next((p for p in data['pairs'] if p['id']==job['pair_id']), None) if 'pair_id' in job else data['pairs'][int(job.get('pair_index', 0))]
    if pair is None:
        raise ValueError('Diagnostic pair_id not found')
    messages = [{'role':'system', 'content':data['neutral_system']},
                {'role':'user', 'content':pair['prompt']}]
    prefix = list(runner.tokens(messages))
    target = runner.tok.encode(pair['catgirl'], add_special_tokens=False)[:max(1, int(job.get('target_tokens', 32)))]
    if not target:
        raise ValueError('Empty diagnostic target')
    labels = mx.array(target, dtype=mx.int32)[None]
    full_ids = mx.array(prefix+target[:-1], dtype=mx.int32)[None]
    fields = ('delta', 'gain', 'early_tokens', 'position_mask', 'capture', 'captured', 'seen', 'boundary', 'tail_scale')
    previous = [{name:getattr(m, name, None) for name in fields} for m in runner.mlps]
    # Parent-before-child order permits train() restoration of mixed modes.
    previous_modes = [(module, bool(module.training)) for _,module in runner.model.named_modules()]
    minimum = min(learned.get('layers', [int(n['layer']) for n in recipe]))
    logits_by_path = {}
    records = []
    started = time.time()

    def emit(record):
        record = dict(record, elapsed_seconds=time.time()-started)
        with output.open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record

    def reset(neurons, mode):
        for m in runner.mlps:
            m.delta = None
            m.gain = None
            m.early_tokens = None
            m.position_mask = None
            m.capture = False
            m.captured = None
            m.seen = 0
            m.boundary = 0
            m.tail_scale = 0.0
        runner.steer(neurons)
        runner.model.eval()
        if mode == 'reference_full':
            runner.model.train()
            for layer in runner.layers[:minimum]:
                layer.eval()

    def run_path(mode):
        if mode != 'fused_cached':
            return runner.model(full_ids, cache=None)[:, len(prefix)-1:, :].astype(mx.float32)
        if hasattr(runner.model, 'make_cache'):
            cache = runner.model.make_cache()
        else:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(runner.model)
        prompt_logits = runner.model(mx.array(prefix, dtype=mx.int32)[None], cache=cache)
        pieces = [prompt_logits[:, -1:, :].astype(mx.float32)]
        mx.eval(pieces[-1])
        for token in target[:-1]:
            value = runner.model(mx.array([[token]], dtype=mx.int32), cache=cache)[:, -1:, :].astype(mx.float32)
            mx.eval(value)
            pieces.append(value)
        return mx.concatenate(pieces, axis=1)

    try:
        for name, neurons in (('baseline', []), ('learned', recipe)):
            for mode in ('reference_full', 'fused_full', 'fused_cached'):
                reset(neurons, mode)
                logits = run_path(mode)
                mx.eval(logits)
                nll = nn.losses.cross_entropy(logits, labels, reduction='none')
                top = mx.argmax(logits, axis=-1)
                mx.eval(nll, top)
                losses = np.array(nll)[0]
                top_ids = [int(i) for i in np.array(top)[0]]
                ce = {str(n):float(losses[:min(n, len(losses))].mean()) for n in (1,4,16)}
                ce['whole'] = float(losses.mean())
                record = emit(dict(event='mode_diagnostic_path', condition=name, mode=mode,
                                   pair=pair['id'], prompt_tokens=len(prefix), target_tokens=len(target),
                                   per_label_nll=losses.tolist(), ce=ce,
                                   target_token_ids=target,
                                   target_tokens_decoded=[runner.tok.decode([i]) for i in target],
                                   top1_token_ids=top_ids,
                                   top1_tokens_decoded=[runner.tok.decode([i]) for i in top_ids]))
                records.append(record)
                logits_by_path[(name, mode)] = logits
        comparisons = []
        for name in ('baseline', 'learned'):
            reference = next(r for r in records if r['condition']==name and r['mode']=='reference_full')
            for mode in ('fused_full', 'fused_cached'):
                other = next(r for r in records if r['condition']==name and r['mode']==mode)
                comparison = dict(condition=name, reference='reference_full', other=mode,
                                  ce_difference={key:other['ce'][key]-reference['ce'][key] for key in reference['ce']},
                                  top1_agreement=float(np.mean(np.asarray(reference['top1_token_ids'])==np.asarray(other['top1_token_ids']))))
                if job.get('logit_metrics', False):
                    a, b = logits_by_path[(name,'reference_full')], logits_by_path[(name,mode)]
                    loga, logb = a-mx.logsumexp(a, axis=-1, keepdims=True), b-mx.logsumexp(b, axis=-1, keepdims=True)
                    kl = mx.sum(mx.exp(loga)*(loga-logb), axis=-1).mean()
                    maximum = mx.max(mx.abs(a-b))
                    mx.eval(kl, maximum)
                    comparison.update(mean_reference_kl=float(kl), max_abs_logit_difference=float(maximum))
                comparisons.append(comparison)
        gains = {}
        for mode in ('reference_full', 'fused_full', 'fused_cached'):
            base = next(r for r in records if r['condition']=='baseline' and r['mode']==mode)
            steered = next(r for r in records if r['condition']=='learned' and r['mode']==mode)
            gains[mode] = {key:steered['ce'][key]-base['ce'][key] for key in base['ce']}
        summary = dict(source=str(source), discovery=str(discovery), pair=pair['id'],
                       messages=messages, target_text=runner.tok.decode(target),
                       learned_condition=condition.get('id'), learned_neurons=len(recipe),
                       mode_comparisons=comparisons, learned_minus_baseline_ce=gains,
                       records=records,
                       interpretation='Negative learned-minus-baseline CE means steering improves target likelihood. Compare this improvement across paths; whole-sequence gains can coexist with weak initial-token gains.')
        destination = output.with_suffix('.summary.json')
        destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        emit(dict(event='mode_diagnostic_complete', summary=str(destination),
                  learned_minus_baseline_ce=gains, mode_comparisons=comparisons))
        return summary
    finally:
        for module, training in previous_modes:
            module.train(training)
        for m, state in zip(runner.mlps, previous):
            for name, value in state.items():
                setattr(m, name, value)
