"""Study112 native-channel measurements. No MLX/model import unless --run.

Run --phase natural first, then use frozen --job for probe/generation.
Raw native scalar units only: no learned vectors or published-dose multipliers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
INGREDIENTS = {'genki': (13, 4810, 1), 'hugs': (4, 8016, -1),
               'paws': (23, 4769, 1), 'female': (22, 8295, -1)}
SYSTEM = 'You are a helpful assistant.'
PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def segmented_tokens(tok, case, probe=False):
    """Canonical encoding; map declared character spans by token overlap.

    The legacy function name is retained; no segmented IDs enter the model.
    """
    messages = case.get('messages') or [{'role': 'system', 'content': SYSTEM},
                                        {'role': 'user', 'content': case['text_prefix']}]
    messages = [dict(m) for m in messages]
    if probe and case.get('probe_question'):
        assert messages[-1]['role'] == 'user'
        messages[-1]['content'] += '\n\n' + case['probe_question']
    base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    readout = '' if probe and case.get('probe_question') else case.get('assistant_prefix', case.get('readout_text', ''))
    rendered = base + readout
    charspans = {'readout_text': [len(base),len(rendered)]}
    for name, field in [('carrier','target_span'),('keyword','keyword_span')]:
        text = case.get(field, case.get('carrier_text') if name == 'carrier' else None)
        if text:
            if rendered.count(text) != 1:
                raise ValueError(field+' must occur exactly once')
            a = rendered.index(text)
            charspans[name] = [a,a+len(text)]
        else:
            charspans[name] = [0,0]
    ids = tok.encode(rendered, add_special_tokens=False)
    underlying = tok if callable(tok) else getattr(tok, '_tokenizer', tok)
    if callable(underlying):
        encoding = underlying(rendered, add_special_tokens=False, return_offsets_mapping=True)
    else:
        encoded = underlying.encode(rendered, add_special_tokens=False)
        encoding = {'input_ids': encoded.ids, 'offset_mapping': encoded.offsets}
    assert list(encoding['input_ids']) == list(ids), 'Offset tokenizer must match actual model IDs'
    offsets = encoding['offset_mapping']
    spans, straddles = {}, {}
    for scope,(a,b) in charspans.items():
        selected = [i for i,(x,y) in enumerate(offsets) if b>a and y>a and x<b and y>x]
        spans[scope] = [min(selected),max(selected)+1] if selected else [len(ids),len(ids)]
        straddles[scope] = [i for i in selected if offsets[i][0]<a or offsets[i][1]>b]
    spans['readout'] = [len(ids)-1,len(ids)]
    spans['all'] = [0,len(ids)]
    boundaries = sorted(set([0,len(rendered)]+[v for ab in charspans.values() for v in ab]))
    segmented = [i for a,b in zip(boundaries,boundaries[1:]) for i in tok.encode(rendered[a:b],add_special_tokens=False)]
    audit = dict(canonical_tokens=True, segmented_differs=segmented!=list(ids),
                 segmented_token_count=len(segmented), canonical_token_count=len(ids),
                 span_policy='all canonical tokens overlapping declared character span; boundary straddlers retained and listed',
                 boundary_straddlers=straddles, character_spans=charspans)
    return list(ids),spans,rendered,audit


def apply_numpy(old, op, active):
    """Reference math for CPU selftests; mirrors MLX hook incl. clipping."""
    import numpy as np
    value = float(op.get('value', 0.))
    mode = op['mode']
    if mode == 'add':
        replacement = old + value
    elif mode == 'toward':
        fraction = op.get('fraction', 1.)
        active = active & ((old < value) if op['direction'] > 0 else (old > value))
        replacement = old + fraction*(value-old)
    elif mode in ('clamp', 'replace', 'suppress_high', 'suppress_low'):
        replacement = np.full_like(old, value)
    elif mode == 'restore':
        replacement = old.copy()
    else:
        raise ValueError(mode)
    if 'clip' in op:
        replacement = np.clip(replacement, *op['clip'])
    if mode == 'suppress_high':
        active = active & (old > op['threshold'])
    if mode == 'suppress_low':
        active = active & (old < op['threshold'])
    return np.where(active, replacement, old)


def validate_job(job):
    if job is None:
        return
    assert job['revision'] == REVISION
    assert len({c['condition_id'] for c in job['conditions']}) == len(job['conditions'])
    for condition in job['conditions']:
        seen = set()
        for op in condition.get('operations', []):
            assert op['ingredient'] in INGREDIENTS
            assert op['ingredient'] not in seen, 'One operation per ingredient per condition'
            seen.add(op['ingredient'])
            assert op['mode'] in {'add', 'toward', 'clamp', 'replace', 'suppress_high', 'suppress_low', 'restore'}
            assert op.get('scope', 'all') in {'carrier', 'keyword', 'readout', 'readout_text', 'all', 'generation'}
            if op['mode'] == 'toward':
                assert 0 < op.get('fraction', 1.) <= 1 and op['direction'] in (-1, 1)
            if op['mode'] == 'add':
                assert 'clip' in op, 'Additive probes must explicitly state observed-range clipping'
            if 'clip' in op:
                assert len(op['clip']) == 2 and op['clip'][0] <= op['clip'][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=['natural', 'probe', 'generation'], default='natural')
    parser.add_argument('--job', type=Path)
    parser.add_argument('--model', default='/Users/arya/Projects/mechinterp/model-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/'+REVISION)
    parser.add_argument('--coordinates', type=Path, help='Explicit replacement name:[layer,neuron,direction] mapping; one channel per layer')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.coordinates:
        global INGREDIENTS
        INGREDIENTS = {k:tuple(v) for k,v in json.loads(args.coordinates.read_text()).items()}
        assert len({v[0] for v in INGREDIENTS.values()}) == len(INGREDIENTS), 'One channel per layer per run'
        assert all(0<=l<32 and 0<=n<9216 and d in (-1,1) for l,n,d in INGREDIENTS.values())
    corpus = json.loads(args.cases.read_text())
    cases = corpus['cases']
    key = lambda c: c.get('case_id', c.get('id'))
    assert len({key(c) for c in cases}) == len(cases)
    job = json.loads(args.job.read_text()) if args.job else None
    validate_job(job)
    if args.phase != 'natural' and job is None:
        raise ValueError('Frozen --job required')
    if not args.run:
        print(json.dumps(dict(valid=True, cases=len(cases), phase=args.phase, model_loaded=False)))
        return
    run(args, corpus, job)


def run(args, corpus, job):
    import importlib.metadata
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    assert Path(args.model).name == REVISION
    mx.set_default_device(mx.gpu)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / (args.phase + '.jsonl')
    if target.exists():
        raise FileExistsError(target)
    model, tok, config = load(args.model, return_config=True)
    cfg = config.get('text_config', config)
    assert len(model.layers) == 32 and cfg['intermediate_size'] == 9216
    for layer in model.layers:
        for attr in ('gate_proj', 'up_proj', 'down_proj'):
            weight = getattr(layer.mlp, attr)
            assert not hasattr(weight, 'scales') and weight.weight.dtype == mx.bfloat16

    class Probe(nn.Module):
        def __init__(self, original, neuron):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = original.gate_proj, original.up_proj, original.down_proj
            self.neuron = neuron
            self.reset()

        def reset(self, op=None, bounds=(0, 0)):
            self.op, self.bounds, self.seen, self.records = op, bounds, 0, []

        def __call__(self, x):
            h = nn.silu(self.gate_proj(x)) * self.up_proj(x)
            old = h[..., self.neuron:self.neuron+1]
            post = old
            if self.op is not None:
                op = self.op
                pos = mx.arange(self.seen, self.seen+h.shape[1])
                active = ((pos >= self.bounds[0]) & (pos < self.bounds[1]))[None, :, None]
                mode = op['mode']
                replacement = old + op.get('value', 0.) if mode == 'add' else mx.full_like(old, op.get('value', 0.))
                if mode == 'toward':
                    replacement = old + op.get('fraction', 1.)*(op['value']-old)
                    active = active & ((old < op['value']) if op['direction'] > 0 else (old > op['value']))
                if mode == 'restore':
                    replacement = old
                if 'clip' in op:
                    replacement = mx.clip(replacement, *op['clip'])
                if mode == 'suppress_high':
                    active = active & (old > op['threshold'])
                if mode == 'suppress_low':
                    active = active & (old < op['threshold'])
                post = mx.where(active, replacement, old)
                onehot = (mx.arange(h.shape[-1]) == self.neuron).astype(h.dtype)
                h = h + (post-old)*onehot
                # Capture achieved scalar after BF16 addition, not requested replacement.
                post = h[..., self.neuron:self.neuron+1]
            self.records.append((self.seen, old[0, :, 0].astype(mx.float32), post[0, :, 0].astype(mx.float32)))
            self.seen += h.shape[1]
            return self.down_proj(h)

    probes = {}
    columns = {}
    for name, (layer, neuron, _) in INGREDIENTS.items():
        probe = Probe(model.layers[layer].mlp, neuron)
        model.layers[layer].mlp = probe
        probes[name] = probe
        col = probe.down_proj.weight[:, neuron].astype(mx.float32)
        columns[name] = float(mx.linalg.norm(col))
    metadata = dict(revision=REVISION, cases_sha256=sha(args.cases), job_sha256=sha(args.job) if args.job else None,
                    runner_sha256=sha(__file__), ingredients=INGREDIENTS, config=config,
                    versions={p: importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers')},
                    hardware=mx.device_info(), down_column_l2=columns, native_units=True,
                    coordinates_sha256=sha(args.coordinates) if args.coordinates else None,
                    tokenization='Canonical full-text encoding; offset overlap spans; boundary-straddling tokens listed per row')
    save(args.output / (args.phase + '_metadata.json'), metadata)
    cases = corpus['cases']
    if job and job.get('case_ids'):
        cases = [c for c in cases if c.get('case_id', c.get('id')) in job['case_ids']]
        assert len(cases) == len(job['case_ids'])
    calibration = {name: {scope: [] for scope in ('carrier','keyword','readout','readout_text','all')} for name in probes}

    def reset(condition, spans, length):
        operations = {o['ingredient']: o for o in condition.get('operations', [])}
        for name, probe in probes.items():
            op = operations.get(name)
            scope = op.get('scope', 'all') if op else 'all'
            bounds = (length-1, 10**9) if scope == 'generation' else tuple(spans[scope])
            if scope == 'all' and args.phase == 'generation':
                bounds = (0, 10**9)
            probe.reset(op, bounds)

    def records():
        result = {}
        for name, probe in probes.items():
            mx.eval([r[1:] for r in probe.records])
            result[name] = {'positions': [i for offset, pre, post in probe.records for i in range(offset, offset+len(pre))],
                            'pre': [float(v) for _, pre, _ in probe.records for v in np.asarray(pre)],
                            'post': [float(v) for _, _, post in probe.records for v in np.asarray(post)]}
        return result

    def forward(ids):
        logits = model(mx.array(ids)[None], cache=None)[0].astype(mx.float32)
        mx.eval(logits)
        return np.asarray(logits)

    count = 0
    with target.open('x') as out:
        for case in cases:
            cid = case.get('case_id', case.get('id'))
            ids, spans, rendered, audit = segmented_tokens(tok, case, probe=args.phase == 'probe')
            conditions = [{'condition_id': 'baseline'}] if args.phase == 'natural' else job['conditions']
            for condition in conditions:
                started = time.monotonic()
                reset(condition, spans, len(ids))
                row = dict(case_id=cid, split=case.get('split'), condition_id=condition['condition_id'],
                           token_ids=ids, rendered_input=rendered, spans=spans, tokenization_audit=audit, operations=condition.get('operations', []))
                if args.phase == 'generation':
                    mx.random.seed(job.get('seed', 42))
                    text, last = [], None
                    for part in stream_generate(model, tok, ids, max_tokens=job.get('max_tokens', 96),
                                                sampler=make_sampler(job.get('temperature', 0.)),
                                                logits_processors=make_logits_processors(repetition_penalty=job.get('repetition_penalty',1.1), repetition_context_size=128)):
                        text.append(part.text); last = part
                    row.update(text=''.join(text), generated_tokens=last.generation_tokens, finish_reason=last.finish_reason)
                    row['activations'] = records()
                else:
                    logits = forward(ids)
                    row['activations'] = records()
                    if args.phase == 'probe':
                        scores = {}
                        for label, continuation in case.get('choices', {}).items():
                            continuation_ids = tok.encode(continuation, add_special_tokens=False)
                            assert continuation_ids
                            reset(condition, spans, len(ids))
                            branch_logits = forward(ids + continuation_ids[:-1])
                            selected = branch_logits[len(ids)-1:len(ids)+len(continuation_ids)-1]
                            peak = selected.max(axis=1)
                            logz = peak + np.log(np.exp(selected-peak[:,None]).sum(axis=1))
                            logps = selected[np.arange(len(continuation_ids)),continuation_ids]-logz
                            scores[label] = dict(continuation=continuation, token_ids=continuation_ids,
                                                 token_logprobs=logps.tolist(), sum_logprob=float(logps.sum()),
                                                 mean_logprob=float(logps.mean()), activations=records())
                        row['choices'] = scores
                        if scores:
                            sums = np.array([v['sum_logprob'] for v in scores.values()])
                            mxlog = float(sums.max())
                            logmass = mxlog + float(np.log(np.exp(sums-mxlog).sum()))
                            row['candidate_logmass'] = logmass
                            row['candidate_mass'] = float(np.exp(logmass))
                            row['conditional_choice_probabilities'] = {label: float(np.exp(v['sum_logprob']-logmass)) for label,v in scores.items()}
                    elif case.get('split') in ('discovery','calibration'):
                        for name, values in row['activations'].items():
                            assert values['pre'] == values['post']
                            for scope, (a,b) in spans.items():
                                calibration[name][scope].extend(values['pre'][a:b])
                row['seconds'] = time.monotonic()-started
                out.write(json.dumps(row, ensure_ascii=False)+'\n'); out.flush()
                count += 1
            print(json.dumps(dict(event='case_complete', phase=args.phase, case_id=cid, rows=count)), flush=True)
    if args.phase == 'natural':
        summary = {}
        for name, scopes in calibration.items():
            summary[name] = {scope: {'count': len(values), 'quantiles': dict(zip(map(str,PERCENTILES),np.percentile(values,PERCENTILES).tolist()))}
                             for scope,values in scopes.items() if values}
        save(args.output/'calibration.json', dict(percentiles=PERCENTILES, values=summary,
             discovery_case_ids=[c.get('case_id',c.get('id')) for c in cases if c.get('split') in ('discovery','calibration')],
             cases_sha256=sha(args.cases), raw_sha256=sha(target), down_column_l2=columns,
             warning='Synthetic discovery-corpus ranges, not universal physiological bounds. Quantiles pooled by token, not case.'))
    save(args.output/(args.phase+'_completion.json'),dict(rows=count,raw_sha256=sha(target),metadata=metadata))


if __name__ == '__main__':
    main()
