"""Frozen native-channel pride/word experiment. GPU imports only under --run.

All intervention values are raw signed SwiGLU scalars estimated from calibration
families only. No normalized steering doses, training, or candidate selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def validate(args):
    corpus = json.loads(args.corpus.read_text())
    plan = json.loads(args.plan.read_text())
    assert sha(args.corpus) == plan['corpus_sha256']
    assert plan['revision'] == REVISION and plan['layer'] == 22 and plan['target'] == 1911
    assert len(corpus['cases']) == 132
    assert len({c['id'] for c in corpus['cases']}) == 132
    assert {c['split'] for c in corpus['cases']} == {'calibration', 'validation', 'validation_lexical_diagnostic'}
    assert len(plan['controls']) == 3 and 1911 not in plan['controls']
    assert len(set(plan['controls'])) == 3
    for c in corpus['cases']:
        assert set(c['choices']) == {'A', 'B'}
        assert c['expected_semantics']['correct_choice'] in c['choices']
        assert c['text'] and c['carrier_suffix'] and c['semantic_question']
    return corpus, plan


def run(args, corpus, plan):
    import importlib.metadata
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_default_device(mx.gpu)
    if Path(args.model).name != REVISION:
        raise ValueError('Pinned original snapshot is required')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    reserve = ['109_causal.jsonl', '109_generations.jsonl'] if args.causal_only else ['109_natural.jsonl', '109_causal.jsonl', '109_calibration.json', '109_generations.jsonl']
    for name in reserve:
        if (out / name).exists():
            raise FileExistsError('Refusing to overwrite ' + name)
    model, tok, config = load(args.model, return_config=True)
    cfg = config.get('text_config', config)
    assert len(model.layers) == 32 and cfg['intermediate_size'] == 9216
    for b in model.layers:
        for k in ['gate_proj', 'up_proj', 'down_proj']:
            p = getattr(b.mlp, k)
            assert not hasattr(p, 'scales') and p.weight.dtype == mx.bfloat16
    channels = [1911] + plan['controls']

    class ProbeMLP(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = original.gate_proj, original.up_proj, original.down_proj
            self.channels = mx.array(channels)
            self.operation = None
            self.bounds = (0, 0)
            self.seen = 0
            self.pre = self.post = None
            self.record = True

        def __call__(self, x):
            h = nn.silu(self.gate_proj(x)) * self.up_proj(x)
            if self.record:
                self.pre = mx.take(h[0], self.channels, axis=-1).astype(mx.float32)
            if self.operation is not None:
                neuron, op, value, threshold = self.operation
                positions = mx.arange(self.seen, self.seen + h.shape[1])
                active = ((positions >= self.bounds[0]) & (positions < self.bounds[1]))[None, :, None]
                old = h[..., neuron:neuron + 1]
                replacement = mx.full_like(old, value)
                if op == 'suppress_high':
                    active = active & (old > threshold)
                elif op == 'restore':
                    replacement = old
                change = mx.where(active, replacement - old, mx.zeros_like(old))
                onehot = (mx.arange(9216) == neuron).astype(h.dtype)
                h = h + change * onehot
            if self.record:
                self.post = mx.take(h[0], self.channels, axis=-1).astype(mx.float32)
            self.seen += h.shape[1]
            return self.down_proj(h)

    probe = ProbeMLP(model.layers[22].mlp)
    model.layers[22].mlp = probe
    metadata = dict(model='Qwen/Qwen3.5-4B', revision=REVISION, corpus_sha256=sha(args.corpus),
                    plan_sha256=sha(args.plan), runner_sha256=sha(__file__), channels=channels,
                    versions={p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm', 'transformers']},
                    capture='Raw signed native gated channel before down_proj; pre and achieved post values',
                    tokenization='Chat-template text, separately encoded at exact carrier boundaries; identical across branches',
                    model_config=config, hardware=mx.device_info())
    if args.causal_only:
        previous = json.loads((out / '109_metadata.json').read_text())
        assert all(previous[k] == metadata[k] for k in ['corpus_sha256', 'plan_sha256', 'runner_sha256', 'channels', 'revision'])
    else:
        write_json(out / '109_metadata.json', metadata)
    print(json.dumps(dict(event='loaded', channels=channels)), flush=True)

    def prompt(case, task, order=0):
        text = case['text'] + case['carrier_suffix']
        pride_letter = None
        if task == 'semantic':
            choices = dict(case['choices'])
            correct = case['expected_semantics']['correct_choice']
            pride_letter = correct if case['expected_semantics']['own_accomplishment_satisfaction'] else ('B' if correct == 'A' else 'A')
            if order:
                choices = dict(A=choices['B'], B=choices['A'])
                pride_letter = 'B' if pride_letter == 'A' else 'A'
            text += '\n\n' + case['semantic_question'] + '\nA: ' + choices['A'] + '\nB: ' + choices['B'] + '\n'
        elif task == 'lexical':
            text += '\n\nDescribe her feeling using one adjective.'
        elif task == 'generation':
            templates = {'en': '\n\nRespond to this account in one or two sentences.',
                         'es': '\n\nResponde a este relato en una o dos frases.',
                         'zh': '\n\n请用一两句话回应这段叙述。'}
            text += templates[case['language']]
        messages = [{'role': 'system', 'content': corpus['task_templates']['system']}, {'role': 'user', 'content': text}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if task == 'lexical':
            rendered += 'The person feels'
        start = rendered.index(case['carrier_suffix'])
        end = start + len(case['carrier_suffix'])
        first = tok.encode(rendered[:start], add_special_tokens=False)
        middle = tok.encode(rendered[start:end], add_special_tokens=False)
        tail = tok.encode(rendered[end:], add_special_tokens=False)
        ids = first + middle + tail
        storychar = rendered.index(case['text'])
        story_start = len(tok.encode(rendered[:storychar], add_special_tokens=False))
        return ids, len(first), len(first) + len(middle), pride_letter, story_start, messages

    def forward(ids, operation=None, bounds=(0, 0)):
        probe.seen = 0
        probe.operation, probe.bounds, probe.record = operation, bounds, True
        started = time.monotonic()
        logits = model(mx.array(ids)[None], cache=None)[0, -1].astype(mx.float32)
        mx.eval(logits, probe.pre, probe.post)
        return np.asarray(logits), np.asarray(probe.pre), np.asarray(probe.post), time.monotonic() - started

    # The next-token choice labels must each be one whole tokenizer token.
    label_tokens = {s: tok.encode(s, add_special_tokens=False) for s in ['A', 'B', ' proud', ' pleased']}
    assert all(len(t) == 1 for t in label_tokens.values()), label_tokens
    metadata['label_token_ids'] = label_tokens
    if not args.causal_only:
        write_json(out / '109_metadata.json', metadata)
    if args.causal_only:
        cal = json.loads((out / '109_calibration.json').read_text())
        assert cal['channels'] == channels
        causal(args, corpus, plan, cal, prompt, forward, label_tokens, mx, np, model, tok, probe, out,
               stream_generate, make_sampler, make_logits_processors)
        return
    # Implementation checks use a calibration case; no behavioral selection.
    check_case = next(c for c in corpus['cases'] if c['split'] == 'calibration')
    ids, start, end, _, _, _ = prompt(check_case, 'natural')
    full, before, after, _ = forward(ids)
    restored, _, _, _ = forward(ids, (1911, 'restore', 0., 0.), (start, end))
    assert np.array_equal(full, restored) and np.array_equal(before, after)
    # Cached/chunked forward is diagnostic; BF16 arithmetic can differ slightly.
    probe.operation, probe.seen, probe.record = None, 0, False
    cache = make_prompt_cache(model)
    half = max(1, len(ids) // 2)
    firstlogits = model(mx.array(ids[:half])[None], cache=cache)
    mx.eval(firstlogits, [c.state for c in cache])
    chunked = model(mx.array(ids[half:])[None], cache=cache)[0, -1].astype(mx.float32)
    mx.eval(chunked)
    error = float(np.max(np.abs(full - np.asarray(chunked))))
    checks = dict(noop_exact=True, no_intervention_capture_exact=True, cached_chunk_max_abs_logit_difference=error,
                  cached_top1_matches=int(full.argmax()) == int(np.asarray(chunked).argmax()))
    if error > 0.5:
        raise RuntimeError('Large full/chunked mismatch; investigate before collecting')
    write_json(out / '109_implementation_checks.json', checks)

    calibration = []
    with (out / '109_natural.jsonl').open('x') as f:
        # Calibration cases explicitly precede validation; only these fix ranges.
        cases = sorted(corpus['cases'], key=lambda c: c['split'] != 'calibration')
        for case in cases:
            ids, start, end, _, story_start, _ = prompt(case, 'natural')
            _, pre, post, seconds = forward(ids)
            assert np.array_equal(pre, post)
            row = {k: case[k] for k in ['language', 'split', 'category', 'family', 'lexeme_present']}
            row.update(case_id=case['id'], prompt_tokens=len(ids), carrier_start=start, carrier_end=end,
                       story_start=story_start, story_end=start, token_ids=ids, channels=channels,
                       pre=pre.tolist(), post=post.tolist(), seconds=seconds)
            f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
            if case['split'] == 'calibration':
                calibration.append(pre[start:end])
        print(json.dumps(dict(event='natural_complete', cases=len(cases))), flush=True)
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    carrier_values = np.concatenate(calibration, axis=0)
    ranges = dict(carrier=np.percentile(carrier_values, percentiles, axis=0).T.tolist())
    output_values = {'semantic': [], 'lexical': []}
    for case in corpus['cases']:
        if case['split'] != 'calibration':
            continue
        for task in ['semantic', 'lexical']:
            if task == 'lexical' and case['language'] != 'en':
                continue
            for order in ([0, 1] if task == 'semantic' else [0]):
                ids, *_ = prompt(case, task, order)
                _, pre, _, _ = forward(ids)
                output_values[task].append(pre[-1])
    for task, values in output_values.items():
        ranges[task + '_output'] = np.percentile(np.stack(values), percentiles, axis=0).T.tolist()
    columns = np.asarray(mx.linalg.norm(probe.down_proj.weight[:, channels].astype(mx.float32), axis=0)).tolist()
    cal = dict(percentiles=percentiles, channels=channels, values=ranges, down_column_l2=columns,
               calibration_case_ids=[c['id'] for c in corpus['cases'] if c['split'] == 'calibration'],
               carrier_token_count=len(carrier_values), output_sample_counts={k: len(v) for k, v in output_values.items()},
               rule='Signed quantiles; per-coordinate pooled calibration carrier positions and separately task-specific final prediction positions',
               warning='Synthetic corpus range, not universal physiological range; interventions unmatched in residual perturbation norm')
    write_json(out / '109_calibration.json', cal)
    print(json.dumps(dict(event='calibrated', calibration_sha256=sha(out / '109_calibration.json'), target=ranges['carrier'][0])), flush=True)
    if args.calibration_only:
        return
    causal(args, corpus, plan, cal, prompt, forward, label_tokens, mx, np, model, tok, probe, out,
           stream_generate, make_sampler, make_logits_processors)


def causal(args, corpus, plan, cal, prompt, forward, label_tokens, mx, np, model, tok, probe, out,
           stream_generate, make_sampler, make_logits_processors):
    channels = cal['channels']
    def quantile(scope, channel_index, p):
        return cal['values'][scope][channel_index][cal['percentiles'].index(p)]
    conditions = ['baseline', 'target_carrier_q10', 'target_carrier_q50', 'target_carrier_q90',
                  'target_carrier_suppress_high', 'target_carrier_zero', 'target_output_q10', 'target_output_q90']
    conditions += [f'control_{i}_carrier_q{q}' for i in range(3) for q in [10, 90]]

    def branch(condition, task, length, start, end):
        if condition == 'baseline':
            return None, (0, 0), None
        parts = condition.split('_')
        ci = int(parts[1]) + 1 if parts[0] == 'control' else 0
        scope = task + '_output' if 'output' in parts else 'carrier'
        bounds = (length - 1, length) if 'output' in parts else (start, end)
        op = 'clamp'
        if condition.endswith('suppress_high'):
            value, threshold, op = quantile(scope, ci, 50), quantile(scope, ci, 90), 'suppress_high'
        elif condition.endswith('zero'):
            value, threshold = 0., 0.
        else:
            value, threshold = quantile(scope, ci, int(parts[-1][1:])), 0.
        return (channels[ci], op, value, threshold), bounds, ci

    # Causal inputs are all fresh validation cases; no calibration behavior used.
    completed = 0
    with (out / '109_causal.jsonl').open('x') as f:
        for case in corpus['cases']:
            if case['split'] == 'calibration':
                continue
            for task in ['semantic', 'lexical']:
                if task == 'lexical' and case['language'] != 'en':
                    continue
                for order in ([0, 1] if task == 'semantic' else [0]):
                    ids, start, end, pride_letter, _, _ = prompt(case, task, order)
                    base_logits = None
                    for condition in conditions:
                        operation, bounds, ci = branch(condition, task, len(ids), start, end)
                        logits, pre, post, seconds = forward(ids, operation, bounds)
                        if condition == 'baseline':
                            base_logits = logits
                        pairs = [pride_letter, 'B' if pride_letter == 'A' else 'A'] if task == 'semantic' else [' proud', ' pleased']
                        maximum = logits.max()
                        logz = float(maximum + np.log(np.exp(logits - maximum).sum()))
                        lp = {s: float(logits[label_tokens[s][0]] - logz) for s in pairs}
                        odds = lp[pairs[0]] - lp[pairs[1]]
                        prob = float(1. / (1. + np.exp(-np.clip(odds, -80, 80))))
                        delta = post - pre
                        if ci is None:
                            assert np.array_equal(pre, post)
                        else:
                            others = [j for j in range(4) if j != ci]
                            assert np.array_equal(pre[:, others], post[:, others])
                            assert not np.any(delta[:bounds[0]]) and not np.any(delta[bounds[1]:])
                            # BF16 achieved value may round the requested scalar.
                            selected = pre[bounds[0]:bounds[1], ci]
                            changed = np.abs(delta[:, ci]) > 0
                            if operation[1] == 'suppress_high':
                                assert np.all(post[changed, ci] <= pre[changed, ci])
                        row = {k: case[k] for k in ['language', 'split', 'category', 'family', 'lexeme_present']}
                        row.update(case_id=case['id'], task=task, order=order, condition=condition,
                                   log_odds_pride=odds, prob_pride=prob, token_logprobs=lp,
                                   choice_correct=(prob >= 0.5) == case['expected_semantics']['own_accomplishment_satisfaction'] if task == 'semantic' else None,
                                   prompt_tokens=len(ids), prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                   operation=operation, intervention_bounds=bounds, seconds=seconds,
                                   changed_positions=int(np.any(delta != 0, axis=1).sum()),
                                   achieved_max_abs_delta=float(np.abs(delta).max()),
                                   achieved_mean_abs_delta=float(np.abs(delta[:, ci]).mean()) if ci is not None else 0.,
                                   achieved_residual_l2_sum=float(np.abs(delta[:, ci]).sum() * cal['down_column_l2'][ci]) if ci is not None else 0.,
                                   max_abs_logit_change=float(np.max(np.abs(logits - base_logits))))
                        f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
                        completed += 1
            print(json.dumps(dict(event='causal_case_complete', case=case['id'], rows=completed)), flush=True)
    # Fixed small readable confirmation; selection from prospective IDs only.
    with (out / '109_generations.jsonl').open('x') as f:
        for cid in plan['generation_case_ids']:
            case = next(c for c in corpus['cases'] if c['id'] == cid)
            ids, start, end, _, _, messages = prompt(case, 'generation')
            for condition in ['baseline', 'target_carrier_q90', 'target_carrier_suppress_high']:
                operation, bounds, _ = branch(condition, 'semantic', len(ids), start, end)
                probe.operation, probe.bounds, probe.seen, probe.record = operation, bounds, 0, False
                mx.random.seed(42)
                chunks, last = [], None
                begun = time.monotonic()
                for part in stream_generate(model, tok, ids, max_tokens=96, sampler=make_sampler(0.),
                                            logits_processors=make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)):
                    chunks.append(part.text); last = part
                row = dict(case_id=cid, language=case['language'], category=case['category'], condition=condition,
                           messages=messages, token_ids=ids, operation=operation, intervention_bounds=bounds,
                           text=''.join(chunks), generated_tokens=last.generation_tokens, finish_reason=last.finish_reason,
                           seconds=time.monotonic() - begun)
                f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
            print(json.dumps(dict(event='generation_case_complete', case=cid)), flush=True)
    write_json(out / '109_completion.json', dict(causal_rows=completed, natural_cases=len(corpus['cases']),
               generation_rows=3 * len(plan['generation_case_ids']), plan_sha256=sha(args.plan),
               files={p.name: sha(p) for p in out.glob('109_*.json*') if p.name != '109_completion.json'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=ROOT / '109_PRIDE_FRESH_CASES.json')
    p.add_argument('--plan', type=Path, default=ROOT / '109_PRIDE_PLAN.json')
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--model', default='/Users/arya/Projects/mechinterp/model-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/' + REVISION)
    phase = p.add_mutually_exclusive_group()
    phase.add_argument('--calibration-only', action='store_true')
    phase.add_argument('--causal-only', action='store_true')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    corpus, plan = validate(args)
    if not args.run:
        print(json.dumps(dict(valid=True, cases=len(corpus['cases']), channels=[1911] + plan['controls'], model_loaded=False)))
        return
    run(args, corpus, plan)


if __name__ == '__main__':
    main()
