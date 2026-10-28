"""Unsteered native SwiGLU association screen; CPU validation/ranking are model-free.

Only --capture imports MLX or loads weights. Capture teacher-forces fixed text,
without generating, fitting, steering, changing weights, or mixing coordinates.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
NORMS_SHA256 = '8f52cd50db336b08a040e48010d0305760d086a7db09ef3cab29e5552c57777e'
CONDITIONS = ['joy', 'neutral', 'sadness', 'cute', 'agreeable']
ATLAS_LEADS = [(13, 4810), (24, 7361), (28, 1478), (25, 7455), (22, 8753)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(corpus_path, plan_path, norms_path):
    corpus = json.loads(corpus_path.read_text())
    plan = json.loads(plan_path.read_text())
    if corpus['conditions'] != CONDITIONS or plan['model_revision'] != MODEL_REVISION:
        raise ValueError('Unexpected condition order/model revision')
    if plan['corpus_sha256'] != digest(corpus_path):
        raise ValueError('Corpus differs from frozen plan')
    contexts = corpus['contexts']
    ids = [c['id'] for c in contexts]
    if len(contexts) != 10 or len(set(ids)) != 10:
        raise ValueError('Need ten unique context topics')
    if [c['split'] for c in contexts].count('discovery') != 6 or [c['split'] for c in contexts].count('confirmation') != 4:
        raise ValueError('Need six discovery and four confirmation contexts')
    leakage = re.compile(r'\b(happ(?:y|ier|iest|iness)|joy\w*|glad|delight\w*|sad\w*|cute\w*|agree\w*|cheer\w*|smil\w*|meow\w*|catgirl\w*)\b', re.I)
    suffix = corpus['shared_suffix']
    if len(suffix) < 40 or leakage.search(suffix):
        raise ValueError('Measured shared suffix contains affect/persona labels or is too short')
    narratives = []
    for c in contexts:
        if set(c['narratives']) != set(CONDITIONS):
            raise ValueError('Each context needs all five conditions')
        if leakage.search(c['prompt']):
            raise ValueError('User prompt contains direct affect label')
        for text in c['narratives'].values():
            if not isinstance(text, str) or not text.strip() or text.endswith(suffix):
                raise ValueError('Invalid narrative')
            narratives.append(text)
    if len(set(narratives)) != 50:
        raise ValueError('Narratives must be unique across splits and conditions')
    norms = json.loads(norms_path.read_text())
    if digest(norms_path) != NORMS_SHA256 or norms['model'] != 'Qwen/Qwen3.5-4B' or norms['dtype'] != 'bfloat16' or len(norms['medians']) != 32:
        raise ValueError('Published norm identity mismatch')
    return corpus, plan


def capture(args, corpus, plan):
    # Kept inside this explicit GPU entry point: importing this file is CPU-safe.
    import mlx.core as mx
    from mlx_lm import load
    import numpy as np
    sys.path.insert(0, str(HERE.parent))
    from runner import SteeredMLP

    model_path = Path(args.model).expanduser().resolve()
    if model_path.name != MODEL_REVISION or not model_path.is_dir():
        raise ValueError('Use the pinned original-model snapshot directory')
    model, tok, config = load(str(model_path), return_config=True)
    cfg = config.get('text_config', config)
    if len(model.layers) != 32 or cfg.get('hidden_size') != 2560 or cfg.get('intermediate_size') != 9216 or cfg.get('model_type') not in ('qwen3_5', 'qwen3_5_text'):
        raise ValueError('Expected original Qwen3.5-4B architecture')
    mlps = []
    for layer, block in enumerate(model.layers):
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            projection = getattr(block.mlp, name)
            if hasattr(projection, 'scales') or projection.weight.dtype != mx.bfloat16:
                raise ValueError('Original unquantized BF16 projections required')
        block.mlp = SteeredMLP(block.mlp, layer)
        mlps.append(block.mlp)
    # Derive the exact suffix token IDs once, and concatenate explicitly, so
    # tokenizer boundary merges cannot put different words into scored tokens.
    suffix_ids = tok.encode(corpus['shared_suffix'], add_special_tokens=False)
    if not suffix_ids:
        raise ValueError('Suffix has no tokens')
    values = np.empty((10, 5, 32, 9216), dtype=np.float32)
    token_log = []
    for ci, context in enumerate(corpus['contexts']):
        messages = [{'role': 'system', 'content': corpus['system']}, {'role': 'user', 'content': context['prompt']}]
        prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        for pi, condition in enumerate(CONDITIONS):
            body_ids = tok.encode(context['narratives'][condition], add_special_tokens=False)
            ids = prefix + body_ids + suffix_ids
            start = len(prefix) + len(body_ids)
            for mlp in mlps:
                if mlp.delta is not None or mlp.gain is not None or mlp.position_mask is not None or mlp.early_tokens is not None:
                    raise RuntimeError('Natural capture must have no intervention')
                mlp.capture = True
                mlp.start = start
                mlp.seen = 0
            try:
                logits = model(mx.array(ids)[None], cache=None)
                mx.eval(logits, [m.captured for m in mlps])
                sample = np.stack([np.asarray(m.captured) for m in mlps])
                if sample.shape != (32, 9216) or not np.isfinite(sample).all():
                    raise RuntimeError('Invalid native activation capture')
                values[ci, pi] = sample
            finally:
                for mlp in mlps:
                    mlp.capture = False
                    mlp.captured = None
            token_log.append(dict(context=context['id'], condition=condition,
                                  prefix_tokens=len(prefix), narrative_tokens=len(body_ids),
                                  scored_token_ids=suffix_ids, scored_start=start,
                                  input_token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
            print(json.dumps(dict(event='captured', context=context['id'], condition=condition)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), values=values,
                        context_ids=np.asarray([c['id'] for c in corpus['contexts']]),
                        splits=np.asarray([c['split'] for c in corpus['contexts']]),
                        conditions=np.asarray(CONDITIONS))
    args.output.with_suffix('.capture.json').write_text(json.dumps(dict(
        model=str(model_path), revision=MODEL_REVISION, model_config=config,
        corpus_sha256=digest(args.corpus), plan_sha256=digest(args.plan),
        norms_sha256=digest(args.norms), norms_role='Provenance only; raw natural activations are not dose-normalized',
        native_coordinate='silu(gate_proj(x))*up_proj(x), before down_proj; zero-based axes',
        scored_text=corpus['shared_suffix'], independent_cache_per_forward=True,
        tokenization='Chat prefix plus separately encoded narrative plus identical separately encoded suffix IDs',
        token_log=token_log), indent=2) + '\n')
    rank(args.output.with_suffix('.npz'), args.output, corpus, args)


def statistics(values, splits):
    """Pure CPU statistics; confirmation cannot affect discovery order."""
    import numpy as np
    discovery = np.asarray([s == 'discovery' for s in splits])
    confirmation = ~discovery
    # Sign is ONE scalar per native coordinate, fixed from discovery joy-neutral.
    # No combinations/rotations or regression coefficients are constructed.
    differences = values[:, :1] - values[:, 1:]
    disc = differences[discovery]
    conf = differences[confirmation]
    mean = disc.mean(0)
    sign = np.where(mean[0] >= 0, 1, -1)
    standardized = sign[None] * mean / (disc.std(0, ddof=1) + 0.05)
    consistency = (disc * sign[None, None] > 0).mean(0)
    worst = standardized.min(0)
    score = worst * consistency.min(0) ** 3
    # Every original axis retained, including negatives, zeros and confounds.
    order = np.argsort(-score.ravel(), kind='stable')
    confirm_mean = conf.mean(0)
    confirm_standardized = sign[None] * confirm_mean / (conf.std(0, ddof=1) + 0.05)
    confirm_consistency = (conf * sign[None, None] > 0).mean(0)
    contrast_orders = np.stack([np.argsort(-s.ravel(), kind='stable') for s in standardized])
    return dict(order=order, score=score, sign=sign, mean_delta=mean,
                standardized=standardized, consistency=consistency,
                per_contrast_orders=contrast_orders,
                confirmation_mean_delta=confirm_mean,
                confirmation_standardized=confirm_standardized,
                confirmation_consistency=confirm_consistency)


def rank(source, output, corpus, args):
    import numpy as np
    with np.load(source, allow_pickle=False) as z:
        values = z['values'].astype(np.float64)
        ids = z['context_ids'].tolist()
        splits = z['splits'].tolist()
        conditions = z['conditions'].tolist()
    if values.shape != (10, 5, 32, 9216) or not np.isfinite(values).all():
        raise ValueError('Expected finite [10,5,32,9216] raw capture')
    if ids != [c['id'] for c in corpus['contexts']] or splits != [c['split'] for c in corpus['contexts']] or conditions != CONDITIONS:
        raise ValueError('Capture/corpus context identity mismatch')
    manifest = json.loads(source.with_suffix('.capture.json').read_text())
    if manifest['corpus_sha256'] != digest(args.corpus) or manifest['plan_sha256'] != digest(args.plan) or manifest['norms_sha256'] != digest(args.norms):
        raise ValueError('Capture/corpus/plan/norm provenance mismatch')
    stats = statistics(values, splits)
    order, score, sign = (stats[k] for k in ['order', 'score', 'sign'])
    standardized = stats['standardized']
    confirm_standardized = stats['confirmation_standardized']
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.all_axes.npz'), **stats)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order)) + 1
    fields = ['rank', 'layer', 'neuron', 'sign', 'discovery_score']
    for pole in CONDITIONS[1:]:
        fields += [f'discovery_vs_{pole}', f'confirmation_vs_{pole}']

    def row(flat):
        layer, neuron = np.unravel_index(flat, score.shape)
        result = dict(rank=int(inverse[flat]), layer=int(layer), neuron=int(neuron),
                      sign=int(sign[layer, neuron]), discovery_score=float(score[layer, neuron]))
        for k, pole in enumerate(CONDITIONS[1:]):
            result[f'discovery_vs_{pole}'] = float(standardized[k, layer, neuron])
            result[f'confirmation_vs_{pole}'] = float(confirm_standardized[k, layer, neuron])
        return result
    with gzip.open(output.with_suffix('.all_axes.jsonl.gz'), 'wt') as f:
        for flat in order:
            f.write(json.dumps(row(flat), separators=(',', ':')) + '\n')
    with output.with_suffix('.top.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row(flat) for flat in order[:64])
    lead_rows = [row(np.ravel_multi_index((l, n), score.shape)) for l, n in ATLAS_LEADS]
    output.with_suffix('.summary.json').write_text(json.dumps(dict(
        capture_sha256=digest(source), corpus_sha256=digest(args.corpus),
        plan_sha256=digest(args.plan), discovery_contexts=6, confirmation_contexts=4,
        coordinate_count=int(score.size), ranking='Discovery only; confirmation never reranks',
        score='min_controls(sign*mean_delta/(sample_sd+0.05))*min_controls(sign_consistency)^3',
        atlas_leads_descriptive_only=lead_rows,
        limitations='Synthetic unsteered teacher-forced text association, not spontaneous emotion, neuron function, causality or a validated intervention. Shared scored tokens remove direct measured-token labels, not all contextual lexical/style/length confounds. Six discovery contexts and multiple comparisons can produce unstable ranks. Negative controls and independent interventions remain necessary.'), indent=2) + '\n')
    print(json.dumps(dict(event='ranked', coordinates=int(score.size), output=str(output),
                         atlas_leads=lead_rows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=HERE / '103_HAPPY_CONTRAST_CORPUS.json')
    parser.add_argument('--plan', type=Path, default=HERE / '103_HAPPY_ACTIVATION_PLAN.json')
    parser.add_argument('--norms', type=Path, default=HERE.parent / 'public_qwen35_norms.json')
    parser.add_argument('--output', type=Path, default=HERE / 'results' / '103_happy_activation')
    parser.add_argument('--model', default=str(Path.home() / 'Projects/mechinterp/model-cache/hub/models--Qwen--Qwen3.5-4B/snapshots' / MODEL_REVISION))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--capture', action='store_true')
    mode.add_argument('--rank-only', type=Path)
    args = parser.parse_args()
    corpus, plan = validate(args.corpus, args.plan, args.norms)
    if args.check_only:
        print(json.dumps(dict(valid=True, model_loaded=False, mlx_imported=False,
                              contexts=10, forwards=50, corpus_sha256=digest(args.corpus),
                              plan_sha256=digest(args.plan), norms_sha256=digest(args.norms))))
        return
    if args.capture:
        for suffix in ['.npz', '.capture.json', '.all_axes.npz', '.all_axes.jsonl.gz', '.top.csv', '.summary.json']:
            if args.output.with_suffix(suffix).exists():
                raise FileExistsError('Refusing to replace existing capture/ranking; choose a new --output')
        capture(args, corpus, plan)
    else:
        rank(args.rank_only, args.output, corpus, args)


if __name__ == '__main__':
    main()
