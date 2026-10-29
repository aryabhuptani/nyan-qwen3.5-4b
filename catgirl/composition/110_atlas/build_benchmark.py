"""Build the prospective catgirl atlas benchmark, before forecasts or inference.

Local packets/raw data are deliberately separate from the compact public record.
"""
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FEATURES = ['feline_assistant_selfrole', 'female_assistant_selfrole',
            'performed_feline_vocalization', 'sustained_cute_register',
            'explicit_virtual_affection', 'coherent', 'task_relevant']
FAMILIES = ['identity_body', 'ordinary_tasks', 'conversation_social', 'lexical_fiction_exclusion']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build():
    import numpy as np
    selected = json.loads((ROOT / 'selected30_atlas.json').read_text())
    assert len(selected) == 30 and len({(x['layer'], x['neuron']) for x in selected}) == 30
    pool = json.loads((ROOT / '110_PROMPTS.json').read_text())
    rng = random.Random(110)
    prompts = []
    for family in FAMILIES:
        choices = [p for p in pool['prompts'] if p['split'] == 'validation' and p['family'] == family]
        assert len(choices) == 3
        prompts.extend(rng.sample(choices, 2))
    save('selected_prompts.json', dict(seed=110, selection='Two per family from the authored twelve-case pool; four calibration cases unused', prompts=prompts))
    coord = lambda x: (x['layer'], x['neuron'])
    known = set()
    for row in json.loads((ROOT / 'public_cooccurring_pairs.local.json').read_text()):
        if row['model'] == 'Qwen/Qwen3.5-4B':
            known.add(tuple(sorted([coord(row['coordinate_a']), coord(row['coordinate_b'])])))
    # Previous local component co-occurrence, independent of observed behavior.
    def visit(value):
        if isinstance(value, dict):
            for key, node in value.items():
                if key in ['neurons', 'recipe', 'effective_recipe'] and isinstance(node, list) and 2 <= len(node) <= 8:
                    axes = [coord(n) for n in node if isinstance(n, dict) and 'layer' in n and 'neuron' in n]
                    known.update(tuple(sorted(p)) for p in itertools.combinations(set(axes), 2))
                if isinstance(node, (list, dict)):
                    visit(node)
        elif isinstance(value, list) and len(value) < 1000:
            for item in value:
                visit(item)
    for folder in [REPO / 'catgirl/composition', REPO / 'catgirl/plans']:
        for path in folder.glob('*.json'):
            if path.stat().st_size < 3_000_000:
                visit(json.loads(path.read_text()))
    anchors = [i for i, s in enumerate(selected) if s['stratum'] == 'known_cat_related_anchor']
    others = [i for i, s in enumerate(selected) if s['stratum'] == 'random_previously_untested']
    eligible = lambda a, b: tuple(sorted([coord(selected[a]), coord(selected[b])])) not in known
    aa = [p for p in itertools.combinations(anchors, 2) if eligible(*p)]
    rng.shuffle(aa)
    pairs = aa[:6]
    # One fresh random partner per anchor; no partner replacement based on outputs.
    used_partners = set()
    for anchor in anchors:
        choices = [i for i in others if i not in used_partners and eligible(anchor, i)]
        partner = rng.choice(choices); used_partners.add(partner)
        pairs.append((anchor, partner))
    remaining = [(a, b) for a in anchors for b in others if eligible(a, b) and (a, b) not in pairs]
    rng.shuffle(remaining)
    pairs += remaining[:12-len(pairs)]
    assert len(pairs) == 12 and all(eligible(*p) for p in pairs)
    save('pair_selection.json', dict(seed=110, rule='Up to six unseen anchor-anchor pairs, one distinct random partner per anchor, then seeded anchor-random fill to twelve; exclude recorded local/public coordinate co-occurrences regardless of dose', pairs=[dict(pair_id=f'P{i+1:02d}', components=[selected[a]['unit_id'], selected[b]['unit_id']]) for i, (a,b) in enumerate(pairs)], exclusions_count=len(known)))
    shuffled = list(range(30)); rng.shuffle(shuffled)
    aliases = {i: f'N{k+1:02d}' for k, i in enumerate(shuffled)}
    components = {}
    for i, s in enumerate(selected):
        components[aliases[i]] = dict(s, opaque_id=aliases[i])
    conditions = []
    for i, s in enumerate(selected):
        for direction in [1, -1]:
            cid = aliases[i] + ('_canonical' if direction == 1 else '_opposite')
            conditions.append(dict(id=cid, kind='single', component_ids=[aliases[i]], directions=[direction],
                neurons=[dict(layer=s['layer'], neuron=s['neuron'], published_dose=direction * .5 * s['dose'])],
                early_tokens=None, prefill_scale=1., tail_scale=1.))
    for k, (a,b) in enumerate(pairs):
        conditions.append(dict(id=f'P{k+1:02d}', kind='pair', component_ids=[aliases[a], aliases[b]], directions=[1,1],
            neurons=[dict(layer=selected[i]['layer'], neuron=selected[i]['neuron'], published_dose=.5 * selected[i]['dose']) for i in [a,b]],
            early_tokens=None, prefill_scale=1., tail_scale=1.))
    baseline = dict(id='baseline', kind='baseline', component_ids=[], directions=[], neurons=[], early_tokens=None, prefill_scale=1., tail_scale=1.)
    save('interventions.json', dict(components=components, conditions=[baseline]+conditions))
    generic_job = dict(action='generate', conditions=[{k:c[k] for k in ['id','neurons','early_tokens','prefill_scale','tail_scale']} for c in [baseline]+conditions],
        prompts=[{k:p[k] for k in ['id','messages','max_tokens']} for p in prompts], temperature=0, seed=42,
        repetition_penalty=1.1, repetition_context_size=128, max_tokens=128,
        metadata=dict(experiment='110 catgirl atlas forecast benchmark', expected_rows=584, no_training=True, plan='110_PLAN.json'))
    save('runtime_job.local.json', generic_job)
    vocabulary = json.loads((REPO / 'explorer/cache/vocab.json').read_text())
    lens = np.load(REPO / 'explorer/cache/lens_index.npz', allow_pickle=False)
    def token_text(i):
        return vocabulary[str(int(i))] if isinstance(vocabulary, dict) else vocabulary[int(i)]
    projections = {}
    for opaque, s in components.items():
        projections[opaque] = {pole: [token_text(n) for n in lens[pole][s['layer'],s['neuron']]] for pole in ['up','down']}
    save('projections.json', dict(source='Original shipped direct vocabulary projection, not contextual steering rollouts', source_sha256=sha(REPO/'explorer/cache/lens_index.npz'), vocab_sha256=sha(REPO/'explorer/cache/vocab.json'), values=projections))
    base = dict(model='Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        scope='Add to original gated MLP scalars before down_proj, on every prompt and generated position; weights unchanged. Published residual/column-norm convention; each tested magnitude is HALF its canonical signed reference. Opposite is sign reversal, not a guaranteed opposite semantic effect.',
        features=FEATURES+['joint_catgirl_coherent'], feature_order_fixed=True, families=FAMILIES,
        prompts=prompts, rubric=(ROOT/'110_RUBRIC.md').read_text(),
        forecasting='For every condition, give four probability vectors, one per family, in feature_order. Each rate forecasts a positive on a uniformly chosen one of that family\'s two fixed prompts. Joint=feline selfrole AND female selfrole AND coherent; task relevance separate. Do not infer joint by multiplying marginals. Include baseline. Negative trait predictions and disruption/coherence matter. No outcome training, lookup or actual inference is permitted.',
        controls='Supplied evidence can be incomplete. A published description concerns its canonical signed reference dose; magnitudes tested here are half of reference. No benchmark generations, historic rollouts, numerical success rates or examples are supplied.',
        output_schema={'predictions':[{'condition_id':'all73conditions', 'probabilities':{'each_of_four_families':'8 finite floats in feature_order'}}], 'method_note':'brief reasoning and uncertainty, not a claim of observed results'})
    packets = []
    for rep in range(2):
        perm = list(range(30)); prng = random.Random(1100+rep)
        while True:
            prng.shuffle(perm)
            if all(i != p for i,p in enumerate(perm)): break
        for arm in ['atlas','shuffled','projection','prompts']:
            source_components = {}
            for i in range(30):
                own = selected[i]; opaque = aliases[i]
                info = dict(component_id=opaque, signed_reference_dose=own['dose'])
                if arm in ['atlas','shuffled']:
                    source = selected[perm[i]] if arm == 'shuffled' else own
                    info.update(neuronary_label=source['publisher_label'], neuronary_words=source['explanation'])
                if arm in ['atlas','shuffled','projection']:
                    info['direct_vocabulary_projection'] = projections[opaque]
                source_components[opaque] = info
            compact_conditions = [dict(condition_id=c['id'], kind=c['kind'], component_ids=c['component_ids'],
                                       directions=c['directions'], actual_doses=[n['published_dose'] for n in c['neurons']]) for c in [baseline]+conditions]
            prng.shuffle(compact_conditions)
            packet = dict(base, components=source_components, conditions=compact_conditions,
                          session_tag=f'{arm}_r{rep+1}')
            # Forecasters see this file only; treatment identity not in content.
            packet['session_tag'] = hashlib.sha256(f'110/{arm}/{rep}'.encode()).hexdigest()[:12]
            packet_name = f'packets.local/session_{packet["session_tag"]}.json'
            save(packet_name, packet)
            packets.append(dict(arm=arm, replicate=rep+1, packet=packet_name, sha256=sha(ROOT/packet_name),
                                output=f'forecasts/session_{packet["session_tag"]}.json'))
    save('forecast_sessions.json', packets)
    plan = dict(experiment='Catgirl science: do Neuronary words improve unseen intervention and composition forecasts?',
        model=base['model'], revision=base['revision'], source_selected_sha256=sha(ROOT/'selected30_atlas.json'),
        prompts_sha256=sha(ROOT/'selected_prompts.json'), interventions_sha256=sha(ROOT/'interventions.json'),
        rubric_sha256=sha(ROOT/'110_RUBRIC.md'), projections_sha256=sha(ROOT/'projections.json'),
        sample=dict(random_previously_untested=24, known_cat_related_anchors=6, pairs=12, eligibility='Published single-coordinate descriptions; NOT a random sample of all model neurons'),
        expected_outputs=584, conditions=73, validation_prompts=8, families=FAMILIES, primary_features=FEATURES,
        dose='Half canonical published reference and equal-magnitude reverse for singles; pair canonical half doses match singleton components. Fixed ex ante, no dose tuning or calibration outputs used.',
        timing='All prompt and answer positions, independent conversations. No natural-role claim from this steering scale.',
        four_arms=dict(atlas='Verbatim Neuronary label/excerpt + correct direct vocabulary projection', shuffled='Deranged label/excerpt together; own correct direct projection retained; no fixed points', projection='Correct direct vocabulary projection only', prompts='No neuron-specific semantic evidence'),
        replication='Two fresh no-history sessions per arm, same model/instructions/tool restrictions; conditional on one forecaster model. Counterbalanced session execution order; all eight locked before inference.',
        label_policy='Two independent condition-label-masked reviews of every reply. A third masked adjudicator resolves disagreements and uncertainty before joining conditions/forecasts. Preserve originals and uncertainty sensitivity. No outcome-driven prompt/rubric changes.',
        primary_score='Paired macro Brier, equal weight to neuron then sign/family/prompt and seven features; mean of separate-session losses (not probability ensemble). Report24random and6known anchors separately; primary inference24random.',
        comparisons=['atlas minus shuffled','atlas minus projection','atlas minus prompts'],
        uncertainty='Fixedseed10000paired cluster bootstrap by24randomneuron, preserving signs/prompts/features/replicates; knownanchors separate descriptive. Composition12pairs share sixanchors: descriptive fixed-benchmark effect and anchor-block/leave-anchor-out sensitivity, no12-independent-pair significance claim. Two forecastreplicate stability reported.',
        practical_threshold='Predeclared0.01absolute Brier reduction against eachcontrol is an engineering-interest threshold, not proof of natural cleanliness. Report confidence intervals and all effects regardless of pass. No pvalue/significance claim from correlated replies.',
        secondary=dict(joint='feline AND female AND coherent; rawfeline+female separately; taskusefulness separate', composition='Direct pair forecast versus clip(pA+pB-pBaseline,0,1) singles-only prediction for eacharm/session; not proof of neuron interaction', imbalance='Report allzero/allone forecasts, feature prevalence, positive/negative losses, rare/absent outcomes and per-family scores; do not switch to favorable balanced metric'),
        tweet_gate='Consider image only if a calibrated forecast benefit or counterexample is stable across both forecastreplicates and both initial reviewers/adjudication;0.01Brier practical threshold with domain effects in>=4interventionunits. Composition-only result labeled twelvefixedpairs; no modelwidecleanliness headline. If positivefeatures absent or only fewcases, call pilot inconclusive and do not marketnull as essayrefutation.',
        limitations=['Published descriptions already reflect steering outcomes; this tests transfer to new prompts/pairs, not zero-shot discovery of natural neuron meanings.', 'Single model, English short replies, eight purpose-built prompts, selected public catalogue; no modelwidecleanliness fraction, long-context or multiturn inference.', 'Intervention doses can be outside native scalar ranges, so failure cannot refute natural neuron interpretability.', 'No naturalactivation primary endpoint; sourceatlas is intervention descriptions. No proof of semanticnecessity, monosemanticity or absence of superposition.', 'Freshagents have no conversation history and must not lookup sources; isolation is procedural, not enforced OSsandbox. Same-token budgets and no cross-session reading.', 'All source/condition identity and generated-output masking are auditable; style may still reveal mechanisms to graders.'],
        publication='Push compact protocol, sourcequotes/coordinates, lockedforecasts, exactscoredCSV, analysis andmethod/results; keepbulkyrawsnapshots/fulltranscripts locally withhashes and selectedexamples.')
    save('110_PLAN.json', plan)
    print(json.dumps(dict(conditions=73, expected_outputs=584, selected_prompts=[p['id'] for p in prompts], anchor_anchor_pairs=len(aa), plan_sha256=sha(ROOT/'110_PLAN.json'))))


if __name__ == '__main__':
    build()
