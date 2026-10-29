"""CPU-only Study115 metadata audit. Never displays or evaluates generated text/messages."""
import csv
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMPOSITION = HERE.parent
CATGIRL = COMPOSITION.parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def finite(v):
    if isinstance(v, dict):
        return all(finite(x) for x in v.values())
    if isinstance(v, list):
        return all(finite(x) for x in v)
    return not isinstance(v, float) or math.isfinite(v)

def bf16(v):
    # Float32 conversion then round-to-nearest, ties-to-even, independent of MLX.
    bits = struct.unpack('I', struct.pack('f', v))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('f', struct.pack('I', bits))[0]

def main():
    job = load(HERE / 'JOB_discovery.json')
    cases = load(HERE / 'CASES_discovery.json')['cases']
    conditions = {c['condition_id']: c for c in job['conditions']}
    require(len(conditions)==len(job['conditions'])==20, '20 unique dose states')
    require({(c['normalized_doses']['paws'],c['normalized_doses']['genki']) for c in conditions.values()} == {(p,g) for p in (0,.1,.2,.4) for g in (0,.1,.2,.4,.8)}, 'complete requested dose grid')
    require(all(c['base'] is True for c in conditions.values()), 'original base in every state')
    expected_cases = {c['case_id']: {k: c[k] for k in ('family_id','split')} for c in cases}
    norms_path = CATGIRL / 'public_qwen35_norms.json'
    sources_path = COMPOSITION / '114_paws_genki/SOURCES.json'
    runner_path = COMPOSITION / '113_catgirl_circuits/run_circuits.py'
    hashes = dict(job_sha256=sha(HERE/'JOB_discovery.json'), cases_sha256=sha(HERE/'CASES_discovery.json'),
                  sources_sha256=sha(sources_path), norms_sha256=sha(norms_path), runner_sha256=sha(runner_path))
    require(hashes['cases_sha256'] == job['cases_sha256'], 'cases job hash')
    require(hashes['sources_sha256'] == job['sources_sha256'], 'sources job hash')
    for freeze_name, key in [('FROZEN_DISCOVERY.json','input_sha256'),('ANALYSIS_FREEZE.json','sha256')]:
        for name, expected in load(HERE/freeze_name)[key].items():
            require(sha(HERE/name) == expected, 'freeze hash '+name)
    replication_paths = dict(wrapper_sha256=COMPOSITION/'114_paws_genki/run_replication.py',
                            helper_sha256=COMPOSITION/'114_paws_genki/build_job.py', original_runner_sha256=runner_path,
                            control_selection_sha256=COMPOSITION/'114_paws_genki/CONTROL_SELECTION.json',
                            original_sources_sha256=COMPOSITION/'113_catgirl_circuits/MODIFIER_SOURCES.json')
    for key, path in replication_paths.items():
        require(sha(path) == job['replication_frozen_files'][key], 'replication hash '+key)
    require(sha(HERE/'CASES_confirmation.json') == load(HERE/'CONFIRMATION_CASE_LOCK.json')['case_sha256'], 'confirmation lock')
    require(job['temperature']==0 and job['seed']==42 and job['max_tokens']==192 and job['repetition_penalty']==1.1 and job['repetition_context_size']==128, 'decoding')
    require(job['model_revision']=='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'revision')
    medians = load(norms_path)['medians']
    base = {(9,9009):(.7,'all'), (26,6173):(.3,'all'), (26,8930):(.7,'all'), (22,8295):(-.4,'prefill_and_first16')}
    source_specs = load(sources_path)['modifiers']
    for name, coordinate in [('paws',(23,4769)),('genki',(13,4810))]:
        source = source_specs[name]
        require((source['coordinate']['layer'],source['coordinate']['neuron']) == coordinate and source['published_reference_dose']==.8, 'source dose/coordinate')
    allowed = ('condition_id','case_id','family_id','split','effective_interventions','intervention_trace_summary','input_hashes','max_tokens','prompt_tokens','generated_tokens','generation_tps','prefill_tps','finish_reason','seconds')
    rows = []
    for line in (HERE/'results.local/discovery.jsonl').open():
        # Only the allowlisted metadata survives deserialization; text/messages are never inspected.
        raw = json.loads(line)
        rows.append({k: raw[k] for k in allowed})
        del raw
    expected = {(c,k) for c in conditions for k in expected_cases}
    require(len(rows)==60 and Counter((r['condition_id'],r['case_id']) for r in rows)==Counter(expected), '60 unique condition/case rows')
    counts = Counter()
    grouped = defaultdict(list)
    for r in rows:
        require(finite(r), 'nonfinite metadata')
        require(r['input_hashes']==hashes, 'row input hashes')
        require({k:r[k] for k in ('family_id','split')}==expected_cases[r['case_id']], 'case metadata')
        require(r['max_tokens']==192 and 0<r['generated_tokens']<=192 and r['prompt_tokens']>0, 'token counts')
        condition = conditions[r['condition_id']]
        expected_recipe = dict(base)
        for name, coord in [('paws',(23,4769)),('genki',(13,4810))]:
            dose = condition['normalized_doses'][name]
            require(math.isclose(condition['modifiers'].get(name,0)*.8,dose,abs_tol=1e-15), 'fraction-to-dose')
            if dose:
                expected_recipe[coord] = (dose,'all')
        specs = {(s['layer'],s['neuron']):s for s in r['effective_interventions']}
        traces = {(s['layer'],s['neuron']):s for s in r['intervention_trace_summary']}
        require(len(specs)==len(r['effective_interventions']) and set(specs)==set(traces)==set(expected_recipe), 'coordinate coverage')
        for coord, (dose,scope) in expected_recipe.items():
            s,t = specs[coord],traces[coord]
            require(s['scope']==t['scope']==scope and math.isclose(s['reference_normalized_dose'],dose,abs_tol=1e-15), 'dose/scope')
            require(s['residual_norm']==medians[coord[0]] and s['column_l2']>0, 'norm factors')
            request = dose*s['residual_norm']/s['column_l2']
            require(math.isclose(s['amount_float32'],request,rel_tol=1e-12), 'raw scalar conversion')
            require(s['amount_bfloat16']==bf16(request), 'BF16 scalar rounding')
            positions = r['prompt_tokens']+r['generated_tokens']
            active = positions if scope=='all' else min(positions,r['prompt_tokens']-1+16)
            require(t['positions']==positions and t['active_positions']==active, 'trace position scope')
            require(t['inactive_nonzero_positions']==0 and t['unchanged_active_positions']==0, 'inactive/active delta integrity')
            require(t['actual_delta_min']<=t['actual_active_delta_mean']<=t['actual_delta_max'] and t['actual_delta_rms']>0, 'delta statistics')
            require(t['actual_active_delta_mean']*dose>0, 'delta sign')
            mean_norm=t['actual_active_delta_mean']*s['column_l2']/s['residual_norm']
            counts['coordinate_traces']+=1
            grouped[(coord[0],coord[1],dose,scope)].append(dict(requested_raw=request,bf16_scalar=s['amount_bfloat16'],scalar_normalized=s['amount_bfloat16']*s['column_l2']/s['residual_norm'],realized_normalized=mean_norm,realized_raw=t['actual_active_delta_mean'],relative_error=(mean_norm-dose)/abs(dose),delta_min=t['actual_delta_min'],delta_max=t['actual_delta_max']))
    dose_rows=[]
    for (layer,neuron,dose,scope), values in sorted(grouped.items()):
        d=dict(layer=layer,neuron=neuron,requested_normalized_dose=dose,scope=scope,traces=len(values))
        for key in values[0]:
            d[key+'_min']=min(x[key] for x in values)
            d[key+'_max']=max(x[key] for x in values)
        dose_rows.append(d)
    out=HERE/'audit'
    with (out/'metadata_dose_audit.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dose_rows[0]));writer.writeheader();writer.writerows(dose_rows)
    report=dict(study=115,status='passed',audited_rows=len(rows),conditions=len(conditions),cases=len(cases),coordinate_traces=counts['coordinate_traces'],
                all_frozen_input_hashes_verified=True,all_metadata_finite=True,all_effective_recipes_match=True,all_trace_scopes_match=True,
                inactive_changed_positions=0,unchanged_active_positions=0,input_hashes=hashes,
                raw_discovery_sha256=sha(HERE/'results.local/discovery.jsonl'),script_sha256=sha(Path(__file__)),
                limitations=['Metadata-only audit: generated text and messages were not inspected.','No model weights were loaded; recorded column norms and traces were checked internally, not independently recaptured.','Job and source hashes bind the declared checkpoint; metadata alone does not authenticate loaded checkpoint weight bytes.','BF16 scalar rounding and activation addition produce realized deltas that differ from requested doses.','Confirmation cases were locked after discovery file creation; the disclosed chronology deviation remains.'])
    (out/'metadata_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
