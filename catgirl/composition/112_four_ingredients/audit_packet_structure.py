"""Independent packet/job structure audit. Never reads forecast probabilities
or validation outcomes. Discovery-selected parameters are inspected only to
verify their identical presentation and scope mapping across arms.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NAMES=('genki','hugs','paws','female')


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mapping=load(ROOT/'forecast_audit_mapping.json')
    sources=load(ROOT/'SOURCES.json')
    doses=load(ROOT/'jobs/SELECTED_DOSES.json')['ingredients']
    natural={c['case_id']:c for c in load(ROOT/'CASES_natural_validation.json')['cases']}
    lexical={c['case_id']:c for c in load(ROOT/'CASES_lexical_validation.json')['cases']}
    generation={c['case_id']:c for c in load(ROOT/'CASES_generation_validation.json')['cases']}
    job=load(ROOT/'jobs/JOB_validation_factorial16.json')
    jobs={c['condition_id']:c for c in job['conditions']}
    old=load(ROOT.parent/'110_atlas/interventions.json')['components']
    projections=load(ROOT.parent/'110_atlas/projections.json')['values']
    projection_by_coord={(v['layer'],v['neuron']):projections[k] for k,v in old.items()}
    common=None;arms={};counts=[]
    for session in mapping['sessions']:
        file=ROOT/'packets.local'/session['packet']
        assert sha(file)==session['packet_sha256']
        packet=load(file);arm=session['arm'];arms.setdefault(arm,set()).add(session['replicate'])
        # Compare all shared information after removing only arm/session fields.
        common_fields={k:v for k,v in packet.items() if k not in ('unit_information','session_id','forecast_output')}
        if common is None:common=common_fields
        else:assert common_fields==common,'Arms differ in shared task/protocol inputs'
        units=mapping['unit_to_ingredient'];original=mapping['case_to_original']
        for unit in packet['units']:
            name=units[unit['unit_id']];selected=doses[name];cal=selected['calibration']
            assert unit['selected_fraction']==selected['selected_fraction'] and unit['direction']==cal['direction']
            for field in ('q05','q50','q95','final_readout_q25','final_readout_q75'):assert unit[field]==cal[field]
        for context in packet['natural_cases']:
            assert set(context)=={'case_id','messages','assistant_readout'}
            case=natural[original[context['case_id']]]
            assert context['messages'][-1]['content']==case['text_prefix'] and context['assistant_readout']==case['readout_text']
        for context in packet['lexical_cases']:
            assert set(context)=={'case_id','messages','assistant_prefix','choices'}
            case=lexical[original[context['case_id']]]
            assert all(context[field]==case[field] for field in ('messages','assistant_prefix','choices'))
        for condition in packet['factorial_conditions']:
            cid=mapping['condition_to_original'][condition['condition_id']]
            actual={units[u] for u in condition['active_units']}
            expected={o['ingredient'] for o in jobs[cid]['operations']}
            assert actual==expected=={name for name,bit in zip(NAMES,cid) if bit=='1'}
        seen_generation=set()
        for family in packet['generation_families']:
            original_family=mapping['family_to_original'][family['family_id']]
            for context in family['cases']:
                cid=original[context['case_id']];case=generation[cid]
                assert context['messages']==case['messages'] and case['family']==original_family
                assert cid not in seen_generation;seen_generation.add(cid)
        assert seen_generation==set(generation)
        for info in packet['unit_information']:
            name=units[info['unit_id']]
            if arm in ('correct_words_projection','shuffled_words_projection'):
                source_name=name if arm=='correct_words_projection' else session['description_permutation'][name]
                if arm=='shuffled_words_projection':assert source_name!=name
                source=sources[source_name]
                assert info['published_label']==source['publisher_label']
                assert info['published_description']==source['source_description_exact']
            else:assert 'published_description' not in info and 'published_label' not in info
            if arm!='no_semantics':
                source=sources[name]
                assert info['direct_vocabulary_projection']==projection_by_coord[source['layer'],source['neuron']]
            else:assert set(info)=={'unit_id'}
        counts.append(dict(arm=arm,replicate=session['replicate'],natural_units=len(packet['units'])*len(packet['natural_cases']),
                           lexical_units=len(packet['units'])*len(packet['lexical_cases']),factorial_family_units=len(packet['factorial_conditions'])*len(packet['generation_families'])))
    assert len(mapping['sessions'])==8 and len(arms)==4 and all(v=={1,2} for v in arms.values())
    assert all(r['natural_units']==192 and r['lexical_units']==48 and r['factorial_family_units']==144 for r in counts)
    # Job scopes/directions must match the forecast's canonical generation task.
    lexical_job=load(ROOT/'jobs/JOB_validation_lexical_generation.json')
    for condition in lexical_job['conditions']:
        for op in condition['operations']:
            if condition['condition_id'].startswith('canonical_'):
                name=op['ingredient'];cal=doses[name]['calibration']
                assert op['scope']=='generation' and op['mode']=='toward' and op['value']==cal['endpoint']
                assert op['fraction']==doses[name]['selected_fraction'] and op['direction']==cal['direction']
    result=dict(study=112,passed=True,sessions=8,arms=4,natural_units_per_session=192,lexical_units_per_session=48,
                factorial_family_units_per_session=144,forecast_probabilities_opened=False,validation_outcomes_opened=False,
                auditor_sha256=sha(__file__),audit_mapping_sha256=sha(ROOT/'forecast_audit_mapping.json'),
                checks=['identical shared inputs','original source words','deranged words without fixed points','own correct projections',
                        'metadata withheld','all coordinates on all contexts','final-readout threshold scope','canonical generation job alignment'])
    target=ROOT/'PACKET_STRUCTURE_AUDIT.json'
    if target.exists():assert target.read_text()==json.dumps(result,indent=2)+'\n'
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
