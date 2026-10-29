"""Build two observation-matched forecast arms and normalize immutable forecasts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
NAMES=['paws','hugs','genki','sounds','pride']
UNITS=['A','B','C','D','E']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def dump(p,x):
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def build(args):
    selected=read(args.selected)
    if set(selected)!=set(NAMES): raise ValueError('Selection keys mismatch')
    spec=read(HERE/'FEATURE_DEFINITIONS.json')
    cases=read(HERE/'CASES_validation.json')['cases']
    sources=read(HERE/'MODIFIER_SOURCES.json')['modifiers']
    job=read(args.validation_job)
    wanted={'base':('base',None)}
    for name,unit in zip(NAMES,UNITS):
        wanted[f'{name}_{selected[name]:g}']=(unit,selected[name])
    raw=[json.loads(s) for s in args.discovery_raw.read_text().splitlines() if s.strip()]
    observations=[]
    seen=set()
    for row in raw:
        if row['condition_id'] not in wanted: continue
        if row.get('split')!='discovery': raise ValueError('Non-discovery observation refused')
        unit,fraction=wanted[row['condition_id']]
        key=unit,row['case_id']
        if key in seen: raise ValueError('Duplicate observation')
        seen.add(key)
        observations.append({'unit':unit,'case_id':row['case_id'],'messages':row['messages'],
                             'text':row['text'],'fraction':fraction})
    if len(observations)!=24 or any(sum(r['unit']==u for r in observations)!=4 for u in ['base']+UNITS):
        raise ValueError('Require base and five selected-single conditions, four discovery prompts each')
    families={}
    for case in cases: families.setdefault(case['family_id'],[]).append({'case_id':case['case_id'],'messages':case['messages']})
    if len(families)!=6 or any(len(v)!=2 for v in families.values()): raise ValueError('Heldout family grid invalid')
    doses=[]
    for name,unit in zip(NAMES,UNITS):
        s=sources[name]
        reference=s.get('published_reference_dose')
        if reference is None: reference=s['project_reference_dose']
        doses.append({'unit':unit,'selected_reference_fraction':selected[name],
                      'signed_layer_normalized_dose':reference*selected[name],
                      'scope':'all input and generation positions'})
    shared={'study_question':'Can effects observed for single native-coordinate interventions predict their combinations on a fixed previously selected catgirl base?',
            'limits':'Known selected neurons, small English prompt set. Do not assume labels are natural functions. Same fixed base remains active in every forecast condition. All discovery observations are supplied; no combined outputs have been observed in this study.',
            'modifier_bit_order':UNITS,'bit_semantics':'Each bit enables that unit at its frozen selected strength; zero disables the modifier, not the fixed base.',
            'fixed_base':{'coordinates':[[9,9009,.7],[26,6173,.3],[26,8930,.7],[22,8295,-.4]],
                          'schedule':'All positions except L22N8295, which is full input prefill and first16 answer-prediction positions only.'},
            'decoding':{k:job.get(k) for k in ('temperature','seed','max_tokens','repetition_penalty','repetition_context_size')},
            'units':doses,'discovery_observations':observations,'heldout_families':families,
            'forecast_states':spec['forecast_states'],'feature_order':spec['forecast_features'],
            'rubric':(HERE/'REVIEW_RUBRIC.md').read_text(),
            'instructions':'Predict the positive fraction for each feature among the two heldout prompts in each family under every listed state. Output every state-family exactly once, seven probabilities in feature_order. There are84 rows and588 probabilities. Do not run inference, browse, inspect any other files, delegate, or infer outcomes from filenames. Use uncertainty honestly. We care about integrated own-voice acts, not vocabulary co-occurrence. Return JSON object with rows:[{state,family,p:[seven probabilities]}], rationale:string. Rationale must be completed before any heldout inference. Do not include any case outcome claims.'}
    descriptions=[]
    for name,unit in zip(NAMES,UNITS):
        s=sources[name]
        descriptions.append({'unit':unit,'label':s['publisher_label'] or 'Tentative project Pride candidate',
                             'description_exact':s['source_description_exact'],'provenance':s['provenance']})
    related=read(HERE/'RELATED_PRIORS.json')
    # Only the source arm receives source claims. Numeric coordinate mapping accompanies those claims.
    source_context={'units':descriptions,'coordinate_mapping':{unit:sources[name]['coordinate'] for name,unit in zip(NAMES,UNITS)},
                    'related_prior_recipes':related}
    args.out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for arm in ('descriptions','observations_only'):
        for rep in (1,2):
            payload=dict(shared)
            payload['session_id']=f'{arm}_{rep}'
            payload['arm']=arm
            if arm=='descriptions': payload['source_information']=source_context
            path=args.out/f'{arm}_{rep}.json';dump(path,payload);paths.append(path)
    inputs=[args.selected,args.discovery_raw,args.validation_job,HERE/'CASES_validation.json',HERE/'FEATURE_DEFINITIONS.json',HERE/'REVIEW_RUBRIC.md',HERE/'MODIFIER_SOURCES.json',HERE/'RELATED_PRIORS.json',Path(__file__)]
    lock={'input_hashes':{str(p):sha(p) for p in inputs},'packet_hashes':{p.name:sha(p) for p in paths},
          'state_count':14,'family_count':6,'features':spec['forecast_features'],'probabilities_per_session':588,
          'mapping':dict(zip(UNITS,NAMES)),'no_heldout_outputs_used':True}
    dump(args.out/'packet_lock.json',lock)
    print(json.dumps({'packets':[p.name for p in paths],'lock_sha256':sha(args.out/'packet_lock.json')}))

def normalize(args):
    packet=read(args.packet); prediction=read(args.prediction)
    features=packet['feature_order']
    expected={(state,family) for state in packet['forecast_states'] for family in packet['heldout_families']}
    seen=set(); rows=[]
    if not isinstance(prediction.get('rationale'),str) or not prediction['rationale'].strip():
        raise ValueError('Rationale required')
    for item in prediction['rows']:
        key=item['state'],item['family']
        if key not in expected or key in seen: raise ValueError('Unexpected/duplicate forecast row')
        seen.add(key)
        if len(item['p'])!=len(features): raise ValueError('Wrong feature count')
        for feature,p in zip(features,item['p']):
            if isinstance(p,bool) or not isinstance(p,(int,float)) or not 0<=p<=1: raise ValueError('Invalid probability')
            rows.append({'session_id':packet['session_id'],'arm':packet['arm'],'family_id':item['family'],
                         'condition_id':item['state'],'feature':feature,'probability':p})
    if seen!=expected: raise ValueError(f'Incomplete predictions:{len(seen)}of{len(expected)}')
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(args.out.with_suffix('.lock.json'),{'packet_sha256':sha(args.packet),'prediction_sha256':sha(args.prediction),
                                          'normalized_sha256':sha(args.out),'rows':len(rows)})
    print(json.dumps({'rows':len(rows),'sha256':sha(args.out)}))

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build'); b.add_argument('--selected',type=Path,required=True); b.add_argument('--discovery-raw',type=Path,required=True)
    b.add_argument('--validation-job',type=Path,required=True);b.add_argument('--out',type=Path,required=True)
    n=sub.add_parser('normalize');n.add_argument('--packet',type=Path,required=True);n.add_argument('--prediction',type=Path,required=True);n.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); {'build':build,'normalize':normalize}[args.action](args)

if __name__=='__main__': main()
