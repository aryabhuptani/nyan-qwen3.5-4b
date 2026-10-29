"""Build isolated forecast packets; validate, seal and export complete forecasts.

Inputs are explicit frozen case/source/job files. Packet metadata whitelists only
quantiles/directions/selected fractions from discovery, never outcome curves.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import uuid
ROOT=Path(__file__).resolve().parent
ARMS=['correct_words_projection','shuffled_words_projection','projection_only','no_semantics']
NAMES=['genki','hugs','paws','female']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,obj):
 p=Path(p);text=json.dumps(obj,indent=2,ensure_ascii=False)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==text,f'Refusing to overwrite {p}'
 else:p.write_text(text)
def opaque(kind,text):return kind+hashlib.sha256(('112/'+text).encode()).hexdigest()[:12]
def build():
 out=ROOT/'packets.local';assert not (out/'sessions.json').exists(),'Packets already built'
 rng=random.Random(112);shuffled=NAMES.copy();rng.shuffle(shuffled)
 units={n:opaque('U',str(i)) for i,n in enumerate(shuffled)}
 sources=load(ROOT/'SOURCES.json');selected=load(ROOT/'jobs/SELECTED_DOSES.json')['ingredients']
 old=load(ROOT.parent/'110_atlas/interventions.json')['components'];projections=load(ROOT.parent/'110_atlas/projections.json')
 projection_by_coordinate={(r['layer'],r['neuron']):projections['values'][k] for k,r in old.items()}
 natural=load(ROOT/'CASES_natural_validation.json')['cases'];lexical=load(ROOT/'CASES_lexical_validation.json')['cases'];generation=load(ROOT/'CASES_generation_validation.json')['cases']
 job=load(ROOT/'jobs/JOB_validation_factorial16.json');features=load(ROOT/'FEATURE_DEFINITIONS.json')
 case_ids={c['case_id']:opaque('C',c['case_id']) for c in natural+lexical+generation}
 families={f:opaque('F',f) for f in sorted({c['family'] for c in generation})}
 condition_ids={c['condition_id']:opaque('T',c['condition_id']) for c in job['conditions']}
 unit_records=[]
 for name in shuffled:
  params=selected[name];cal=params['calibration']
  unit_records.append(dict(unit_id=units[name],direction=cal['direction'],selected_fraction=params['selected_fraction'],
   q05=cal['q05'],q50=cal['q50'],q95=cal['q95'],final_readout_q25=cal['final_readout_q25'],final_readout_q75=cal['final_readout_q75']))
 common=dict(study=112,model='Qwen/Qwen3.5-4B',revision=job['revision'],units=unit_records,
  intervention='Native post-gating MLP scalar before down projection. At every assistant prediction position, move fraction of distance toward q95 for positive direction orq05 for negative direction only when current activation is less canonical. Never move already-more-canonical values backwards. No prompt steering, no weight training. Natural threshold task has no intervention.',
  quantile_note='q05/q50/q95 from discovery neutral readout-text positions; final_readout_q25/q75 from final discovery readout token. These are parameters, not evidence that the published effect transferred. Selected fraction was fixed by a prespecified discovery-only procedure; no discovery outputs/pass flags are supplied.',
  decoding={k:job[k] for k in ['seed','temperature','max_tokens','repetition_penalty']},
  source_caution='Any supplied descriptions are exact prior published steering reports, not new study112 outputs or proofs of natural meaning. In particular a quoted example inside a source description is prior publisher text. Information may be incomplete or inaccurate.',
  natural_cases=[dict(case_id=case_ids[c['case_id']],messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':c['text_prefix']}],assistant_readout=c['readout_text']) for c in natural],
  lexical_cases=[dict(case_id=case_ids[c['case_id']],messages=c['messages'],assistant_prefix=c['assistant_prefix'],choices=c['choices']) for c in lexical],
  factorial_conditions=[dict(condition_id=condition_ids[c['condition_id']],active_units=[units[o['ingredient']] for o in c['operations']]) for c in job['conditions']],
  generation_families=[dict(family_id=families[f],cases=[dict(case_id=case_ids[c['case_id']],messages=c['messages']) for c in generation if c['family']==f]) for f in families],
  feature_definitions=features,behavior_rubric=(ROOT/'BEHAVIOR_RUBRIC.md').read_text(),
  output_instructions='Predict every grid cell without generating model replies. Natural: probability that canonical signed FINAL neutral-readout activation crosses its frozen q75 positive/q25 negative threshold for eachunit×eachnaturalcase. Lexical: selected canonical generation intervention minus baseline change in mean-token-logp(target)-mean-token-logp(control), for EVERYunit×EVERYwritingcase, plus probability change>0; not semantic accuracy. Factorial: feature probabilities percondition×family, one common probability applied to each constituent case. Quality endpoints remain separate. All prediction cells are required, including controls/near misses. Use only this packet, no other files/web/model inference/messages.',
  output_schema={'natural':[{'unit_id':'U...','case_id':'C...','probability':'float[0,1]'}], 'lexical':[{'unit_id':'U...','case_id':'C...','predicted_change':'finitefloat[-10,10]nat/token','p_positive':'float[0,1]'}], 'factorial':[{'condition_id':'T...','family_id':'F...','probabilities':'12 floats in feature_definitions.forecast_feature_order'}]})
 sessions=[]
 order=[(arm,rep) for rep in [1,2] for arm in ARMS];rng.shuffle(order)
 for arm,rep in order:
  token=uuid.UUID(int=rng.getrandbits(128)).hex;packet=copy.deepcopy(common)
  perm=NAMES.copy()
  if arm=='shuffled_words_projection':
   while True:
    rng.shuffle(perm)
    if all(a!=b for a,b in zip(NAMES,perm)):break
  word_mapping=dict(zip(NAMES,perm));semantics=[]
  for name in shuffled:
   entry={'unit_id':units[name]}
   if arm in ARMS[:2]:
    src=sources[word_mapping[name] if arm==ARMS[1] else name]
    entry['published_label']=src['publisher_label'];entry['published_description']=src['source_description_exact']
   if arm!=ARMS[3]:entry['direct_vocabulary_projection']=projection_by_coordinate[(sources[name]['layer'],sources[name]['neuron'])]
   semantics.append(entry)
  packet['unit_information']=semantics
  packet['session_id']=token;packet['forecast_output']=str(ROOT/'forecasts'/f'session_{token}.json')
  path=out/f'session_{token}.json';save(path,packet)
  sessions.append(dict(session_id=token,arm=arm,replicate=rep,packet=path.name,packet_sha256=sha(path),forecast=f'session_{token}.json',description_permutation=word_mapping if arm==ARMS[1] else None))
 mapping=dict(unit_to_ingredient={v:k for k,v in units.items()},case_to_original={v:k for k,v in case_ids.items()},family_to_original={v:k for k,v in families.items()},condition_to_original={v:k for k,v in condition_ids.items()})
 save(out/'mapping.local.json',mapping)
 inputs=['112_PLAN.json','SOURCES.json','FEATURE_DEFINITIONS.json','BEHAVIOR_RUBRIC.md','jobs/SELECTED_DOSES.json'] + [p.name for p in sorted(ROOT.glob('CASES_*.json'))] + [str(p.relative_to(ROOT)) for p in sorted((ROOT/'jobs').glob('JOB_validation_*.json'))]
 save(out/'sessions.json',dict(sessions=sessions,input_sha256={p:sha(ROOT/p) for p in inputs},projection_sha256=sha(ROOT.parent/'110_atlas/projections.json'),mapping_sha256=sha(out/'mapping.local.json')))
 # Audit by construction: no original IDs or case-ingredient/class labels exist in packets.
 for session in sessions:
  packet=load(out/session['packet'])
  for c in packet['natural_cases']+packet['lexical_cases']:
   assert set(c)<= {'case_id','messages','assistant_readout','assistant_prefix','choices'}
  assert all('ingredient' not in c and 'semantic_present' not in c for c in packet['natural_cases'])
 print(json.dumps({'sessions':len(sessions),'natural_cells':len(units)*len(natural),'lexical_cells':len(units)*len(lexical),'factorial_cells':len(job['conditions'])*len(families)*12}))

def probability(x):assert type(x) in (int,float) and math.isfinite(x) and 0<=x<=1

def validate(session):
 base=ROOT/'packets.local';path=base/session['packet'];assert sha(path)==session['packet_sha256'];p=load(path)
 fpath=ROOT/'forecasts'/session['forecast'];forecast=load(fpath)
 expected_n={(u['unit_id'],c['case_id']) for u in p['units'] for c in p['natural_cases']}
 expected_l={(u['unit_id'],c['case_id']) for u in p['units'] for c in p['lexical_cases']}
 expected_f={(c['condition_id'],f['family_id']) for c in p['factorial_conditions'] for f in p['generation_families']}
 for task,expected,fields in [('natural',expected_n,('unit_id','case_id')),('lexical',expected_l,('unit_id','case_id')),('factorial',expected_f,('condition_id','family_id'))]:
  seen=set()
  for r in forecast[task]:
   key=tuple(r[k] for k in fields);assert key in expected and key not in seen,(task,key);seen.add(key)
   if task=='natural':probability(r['probability'])
   elif task=='lexical':
    probability(r['p_positive']);x=r['predicted_change'];assert type(x) in (int,float) and math.isfinite(x) and -10<=x<=10
   else:
    assert len(r['probabilities'])==12
    for x in r['probabilities']:probability(x)
  assert seen==expected,(task,'incomplete',len(seen),len(expected))
 return forecast

def seal():
 base=ROOT/'packets.local';manifest=load(base/'sessions.json');mapping=load(base/'mapping.local.json')
 # Initial sealing must precede all validation output; discovery JSONL is allowed.
 if not (ROOT/'forecast_lock.json').exists():
  allowed={
   'discovery_capture/natural.jsonl':'CASES_capture_discovery.json',
   'control_capture/natural.jsonl':'CASES_capture_discovery.json',
   'discovery_writing/probe.jsonl':'CASES_lexical_discovery.json',
   'discovery_generation_v1/generation.jsonl':'CASES_generation_discovery.json'}
  unexpected=[]
  for path in (ROOT/'results.local').rglob('*.jsonl'):
   if not path.stat().st_size:continue
   key=str(path.relative_to(ROOT/'results.local'))
   if key not in allowed:
    unexpected.append(path);continue
   expected={c['case_id'] for c in load(ROOT/allowed[key])['cases']}
   for line in path.read_text().splitlines():
    row=json.loads(line)
    assert row['split']=='discovery' and row['case_id'] in expected, 'Non-discovery case before sealing'
  assert not unexpected, f'Non-discovery outputs already exist: {unexpected}'
 assert sha(base/'mapping.local.json')==manifest['mapping_sha256']
 assert sha(ROOT.parent/'110_atlas/projections.json')==manifest['projection_sha256']
 for path,h in manifest['input_sha256'].items():assert sha(ROOT/path)==h, f'Frozen input changed:{path}'
 assert len(manifest['sessions'])==8
 rows=[];hashes={}
 for session in manifest['sessions']:
  forecast=validate(session);hashes[session['forecast']]=sha(ROOT/'forecasts'/session['forecast'])
  shared=dict(arm=session['arm'],replicate=session['replicate'])
  for r in forecast['natural']:
   rows.append(dict(**shared,task='natural',ingredient=mapping['unit_to_ingredient'][r['unit_id']],case_id=mapping['case_to_original'][r['case_id']],probability=r['probability']))
  for r in forecast['lexical']:
   rows.append(dict(**shared,task='lexical',ingredient=mapping['unit_to_ingredient'][r['unit_id']],case_id=mapping['case_to_original'][r['case_id']],probability=r['p_positive'],predicted_change=r['predicted_change']))
  packet=load(base/session['packet']);features=packet['feature_definitions']['forecast_feature_order']
  for r in forecast['factorial']:
   for endpoint,value in zip(features,r['probabilities']):
    rows.append(dict(**shared,task='factorial',condition_id=mapping['condition_to_original'][r['condition_id']],family=mapping['family_to_original'][r['family_id']],endpoint=endpoint,probability=value))
 csvpath=ROOT/'forecasts_normalized.csv';buffer=io.StringIO(newline='');fields=['arm','replicate','task','ingredient','case_id','condition_id','family','endpoint','probability','predicted_change']
 w=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 if csvpath.exists():assert csvpath.read_text()==buffer.getvalue()
 else:csvpath.write_text(buffer.getvalue())
 save(ROOT/'forecast_lock.json',dict(study=112,sessions_sha256=sha(base/'sessions.json'),forecast_sha256=hashes,normalized_sha256=sha(csvpath),rows=len(rows),forecast_cells_per_session=1968,input_sha256=manifest['input_sha256'],projection_sha256=manifest['projection_sha256']))
 # Public audit mapping released only after every forecast has validated and sealed.
 save(ROOT/'forecast_audit_mapping.json',dict(**mapping,sessions=manifest['sessions']))
 print(json.dumps(dict(sealed=True,sessions=8,cells=len(rows),lock_sha256=sha(ROOT/'forecast_lock.json'))))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['build','validate','seal']);p.add_argument('--session');args=p.parse_args()
 if args.command=='build':build()
 elif args.command=='seal':seal()
 else:
  sessions=load(ROOT/'packets.local/sessions.json')['sessions'];selected=[s for s in sessions if not args.session or s['session_id']==args.session]
  assert selected
  for s in selected:validate(s)
  print(json.dumps(dict(validated=len(selected))))
if __name__=='__main__':main()
