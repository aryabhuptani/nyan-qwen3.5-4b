"""CPU-only reproduction of published selection from complete scored CSVs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
decision=json.loads((ROOT/'dose_selection.json').read_text())
summaries=json.loads((ROOT/'calibration_review_sensitivity.json').read_text())
features=json.loads((ROOT/'111_CALIBRATION_PLAN.json').read_text())['rating_order']
for review in summaries['reviews']:
    path=ROOT/f"calibration_{review['review']}_observations.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==review['observations_sha256']
    rows=list(csv.DictReader(path.open()))
    assert len(rows)==len({r['review_id'] for r in rows})==120
    for saved in review['counts']:
        selected=[r for r in rows if float(r['multiplier'])==saved['multiplier']]
        assert len(selected)==40
        coherent=sum(r['coherent']=='1' for r in selected)
        units={r['unit_id'] for r in selected if any(r[f]=='1' for f in features[:5])}
        types={f for r in selected for f in features[:5] if r[f]=='1'}
        assert coherent==saved['coherent'] and sorted(units)==saved['persona_units'] and sorted(types)==saved['persona_types']
        assert sum(r['hard_loop']=='True' for r in selected)==saved['hard_loops']
        assert sum(r['task_relevant']=='1' for r in selected)==saved['task_relevant']
        assert saved['would_meet_frozen_gate']==(coherent>=36 and len(units)>=4 and len(types)>=2)
        if review['review']=='primary':
            original=next(x for x in decision['selection_counts'] if x['multiplier']==saved['multiplier'])
            assert original['coherent_replies']==coherent and original['qualifying_units']==len(units)
            assert original['primary_gate']==saved['would_meet_frozen_gate']
    print(review['review'],'120 replies, dose/trait/coherence/loop/task counts verified')
assert decision['rule_status']=='stop_no_coherent_dose' and decision['selected_multiplier'] is None and decision['confirmatory_allowed'] is False
assert all(not x['coherence_eligible'] for x in decision['selection_counts'])
print('Frozen primary stop reproduced; sensitivity does not replace selection.')
