"""Queue the two assistant-position residual screens after capture completes."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
sources=[ROOT/'results'/f'{name}.recipes.json' for name in ['028a_response_matched','028b_response_broad']]
while not all(p.exists() for p in sources):
    time.sleep(2)
conditions=[]
for source in sources:
    data=json.loads(source.read_text())
    for condition in data['conditions']:
        condition=dict(condition,id=source.stem+'_'+condition['id'])
        conditions.append(condition)
system=json.loads((ROOT/'discovery.json').read_text())['neutral_system']
prompts=[{'id':i,'messages':[{'role':'system','content':system},{'role':'user','content':p}]} for i,p in [
    ('identity','Tell me about yourself and what you enjoy.'),
    ('rain','I am bored on a rainy afternoon. Suggest something fun to do.')]]
destination=ROOT/'jobs/031_response_residual_screen.json'
pending=destination.with_suffix('.pending')
pending.write_text(json.dumps({'conditions':conditions,'prompts':prompts,'max_tokens':192},indent=2))
pending.replace(destination)
print(f'Queued {len(conditions)} response-residual conditions',flush=True)
