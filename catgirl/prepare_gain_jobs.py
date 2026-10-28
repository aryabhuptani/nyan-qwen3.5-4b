"""Queue a frozen discovery screen when the current gain optimization finishes."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent

def prepare(source,destination):
    data=json.loads(source.read_text())
    system=json.loads((ROOT/'discovery.json').read_text())['neutral_system']
    conditions=[]
    for c in data['conditions']:
        size=len(c['neurons'])
        for strength in ([.5,1,1.5,2] if size==512 else [.5,1,2]):
            neurons=[{'layer':n['layer'],'neuron':n['neuron'],'mode':'mul',
                      'factor':max(0,min(8,1+strength*(n['factor']-1)))} for n in c['neurons']]
            conditions.append({'id':c['id']+f'_x{strength}','neurons':neurons})
    prompts=[]
    for identifier,text in [
        ('identity','Tell me about yourself and what you enjoy.'),
        ('rain','I am bored on a rainy afternoon. Suggest something fun to do.'),
        ('arithmetic','I have 15 apples and give away 3. How many apples are left? Explain briefly.'),
        ('technical','Explain what a computer cache is in three sentences.')]:
        prompts.append({'id':identifier,'messages':[{'role':'system','content':system},{'role':'user','content':text}]})
    job={'conditions':[{'id':'neutral_system_baseline','neurons':[]}]+conditions,'prompts':prompts,'max_tokens':192}
    # Rename atomically so the GPU worker never reads a partial JSON file.
    pending=destination.with_suffix('.pending')
    pending.write_text(json.dumps(job,indent=2))
    pending.replace(destination)
    print(f'Queued {len(conditions)} gain conditions at {destination}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'results/020_gain_optimization.recipes.json')
    parser.add_argument('--output',type=Path,default=ROOT/'jobs/021_gain_screen.json')
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    while args.wait and not args.source.exists():
        time.sleep(2)
    prepare(args.source,args.output)
