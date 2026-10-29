"""Serial laptop-GPU queue. Requires all forecasts and frozen inputs to match."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
TASKS=[
 ('factorial','generation','generation','JOB_validation_factorial16.json',False),
 ('strong_singletons','generation','generation','JOB_validation_strong_singletons.json',False),
 ('random_four','generation','generation','JOB_validation_random_four.json',True),
 ('natural','capture','natural',None,False),
 ('natural_random','capture','natural',None,True),
 ('lexical_generation','lexical','probe','JOB_validation_lexical_generation.json',False),
 ('lexical_readout','lexical','probe','JOB_validation_lexical_readout.json',False),
 ('semantic_carrier','natural','probe','JOB_validation_semantic_carrier.json',False),
 ('semantic_readout','natural','probe','JOB_validation_semantic_readout.json',False),
 ('lexical_generation_random','lexical','probe','JOB_validation_lexical_generation_random.json',True),
 ('lexical_readout_random','lexical','probe','JOB_validation_lexical_readout_random.json',True),
 ('semantic_carrier_random','natural','probe','JOB_validation_semantic_carrier_random.json',True),
 ('semantic_readout_random','natural','probe','JOB_validation_semantic_readout_random.json',True)]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 lock=json.loads((ROOT/'forecast_lock.json').read_text())
 for name,h in lock['input_sha256'].items():assert sha(ROOT/name)==h,name
 for name,h in lock['forecast_sha256'].items():assert sha(ROOT/'forecasts'/name)==h,name
 assert sha(ROOT/'forecasts_normalized.csv')==lock['normalized_sha256']
 plan=json.loads((ROOT/'112_PLAN.json').read_text())
 for name,h in plan['input_sha256'].items():assert sha(ROOT/name)==h,name
 return lock

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--model');p.add_argument('--output-root',type=Path,default=ROOT/'results.local');args=p.parse_args()
 verify();print(json.dumps(dict(event='frozen_inputs_verified',tasks=len(TASKS))),flush=True)
 if not args.run:return
 args.output_root.mkdir(parents=True,exist_ok=True)
 start=time.monotonic()
 for name,corpus,phase,job,random in TASKS:
  verify()
  output=args.output_root/('validation_'+name)
  completion=output/(phase+'_completion.json')
  if completion.exists():
   d=json.loads(completion.read_text());assert sha(output/(phase+'.jsonl'))==d['raw_sha256']
   print(json.dumps(dict(event='already_complete',task=name,rows=d['rows'])),flush=True);continue
  command=[sys.executable,'-u',str(ROOT/'measure.py'),'--cases',str(ROOT/f'CASES_{corpus}_validation.json'),
           '--output',str(output),'--phase',phase,'--run']
  if job:command+=['--job',str(ROOT/'jobs'/job)]
  if random:command+=['--coordinates',str(ROOT/'CONTROL_COORDINATES.json')]
  if args.model:command+=['--model',args.model]
  log=args.output_root/f'validation_{name}.log'
  print(json.dumps(dict(event='starting',task=name,log=str(log))),flush=True)
  with log.open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
  print(json.dumps(dict(event='completed',task=name)),flush=True)
 command=[sys.executable,'-u',str(ROOT/'precision_audit.py'),'--output',str(args.output_root/'precision_audit'),
          '--bf16-results',str(args.output_root/'validation_lexical_generation/probe.jsonl'),
          '--forecast-lock',str(ROOT/'forecast_lock.json'),'--run']
 if args.model:command+=['--model',args.model]
 marker=args.output_root/'precision_queue_complete.local.json'
 if not marker.exists():
  with (args.output_root/'precision_audit.log').open('a') as f:
   subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
  marker.write_text(json.dumps(dict(finished=True))+'\n')
 print(json.dumps(dict(event='all_validation_complete',seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
