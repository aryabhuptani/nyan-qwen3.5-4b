"""Build an exact-numeric, CPU-replay archive from the locked primary manifest.

From study112:
 python3 build_measurement_archive.py --manifest analysis_inputs.local.json --output data/measurements.zip
The explicit collector includes only measurement data, locks, source definitions,
phase metadata and reproduction code. No weights, logs or agent transcripts.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
from public_export import safe_metadata,ensure_no_private_text
ROOT=Path(__file__).resolve().parent
PATH_KEYS={'raw','cases','metadata','selection','calibration','conditions','forecasts_csv','labels_lock','labels_csv'}

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def manifest_paths(obj,key=None):
 if isinstance(obj,dict):
  for k,v in obj.items():yield from manifest_paths(v,k)
 elif isinstance(obj,list):
  for v in obj:yield from manifest_paths(v,key)
 elif isinstance(obj,str) and (key in PATH_KEYS or key in {'final','reviewer_A','reviewer_B','uncertain_zero','uncertain_one'}):
  yield obj

def numbers(obj):
 if isinstance(obj,dict):return [n for v in obj.values() for n in numbers(v)]
 if isinstance(obj,list):return [n for v in obj for n in numbers(v)]
 return [obj] if type(obj) in (float,int,bool) else []

def check_bytes(path,data):
 text=data.decode('utf-8')
 if path.endswith('.jsonl'):
  for line in text.splitlines():
   if line.strip():ensure_no_private_text(json.loads(line))
 elif path.endswith('.json'):ensure_no_private_text(json.loads(text))
 elif path.endswith('.csv'):ensure_no_private_text(list(csv.DictReader(io.StringIO(text))))
 else:ensure_no_private_text(text)

def build(args):
 base=args.manifest.resolve().parent;manifest=load(args.manifest)
 assert manifest.get('labels_lock'),'A locked primary manifest is required'
 lock=load(base/manifest['labels_lock'])
 for name,h in lock['labels_sha256'].items():
  assert sha_bytes((base/manifest['labels_lock']).with_name(name).read_bytes())==h,'A label changed after locking'
 collected={}
 def add(relative,role='scientific',required=True):
  p=base/relative
  if not p.exists():
   assert not required,f'Missing required archive input:{relative}'
   return
  assert p.is_file() and not Path(relative).is_absolute() and '..' not in Path(relative).parts
  if relative in collected:
   if role=='metadata':collected[relative]='metadata'
  else:collected[relative]=role
 for relative in manifest_paths(manifest):add(relative)
 # Every phase containing raw measurements gets its exact executed metadata.
 for relative in list(collected):
  p=Path(relative)
  if p.suffix=='.jsonl':
   phase={'natural':'natural','probe':'probe','generation':'generation'}.get(p.stem)
   if phase:add(str(p.with_name(phase+'_metadata.json')),'metadata')
 # Independently checkable discovery range/dose selection, including controls.
 for folder in ['discovery_capture','control_capture']:
  for name in ['natural.jsonl','natural_metadata.json','calibration.json']:
   add(f'results.local/{folder}/{name}','metadata' if name.endswith('_metadata.json') else 'scientific')
 for name in ['probe.jsonl','probe_metadata.json']:
  add('results.local/discovery_writing/'+name,'metadata' if name.endswith('_metadata.json') else 'scientific')
 for name in ['float32.jsonl','comparison.csv','metadata.json','summary.json','summary_complete_set.json']:
  add('results.local/precision_audit/'+name,'metadata' if name.endswith('.json') else 'scientific',required=name!='summary_complete_set.json')
 # Root's natural association control is separate from the primary manifest.
 for name in ['natural.jsonl','natural_metadata.json']:
  add('results.local/validation_natural_random/'+name,'metadata' if name.endswith('_metadata.json') else 'scientific')
 for name in ['analyze.py','build_jobs.py','CONTROL_COORDINATES.json',
              '112_PLAN.json','FEATURE_DEFINITIONS.json','BEHAVIOR_RUBRIC.md','ANALYSIS_INTERFACE.md','PRECISION_PLAN.md',
              'analysis_conditions.json','jobs/JOB_discovery_writing.json','jobs/DISCOVERY_CALIBRATION.json',
              'jobs/SELECTED_DOSES.json','jobs/MATCHED_CONTROLS.json']:
  add(name)
 for p in sorted(base.glob('CASES_*.json')):add(p.name)
 for name in ['forecast_lock.json','forecast_audit_mapping.json']:add(name)
 for p in sorted((base/'jobs').glob('JOB_validation_*.json')):add(str(p.relative_to(base)))
 labels_dir=Path(manifest['labels_lock']).parent
 add(str(labels_dir/'label_evidence.json'))
 evidence=base/labels_dir/'label_evidence.json';assert sha_bytes(evidence.read_bytes())==lock['evidence_sha256']
 # Record the public manifest under its exact relative archive path.
 reproduction=dict(manifest)
 reproduction['reproduction_note']='Extract archive into this study112 directory. All analysis inputs remain relative; numeric and verbatim scientific inputs are unchanged. Host paths are sanitized only in phase metadata. Persistence is exported separately.'
 repro_bytes=(json.dumps(reproduction,indent=2,ensure_ascii=False)+'\n').encode()
 ensure_no_private_text(reproduction)
 payload={'analysis_inputs.reproduction.json':repro_bytes};entries=[]
 for relative,role in sorted(collected.items()):
  original=(base/relative).read_bytes();published=original
  if role=='metadata':
   value=json.loads(original);clean=safe_metadata(value)
   assert numbers(value)==numbers(clean),'Metadata numeric values changed'
   if value!=clean:published=(json.dumps(clean,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
  try:check_bytes(relative,published)
  except AssertionError as error:raise AssertionError(f'Private path found in scientific file {relative}; no data rewritten') from error
  payload[relative]=published
  entries.append(dict(path=relative,kind=role,original_sha256=sha_bytes(original),published_sha256=sha_bytes(published),
   original_bytes=len(original),published_bytes=len(published),metadata_path_sanitization=original!=published))
 info=dict(study=112,archive_schema=1,source_manifest_sha256=sha_bytes(args.manifest.read_bytes()),
  archive_builder_sha256=sha_bytes(Path(__file__).read_bytes()),files=entries,
  source_code_sha256={name:sha_bytes((base/name).read_bytes()) for name in ['measure.py','precision_audit.py']},
  reproduction_manifest_sha256=sha_bytes(repro_bytes),numeric_data_policy='Scientific data, generated text, token IDs, logprobs and activation values remain byte-identical. Only host paths in metadata can change; original and published hashes are both retained.',
  excludes=['model weights','logs','full agent transcripts','persistence outcomes not yet joined'])
 payload['MEASUREMENTS_MANIFEST.json']=(json.dumps(info,indent=2)+'\n').encode()
 readme='''# CPU reproduction of study112\n\nDo not extract over the live repository or an existing result directory. From the repository study112 directory, choose a NEW separate directory (the mkdir command must succeed):\n\n    mkdir ../study112-reproduction\n    python3 -m zipfile -e data/measurements.zip ../study112-reproduction\n    cd ../study112-reproduction\n\nUsing a Python interpreter with NumPy installed, run:\n\n    python3 analyze.py --manifest analysis_inputs.reproduction.json --output analysis_reproduced\n\nUse a new output directory. No model download, GPU or inference is required. All five primary label versions are included. Each archive entry's original and published SHA256 appears in MEASUREMENTS_MANIFEST.json; only phase-metadata host paths may differ. All scientific raw data are byte-identical to the executed experiment. The archive includes discovery calibration and dose-choice inputs, control measurements, all primary analysis inputs, and the float32 precision sensitivity data. Compare analysis CSVs with the published analysis; summary input-path hashes can differ because the extraction directory differs. Persistence is a separate extension and is not included here.\n'''
 payload['MEASUREMENTS_README.md']=readme.encode()
 for relative,data in payload.items():check_bytes(relative,data)
 args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists(),'Archive already exists'
 with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for relative,data in sorted(payload.items()):
   zi=zipfile.ZipInfo(relative,date_time=(2026,10,29,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o644<<16
   z.writestr(zi,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
 with zipfile.ZipFile(args.output) as z:
  assert set(z.namelist())==set(payload) and z.testzip() is None
  for relative,data in payload.items():assert z.read(relative)==data
 # Publish the relative analyzer manifest beside the archive as a reviewable file.
 repro_path=base/'analysis_inputs.reproduction.json'
 if repro_path.exists():assert repro_path.read_bytes()==repro_bytes
 else:repro_path.write_bytes(repro_bytes)
 print(json.dumps(dict(archive=str(args.output),files=len(payload),compressed_bytes=args.output.stat().st_size,
  uncompressed_bytes=sum(len(v) for v in payload.values()),sha256=sha_bytes(args.output.read_bytes()),metadata_sanitized=sum(e['metadata_path_sanitization'] for e in entries))))
 return info

def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);return p
if __name__=='__main__':build(parser().parse_args())
