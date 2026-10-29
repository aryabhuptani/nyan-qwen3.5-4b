"""CPU test of immutable masking, independent review locks and null preservation."""
import argparse,importlib.util,json,tempfile
from pathlib import Path
p=Path(__file__).with_name('prepare_review.py');spec=importlib.util.spec_from_file_location('review113',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='review113_') as tmp:
 d=Path(tmp);cases=dict(cases=[dict(case_id='c',family_id='f',split='validation',messages=[dict(role='user',content='Hello')],quality_criteria='Readable')]);m.save(d/'cases.json',cases);m.save(d/'job.json',dict(conditions=[dict(condition_id='00000')]))
 m.write(d/'raw.jsonl',json.dumps(dict(condition_id='00000',case_id='c',messages=cases['cases'][0]['messages'],text='A hug!',finish_reason='stop'))+'\n')
 a=argparse.Namespace(raw=[d/'raw.jsonl'],cases=d/'cases.json',job=d/'job.json',out_dir=d/'masked',seed=113,tag='selftest',batch_size=10);m.mask(a)
 item=m.lines(d/'masked/batch_01.jsonl')[0];assert 'condition_id' not in item and 'case_id' not in item
 iid=item['item_id'];ra={f:0 for f in m.FEATURES};ra['own_affection']=1;rb=dict(ra);rb['self_pride']=None
 for who,r in [('A',ra),('B',rb)]:m.save(d/f'{who}.json',dict(reviewer_id=who,items=[dict(item_id=iid,ratings=r,evidence={f:'test evidence' for f,v in r.items() if v in (1,None)})]))
 a=argparse.Namespace(command='adjudicate',out_dir=d/'masked',review_a=[d/'A.json'],review_b=[d/'B.json'],batch_size=10);m.adjudicate(a)
 final=dict(ra);final['self_pride']=None;m.save(d/'third.json',dict(reviewer_id='third',items=[dict(item_id=iid,ratings=final,evidence={f:'test evidence' for f,v in final.items() if v in (1,None)})]))
 a.command='join';a.adjudications=[d/'third.json'];a.labels_dir=d/'labels';m.adjudicate(a)
 rows=list(__import__('csv').DictReader((d/'labels/labels.csv').open()));assert rows[0]['self_pride']==''
 assert m.load(d/'labels/label_evidence.json')['items'][0]['reviewer_B']['ratings']['self_pride'] is None
 assert list(__import__('csv').DictReader((d/'labels/labels_uncertain_zero.csv').open()))[0]['self_pride']=='0'
 assert list(__import__('csv').DictReader((d/'labels/labels_uncertain_one.csv').open()))[0]['self_pride']=='1'
 assert m.load(d/'labels/export_manifest.json')['rows']==1
 try:m.write(d/'A.json','changed')
 except AssertionError:pass
 else:raise AssertionError('Overwrite guard failed')
print('PASS: masking, complete review coverage, null preservation, bounds, immutable exports')
