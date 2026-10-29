"""CPU-only complete-set summary of all fixed24 numerical sensitivity contrasts.

Reads results only on explicit --comparisons invocation. --self-test uses synthetic
fixtures and frozen authored case metadata, never model outcomes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
NAMES=('genki','hugs','paws','female')
OPERATIONS=('canonical','attenuate')
METHOD='Float32 computations on exactly cast original BF16 checkpoint weight values, using unchanged BF16-discovery endpoints/fractions; compared with original BF16 computations. Numerical sensitivity, not an independent checkpoint or replication.'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sign(v): return 'positive' if v>0 else ('negative' if v<0 else 'zero')


def summarize(rows,cases):
    assert len(cases)==12 and {c['ingredient'] for c in cases}==set(NAMES)
    expected={(c['case_id'],op+'_'+c['ingredient']) for c in cases for op in OPERATIONS}
    lookup={(r['case_id'],r['condition_id']):r for r in rows}
    assert len(rows)==len(lookup)==24 and set(lookup)==expected, 'All fixed24 unique matched contrasts required; no subset selection'
    ingredients={c['case_id']:c['ingredient'] for c in cases}
    parsed=[]
    for r in rows:
        assert r['ingredient']==ingredients[r['case_id']]
        b=float(r['bf16_change']); f=float(r['float32_change'])
        assert math.isfinite(b) and math.isfinite(f)
        diff=f-b
        if 'difference' in r: assert abs(float(r['difference'])-diff)<1e-10
        if 'absolute_difference' in r: assert abs(float(r['absolute_difference'])-abs(diff))<1e-10
        parsed.append(dict(case_id=r['case_id'],condition_id=r['condition_id'],ingredient=r['ingredient'],
                           bf16_change=b,float32_change=f,difference=diff,absolute_difference=abs(diff)))
    def group(rs):
        matrix={b:{f:0 for f in ('negative','zero','positive')} for b in ('negative','zero','positive')}
        for r in rs: matrix[sign(r['bf16_change'])][sign(r['float32_change'])]+=1
        return dict(contrasts=len(rs),mean_bf16_change=statistics.mean(r['bf16_change'] for r in rs),
                    mean_float32_change=statistics.mean(r['float32_change'] for r in rs),
                    mean_absolute_discrepancy=statistics.mean(r['absolute_difference'] for r in rs),
                    maximum_absolute_discrepancy=max(r['absolute_difference'] for r in rs),
                    same_nonzero_sign=sum(r['bf16_change']*r['float32_change']>0 for r in rs),
                    opposite_nonzero_sign=sum(r['bf16_change']*r['float32_change']<0 for r in rs),
                    bf16_exact_zero=sum(r['bf16_change']==0 for r in rs),float32_exact_zero=sum(r['float32_change']==0 for r in rs),
                    sign_matrix_bf16_rows_float32_columns=matrix)
    return dict(method=METHOD,endpoint='Intervention-minus-baseline change in target-minus-control mean-token log probability, nat/token',
                selection='All12fixedheldoutlexicalcases;matchedcanonicalandattenuation;all24contrastsretained',
                all=group(parsed),by_ingredient={n:group([r for r in parsed if r['ingredient']==n]) for n in NAMES},
                by_operation={op:group([r for r in parsed if r['condition_id'].startswith(op+'_')]) for op in OPERATIONS},
                all_comparisons=parsed,
                limits='Sign agreement is descriptive and ignores magnitude; exact-zero counts are separate. No numerical threshold excludes outcomes or revises primary BF16 findings.')


def selftest(cases):
    fixture=[]
    for i,c in enumerate(cases):
        for op in OPERATIONS:
            b=0. if i==0 else (.02 if op=='canonical' else -.01)
            f=-b if i==1 else b+.001
            fixture.append(dict(case_id=c['case_id'],condition_id=op+'_'+c['ingredient'],ingredient=c['ingredient'],bf16_change=b,float32_change=f))
    result=summarize(fixture,cases)
    assert result['all']['contrasts']==24 and result['all']['opposite_nonzero_sign']==2
    assert result['all']['bf16_exact_zero']==2
    try: summarize(fixture[:-1],cases)
    except AssertionError: pass
    else: raise AssertionError('Subset must fail')
    try: summarize(fixture[:-1]+fixture[:1],cases)
    except AssertionError: pass
    else: raise AssertionError('Duplicate must fail')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparisons',type=Path)
    p.add_argument('--cases',type=Path,default=ROOT/'CASES_lexical_validation.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(); cases=json.loads(args.cases.read_text())['cases']
    if args.self_test:
        selftest(cases)
        print(json.dumps(dict(selftest_passed=True,synthetic_only=True,validation_outcomes_read=False)))
        return
    assert args.comparisons and args.output
    assert not args.output.exists()
    with args.comparisons.open() as f: rows=list(csv.DictReader(f))
    result=summarize(rows,cases)
    result['provenance']=dict(comparisons_sha256=sha(args.comparisons),cases_sha256=sha(args.cases),helper_sha256=sha(__file__),precision_plan_sha256=sha(ROOT/'PRECISION_PLAN.md'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,contrasts=24,output=str(args.output))))

if __name__=='__main__':main()
