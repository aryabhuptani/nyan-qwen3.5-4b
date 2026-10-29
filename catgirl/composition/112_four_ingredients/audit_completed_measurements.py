"""CPU-only complete-set precision and unrelated-coordinate natural diagnostics.

Explicitly separate random-coordinate diagnostics from original target forecasts.
Does not mutate primary analysis, study settings, model outputs or labels.
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


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write_csv(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'measurement_diagnostics')
    args=p.parse_args()
    assert not args.output.exists(),'Fresh output directory required'
    from precision_audit import contrasts
    from precision_summary import summarize
    import analyze
    paths=dict(precision_comparisons=ROOT/'results.local/precision_audit/comparison.csv',
               precision_raw=ROOT/'results.local/precision_audit/float32.jsonl',
               precision_metadata=ROOT/'results.local/precision_audit/metadata.json',
               bf16_raw=ROOT/'results.local/validation_lexical_generation/probe.jsonl',
               random_bf16_raw=ROOT/'results.local/validation_lexical_generation_random/probe.jsonl',
               lexical_cases=ROOT/'CASES_lexical_validation.json',
               control_discovery=ROOT/'results.local/control_capture/natural.jsonl',
               control_validation=ROOT/'results.local/validation_natural_random/natural.jsonl',
               control_metadata=ROOT/'results.local/control_capture/natural_metadata.json',
               control_coordinates=ROOT/'CONTROL_COORDINATES.json',natural_cases=ROOT/'CASES_natural_validation.json',
               precision_plan=ROOT/'PRECISION_PLAN.md',analyzer=ROOT/'analyze.py',helper=Path(__file__))
    lex=load(paths['lexical_cases'])['cases']; fp=rows(paths['precision_raw']);bf=rows(paths['bf16_raw'])
    with paths['precision_comparisons'].open() as f: comparisons=list(csv.DictReader(f))
    summary=summarize(comparisons,lex)
    fpcontrasts=contrasts(fp,lex);bfcontrasts=contrasts(bf,lex)
    assert len(fp)==36 and len(fpcontrasts)==len(bfcontrasts)==24
    metadata=load(paths['precision_metadata'])
    assert metadata['bf16_results_sha256']==sha(paths['bf16_raw'])
    fptable={(r['case_id'],r['condition_id']):r for r in fp}
    bftable={(r['case_id'],r['condition_id']):r for r in bf}
    complete=[]
    for r in comparisons:
        key=(r['case_id'],r['condition_id'])
        assert float(r['bf16_change'])==bfcontrasts[key] and float(r['float32_change'])==fpcontrasts[key]
        name=r['ingredient']; b=fptable[key];a=bftable[key]
        assert a['token_ids']==b['token_ids'] and a['operations']==b['operations']
        complete.append(dict(case_id=r['case_id'],ingredient=name,condition_id=r['condition_id'],
             bf16_change=float(r['bf16_change']),float32_change=float(r['float32_change']),
             same_nonzero_sign=float(r['bf16_change'])*float(r['float32_change'])>0,
             bf16_baseline_candidate_mass=bftable[(r['case_id'],'baseline')]['candidate_mass'],
             bf16_intervention_candidate_mass=a['candidate_mass'],
             float32_baseline_candidate_mass=fptable[(r['case_id'],'baseline')]['candidate_mass'],
             float32_intervention_candidate_mass=b['candidate_mass']))
    random_rows=rows(paths['random_bf16_raw'])
    random_table={(r['case_id'],r['condition_id']):r for r in random_rows}
    pref=lambda r:r['choices']['target']['mean_logprob']-r['choices']['control']['mean_logprob']
    random_changes={name:[pref(random_table[(c['case_id'],'random_'+name)])-pref(random_table[(c['case_id'],'baseline')]) for c in lex if c['ingredient']==name] for name in NAMES}
    table=[]
    for name in NAMES:
        for op in ('canonical','attenuate'):
            rs=[r for r in complete if r['ingredient']==name and r['condition_id']==op+'_'+name]
            bmean=statistics.mean(r['bf16_change'] for r in rs);fmean=statistics.mean(r['float32_change'] for r in rs)
            table.append(dict(ingredient=name,operation=op,cases=3,bf16_mean_change=bmean,float32_mean_change=fmean,
             bf16_positive_cases=sum(r['bf16_change']>0 for r in rs),float32_positive_cases=sum(r['float32_change']>0 for r in rs),
             sign_agreement=sum(r['same_nonzero_sign'] for r in rs),
             random_bf16_matched_mean=statistics.mean(random_changes[name]) if op=='canonical' else None,
             random_bf16_positive_cases=sum(v>0 for v in random_changes[name]) if op=='canonical' else None,
             bf16_practical_enhancement_marker=op=='canonical' and bmean>=.01 and sum(r['bf16_change']>0 for r in rs)>=2,
             float32_practical_enhancement_marker=op=='canonical' and fmean>=.01 and sum(r['float32_change']>0 for r in rs)>=2,
             bf16_candidate_mass_min=min(r['bf16_intervention_candidate_mass'] for r in rs),
             bf16_candidate_mass_max=max(r['bf16_intervention_candidate_mass'] for r in rs),
             float32_candidate_mass_min=min(r['float32_intervention_candidate_mass'] for r in rs),
             float32_candidate_mass_max=max(r['float32_intervention_candidate_mass'] for r in rs)))
    assert all(r['split']=='discovery' for r in rows(paths['control_discovery']))
    discovery=analyze.discovery_summary(rows(paths['control_discovery']))
    assert all(r['count']==60 for r in discovery.values())
    caseindex=analyze.cases_index(paths['natural_cases'])
    random=analyze.natural_analysis(rows(paths['control_validation']),caseindex,discovery)
    cm=load(paths['control_metadata']);coordinates=load(paths['control_coordinates'])
    assert {k:list(v) for k,v in cm['ingredients'].items()}==coordinates
    natural=[]
    for r in random['effects']:
        if r['scope']=='readout' and r['family']=='ALL':
            name=r['ingredient'];sd=discovery[name]['sd_population']
            natural.append(dict(matched_context_ingredient=name,layer=coordinates[name][0],neuron=coordinates[name][1],
             arbitrary_matched_direction=coordinates[name][2],discovery_final_sd=sd,
             semantic_signed_raw=r['semantic_signed'],semantic_discovery_sd=r['semantic_discovery_sd'],
             keyword_signed_raw=r['keyword_signed'],keyword_discovery_sd=r['keyword_signed']/sd if sd else None,
             semantic_positive_families=r['positive_families'],families=r['families'],
             descriptive_semantic_marker=r['practical_semantic_marker']))
    args.output.mkdir(parents=True)
    write_csv(args.output/'precision_matched_table.csv',table)
    write_csv(args.output/'precision_all24_with_mass.csv',complete)
    write_csv(args.output/'random_natural_summary.csv',natural)
    analyze.write_csv(args.output/'random_natural_family_effects.csv',random['effects'])
    analyze.write_csv(args.output/'random_natural_case_scalars.csv',random['details'])
    result=dict(precision=summary,precision_matched_table=table,random_natural_summary=natural,
                random_discovery_final_stats=discovery,
                candidate_mass_limit='Full candidate continuation masses are tiny and vary with length; exact-sequence probabilities naturally shrink with length, so this does not imply a meaning is impossible. These are relative teacher-forced wording preferences, not free-generation persona probabilities. Conditional normalization over supplied sequences does not establish semantic interpretation.',
                random_writing_limit='Random writing controls were measured in BF16 only, not float32. Similar matched random gains prevent interpreting native sign stability as semantic specificity.',
                random_limit='Four fixed unrelated same-layer coordinates, signs inherited from matched target interventions. Context labels are original target hypotheses, not meanings of control neurons. No random-control forecasts were collected or pooled into target forecast scores. No population inference.',
                numerical_limit='Float32 computations preserve original BF16 checkpoint weight values exactly; no dose refit. Precision sensitivity alone does not identify which path is closer to an exact model computation.',
                provenance={k:dict(path=str(v.relative_to(ROOT)),sha256=sha(v)) for k,v in paths.items()})
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Completed measurement audit','',
           'All 24 fixed matched contrasts were recomputed from raw rows and agree exactly with the comparison CSV. Float32 means computations on the exactly cast original BF16 checkpoint values, with unchanged discovery doses.','',
           '| Ingredient | Operation | BF16 mean | Float32 mean | Positive cases BF16/float32 | Sign agreement |',
           '|---|---|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['ingredient']} | {r['operation']} | {r['bf16_mean_change']:+.6f} | {r['float32_mean_change']:+.6f} | {r['bf16_positive_cases']}/3; {r['float32_positive_cases']}/3 | {r['sign_agreement']}/3 |")
    lines.extend(['','Units: intervention-minus-baseline change in target-minus-control mean-token log probability, nat/token. Canonical and attenuation means are distinct and must not be pooled as a desired-behavior score.','',
      'No canonical float32 mean reaches the predeclared +0.01 nat/token marker. Paws retains positive enhancement and negative attenuation in all three contexts, but its mean enhancement shrinks from +0.014145 to +0.005617. Hugs attenuation increases target preference in two of three contexts, with mean +0.056989 in float32; this opposes a simple monotonic necessity interpretation. Genki canonical is tiny despite three positive float32 signs. Female canonical changes from a negative BF16 mean to a near-zero positive float32 mean.','',
      f"Across all 24 contrasts, {summary['all']['same_nonzero_sign']}/24 nonzero signs agree. Mean absolute discrepancy is {summary['all']['mean_absolute_discrepancy']:.6f}, maximum {summary['all']['maximum_absolute_discrepancy']:.6f} nat/token.",'',
      'Candidate continuation masses are extremely small (see all24 table), so these effects concern relative preferences among supplied completions. They do not measure the probability of spontaneously producing a catgirl role. Exact-sequence probabilities shrink with length; low mass does not show these meanings are impossible. Variable continuation length also limits mass comparisons across cases.','',
      f"The unrelated same-layer Paws control has BF16 matched writing mean {statistics.mean(random_changes['paws']):+.6f} nat/token ({sum(v>0 for v in random_changes['paws'])}/3 positive), close to the native Paws mean +0.014145. Random writing controls were not repeated in float32. Native sign consistency alone is not evidence of semantic specificity or a clean mechanism. All matched random BF16 writing means are retained in precision_matched_table.csv.",'',
      '## Unrelated-coordinate natural diagnostic','',
      '| Matched target context | Unrelated coordinate | Semantic contrast / own discovery final SD | Same-direction families | Descriptive marker |',
      '|---|---|---:|---:|---|'])
    for r in natural:
        lines.append(f"| {r['matched_context_ingredient']} | L{r['layer']} N{r['neuron']} | {r['semantic_discovery_sd']:+.6f} | {r['semantic_positive_families']}/3 | {r['descriptive_semantic_marker']} |")
    lines.extend(['','All four fail the directional 0.25 SD plus two-of-three-family marker. This is a fixed four-control diagnostic, not a statistical population test or evidence that published target meanings are correct. Standardization uses each control coordinate’s own final-position population SD over all 60 discovery capture cases. Validation uses the 48 natural cases only. Control signs follow the paired target convention.','',
      'Family effects and all case scalars are retained separately. Control activation outcomes were not added to the original target forecasts. Source hashes and complete numerical tables are in summary.json.'])
    (args.output/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output),precision_contrasts=24,random_natural_cases=48,inputs_hashed=len(paths))))

if __name__=='__main__':main()
