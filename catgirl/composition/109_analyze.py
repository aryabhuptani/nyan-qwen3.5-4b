"""CPU-only, calibration-separated analysis of frozen109 natural/causal probes.

No model libraries, fitting, significance tests, or candidate selection. Semantic
choice orders are averaged before pairing, language summaries, or family means.
Run --self-test for synthetic checks without reading experimental outcomes.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUANTILES = {'q05': .05, 'q10': .10, 'q25': .25, 'q50': .50,
             'q75': .75, 'q90': .90, 'q95': .95}
METRICS = ('carriermean', 'lastmean', 'wholetextmax')
NEIGHBORS = ('neutral_achievement', 'joy', 'confidence', 'arrogance')


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric value, got %r' % value)
    return float(value)


def mean(values):
    return statistics.fmean(values) if values else None


def percentile(values, fraction):
    """Linear/type7 quantile, identical to the usual NumPy default."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(values):
    result = dict(n=len(values), mean=mean(values), minimum=min(values) if values else None,
                  maximum=max(values) if values else None)
    result.update({name: percentile(values, q) for name, q in QUANTILES.items()})
    result['iqr'] = result['q75'] - result['q25'] if values else None
    return result


def read_jsonl(path):
    if not path.exists():
        return []
    result = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError as error:
                raise ValueError('%s:%s: %s' % (path, line_number, error)) from error
    return result


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def case_meta(record, cases):
    case_id = record['case_id']
    if case_id not in cases:
        raise ValueError('Unregistered case: ' + case_id)
    case = cases[case_id]
    result = dict(case_id=case_id)
    for field in ('family', 'language', 'split', 'category', 'lexeme_present'):
        if field in record and record[field] != case[field]:
            raise ValueError('Record/corpus mismatch for %s %s' % (case_id, field))
        result[field] = case[field]
    result['expected_pride_semantics'] = case['expected_semantics']['own_accomplishment_satisfaction']
    return result


def check_window(start, end, length, label):
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= length:
        raise ValueError('Invalid %s window [%r,%r), token count%s' % (label, start, end, length))


def natural_analysis(records, cases):
    rows, pool, warnings = [], defaultdict(list), []
    seen, channel_order = set(), None
    for record in records:
        meta = case_meta(record, cases)
        if meta['case_id'] in seen:
            raise ValueError('Duplicate natural case: ' + meta['case_id'])
        seen.add(meta['case_id'])
        channels = record['channels']
        if len(channels) != 4 or len(set(channels)) != 4 or 1911 not in channels:
            raise ValueError('Natural capture must include target1911 and three unique controls')
        if channel_order is None:
            channel_order = channels
        elif channels != channel_order:
            raise ValueError('Natural channel order changed between records')
        pre = [[finite(v) for v in token] for token in record['pre']]
        post = [[finite(v) for v in token] for token in record['post']]
        size = len(record['token_ids'])
        if not len(pre) == len(post) == record['prompt_tokens'] == size or any(len(t) != 4 for t in pre + post):
            raise ValueError('Natural tensor/token shape mismatch: ' + meta['case_id'])
        a, z = record['carrier_start'], record['carrier_end']
        check_window(a, z, size, 'carrier')
        explicit_story = 'story_start' in record and 'story_end' in record
        story_a = record.get('story_start', record.get('text_start', 0))
        story_z = record.get('story_end', record.get('text_end', z))
        check_window(story_a, story_z, size, 'story/text')
        if not explicit_story:
            warnings.append('Wholetext window fallback for ' + meta['case_id'] + '; exact story bounds absent')
        for j, channel in enumerate(channels):
            carrier = [token[j] for token in pre[a:z]]
            modified = [token[j] for token in post[a:z]]
            row = dict(**meta, channel=channel, is_target=channel == 1911,
                       prompt_tokens=size, carrier_start=a, carrier_end=z,
                       carrier_tokens=z-a, wholetext_start=story_a, wholetext_end=story_z,
                       wholetext_tokens=story_z-story_a,
                       wholetext_excludes_chat_prefix=explicit_story or story_a > 0,
                       wholetext_window_source='story_start/end' if explicit_story else 'fallback',
                       carriermean=mean(carrier), lastmean=carrier[-1],
                       wholetextmax=max(token[j] for token in pre[story_a:story_z]),
                       story_and_carrier_max=max(token[j] for token in pre[story_a:max(story_z,z)]),
                       carrier_last4_mean=mean(carrier[-4:]),
                       post_carriermean=mean(modified), post_lastmean=modified[-1],
                       max_abs_pre_post_change=max(abs(pre[t][j]-post[t][j]) for t in range(size)))
            rows.append(row)
            if meta['split'] == 'calibration':
                pool[(channel, 'all')].extend(carrier)
                pool[(channel, meta['language'])].extend(carrier)
    quantiles = [dict(channel=channel, language=language, split='calibration',
                      window='carrier', statistic='pooled_pre_token_quantiles', **distribution(values))
                 for (channel, language), values in sorted(pool.items())]
    q90 = {x['channel']: x['q90'] for x in quantiles if x['language'] == 'all'}
    for row in rows:
        if row['channel'] in q90:
            record = next(x for x in records if x['case_id'] == row['case_id'])
            j = record['channels'].index(row['channel'])
            count = sum(token[j] > q90[row['channel']] for token in record['pre'][row['carrier_start']:row['carrier_end']])
            row['calibration_q90'] = q90[row['channel']]
            row['high_token_count'] = count
            row['high_token_fraction'] = count / row['carrier_tokens']
    paired = []
    index = {(x['family'], x['language'], x['category'], x['channel']): x for x in rows if x['split'] == 'validation'}
    for row in rows:
        if row['split'] != 'validation' or row['category'] != 'own_accomplishment':
            continue
        for neighbor in NEIGHBORS:
            other = index.get((row['family'], row['language'], neighbor, row['channel']))
            if other is None:
                warnings.append('Missing natural pair: %s versus %s' % (row['case_id'], neighbor))
                continue
            for metric in METRICS:
                paired.append(dict(split='validation', family=row['family'], language=row['language'],
                                   channel=row['channel'], contrast='own_accomplishment_minus_'+neighbor,
                                   metric=metric, target_case=row['case_id'], comparison_case=other['case_id'],
                                   difference=row[metric]-other[metric]))
    lexical = []
    diagnostic = {(x['family'], x['language'], x['category'], x['channel']): x for x in rows if x['split'] == 'validation_lexical_diagnostic'}
    for (family, language, category, channel), row in diagnostic.items():
        if category not in ('lexical_trap', 'own_accomplishment', 'own_accomplishment_explicit'):
            continue
        for other_category in ('neutral_lexical_control', 'lexical_trap', 'own_accomplishment'):
            if other_category == category:
                continue
            other = diagnostic.get((family, language, other_category, channel))
            if other:
                for metric in METRICS:
                    lexical.append(dict(family=family, language=language, channel=channel,
                                        contrast=category+'_minus_'+other_category, metric=metric,
                                        difference=row[metric]-other[metric]))
    summaries = group_differences(paired, ('channel', 'contrast', 'metric'))
    return rows, dict(channels=channel_order, quantiles_calibration_only=quantiles,
                      paired_validation=paired, paired_validation_summaries=summaries,
                      lexical_diagnostics=lexical, warnings=sorted(set(warnings)),
                      natural_pre_post_equal=all(x['max_abs_pre_post_change'] == 0 for x in rows))


def effect_summary(rows, field):
    values = [x[field] for x in rows]
    return dict(n=len(values), mean=mean(values), median=percentile(values, .5),
                fraction_positive=sum(v > 0 for v in values)/len(values) if values else None,
                fraction_negative=sum(v < 0 for v in values)/len(values) if values else None,
                fraction_zero=sum(v == 0 for v in values)/len(values) if values else None)


def group_differences(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        for language in ('all', row['language']):
            groups[tuple(row[k] for k in keys)+(language,)].append(row)
    result = []
    for values, members in sorted(groups.items()):
        families = defaultdict(list)
        for row in members:
            families[row['family']].append(row['difference'])
        clustered = [dict(family=f, difference=mean(v)) for f,v in sorted(families.items())]
        result.append(dict(**dict(zip(keys+('language',), values)),
                           pair_summary=effect_summary(members, 'difference'),
                           family_summary=effect_summary(clustered, 'difference'), family_effects=clustered))
    return result


def average_causal_orders(records, cases):
    groups = defaultdict(list)
    for record in records:
        case_meta(record, cases)
        if record['task'] not in ('semantic', 'lexical'):
            raise ValueError('Unknown causal task')
        if type(record.get('order')) is not int or record['order'] not in (0,1):
            raise ValueError('Causal order must be0/1')
        finite(record['log_odds_pride'])
        if not 0 <= finite(record['prob_pride']) <= 1:
            raise ValueError('Probability outside[0,1]')
        if record['task'] == 'semantic' and type(record.get('choice_correct')) is not bool:
            raise ValueError('Semantic choice_correct must be boolean')
        if not isinstance(record.get('token_logprobs'), dict) or len(record['token_logprobs']) != 2:
            raise ValueError('Expected exactly two fixed-option token log-probabilities')
        for value in record['token_logprobs'].values():
            finite(value)
        groups[(record['case_id'],record['task'],record['condition'])].append(record)
    rows = []
    for (case_id, task, condition), members in sorted(groups.items()):
        orders = [x['order'] for x in members]
        if len(set(orders)) != len(orders):
            raise ValueError('Duplicate causal case/task/condition/order')
        row = dict(**case_meta(members[0],cases), task=task, condition=condition,
                   orders=sorted(orders), orders_complete=task != 'semantic' or set(orders)=={0,1},
                   prob_pride=mean([x['prob_pride'] for x in members]),
                   log_odds_pride=mean([x['log_odds_pride'] for x in members]),
                   choice_accuracy=mean([int(x['choice_correct']) for x in members]) if task=='semantic' else None,
                   order_probability_gap=abs(members[0]['prob_pride']-members[1]['prob_pride']) if len(members)==2 else None,
                   per_order=sorted(members,key=lambda x:x['order']))
        masses = [sum(math.exp(v) for v in x['token_logprobs'].values()) for x in members]
        row['option_token_mass_mean'] = mean(masses)
        row['option_token_mass_min'] = min(masses)
        row['option_token_mass_max'] = max(masses)
        row['unconditional_pride_option_probability'] = mean([x['prob_pride']*mass for x,mass in zip(members,masses)])
        for field in ('changed_positions', 'achieved_max_abs_delta', 'achieved_mean_abs_delta', 'achieved_residual_l2_sum', 'max_abs_logit_change'):
            values = [finite(x[field]) for x in members if field in x]
            row[field+'_mean'] = mean(values)
        rows.append(row)
    index = {(x['case_id'],x['task'],x['condition']):x for x in rows}
    for row in rows:
        baseline = index.get((row['case_id'],row['task'],'baseline'))
        row['paired_baseline_complete'] = bool(baseline and row['orders_complete'] and baseline['orders_complete'])
        if row['paired_baseline_complete']:
            row['delta_probability'] = row['prob_pride']-baseline['prob_pride']
            row['delta_log_odds'] = row['log_odds_pride']-baseline['log_odds_pride']
            row['delta_accuracy'] = row['choice_accuracy']-baseline['choice_accuracy'] if row['task']=='semantic' else None
            row['delta_option_token_mass'] = row['option_token_mass_mean']-baseline['option_token_mass_mean']
            row['delta_unconditional_pride_probability'] = row['unconditional_pride_option_probability']-baseline['unconditional_pride_option_probability']
    return rows


def summarize_causal(rows, natural_rows):
    high = {x['case_id']:x.get('high_token_count',0) for x in natural_rows if x['is_target']}
    for row in rows:
        row['natural_target_high_tokens'] = high.get(row['case_id'])
    groups = defaultdict(list)
    for row in rows:
        if not row['paired_baseline_complete']:
            continue
        for language in ('all', row['language']):
            for category in ('all', row['category']):
                for high_stratum in ('all', 'natural_high') if (high.get(row['case_id'],0)>0) else ('all',):
                    splits = (row['split'], 'all_fresh') if row['split'].startswith('validation') else (row['split'],)
                    for split in splits:
                        groups[(split, row['task'], row['condition'],language,category,high_stratum)].append(row)
    summaries = []
    for key, members in sorted(groups.items()):
        families = defaultdict(list)
        for row in members:
            families[row['family']].append(row)
        family_effects = [dict(family=f, n_cases=len(v), delta_probability=mean([x['delta_probability'] for x in v]),
                               delta_log_odds=mean([x['delta_log_odds'] for x in v])) for f,v in sorted(families.items())]
        summary = dict(zip(('split','task','condition','language','category','high_stratum'),key))
        summary.update(n_cases=len(members), n_families=len(family_effects),
                       probability_change=effect_summary(members,'delta_probability'),
                       log_odds_change=effect_summary(members,'delta_log_odds'),
                       family_probability_change=effect_summary(family_effects,'delta_probability'),
                       family_log_odds_change=effect_summary(family_effects,'delta_log_odds'),
                       family_effects=family_effects, mean_probability=mean([x['prob_pride'] for x in members]),
                       mean_choice_accuracy=mean([x['choice_accuracy'] for x in members]) if key[1]=='semantic' else None,
                       mean_accuracy_change=mean([x['delta_accuracy'] for x in members]) if key[1]=='semantic' else None,
                       option_token_mass_mean=mean([x['option_token_mass_mean'] for x in members]),
                       option_token_mass_min=min(x['option_token_mass_min'] for x in members),
                       option_token_mass_max=max(x['option_token_mass_max'] for x in members),
                       mean_unconditional_pride_probability=mean([x['unconditional_pride_option_probability'] for x in members]),
                       mean_unconditional_pride_probability_change=mean([x['delta_unconditional_pride_probability'] for x in members]))
        summaries.append(summary)
    contrasts = []
    index = {(x['case_id'],x['task'],x['condition']):x for x in rows if x['orders_complete']}
    for row in rows:
        if not row['orders_complete'] or not row['condition'].endswith('_q90'):
            continue
        low = index.get((row['case_id'],row['task'],row['condition'][:-3]+'q10'))
        if low:
            contrasts.append(dict(case_id=row['case_id'], family=row['family'],language=row['language'],
                                  split=row['split'],task=row['task'],condition=row['condition']+'_minus_q10',
                                  probability_difference=row['prob_pride']-low['prob_pride'],
                                  log_odds_difference=row['log_odds_pride']-low['log_odds_pride']))
    return dict(summaries=summaries, q90_minus_q10_case_contrasts=contrasts,
                incomplete_semantic_orders=[dict(case_id=x['case_id'],condition=x['condition'],orders=x['orders']) for x in rows if not x['orders_complete']],
                missing_baselines=[dict(case_id=x['case_id'],task=x['task'],condition=x['condition']) for x in rows if not x['paired_baseline_complete']],
                gate_status='Inputs only. Apply the prospectively frozen root plan; no inferred pass rule or p-values.',
                aggregation='Mean each semantic order pair first. Case means then family means across related languages/categories. Translation rows are not independent topics.')


def frozen_gate_analysis(cases, natural_result, causal_rows):
    """Literal frozen109 gates; no language/event/condition selection from outcomes."""
    primary_ids = {k for k,v in cases.items() if v['split']=='validation' and v['category']=='own_accomplishment'}
    pairs = [x for x in natural_result['paired_validation'] if x['channel']==1911 and x['metric']=='carriermean'
             and x['contrast']=='own_accomplishment_minus_neutral_achievement']
    association = effect_summary(pairs, 'difference')
    association['expected_pairs'] = len(primary_ids)
    association['passes'] = association['n']==len(primary_ids) and association['fraction_positive']>=.75
    conditions = ['target_carrier_q90','target_carrier_suppress_high'] + [
        'control_%s_carrier_q%s'%(i,q) for i in range(3) for q in (10,90)]
    effects={}
    for condition in conditions:
        members=[x for x in causal_rows if x['case_id'] in primary_ids and x['task']=='semantic'
                 and x['condition']==condition and x['paired_baseline_complete']]
        effects[condition]=dict(n=len(members), probability=effect_summary(members,'delta_probability'),
            log_odds=effect_summary(members,'delta_log_odds'), case_effects=[dict(case_id=x['case_id'],
            family=x['family'],language=x['language'],delta_probability=x['delta_probability'],
            delta_log_odds=x['delta_log_odds'],natural_high_tokens=x['natural_target_high_tokens'],
            changed_positions_mean=x['changed_positions_mean'], option_token_mass=x['option_token_mass_mean']) for x in members])
    complete=all(x['n']==len(primary_ids) for x in effects.values())
    semantic=dict(expected_pairs=len(primary_ids),complete=complete,effects=effects,
                  suppression_control_operation='Target conditional h>q90 toq50 versus prespecified random all-carrier q10 downward proxies. Not operation- or residual-norm-matched.')
    if complete:
        high=effects['target_carrier_q90']['probability']; low=effects['target_carrier_suppress_high']['probability']
        high_controls=[effects['control_%s_carrier_q90'%i]['probability']['mean'] for i in range(3)]
        low_controls=[effects['control_%s_carrier_q10'%i]['probability']['mean'] for i in range(3)]
        suppression=effects['target_carrier_suppress_high']['case_effects']
        eligible=[x for x in suppression if x['natural_high_tokens'] is not None and x['natural_high_tokens']>0]
        changed=[x for x in suppression if x['changed_positions_mean'] is not None and x['changed_positions_mean']>0]
        checks=dict(q90_direction_at_least_75pct=high['fraction_positive']>=.75,
                    suppression_direction_at_least_75pct=low['fraction_negative']>=.75,
                    q90_mean_exceeds_every_random_q90=high['mean']>0 and all(high['mean']>v for v in high_controls),
                    suppression_negative_mean_exceeds_every_random_q10=low['mean']<0 and all(low['mean']<v for v in low_controls),
                    nonzero_naturally_eligible_events=bool(eligible))
        semantic.update(checks=checks,passes=all(checks.values()),natural_eligible_cases=len(eligible),
                        natural_eligible_case_fraction=len(eligible)/len(primary_ids),changed_cases=len(changed),
                        changed_case_fraction=len(changed)/len(primary_ids),
                        natural_high_tokens_total=sum(x['natural_high_tokens'] or 0 for x in suppression),
                        changed_positions_total_order_averaged=sum(x['changed_positions_mean'] or 0 for x in suppression),
                        eligibility_agrees_with_changed_positions=all((x['natural_high_tokens']>0)==(x['changed_positions_mean']>0)
                            for x in suppression if x['natural_high_tokens'] is not None and x['changed_positions_mean'] is not None),
                        insufficient_eligible_case_coverage_for_75pct=len(eligible)<math.ceil(.75*len(primary_ids)))
    else:
        semantic['passes']=None
    return dict(association=association,semantic=semantic,
                interpretation='Descriptive thresholds, not p-values or independent translations. No broad necessity or monosemanticity claim; absent high events limit suppression evidence.')


def audit_calibration(path, cases, natural_result):
    if path is None or not path.exists():
        return dict(available=False)
    saved=json.loads(path.read_text())
    expected={k for k,v in cases.items() if v['split']=='calibration'}
    actual=set(saved['calibration_case_ids'])
    if actual!=expected:
        raise ValueError('Saved calibration IDs do not match calibration-only corpus split')
    observed={x['channel']:x for x in natural_result['quantiles_calibration_only'] if x['language']=='all'}
    differences=[]
    for i,channel in enumerate(saved['channels']):
        for j,pct in enumerate(saved['percentiles']):
            differences.append(abs(observed[channel]['q%02d'%pct]-saved['values']['carrier'][i][j]))
    maximum=max(differences) if differences else None
    if maximum is not None and maximum>1e-10:
        raise ValueError('Saved carrier calibration differs from unsteered calibration-token recomputation')
    target=saved['channels'].index(1911)
    quant=dict(zip(saved['percentiles'],saved['values']['carrier'][target]))
    spread=quant[90]-quant[50]
    return dict(available=True,path=str(path),sha256=source_hash(path),calibration_only_ids_verified=True,
                maximum_carrier_quantile_recomputation_error=maximum,saved_calibration=saved,
                historical_additive_comparison=dict(previous_published_dose=.4,previous_raw_offset_approx=19.6603,
                current_carrier_q90_minus_q50=spread,offset_over_q90_minus_q50=19.6603/spread if spread else None,
                caveat='Previous additive delta was about51.2times this synthetic carrier q90-q50 spread, not natural SDs. Additive prompt/first8 scope differs from carrier-only replacement; clamp displacement depends on original h. Context/position-specific descriptive comparison only.'))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        if not fields:
            return
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def analyze(corpus_path, natural_path, causal_path, plan_path=None, calibration_path=None):
    corpus=json.loads(corpus_path.read_text());cases={x['id']:x for x in corpus['cases']}
    if len(cases)!=len(corpus['cases']):
        raise ValueError('Duplicate corpus case IDs')
    natural=read_jsonl(natural_path);causal=read_jsonl(causal_path)
    natural_rows,natural_result=natural_analysis(natural,cases)
    causal_rows=average_causal_orders(causal,cases)
    causal_result=summarize_causal(causal_rows,natural_rows)
    plan=json.loads(plan_path.read_text()) if plan_path and plan_path.exists() else None
    if plan and source_hash(corpus_path)!=plan['corpus_sha256']:
        raise ValueError('Corpus SHA differs from frozen109 plan')
    expected_conditions=['baseline','target_carrier_q10','target_carrier_q50','target_carrier_q90',
                         'target_carrier_suppress_high','target_carrier_zero','target_output_q10','target_output_q90']
    expected_conditions += ['control_%s_carrier_q%s'%(i,q) for i in range(3) for q in (10,90)]
    expected_keys=set()
    for case_id,case in cases.items():
        if not case['split'].startswith('validation'):
            continue
        for condition in expected_conditions:
            expected_keys.update((case_id,'semantic',condition,order) for order in (0,1))
            if case['language']=='en':
                expected_keys.add((case_id,'lexical',condition,0))
    actual_keys={(x['case_id'],x['task'],x['condition'],x['order']) for x in causal}
    completeness=dict(expected_causal_order_records=len(expected_keys),missing_causal_keys=sorted(expected_keys-actual_keys),
                      unexpected_causal_keys=sorted(actual_keys-expected_keys),complete=expected_keys==actual_keys)
    calibration=audit_calibration(calibration_path,cases,natural_result)
    if plan and natural_result['channels'] and natural_result['channels']!=[plan['target']]+plan['controls']:
        raise ValueError('Natural channels differ from frozen target/control order')
    result=dict(schema_version=1,sources=dict(corpus=str(corpus_path),corpus_sha256=source_hash(corpus_path),
                 natural=str(natural_path),natural_sha256=source_hash(natural_path),causal=str(causal_path),causal_sha256=source_hash(causal_path)),
                counts=dict(corpus_cases=len(cases),natural_cases=len(natural),causal_order_records=len(causal),causal_case_condition_rows=len(causal_rows)),
                primary_split='validation',diagnostic_split='validation_lexical_diagnostic',
                calibration_only_descriptive=True,missing_natural_cases=sorted(set(cases)-{x['case_id'] for x in natural}),
                natural=natural_result,causal=causal_result,completeness=completeness,
                plan_sha256=source_hash(plan_path) if plan_path else None,calibration_audit=calibration,
                frozen_gates=frozen_gate_analysis(cases,natural_result,causal_rows),
                definitions=dict(carriermean='Mean pre scalar over[carrier_start,carrier_end).',lastmean='Pre scalar at final carrier token, not last generated token.',
                wholetextmax='Maximum signed pre scalar over logged story_start/end; fallback explicitly marked. Not maximum absolute value.',
                quantiles='Type7/linear quantiles of calibration carrier-token values only, pooled or language-specific. Never validation-derived.',
                direction='Positive/negative fractions use exact difference>0/<0; zero retained in denominator.',
                uncertainty='No p-values. Four core validation families; translations and category variants share their family.'),
                limitations=['Synthetic repeated templates and translation dependencies; not broad naturalistic pride detection.',
                'Pooled per-token quantiles weight longer carriers/languages by token count; language-specific quantiles are separate diagnostics.',
                'Range matching and residual-perturbation matching differ; this analyzer does not silently treat random controls as norm matched.',
                'Incomplete semantic order pairs and missing baselines are retained but excluded from paired summaries.',
                'No coefficient, event, family, language or condition selection from outcomes.'])
    return result,natural_rows,causal_rows


def write_report(path, result):
    counts=result['counts']
    lines=['#109 pride probe: CPU analysis','',
           'This report is descriptive. It does not establish a pride neuron or choose a new model intervention. Primary results use only the fresh validation split; calibration and lexical diagnostics remain separate.',
           '',f"Captured {counts['natural_cases']}/{counts['corpus_cases']} natural cases and {counts['causal_order_records']} causal order records. Missing/incomplete data and the literal frozen gate calculations are exposed in109_summary.json.",'',
           'Calibration quantiles are linear/type7 quantiles over calibration carrier tokens only. Lastmean is the final carrier-token scalar. Wholetextmax uses logged story boundaries, with explicit fallback warnings. Natural pre/post equality is an implementation check.', '',
           '| Primary own-accomplishment semantic condition | Cases | Families | Mean probability change | Mean log-odds change | Positive fraction | Negative fraction |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['causal']['summaries']:
        if (row['split'],row['task'],row['language'],row['category'],row['high_stratum'])!=('validation','semantic','all','own_accomplishment','all'):
            continue
        p,l=row['probability_change'],row['log_odds_change']
        lines.append('|%s|%s|%s|%.8g|%.8g|%.3f|%.3f|'%(row['condition'],row['n_cases'],row['n_families'],p['mean'],l['mean'],p['fraction_positive'],p['fraction_negative']))
    gates=result['frozen_gates']
    lines+=['', 'Frozen association gate: %s. Frozen semantic gate: %s. These are prospective descriptive thresholds, not statistical significance.'%(gates['association']['passes'],gates['semantic']['passes']),
            '', 'The primary semantic gate retains all12 own-accomplishment family-language cases. Conditional target suppression is compared with the prespecified downward proxy of each random all-carrier q10 clamp. These operations and residual perturbations are unmatched; no matched-operation claim is made.',
            '', 'Both semantic choice orders are averaged before condition-minus-baseline pairing. Each family cluster averages its related language/category rows; translations are not independent replications. Positive/negative fractions and individual effects are preserved, including zeros and failed predictions. High-event suppression is reported on all cases and separately on independently selected natural-high cases, without silently changing denominators.',
            '', 'Semantic and lexical probabilities are conditional on the two specified answer tokens. Their absolute total token mass, exp(logP(option1))+exp(logP(option2)), is reported by condition/language/category, alongside unconditional pride-option probability. Low mass limits free-response interpretation; the frozen endpoint is not changed because mass is low.',
            '', 'Previous published additive dose+0.4 corresponded to raw offset about19.6603, roughly51.2times the calibration carrier q90-q50 spread of about0.38423. That context/scope-specific additive displacement is not identical to a natural-range replacement and is not a standard-deviation claim.', '',
            'All random controls, lexical tasks, language/category strata, natural contrasts and q90-minus-q10 differences are in [summary](109_summary.json), [natural CSV](109_natural.csv) and [causal CSV](109_causal.csv). Suppression is expected to decrease pride odds; a lexical effect alone does not establish the nonlexical semantic claim. Apply the separately frozen gate, including every prescribed random control, without post-hoc selections.']
    path.write_text('\n'.join(lines)+'\n')


def self_test():
    assert percentile([0,10],.9)==9
    def case(i,split,language='en',family='v'):
        return dict(id=i,split=split,language=language,family=family,category='own_accomplishment',lexeme_present=False,expected_semantics={'own_accomplishment_satisfaction':True})
    cases={x['id']:x for x in [case('c','calibration',family='c'),case('v1','validation'),case('v2','validation','es')]}
    def record(i,v):
        values=[[99999]*4,[v]*4,[v+10]*4,[77777]*4]
        return dict(case_id=i,channels=[1911,1,2,3],token_ids=[1,2,3,4],prompt_tokens=4,carrier_start=1,carrier_end=3,story_start=1,story_end=3,pre=values,post=values)
    rows,summary=natural_analysis([record('c',0),record('v1',1000),record('v2',2000)],cases)
    q=next(x for x in summary['quantiles_calibration_only'] if x['channel']==1911 and x['language']=='all')
    assert q['q90']==9 and q['n']==2, 'Validation must never enter calibration'
    assert next(x for x in rows if x['case_id']=='v1' and x['is_target'])['high_token_count']==2
    target_window=next(x for x in rows if x['case_id']=='v1' and x['is_target'])
    assert target_window['lastmean']==1010 and target_window['wholetextmax']==1010, 'Exclude chat-prefix and post-carrier tokens'
    causal=[]
    for i in ('v1','v2'):
        for condition,probs,odds in [('baseline',[.1,.9],[0,4]),('target_carrier_q90',[.3,.9],[2,6])]:
            for order in (0,1):
                mass=.1 if order==0 else .2
                causal.append(dict(case_id=i,task='semantic',order=order,condition=condition,prob_pride=probs[order],log_odds_pride=odds[order],choice_correct=order==1,
                                   token_logprobs={'A':math.log(mass*.25),'B':math.log(mass*.75)}))
    averaged=average_causal_orders(causal,cases)
    target=next(x for x in averaged if x['case_id']=='v1' and x['condition']=='target_carrier_q90')
    assert math.isclose(target['delta_probability'],.1) and target['delta_log_odds']==2 and target['choice_accuracy']==.5
    assert math.isclose(target['option_token_mass_mean'],.15)
    assert math.isclose(target['unconditional_pride_option_probability'],.105), 'Average per-order products, not product of means'
    summaries=summarize_causal(averaged,rows)['summaries']
    group=next(x for x in summaries if x['condition']=='target_carrier_q90' and x['language']=='all' and x['category']=='all' and x['high_stratum']=='all')
    assert group['n_cases']==2 and group['n_families']==1, 'Translations must share family cluster'
    try:
        average_causal_orders(causal+[causal[0]],cases)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate order must fail')
    incomplete=average_causal_orders(causal[1:],cases)
    assert any(not x['orders_complete'] for x in incomplete)
    gate_cases={}
    gate_pairs=[]
    gate_rows=[]
    for i in range(12):
        identifier='gate_%s'%i
        gate_cases[identifier]=case(identifier,'validation',language=('en','es','zh')[i%3],family='family_%s'%(i//3))
        gate_pairs.append(dict(channel=1911,metric='carriermean',contrast='own_accomplishment_minus_neutral_achievement',difference=1 if i<9 else 0))
        for condition in ['target_carrier_q90','target_carrier_suppress_high']+['control_%s_carrier_q%s'%(k,q) for k in range(3) for q in (10,90)]:
            value=.1 if condition=='target_carrier_q90' else -.1 if condition=='target_carrier_suppress_high' else .01 if condition.endswith('q90') else -.01
            gate_rows.append(dict(case_id=identifier,task='semantic',condition=condition,paired_baseline_complete=True,
                delta_probability=value,delta_log_odds=value*3,language=gate_cases[identifier]['language'],family=gate_cases[identifier]['family'],
                natural_target_high_tokens=1,changed_positions_mean=1,option_token_mass_mean=.05))
    g=frozen_gate_analysis(gate_cases,dict(paired_validation=gate_pairs),gate_rows)
    assert g['association']['passes'] and g['semantic']['passes']
    for row in gate_rows:
        if row['condition']=='control_2_carrier_q90':
            row['delta_probability']=.2
    assert not frozen_gate_analysis(gate_cases,dict(paired_validation=gate_pairs),gate_rows)['semantic']['passes'], 'Every random control must be exceeded'
    for row in gate_rows:
        if row['condition']=='control_2_carrier_q90':
            row['delta_probability']=.01
        if row['condition']=='target_carrier_suppress_high':
            row['natural_target_high_tokens']=0
            row['changed_positions_mean']=0
    assert not frozen_gate_analysis(gate_cases,dict(paired_validation=gate_pairs),gate_rows)['semantic']['passes'], 'Absent events cannot establish suppression support'
    print('109 CPU self-tests passed: calibration isolation, exact windows, order/product averaging, baseline pairing, family clustering, duplicates/incomplete orders, all-control gate and absent-high-event rejection')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,default=ROOT/'109_PRIDE_FRESH_CASES.json')
    parser.add_argument('--natural',type=Path,default=ROOT/'results/109_natural.jsonl')
    parser.add_argument('--causal',type=Path,default=ROOT/'results/109_causal.jsonl')
    parser.add_argument('--plan',type=Path,default=ROOT/'109_PRIDE_PLAN.json')
    parser.add_argument('--calibration',type=Path,default=ROOT/'results/109_calibration.json')
    parser.add_argument('--completion',type=Path,default=ROOT/'results/109_completion.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--require-complete',action='store_true',help='Require all natural cases and complete paired causal orders; expected absent conditions still require a frozen-plan count check')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not args.completion.exists():
        raise SystemExit('Wait for109_completion.json before analyzing experimental results; no outputs written')
    result,natural,causal=analyze(args.cases,args.natural,args.causal,args.plan,args.calibration)
    result['completion_metadata']=json.loads(args.completion.read_text())
    result['completion_sha256']=source_hash(args.completion)
    if args.require_complete and (result['missing_natural_cases'] or not causal or not result['completeness']['complete'] or result['causal']['incomplete_semantic_orders'] or result['causal']['missing_baselines']):
        raise SystemExit('Incomplete natural/causal data; no outputs written under --require-complete')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'109_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(args.output_dir/'109_natural.csv',natural)
    write_csv(args.output_dir/'109_causal.csv',causal)
    write_report(args.output_dir/'109_PRIDE_REPORT.md',result)
    print(json.dumps(result['counts']))


if __name__=='__main__':
    main()
