"""Prespecified specificity/CAR associations from frozen, existing outputs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import warnings

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'specificity_car_test'
INPUT = BASE / 'inputs'
OUT = BASE / 'results'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime' / 'matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime' / 'numba'))
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wildboottest.wildboottest import wildboottest

MAP = {
    'AIG': 'American International Group (AIG)', 'Chubb': 'Chubb',
    'MetLife': 'MetLife, Inc.', 'Prudentials': 'Prudential Financial, Inc.',
    'Allstate': 'The Allstate Corporation', 'Progressive': 'The Progressive Corporation',
    'Travelers': 'The Travelers Companies, Inc.',
}
TICKERS = dict(zip(MAP.values(), ['AIG', 'CB', 'MET', 'PRU', 'ALL', 'PGR', 'TRV']))
OUTCOMES = ['CAR[-1,+1]', '|CAR|[-1,+1]', 'CAR[-2,+2]', 'CAR[0,+1]']
KEYS = ['Company', 'Fiscal Year']


def save(frame, name):
    frame.to_csv(OUT / name, index=False)


def load_and_audit():
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        assert hashlib.sha256((INPUT / name).read_bytes()).hexdigest() == info['sha256'], name
    event = pd.read_csv(INPUT / 'ceo_letter_event_study_panel_US.csv')
    spec = pd.read_csv(INPUT / 'panel_reg_export.csv').rename(columns={'Year': 'Fiscal Year'})
    narrative = pd.read_csv(INPUT / 'df_ar_cluster.csv').rename(
        columns={'Company Name': 'Company', 'Year': 'Fiscal Year'})
    narrative.Company = narrative.Company.replace({'MET': 'MetLife', 'Generali': 'Assicurazioni Generali',
        'PIC': 'People Insurance China', 'Prudential Financials': 'Prudentials'})
    duplicates = pd.concat([d.loc[d.duplicated(KEYS, keep=False), KEYS].assign(input=name)
                            for name, d in [('event', event), ('specificity', spec), ('narrative', narrative)]])
    save(duplicates, 'duplicate_keys.csv')
    assert duplicates.empty, 'Duplicate company-year keys: see duplicate_keys.csv'
    expected = pd.MultiIndex.from_product([list(MAP.values()), range(2012, 2024)], names=KEYS)
    assert len(event) == 84 and set(map(tuple, event[KEYS].values)) == set(expected)
    assert np.isfinite(event[OUTCOMES + ['Alpha', 'Beta', 'Estimation N']].to_numpy()).all()
    assert (event['Estimation N'] >= 100).all()
    np.testing.assert_allclose(event[OUTCOMES[1]], event[OUTCOMES[0]].abs(), atol=1e-14)
    comparison = spec.merge(narrative, on=KEYS, validate='one_to_one', suffixes=('_branch', '_drive'))
    assert len(comparison) == len(spec) == len(narrative)
    for col in ['Specificity_mean', 'total_sent', 'ESGSentiment_Corporate']:
        np.testing.assert_allclose(comparison[col+'_branch'], comparison[col+'_drive'],
                                   equal_nan=True, atol=1e-12)
    spec['Specificity source company'] = spec.Company
    spec.Company = spec.Company.replace(MAP)
    spec = spec[KEYS + ['Specificity source company', 'Specificity_mean', 'total_sent',
                       'ESGSentiment_Corporate', 'Corporate_ESG_share', 'Verified_ESG_share']]
    joined = event.merge(spec, on=KEYS, how='outer', indicator=True, validate='one_to_one')
    unmatched = joined.loc[joined._merge.ne('both'), KEYS + ['_merge']].copy()
    unmatched['reason'] = np.where(unmatched._merge.eq('left_only'),
        'No narrative firm-year output', 'Specificity firm-year outside seven-firm event-study scope')
    save(unmatched, 'unmatched_company_years.csv')
    merged = joined.loc[joined._merge.ne('right_only')].copy()
    merged['exclusion_reason'] = np.select(
        [merged._merge.eq('left_only'), merged.Specificity_mean.isna()],
        ['No narrative firm-year output', 'Specificity undefined: no qualifying corporate ESG sentences'],
        default='')
    merged['regression_eligible'] = merged.exclusion_reason.eq('')
    merged['Event Date (adjusted)'] = pd.to_datetime(merged['Event Date (adjusted)'])
    merged['COVID_period'] = merged['Event Date (adjusted)'].between('2020-03-01', '2020-04-30')
    sample = merged.loc[merged.regression_eligible].copy()
    mean, sd = sample.Specificity_mean.mean(), sample.Specificity_mean.std(ddof=1)
    assert sd > 0
    merged['Specificity_z'] = (merged.Specificity_mean - mean) / sd
    merged['log_total_sent'] = np.log(merged.total_sent)
    merged = merged.sort_values(KEYS).reset_index(drop=True)
    merged.to_csv(ROOT / 'specificity_car_US_merged.csv', index=False)
    save(merged.loc[~merged.regression_eligible, KEYS + ['exclusion_reason']], 'regression_exclusions.csv')
    sample = merged.loc[merged.regression_eligible].copy()
    sample['firm'] = sample.Company
    sample['year'] = sample['Fiscal Year'].astype(int)
    summary = {'event_input_rows': len(event), 'specificity_input_rows': len(spec),
        'narrative_input_rows': len(narrative), 'matched': int(joined._merge.eq('both').sum()),
        'event_only': int(joined._merge.eq('left_only').sum()),
        'specificity_only_outside_scope': int(joined._merge.eq('right_only').sum()),
        'duplicates': len(duplicates), 'undefined_specificity_in_matched': int(
            (merged._merge.eq('both') & merged.Specificity_mean.isna()).sum()),
        'pooled_N': len(sample), 'specificity_mean': mean, 'specificity_sd_ddof1': sd,
        'specificity_definition': 'Mean tier among Corporate ESG sentences with tier >= 1; E and S/G combined. No lag.',
        'branch_drive_specificity_match': True}
    (OUT / 'merge_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('MERGE AUDIT (before estimation)\n', json.dumps(summary, indent=2))
    print('\nEvery excluded event\n', merged.loc[~merged.regression_eligible, KEYS+['exclusion_reason']].to_string(index=False))
    print('\nSpecificity summary\n', sample.Specificity_mean.describe().to_string())
    return merged, sample, summary


def audit_event_prices(event):
    """Independent re-computation for checking only; never replaces saved CARs."""
    closes, coverage = {}, []
    for ticker in list(TICKERS.values()) + ['URTH']:
        raw = pd.read_csv(INPUT / 'raw_prices' / f'{ticker}.csv', header=[0, 1], index_col=0)
        raw.index = pd.to_datetime(raw.index)
        assert raw.index.is_unique and raw.index.is_monotonic_increasing
        closes[ticker] = raw['Close'].iloc[:, 0]
        coverage.append({'ticker': ticker, 'first_date': str(raw.index.min().date()),
                         'last_date': str(raw.index.max().date()), 'rows': len(raw)})
    save(pd.DataFrame(coverage), 'price_coverage.csv')
    calendar = closes['URTH'].dropna().index
    returns = {t: c.reindex(calendar).pct_change(fill_method=None) for t, c in closes.items()}
    dates = pd.read_excel(ROOT / 'Combined_Insurer_Publication_Dates.xlsx')
    dates = dates.loc[dates.Region.eq('United States'), KEYS+['Publication Date']]
    checked = event.merge(dates, on=KEYS, validate='one_to_one')
    checks = []
    for _, row in checked.iterrows():
        original = pd.Timestamp(row['Publication Date'])
        pos = calendar.searchsorted(original)
        adjusted = calendar[pos]
        assert adjusted == pd.Timestamp(row['Event Date (adjusted)'])
        est_dates = calendar[max(0, pos-250):pos-29]
        r = returns[TICKERS[row.Company]]
        b = returns['URTH']
        est = pd.concat([r.loc[est_dates], b.loc[est_dates]], axis=1).dropna()
        y, x = est.iloc[:, 0].to_numpy(), est.iloc[:, 1].to_numpy()
        beta = np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
        alpha = y.mean()-beta*x.mean()
        assert len(est) == row['Estimation N']
        np.testing.assert_allclose([alpha, beta], [row.Alpha, row.Beta], atol=1e-10)
        errors = []
        for col, lo, hi in [(OUTCOMES[0], -1, 1), (OUTCOMES[2], -2, 2), (OUTCOMES[3], 0, 1)]:
            idx = calendar[pos+lo:pos+hi+1]
            assert len(idx) == hi-lo+1
            ar = r.loc[idx]-(alpha+beta*b.loc[idx])
            assert ar.notna().all(), f'Incomplete event window: {row.Company} {row["Fiscal Year"]}'
            error = float(ar.sum()-row[col])
            assert abs(error) < 1e-10, (row.Company, row['Fiscal Year'], col, error)
            errors.append(abs(error))
        checks.append({**row[KEYS].to_dict(), 'original_event_date': str(original.date()),
            'adjusted_event_date': str(adjusted.date()), 'shifted': adjusted != original,
            'estimation_N': len(est), 'max_CAR_absolute_error': max(errors)})
    result = pd.DataFrame(checks)
    save(result, 'event_reproduction_checks.csv')
    print(f'Independently reproduced {len(result)} saved event rows; max CAR error '
          f'{result.max_CAR_absolute_error.max():.3g}; shifted dates: {result.shifted.sum()}')
    return result


def descriptives(merged, sample):
    summaries = []
    for name, d in [('all_events', merged), ('regression_eligible', sample)]:
        x = d[OUTCOMES+['Specificity_mean', 'total_sent', 'ESGSentiment_Corporate']].describe().T
        summaries.append(x.reset_index(names='variable').assign(sample=name))
    save(pd.concat(summaries), 'descriptive_statistics.csv')
    correlations = []
    for col in OUTCOMES:
        for method, func in [('Pearson', stats.pearsonr), ('Spearman', stats.spearmanr)]:
            r, p = func(sample.Specificity_mean, sample[col])
            correlations.append({'outcome': col, 'method': method, 'N': len(sample),
                                 'correlation': r, 'p_value_iid_descriptive': p})
    corr = pd.DataFrame(correlations)
    save(corr, 'correlations.csv')
    save(merged.sort_values(OUTCOMES[1], ascending=False)[KEYS+OUTCOMES+['regression_eligible']],
         'largest_absolute_CAR.csv')
    print('\nDescriptive correlations (p-values assume independent observations)\n', corr.to_string(index=False))
    return corr


def remove_singletons(d, spec):
    d = d.copy()
    removed = []
    effects = [] if spec == 'pooled' else ['firm'] + (['year'] if spec == 'firm_year_FE' else [])
    while effects:
        bad = pd.Series(False, index=d.index)
        for effect in effects:
            bad |= d.groupby(effect)[effect].transform('size').eq(1)
        if not bad.any():
            break
        removed.extend(d.loc[bad, KEYS].to_dict('records'))
        d = d.loc[~bad].copy()
    return d, removed


def fit(d, outcome, spec, predictor, controls=False):
    d, removed = remove_singletons(d, spec)
    d = d.copy()
    d['outcome'] = d[outcome]
    formula = 'outcome ~ ' + predictor
    if spec != 'pooled':
        formula += ' + C(firm)'
    if spec == 'firm_year_FE':
        formula += ' + C(year)'
    if controls:
        formula += ' + log_total_sent + ESGSentiment_Corporate'
    model = smf.ols(formula, data=d, missing='raise')
    assert np.linalg.matrix_rank(model.exog) == model.exog.shape[1], 'Rank deficient model'
    result = model.fit()
    assert result.get_influence().hat_matrix_diag.max() < 1-1e-10
    return d, result, removed


def regressions(sample):
    rows, influences, exclusions, wild_rows = [], [], [], []
    fitted = {}

    def run(outcome, spec, scenario='main', d=None, controls=False):
        d = sample if d is None else d
        raw_result = None
        for predictor in ['Specificity_mean', 'Specificity_z']:
            used, result, removed = fit(d, outcome, spec, predictor, controls)
            if predictor == 'Specificity_mean':
                exclusions.extend([{**r, 'outcome': outcome, 'specification': spec,
                    'scenario': scenario, 'reason': 'Singleton fixed-effect group; no within-group slope information'}
                    for r in removed])
                raw_result = result
            else:
                np.testing.assert_allclose(result.fittedvalues, raw_result.fittedvalues, atol=1e-11)
                np.testing.assert_allclose(result.params[predictor],
                    raw_result.params['Specificity_mean'] * sample.Specificity_mean.std(ddof=1), atol=1e-11)
            j = result.model.exog_names.index(predictor)
            group = pd.Categorical(used.firm).codes
            robust_results = {
                'HC3': result.get_robustcov_results(cov_type='HC3', use_t=True),
                'firm_cluster_CR1': result.get_robustcov_results(cov_type='cluster', groups=group,
                    use_correction=True, df_correction=True, use_t=True),
            }
            # Independent HC3 sandwich cross-check.
            X = result.model.exog
            bread = np.linalg.inv(X.T @ X)
            h = np.sum((X @ bread) * X, axis=1)
            xe = X * (np.asarray(result.resid)/(1-h))[:, None]
            manual_se = np.sqrt((bread @ (xe.T @ xe) @ bread)[j, j])
            np.testing.assert_allclose(manual_se, robust_results['HC3'].bse[j], rtol=1e-8)
            # Frisch-Waugh-Lovell: independently partial out all nuisance regressors.
            nuisance = np.delete(X, j, axis=1)
            xr = X[:, j] - nuisance @ np.linalg.lstsq(nuisance, X[:, j], rcond=None)[0]
            yr = result.model.endog - nuisance @ np.linalg.lstsq(nuisance, result.model.endog, rcond=None)[0]
            np.testing.assert_allclose(xr @ yr / (xr @ xr), result.params[predictor], atol=1e-11)
            for cov, robust in robust_results.items():
                ci = robust.conf_int()[j]
                rows.append({'outcome': outcome, 'specification': spec, 'scenario': scenario,
                    'predictor': predictor, 'covariance': cov, 'N': len(used),
                    'firms': used.firm.nunique(), 'years': used.year.nunique(),
                    'coefficient': robust.params[j], 'SE': robust.bse[j], 'p_value': robust.pvalues[j],
                    'CI95_low': ci[0], 'CI95_high': ci[1], 'R_squared': result.rsquared,
                    'inference_df': result.df_resid if cov == 'HC3' else used.firm.nunique()-1,
                    'controls': 'log_total_sent + ESGSentiment_Corporate' if controls else '',
                    'standardization': 'fixed pooled eligible sample; ddof=1' if predictor.endswith('_z') else 'none'})
            if predictor == 'Specificity_mean':
                with warnings.catch_warnings(record=True) as captured:
                    boot = wildboottest(result.model, B=9999, cluster=group,
                        param=predictor, weights_type='rademacher', impose_null=True,
                        bootstrap_type='11', seed=20260919, parallel=False, show=False)
                wild_rows.append({'outcome': outcome, 'specification': spec, 'scenario': scenario,
                    'N': len(used), 'firms': used.firm.nunique(),
                    'p_value': float(boot['p-value'].iloc[0]),
                    'method': 'Null-imposed WCR11; Rademacher; full enumeration',
                    'weight_patterns': 2**used.firm.nunique(),
                    'warnings': '; '.join(str(w.message) for w in captured)})
        fitted[(outcome, spec, scenario)] = (used, raw_result)
        return used, raw_result

    for outcome in OUTCOMES[:2]:
        for spec in ['pooled', 'firm_FE', 'firm_year_FE']:
            used, result = run(outcome, spec)
            inf = result.get_influence()
            influence = used[KEYS].copy()
            influence['outcome'] = outcome
            influence['specification'] = spec
            influence['residual'] = result.resid.to_numpy()
            influence['studentized_internal'] = inf.resid_studentized_internal
            influence['studentized_external'] = inf.resid_studentized_external
            influence['cooks_distance'] = inf.cooks_distance[0]
            influence['leverage'] = inf.hat_matrix_diag
            influence['specificity_dfbeta'] = inf.dfbetas[:, result.model.exog_names.index('Specificity_mean')]
            influences.append(influence)
    influence = pd.concat(influences)
    save(influence, 'influence_diagnostics.csv')
    for outcome in OUTCOMES[2:]:
        run(outcome, 'firm_year_FE')
    for outcome in OUTCOMES[:2]:
        diag = influence.loc[influence.outcome.eq(outcome) & influence.specification.eq('firm_year_FE')]
        worst = diag.loc[diag.cooks_distance.idxmax()]
        drop = sample.Company.eq(worst.Company) & sample['Fiscal Year'].eq(worst['Fiscal Year'])
        run(outcome, 'firm_year_FE', 'exclude_most_influential', sample.loc[~drop])
        run(outcome, 'firm_year_FE', 'exclude_March_April_2020', sample.loc[~sample.COVID_period])
        assert sample[['log_total_sent', 'ESGSentiment_Corporate']].notna().all().all()
        run(outcome, 'firm_year_FE', 'existing_text_controls', controls=True)
    results = pd.DataFrame(rows)
    assert np.isfinite(results[['coefficient', 'SE', 'p_value', 'CI95_low', 'CI95_high']].to_numpy()).all()
    results.to_csv(ROOT / 'specificity_car_regression_results.csv', index=False)
    save(pd.DataFrame(exclusions), 'model_singleton_exclusions.csv')
    save(pd.DataFrame(wild_rows), 'wild_cluster_bootstrap.csv')
    save(sample.loc[sample.COVID_period, KEYS+['Event Date (adjusted)']+OUTCOMES], 'COVID_sensitivity_exclusions.csv')
    return results, influence, pd.DataFrame(wild_rows), fitted


def plot(sample, fitted):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    colors = dict(zip(sorted(sample.firm.unique()), plt.get_cmap('tab10').colors))
    for ax, outcome, title in zip(axes, OUTCOMES[:2], ['Signed reaction', 'Reaction magnitude']):
        used, result = fitted[(outcome, 'pooled', 'main')]
        for firm, group in sample.groupby('firm'):
            ax.scatter(group.Specificity_mean, 100*group[outcome], s=32, alpha=.8,
                       color=colors[firm], label=next(k for k,v in MAP.items() if v == firm))
        x = np.linspace(sample.Specificity_mean.min(), sample.Specificity_mean.max(), 100)
        ax.plot(x, 100*(result.params.Intercept+result.params.Specificity_mean*x), color='#26374a', lw=1.5)
        inf = result.get_influence()
        positions = set(np.argsort(inf.cooks_distance[0])[-2:]) | set(np.argsort(np.abs(inf.resid_studentized_external))[-2:])
        for n, pos in enumerate(sorted(positions)):
            row = used.iloc[pos]
            label = next(k for k,v in MAP.items() if v == row.Company) + f' {int(row["Fiscal Year"])}'
            ax.annotate(label, (row.Specificity_mean, 100*row[outcome]),
                        xytext=(8, 10 if n % 2 == 0 else -17), textcoords='offset points', fontsize=9,
                        arrowprops={'arrowstyle': '-', 'color': '#667788', 'lw': .6})
        ax.set(title=title, xlabel='Corporate ESG specificity (mean tier)', ylabel=f'{outcome} (percentage points)')
        ax.axhline(0, color='#999999', linewidth=.5)
        ax.grid(alpha=.16)
    axes[0].legend(fontsize=8, loc='lower right', frameon=False)
    fig.suptitle(f'CEO-letter specificity and abnormal returns | N = {len(sample)}\nDescriptive pooled fits; associations are not causal', fontsize=13)
    fig.savefig(OUT/'specificity_CAR_scatterplots.png', dpi=180)
    fig.savefig(OUT/'specificity_CAR_scatterplots.pdf')
    plt.close(fig)


def report(audit, correlations, results, influence, wild, event_checks):
    main = results.loc[results.scenario.eq('main') & results.predictor.eq('Specificity_mean') & results.covariance.eq('HC3')]
    sensitivity = results.loc[results.scenario.ne('main') & results.predictor.eq('Specificity_mean') & results.covariance.eq('HC3')]
    top = influence.loc[influence.specification.eq('firm_year_FE')].sort_values('cooks_distance', ascending=False).groupby('outcome').head(5)
    text = '''# Specificity and abnormal equity returns — US insurers

This analysis uses existing outputs; it does not rerun the classifier or modify event-study construction.
The preferred specification was fixed before estimation: firm and fiscal-year fixed effects. Publication dates determine the COVID exclusion, while fiscal years define year fixed effects.

## Measurement and provenance

`Specificity_mean` is the mean ordinal specificity tier (1 vague, 2 named/unquantified, 3 quantified) among **Corporate** ESG sentences with tier >= 1, combining E and S/G classifications. Contextual sentences and tier-0 false positives are excluded. Undefined means remain missing. This is the existing measure, not the distinct `Verified_ESG_share` or the all-relevance measure.

Source: `specificity-pipeline-3d-regression` commit `2428ae52895ffba97d280f5d776c9e88b376bfc9`, `null_check/data/panel_reg_export.csv`. All 238 specificity values (including missingness), sentence counts, and corporate sentiment agree with the original Drive `df_ar_cluster.csv`. The latter also confirms narrative coverage, so missing years are not manufactured by this integration. No valuation-model lag is carried over: each letter's own fiscal year is joined to its publication event.

CAR source: the supplied Drive `ceo_letter_event_study_panel_US.csv`, snapshotted without alteration. Event-study code at commit `6693c1d60844c31c76dc6a129cb40863dc50b05c` sets `PRICE_END = "2024-07-31"`; every saved stock series and URTH ends 2024-07-30. Raw prices independently reproduce all 84 panels' alpha, beta, estimation counts and three CAR windows within 1e-10. This checks calculations against saved inputs, not the historical accuracy of publication dates or vendor prices.

## Merge and sample

'''
    text += f"Event rows: {audit['event_input_rows']}; specificity rows: {audit['specificity_input_rows']}; matches: {audit['matched']}; event-only rows: {audit['event_only']}; specificity-only rows: {audit['specificity_only_outside_scope']} (outside the seven target firms); duplicate keys: {audit['duplicates']}. Three matched rows have undefined specificity. Pooled regression N = {audit['pooled_N']}. All 84 events remain in the merged CSV with eligibility and exclusion reasons. Full unmatched keys and exclusions are saved separately.\n\n"
    text += 'MetLife contributes only FY2012 after missing specificity is excluded. This singleton is removed explicitly from firm-FE and firm/year-FE models: it contributes no within-firm slope information and has leverage one. Main FE N = 61 and **six**, not seven, firm clusters. Model-specific removals are saved.\n\n'
    text += f"Specificity mean = {audit['specificity_mean']:.6f}, sample SD = {audit['specificity_sd_ddof1']:.6f}. Standardized versions use this fixed pooled-sample mean and SD (ddof=1), also in sensitivities. CARs are decimal returns; multiply coefficients by 100 for percentage points.\n\n"
    text += '## Descriptive correlations\n\n'+correlations.to_markdown(index=False)+'\n\nCorrelation p-values assume independent observations and are descriptive, not panel-robust inference.\n\n'
    cols = ['outcome','specification','N','firms','coefficient','SE','p_value','CI95_low','CI95_high']
    text += '## Main models: HC3 inference\n\n'+main[cols].to_markdown(index=False, floatfmt='.6f')+'\n\n'
    text += '''Neither primary outcome shows a statistically detectable association in pooled, firm-FE, or firm/year-FE models. This is not evidence that the true association is zero: the confidence intervals are wide.

In the preferred model, the signed-CAR coefficient is −0.021111 (HC3 SE 0.021455, p=0.330649, 95% CI [−0.064380, 0.022158]). The absolute-CAR coefficient is +0.010716 (SE 0.015366, p=0.489344, CI [−0.020274, 0.041705]). A one-SD specificity increase corresponds to approximately −0.485 percentage points of signed CAR and +0.246 percentage points of absolute CAR; both intervals include zero. Magnitude's coefficient changes from slightly negative pooled to positive with firm effects, so its sign is not stable across specifications.

Alternative signed windows remain negative and nonsignificant under HC3. The five-day window has a coarse wild-cluster p-value of 0.0625, contrasting with HC3 p=0.2371; report this sensitivity without treating it as reliable confirmatory evidence.

AIG FY2019 is the most influential observation in both preferred models. Removing it reduces the signed coefficient to −0.005833 (p=0.6789) and changes the magnitude coefficient to −0.000786 (p=0.9372). Removing all March–April 2020 events gives −0.009814 (p=0.4929) and −0.000239 (p=0.9813), respectively. The magnitude estimate's positive sign and much of its size depend on these observations; the absence of conventional statistical evidence persists. The signed estimate remains negative but is materially attenuated. Existing length/tone controls likewise do not produce a statistically detectable association.

'''
    text += '## Prespecified sensitivities: HC3 inference\n\n'+sensitivity[['outcome','scenario']+cols[2:]].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text += '## Wild-cluster inference\n\n'+wild.to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text += 'HC3 uses Student-t reference with OLS residual degrees of freedom. Firm-clustered CR1 uses the finite-sample correction and t(G−1); it is a sensitivity check, with only seven pooled clusters and six main FE clusters. HC3 does not address within-firm dependence or contemporaneous cross-firm dependence. Wild-cluster tests impose a zero-specificity-coefficient null, use WCR11 and enumerate Rademacher weights (128 patterns for seven firms; 64 for six); their p-values are coarse, and they do not eliminate few-cluster limitations. Standardized models have the same null tests. No bootstrap confidence intervals are claimed.\n\n'
    text += '## Influence\n\n'+top[KEYS+['outcome','studentized_external','cooks_distance','leverage','specificity_dfbeta']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text += 'Influence diagnostics use conventional OLS studentization, separately from robust inference. The exclusion sensitivity removes the observation with the largest Cook’s distance in each outcome’s preferred model. COVID sensitivity separately removes publication events from March 1 through April 30, 2020 inclusive. No observation is winsorized or removed from the primary pooled sample for being an outlier. The control sensitivity adds existing log sentence count and corporate ESG sentiment; it adjusts jointly for document length and tone. Sentiment is itself part of narrative content, so this is a sensitivity specification, not an identified causal adjustment.\n\n'
    text += '## Data-quality limitations\n\n'
    text += f"- {event_checks.shifted.sum()} publication dates shift to the next observed URTH session. All tested event windows have complete returns.\n"
    text += '- URTH data begin January 12, 2012, despite the January 2011 requested start. Paired estimation counts were checked directly.\n'
    text += '- Publication dates are taken as supplied; their source documents and time-of-day have not been independently authenticated. Annual-report announcements can include other value-relevant news.\n'
    text += '- Chubb calculations reproduce using CB only. ACE became Chubb Limited and its shares began trading as CB on January 15, 2016, per the [issuer announcement](https://investors.chubb.com/News--Events/news/news-details/2016/ACE-Completes-Acquisition-of-Chubb-Adopts-Chubb-Name-and-Launches-New-Chubb-Brand-01-14-2016/default.aspx). This establishes legal/ticker continuity, but does not independently authenticate every pre-2016 Yahoo historical quote. The upstream notebook itself asks for a historical-price spot-check; that limitation remains.\n'
    text += '- The upstream event code drops missing firm prices before calculating returns and sums abnormal returns with default missing-value handling. The independent audit checks a common benchmark calendar, no forward filling, and complete windows; no effect on these supplied CARs was found.\n'
    text += '- The regression sample is small and unbalanced, with limited generalizability. Multiple outcomes/specifications are reported as prespecified; no claim rests on a selected minimum p-value. No multiplicity-adjusted confirmatory claim is made.\n\n'
    text += 'Interpret results only as associations between narrative specificity and the direction or magnitude of abnormal equity returns surrounding publication. Signed CAR and absolute CAR answer different questions.\n'
    (BASE/'REPORT.md').write_text(text)
    versions = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','matplotlib','wildboottest','nbformat','nbclient']}
    (BASE/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')


def main():
    merged, sample, audit = load_and_audit()
    event_checks = audit_event_prices(merged)
    corr = descriptives(merged, sample)
    results, influence, wild, fitted = regressions(sample)
    plot(sample, fitted)
    report(audit, corr, results, influence, wild, event_checks)
    print('\nCompleted. See specificity_car_test/REPORT.md and root output CSVs.')
    return merged, results


if __name__ == '__main__':
    main()
