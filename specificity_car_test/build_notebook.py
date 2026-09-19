"""Build the readable, executable integration notebook, without upstream imports."""
from pathlib import Path
import json
import sys
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {'display_name': 'Specificity CAR analysis', 'language': 'python', 'name': 'specificity-car'}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md('''# CEO-letter specificity and abnormal equity returns — US insurers

This integration uses frozen **existing** outputs from the specificity and event-study work. It does not rerun the classifier, construct replacement CARs, or modify either upstream notebook/branch.

Primary outcomes: signed CAR[-1,+1] and absolute CAR[-1,+1]. Prespecified models: pooled, firm FE, firm + fiscal-year FE. Preferred specification: firm + fiscal-year FE. Associations are not causal.

The executable implementation lives in `specificity_car_test/analysis.py`; inputs, hashes, diagnostics and a full report are saved alongside it. Run from the repository root after installing `specificity_car_test/requirements.txt`.'''),
    code('''from pathlib import Path
import sys
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
assert (ROOT / 'specificity_car_test' / 'analysis.py').exists(), 'Run from repository root'
sys.path.insert(0, str(ROOT))
from specificity_car_test.analysis import (
    load_and_audit, audit_event_prices, descriptives, regressions, plot, report, OUT, BASE
)
import pandas as pd'''),
    md('''## 1. Source definition and merge audit — before estimation

`Specificity_mean` is the existing mean tier across **Corporate** ESG sentences with tier ≥1, combining E and S/G. It excludes contextual and false-positive sentences. Empty populations remain missing. We use the contemporaneous letter fiscal year, without the lag used in earlier valuation regressions.

All input duplicates and unmatched keys are reported. The merged export retains all 84 event rows. The source narrative export is checked against the committed 238-row specificity panel before use.'''),
    code('merged, sample, audit = load_and_audit()\ndisplay(pd.read_csv(OUT / "unmatched_company_years.csv"))'),
    md('## 2. Independently verify saved CARs and date coverage\n\nReproduce every saved row using the archived adjusted prices, a common URTH calendar, complete event windows and an independent market-model calculation. Saved CARs are never replaced.'),
    code('event_checks = audit_event_prices(merged)\ndisplay(pd.read_csv(OUT / "price_coverage.csv"))\ndisplay(event_checks)'),
    md('## 3. Describe associations\n\nPearson/Spearman correlations are descriptive. Their ordinary p-values do not account for the panel structure.'),
    code('correlations = descriptives(merged, sample)\ndisplay(pd.read_csv(OUT / "descriptive_statistics.csv"))'),
    md('''## 4. Prespecified regressions, influence, and sensitivities

HC3 is the main conventional inference (t reference with residual degrees of freedom). Firm-clustered CR1 uses a finite-sample correction and t(G−1). Null-imposed WCR11 wild-cluster tests enumerate all Rademacher patterns. Few-cluster uncertainty remains.

MetLife has only one eligible year: retain it in pooled models, explicitly exclude it as a singleton in firm-FE models, and report the resulting six clusters. No outliers are removed from main models. Diagnostics include studentized residuals, Cook's distance, leverage and specificity DFBETAs.

Separate preferred-model sensitivities remove (1) the single highest-Cook's-distance row and (2) events published March 1–April 30, 2020. One optional controls model adds existing log sentence count and corporate ESG sentiment. Standardization uses the fixed 62-row pooled sample SD (ddof=1).'''),
    code('''results, influence, wild, fitted = regressions(sample)
display(results[(results.predictor == 'Specificity_mean') & (results.covariance == 'HC3')])
display(wild)
display(pd.read_csv(OUT / 'model_singleton_exclusions.csv'))
display(influence[influence.specification.eq('firm_year_FE')].sort_values('cooks_distance', ascending=False).groupby('outcome').head(5))'''),
    md('## 5. Visual inspection and interpretation'),
    code('''plot(sample, fitted)
display(Image(filename=str(OUT / 'specificity_CAR_scatterplots.png')))
report(audit, correlations, results, influence, wild, event_checks)
display(Markdown((BASE / 'REPORT.md').read_text()))'''),
    md('''## 6. Completion checks

Every regression is checked against an independent partial-regression coefficient and HC3 sandwich calculation. Standardized and original fits must have equal predictions. No branch is merged into main.'''),
    code('''assert len(merged) == 84
assert merged.regression_eligible.sum() == 62
assert len(event_checks) == 84
assert event_checks.max_CAR_absolute_error.max() < 1e-10
assert len(results) == 56  # 14 specifications/scenarios × 2 scales × 2 covariance estimators
assert len(wild) == 14
assert results[['coefficient','SE','p_value','CI95_low','CI95_high']].notna().all().all()
print('Validated outputs: specificity_car_US_merged.csv; specificity_car_regression_results.csv')'''),
]
nbf.validate(nb)
nbf.write(nb, root / 'CEO_Letter_Specificity_CAR_Test.ipynb')
kernel = root / '.runtime' / 'jupyter' / 'kernels' / 'specificity-car'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Specificity CAR analysis','language':'python'}))
print('Created notebook and project-local kernel specification.')
