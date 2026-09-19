# Specificity / CAR integration

Branch: `specificity-car-test`, based on `event-study`. Neither source branch is modified. No merge into main.

Read `REPORT.md` for findings and limitations. Root deliverables:

- `CEO_Letter_Specificity_CAR_Test.ipynb`: executed analysis and audit.
- `specificity_car_US_merged.csv`: all 84 event rows, source measure, controls, eligibility and exclusion reasons.
- `specificity_car_regression_results.csv`: original and standardized specificity coefficients, HC3 and firm-clustered SEs, p-values and 95% confidence intervals for every requested model.

`inputs/` contains unchanged snapshots with SHA-256 hashes in `manifest.json`. The final specificity measure comes from the committed regression panel and is checked against the original narrative export. `results/` contains all unmatched keys, exclusions, descriptives, correlations, influence diagnostics, wild-cluster tests, independent CAR reproduction checks, raw-price coverage, and the plotted PNG/PDF.

## Reproduce

Use Python 3.12 and a virtual environment. Install `requirements.txt`, then run from repository root:

```sh
python specificity_car_test/analysis.py
```

To rebuild and execute the notebook with a project-local kernel:

```sh
python specificity_car_test/build_notebook.py
python specificity_car_test/execute_notebook.py
```

No API keys, Google Drive access, downloads or classifier calls are needed to reproduce the frozen analysis. `snapshot_inputs.py` records the initial local capture; it is not part of routine reproduction and should not be rerun unless intentionally updating sources and hashes. Source-branch hashes are in the manifest. The notebook only imports the new integration module.

## Prespecified interpretation

Signed CAR concerns direction; absolute CAR concerns magnitude. All outcomes are decimal returns. Multiply coefficients and intervals by 100 for percentage points. No causal interpretation. Year FE use letter fiscal year; COVID exclusions use publication date.

The 84-event panel is complete, but narrative data match only 65 events, with three undefined specificity values. Pooled N=62 (seven firms); main FE N=61 (six firms) after explicit removal of a MetLife singleton. This is a sample limitation, not a missing-price issue.

## Sentiment extension

`CEO_Letter_Specificity_Sentiment_CAR_Test.ipynb` and `REPORT_SENTIMENT.md` extend the analysis with overall FinBERT sentiment, ESG-specific sentiment, joint models, interactions, and `[0,+5]`/`[0,+20]` CAR windows. Root outputs are `specificity_sentiment_car_US_merged.csv` and `specificity_sentiment_car_regression_results.csv`; detailed tables and diagnostics are under `results_sentiment/`.

HC3 is conventional heteroskedasticity-robust inference, not protection against serial or cross-sectional dependence. Few-cluster sensitivity estimates and coarse wild-cluster p-values must be interpreted cautiously. No automatic winsorization, imputation, or variable selection is used.
