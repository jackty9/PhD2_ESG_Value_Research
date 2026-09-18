# null_check

Independent null-result check for Table 5a (`PHDp2_FinancialData_Regression.ipynb`).
Does not modify the notebook, its data pipeline, or any export file — this is a
standalone, from-source re-derivation.

## Contents

- `data/panel_reg_export.csv` — raw pre-lag panel (238 rows), exported by the
  notebook's own cell 33 (`panel.to_csv(...)`), uploaded by the user.
- `common.py` — line-for-line ports of the notebook's `safe_lag()` and
  `build_model_sample()`, plus the exact rename/lag/filter steps that turn the
  raw export into what the notebook calls `panel_reg` at the point Table 5a
  fits its models.
- `01_reproduce.py` — **run first.** Refits all four Table 5a models (Q and
  ROE) and checks every coefficient/SE/N against the values already reported
  in the paper. Exits nonzero if anything fails to reproduce.
- `02_analysis.py` — effect sizes, standardized effects, 95% CIs, minimum
  detectable effect (80% power), TOST equivalence at ±0.05/0.10/0.20 SD, and
  Bayes factors (BF01) under two priors. Writes `results_q.csv`.
- `03_robustness.py` — re-fits under four alternative specifications
  (cluster-robust SE, excl. Progressive, ROE as DV, drop ROA control). Writes
  `results_robustness.csv`.
- `04_report.py` — assembles `report.md` from the two results CSVs (no new
  estimation).
- `versions.txt` — exact package versions used.

## Run order

```
python3 null_check/01_reproduce.py
python3 null_check/02_analysis.py
python3 null_check/03_robustness.py
python3 null_check/04_report.py
```

## Scope

Treats Table 5a's four nested Q models as the "key coefficients" — the
request's bracketed placeholders for this were never filled in. See
`report.md`'s Scope note for the full reasoning and how to extend this to a
different set of coefficients if that assumption is wrong.
