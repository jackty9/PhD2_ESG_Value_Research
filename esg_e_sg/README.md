# esg_e_sg

Exploratory E vs. S+G-combined check — splits `Corporate_ESG_share`/
`Verified_ESG_share`/`Specificity_mean` back into their two actual GPT
specificity-classification source corpora (E alone, S+G combined). No new
GPT classification needed. Self-contained, does not touch the notebooks,
`null_check/`, or `esg_pillars/`.

## Contents

- `data/panel_reg_export.csv`, `data/df_esg_pillars_firmyear.csv` — copied
  from `null_check/`/`esg_pillars/` for self-containment.
- `data/df_esg_combined_specificity.csv` — sentence-level combined E+S/G
  specificity classification (5544 rows, from the notebook's `df_esg_spec`
  export).
- `data/df_e_sg_specificity_firmyear.csv` — firm-year E vs. SG aggregation,
  built by `00_build_e_sg_firmyear.py` directly from the sentence-level
  file (splitting on `env_flag`, confirmed to exactly partition the file).
- `common.py` — shared reconstruction (`safe_lag()`, `build_model_sample()`,
  the E/SG merge with the same zero-fill convention as `Corporate_ESG_share`).
- `00_build_e_sg_firmyear.py` — builds the firm-year E/SG file, with a
  built-in sanity check (`Corporate_E_share + Corporate_SG_share` must
  exactly equal `Corporate_ESG_share` — confirmed, max diff 0.000000).
- `01_reproduce.py` — **run first.** Reproduces Table 5a.
- `02_regression.py` — Part 1: E and SG shares, combined in one regression
  (plus each alone for context), Q and ROE.
- `03_pillar_specificity.py` — Part 2: `Corporate_{X}_share_lag` +
  `Specificity_{X}_mean_lag` jointly per pillar, with and without a
  <5-sentence exclusion, full MDE/TOST/Holm/BH/BF01 battery.
- `04_report.py` — assembles `report.md` from the results CSVs.
- `versions.txt` — exact package versions used.

## Run order

```
python3 esg_e_sg/01_reproduce.py
python3 esg_e_sg/02_regression.py
python3 esg_e_sg/03_pillar_specificity.py
python3 esg_e_sg/04_report.py
```

(`00_build_e_sg_firmyear.py` has already been run — its output is committed
in `data/df_e_sg_specificity_firmyear.csv` — but is included for
reproducibility; re-run it if `data/df_esg_combined_specificity.csv` ever
changes.)
