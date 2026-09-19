# esg_pillars

Exploratory E/S/G pillar breakdown of the paper's ESG-disclosure measure.
Does not modify the notebooks, `null_check/`, or any export file.

## Contents

- `data/panel_reg_export.csv` — same raw pre-lag panel used in `null_check/`
  (copied here for self-containment, not re-derived).
- `data/df_esg_pillars_firmyear.csv` — firm-year E/S/G sentence ratios
  (238 rows), exported by the notebook's new cell 8b (`env_flag`/`soc_flag`/
  `gov_flag` shares — see Step 1 of `report.md` for why this is narrower
  than Table 5a's Corporate/Verified/Specificity measures).
- `common.py` — shared reconstruction (`safe_lag()`, `build_model_sample()`,
  the pillar merge with the same `name_fixes` mapping the main notebook
  uses).
- `01_reproduce.py` — **run first.** Reproduces Table 5a before touching
  the pillar data.
- `02_analysis.py` — fits the six primary pillar-alone models (E/S/G × Q/
  ROE), the two secondary joint three-pillar models, correlations, VIF,
  standardized effects/CIs/MDE, Holm + BH correction, TOST at ±0.10/±0.20
  SD.
- `03_report.py` — assembles `report.md` from the results CSVs.
- `versions.txt` — exact package versions used.

## Run order

```
python3 esg_pillars/01_reproduce.py
python3 esg_pillars/02_analysis.py
python3 esg_pillars/03_report.py
```

## Key caveat (see report.md Step 1)

These E/S/G measures are simple sentence ratios (pillar-flagged sentences /
total sentences), matching `ESG_sentence_ratio`'s definition split by
dimension — **not** a pillar breakdown of `Corporate_ESG_share`/
`Verified_ESG_share`/`Specificity_mean`, since the GPT specificity/
corporate-relevance classification underlying those was only ever run
E-alone vs. S+G-combined, never S and G separately.
