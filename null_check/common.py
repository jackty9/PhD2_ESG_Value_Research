"""
Shared reconstruction logic for the null-result check.

This module independently re-derives the exact analysis-ready panel that
PHDp2_FinancialData_Regression.ipynb builds internally (Table 5a and its
ROE-DV parallel, Section 13), starting from the RAW pre-lag export
(null_check/data/panel_reg_export.csv, written by the notebook's cell 33,
`panel.to_csv(...)`, 238 rows / 35 columns).

Every function here is a line-for-line port of the corresponding notebook
cell, not a reinterpretation -- so that fitting the models on the output of
this module and comparing to the coefficients already reported in the paper
is a genuine independent reproduction check, not a restatement of numbers
already trusted on faith.

Ported from (cell numbers as of branch specificity-pipeline-3d-regression,
commit 10d6b3b):
  - cell 36: Revenue Growth rename, safe_lag(), cluster_lag construction,
    SR_CEO_Letter_lag1, the dropna(subset=["cluster_lag"]) filter that
    defines panel_reg's row set (238 -> 209 rows).
  - cell 37: Tobins Q -> Q rename.
  - cell 87 (5a.1): ESG_sentence_ratio = esg_ratio, the three ESG-share
    lags, ctrl_cols_5a, build_model_sample() (per-model complete-case +
    firm-count guard, fixed in commits a2ef69c/4e418dc to return the
    row-complete sample rather than a company-filtered slice of the
    original data -- see that commit message for why the earlier version
    was silently rank-deficient).
  - Section 13 (cell ~21 / ~113): ROE construction (already present as a
    raw column in the export, since it's built upstream of cell 33) and
    Specificity_mean_lag.
"""

import numpy as np
import pandas as pd

RAW_PANEL_PATH = "null_check/data/panel_reg_export.csv"

CTRL_COLS_5A = ["SR_CEO_Letter_lag1", "Leverage", "Size", "ROA", "Revenue_Growth"]
CTRL_5A_FORMULA = "SR_CEO_Letter_lag1 + Leverage + Size + ROA + Revenue_Growth + C(Year) + C(Company)"

ESG_MEASURES_5A = [
    ("ESG_sentence_ratio_lag", "Model 1 (baseline ESG_sentence_ratio)"),
    ("Corporate_ESG_share_lag", "Model 2 (Corporate_ESG_share)"),
]
# Models 3 and 4 use two regressors each; handled explicitly in the fit step.


def safe_lag(df, col, group_col="Company", year_col="Year", periods=1):
    """Verbatim port of the notebook's gap-aware lag helper (cell 36)."""
    lagged = df.groupby(group_col)[col].shift(periods)
    year_shifted = df.groupby(group_col)[year_col].shift(periods)
    expected_year = df[year_col] - periods
    valid = year_shifted == expected_year
    return lagged.where(valid)


def load_and_reconstruct(path=RAW_PANEL_PATH):
    """
    Reproduces panel_reg as it stands right before Table 5a / Section 13
    fit their models: cluster_lag built + the dropna(subset=["cluster_lag"])
    filter applied (cell 36), Tobins Q -> Q renamed (cell 37), the three
    ESG-share lags + Specificity_mean_lag + Post2017 built (cells 87 / 12.1).
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Revenue Growth YoY (%)": "Revenue_Growth"})
    df = df.rename(columns={"Tobins Q": "Q"})
    df = df.sort_values(["Company", "Year"]).reset_index(drop=True)

    n_raw = len(df)

    # cluster_lag: gap-aware lag of the (3-D, per cluster_2d_legacy's
    # presence -- see null_check/01_reproduce.py's confirmation) cluster
    # assignment, then the row filter cell 36 applies BEFORE any Step 5/
    # Section 13 construction.
    df["cluster_lag"] = safe_lag(df, "cluster")
    df = df.dropna(subset=["cluster_lag"]).reset_index(drop=True)
    n_after_cluster_lag_drop = len(df)

    df["SR_CEO_Letter_lag1"] = safe_lag(df, "SR_CEO_Letter")
    df["SR_CEO_Letter_lag1"] = df["SR_CEO_Letter_lag1"].fillna(0).astype(int)

    df["ESG_sentence_ratio"] = df["esg_ratio"]
    for col in ["ESG_sentence_ratio", "Corporate_ESG_share", "Verified_ESG_share", "Specificity_mean"]:
        df[col + "_lag"] = safe_lag(df, col)

    df["Post2017"] = (df["Year"] >= 2017).astype(int)

    meta = {"n_raw": n_raw, "n_after_cluster_lag_drop": n_after_cluster_lag_drop}
    return df, meta


def build_model_sample(df, esg_cols, dv="Q", label=""):
    """
    Verbatim port of build_model_sample() (notebook cell 87, post-fix
    a2ef69c + 4e418dc): row-level complete-case on [dv] + esg_cols +
    CTRL_COLS_5A, THEN drop any company with <3 rows in that complete-case
    set OR zero rows at all (the latter catches firms invisible to a naive
    groupby().size(), the original bug this function was written to fix).
    Returns the row-complete, firm-filtered sample -- not a company-filter
    of the original frame.
    """
    required = [dv] + esg_cols + CTRL_COLS_5A
    sample = df.dropna(subset=required)
    counts = sample.groupby("Company").size()
    firms_to_drop = set(counts[counts < 3].index)
    all_firms = set(df["Company"].unique())
    zero_obs_firms = all_firms - set(counts.index)
    firms_to_drop |= zero_obs_firms
    result = sample[~sample["Company"].isin(firms_to_drop)].copy()
    return result, sorted(firms_to_drop)


MODEL_SPECS = [
    ("Model 1", ["ESG_sentence_ratio_lag"]),
    ("Model 2", ["Corporate_ESG_share_lag"]),
    ("Model 3", ["Corporate_ESG_share_lag", "Verified_ESG_share_lag"]),
    ("Model 4", ["Corporate_ESG_share_lag", "Specificity_mean_lag"]),
]
