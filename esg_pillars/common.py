"""
Shared reconstruction logic for the E/S/G pillar check. Self-contained
(does not import from null_check/), per instruction to write new scripts
in esg_pillars/ without touching or depending on prior work.

Same source data and same line-for-line ported logic as null_check/common.py
(safe_lag, build_model_sample, Table 5a's ctrl_5a formula) -- see that
module's docstring for the cell-by-cell provenance. This module adds the
E/S/G pillar merge on top.

Pillar data (data/df_esg_pillars_firmyear.csv, 238 firm-years) is a SIMPLE
sentence-ratio measure (pillar-flagged sentences / total sentences),
matching ESG_sentence_ratio's own definition split by dimension -- NOT the
Corporate/Verified/Specificity-restricted measures Table 5a uses for the
combined ESG variables. That richer classification was only ever run
E-alone vs. S+G-combined, never split by S and G separately, so a pillar-
level equivalent isn't available without new GPT classification work (out
of scope here -- see the conversation's Step 1 finding).
"""

import numpy as np
import pandas as pd

RAW_PANEL_PATH = "esg_pillars/data/panel_reg_export.csv"
PILLARS_PATH = "esg_pillars/data/df_esg_pillars_firmyear.csv"

CTRL_COLS_5A = ["SR_CEO_Letter_lag1", "Leverage", "Size", "ROA", "Revenue_Growth"]
CTRL_5A_FORMULA = "SR_CEO_Letter_lag1 + Leverage + Size + ROA + Revenue_Growth + C(Year) + C(Company)"

# Same name_fixes mapping the FinancialData notebook applies (cell 27) when
# merging df_cluster into panel -- the pillar export uses the CEO-letters
# notebook's own "Company Name" values, which predate that mapping.
NAME_FIXES = {
    "Generali": "Assicurazioni Generali",
    "MET": "MetLife",
    "PIC": "People Insurance China",
    "Prudential Financials": "Prudentials",
}


def safe_lag(df, col, group_col="Company", year_col="Year", periods=1):
    """Verbatim port of the notebook's gap-aware lag helper (cell 36)."""
    lagged = df.groupby(group_col)[col].shift(periods)
    year_shifted = df.groupby(group_col)[year_col].shift(periods)
    expected_year = df[year_col] - periods
    valid = year_shifted == expected_year
    return lagged.where(valid)


def load_and_reconstruct(path=RAW_PANEL_PATH, pillars_path=PILLARS_PATH):
    """
    Reproduces panel_reg as it stands right before Table 5a fits its models
    (cell 36's cluster_lag filter, cell 37's Q rename, cell 87's ESG-share
    lags), THEN merges in the E/S/G pillar ratios and builds their lags the
    same way.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Revenue Growth YoY (%)": "Revenue_Growth"})
    df = df.rename(columns={"Tobins Q": "Q"})
    df = df.sort_values(["Company", "Year"]).reset_index(drop=True)

    n_raw = len(df)

    df["cluster_lag"] = safe_lag(df, "cluster")
    df = df.dropna(subset=["cluster_lag"]).reset_index(drop=True)
    n_after_cluster_lag_drop = len(df)

    df["SR_CEO_Letter_lag1"] = safe_lag(df, "SR_CEO_Letter")
    df["SR_CEO_Letter_lag1"] = df["SR_CEO_Letter_lag1"].fillna(0).astype(int)

    df["ESG_sentence_ratio"] = df["esg_ratio"]
    for col in ["ESG_sentence_ratio", "Corporate_ESG_share", "Verified_ESG_share", "Specificity_mean"]:
        df[col + "_lag"] = safe_lag(df, col)

    # ---- pillar merge ----
    pillars = pd.read_csv(pillars_path)
    pillars = pillars.rename(columns={"Company Name": "Company"})
    pillars["Company"] = pillars["Company"].replace(NAME_FIXES)
    pillars = pillars[["Company", "Year", "E_ratio", "S_ratio", "G_ratio"]]

    n_before_pillar_merge = len(df)
    df = df.merge(pillars, on=["Company", "Year"], how="left")
    n_missing_pillar = df[["E_ratio", "S_ratio", "G_ratio"]].isna().any(axis=1).sum()

    for col in ["E_ratio", "S_ratio", "G_ratio"]:
        df[col + "_lag"] = safe_lag(df, col)

    meta = {
        "n_raw": n_raw,
        "n_after_cluster_lag_drop": n_after_cluster_lag_drop,
        "n_before_pillar_merge": n_before_pillar_merge,
        "n_missing_pillar_after_merge": int(n_missing_pillar),
    }
    return df, meta


def build_model_sample(df, esg_cols, dv="Q", label=""):
    """
    Verbatim port of build_model_sample() (notebook cell 87, post-fix
    a2ef69c + 4e418dc): row-level complete-case on [dv] + esg_cols +
    CTRL_COLS_5A, THEN drop any company with <3 rows in that complete-case
    set OR zero rows at all. Returns the row-complete, firm-filtered
    sample.
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


TABLE5A_MODEL_SPECS = [
    ("Model 1", ["ESG_sentence_ratio_lag"]),
    ("Model 2", ["Corporate_ESG_share_lag"]),
    ("Model 3", ["Corporate_ESG_share_lag", "Verified_ESG_share_lag"]),
    ("Model 4", ["Corporate_ESG_share_lag", "Specificity_mean_lag"]),
]

PILLARS = ["E", "S", "G"]
PILLAR_LAG_COLS = {p: f"{p}_ratio_lag" for p in PILLARS}
