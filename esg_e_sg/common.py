"""
Shared reconstruction logic for the E vs. S+G-combined check. Self-
contained (no imports from null_check/ or esg_pillars/), per instruction.
Same safe_lag/build_model_sample/ctrl_5a logic as those folders -- see
null_check/common.py's docstring for the cell-by-cell provenance.
"""

import numpy as np
import pandas as pd

RAW_PANEL_PATH = "esg_e_sg/data/panel_reg_export.csv"
E_SG_PATH = "esg_e_sg/data/df_e_sg_specificity_firmyear.csv"

CTRL_COLS_5A = ["SR_CEO_Letter_lag1", "Leverage", "Size", "ROA", "Revenue_Growth"]
CTRL_5A_FORMULA = "SR_CEO_Letter_lag1 + Leverage + Size + ROA + Revenue_Growth + C(Year) + C(Company)"

NAME_FIXES = {
    "Generali": "Assicurazioni Generali",
    "MET": "MetLife",
    "PIC": "People Insurance China",
    "Prudential Financials": "Prudentials",
}


def safe_lag(df, col, group_col="Company", year_col="Year", periods=1):
    lagged = df.groupby(group_col)[col].shift(periods)
    year_shifted = df.groupby(group_col)[year_col].shift(periods)
    expected_year = df[year_col] - periods
    valid = year_shifted == expected_year
    return lagged.where(valid)


def load_and_reconstruct(path=RAW_PANEL_PATH, e_sg_path=E_SG_PATH):
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

    e_sg = pd.read_csv(e_sg_path)
    e_sg = e_sg.rename(columns={"Company Name": "Company"})
    e_sg["Company"] = e_sg["Company"].replace(NAME_FIXES)

    n_before_merge = len(df)
    df = df.merge(e_sg, on=["Company", "Year"], how="left")
    esg_cols = [c for c in e_sg.columns if c not in ("Company", "Year")]
    n_missing_before_fill = df[esg_cols].isna().any(axis=1).sum()

    # Firm-years with ZERO any_esg_flag==1 sentences never appear in the
    # sentence-level file at all (there's nothing to aggregate), so the
    # left-merge above leaves NaN for them -- but that's "zero qualifying
    # sentences over a valid denominator" (i.e. 0), not missing data. Same
    # convention as Corporate_ESG_share's own zero-fill in Section 8.
    # Specificity_*_mean / ESGSentiment_*_Corporate stay NaN (genuinely
    # undefined for an empty set).
    for share_col in ["Corporate_E_share", "Verified_E_share",
                       "Corporate_SG_share", "Verified_SG_share"]:
        df[share_col] = df[share_col].fillna(0)

    for col in ["Corporate_E_share", "Verified_E_share", "Specificity_E_mean",
                "Corporate_SG_share", "Verified_SG_share", "Specificity_SG_mean"]:
        df[col + "_lag"] = safe_lag(df, col)

    meta = {
        "n_raw": n_raw,
        "n_after_cluster_lag_drop": n_after_cluster_lag_drop,
        "n_before_e_sg_merge": n_before_merge,
        "n_missing_e_sg_before_zero_fill": int(n_missing_before_fill),
    }
    return df, meta


def build_model_sample(df, esg_cols, dv="Q", label=""):
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
