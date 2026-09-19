"""
Builds the firm-year E vs. S+G-combined specificity aggregation directly
from the sentence-level combined file (data/df_esg_combined_specificity.csv,
5544 rows -- the same df_esg_spec the notebook's "which df combines E and
SG" export produces), instead of waiting for the separate notebook export
cell. Splits by env_flag (==1 -> E corpus / 1823 rows, ==0 -> S+G corpus /
3721 rows), confirmed to exactly partition the file (cross-checked against
each row's custom_id prefix -- zero mismatches).

Mirrors the exact aggregation logic already in the notebook (and in the
"Firm-year E vs. S+G-combined specificity export" cell pushed to GitHub)
-- is_corporate_tier1plus/tier2plus, tier_if_corp_t1p, and
ESGSentiment_*_Corporate's positive-minus-negative-share construction are
all already-computed columns in this file, not re-derived differently here.

Run:  python3 esg_e_sg/00_build_e_sg_firmyear.py
"""

import numpy as np
import pandas as pd

sentences = pd.read_csv("esg_e_sg/data/df_esg_combined_specificity.csv")
pillars = pd.read_csv("esg_e_sg/data/df_esg_pillars_firmyear.csv")  # for total_sent

n_env = (sentences["env_flag"] == 1).sum()
n_sg = (sentences["env_flag"] == 0).sum()
print(f"E corpus (env_flag==1): {n_env} sentences")
print(f"S+G corpus (env_flag==0): {n_sg} sentences")
print(f"Total: {len(sentences)}")


def build_fy(df_corpus, prefix, total_sent_df):
    fy = (
        df_corpus.groupby(["Company Name", "Year"])
        .agg(
            n_corporate_tier1plus=("is_corporate_tier1plus", "sum"),
            n_corporate_tier2plus=("is_corporate_tier2plus", "sum"),
            Specificity_mean=("tier_if_corp_t1p", "mean"),
            n_corp_t1p_positive=("is_corp_t1p_positive", "sum"),
            n_corp_t1p_negative=("is_corp_t1p_negative", "sum"),
        )
        .reset_index()
    )
    fy = fy.merge(total_sent_df[["Company Name", "Year", "total_sent"]],
                   on=["Company Name", "Year"], how="left")

    fy[f"Corporate_{prefix}_share"] = fy["n_corporate_tier1plus"] / fy["total_sent"]
    fy[f"Verified_{prefix}_share"] = fy["n_corporate_tier2plus"] / fy["total_sent"]
    fy[f"Specificity_{prefix}_mean"] = fy["Specificity_mean"]
    fy[f"ESGSentiment_{prefix}_Corporate"] = np.where(
        fy["n_corporate_tier1plus"] > 0,
        (fy["n_corp_t1p_positive"] - fy["n_corp_t1p_negative"]) / fy["n_corporate_tier1plus"],
        np.nan,
    )
    fy[f"n_{prefix}_corporate_tier1plus"] = fy["n_corporate_tier1plus"]

    keep = ["Company Name", "Year", f"n_{prefix}_corporate_tier1plus",
            f"Corporate_{prefix}_share", f"Verified_{prefix}_share",
            f"Specificity_{prefix}_mean", f"ESGSentiment_{prefix}_Corporate"]
    return fy[keep]


fy_e = build_fy(sentences[sentences["env_flag"] == 1], "E", pillars)
fy_sg = build_fy(sentences[sentences["env_flag"] == 0], "SG", pillars)

df_e_sg = fy_e.merge(fy_sg, on=["Company Name", "Year"], how="outer")

for prefix in ["E", "SG"]:
    for col in [f"Corporate_{prefix}_share", f"Verified_{prefix}_share"]:
        df_e_sg[col] = df_e_sg[col].fillna(0)

print(f"\nFirm-years: {len(df_e_sg)}")
print(df_e_sg.head())

# ---- sanity check: does Corporate_E_share + Corporate_SG_share's
# numerator match Table 5a's own Corporate_ESG_share numerator? ----
raw_panel = pd.read_csv("esg_e_sg/data/panel_reg_export.csv")
raw_panel = raw_panel.rename(columns={"Company Name": "Company"}) if "Company Name" in raw_panel.columns else raw_panel
check = df_e_sg.rename(columns={"Company Name": "Company"}).merge(
    raw_panel[["Company", "Year", "Corporate_ESG_share"]], on=["Company", "Year"], how="inner"
)
check["Corporate_E_plus_SG_share"] = check["Corporate_E_share"] + check["Corporate_SG_share"]
check["diff"] = (check["Corporate_E_plus_SG_share"] - check["Corporate_ESG_share"]).abs()
print(f"\nSanity check -- Corporate_E_share + Corporate_SG_share vs. Corporate_ESG_share "
      f"(should match, since E and S+G corpora are a clean partition of the ESG-flagged sentences):")
print(f"Max abs diff across {len(check)} firm-years: {check['diff'].max():.6f}")
print(f"Firm-years with diff > 1e-6: {(check['diff'] > 1e-6).sum()}")

df_e_sg.to_csv("esg_e_sg/data/df_e_sg_specificity_firmyear.csv", index=False)
print("\nSaved: esg_e_sg/data/df_e_sg_specificity_firmyear.csv")
