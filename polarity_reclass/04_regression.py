"""
Regression on Positive_Specific_share / Negative_Specific_share (GPT-relabeled,
Revision 6, tier>=2), on Q and ROE. Same controls/FE/HC3 pattern as
esg_e_sg/02_regression.py's E+SG joint-vs-alone spec (CTRL_5A_FORMULA,
build_model_sample, HC3 OLS) -- reused directly from esg_e_sg/common.py.

Builds the firm-year share variables fresh from the full sentence-level GPT
output (polarity_reclass/data/df_ar_ceo_sentences_polarity_reclass_v6.csv),
restricted to specificity_tier_pred >= 2 (the threshold these two share
variables use), with total_sent (from panel_reg_export.csv) as the
denominator -- share of ALL sentences, same convention as Corporate_ESG_share.

kappa (GPT vs. manual) for this Revision 6 classification was accepted at
0.77-0.80 (see polarity_reclass/README.md) rather than chased past the 0.80
target -- noted here as a caveat on result interpretation, not a blocker.

Negative_Specific_share is sparse (17 of 238 firm-years have >=1 qualifying
negative-specific sentence, tier>=2) -- flagged explicitly in the output,
since a regression coefficient built on that few non-zero observations
warrants caution regardless of its p-value.

Run:  python3 polarity_reclass/04_regression.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, "esg_e_sg")
from common import CTRL_5A_FORMULA, CTRL_COLS_5A, NAME_FIXES, build_model_sample, load_and_reconstruct, safe_lag

Z = stats.norm.ppf(0.975)

GPT_RESULTS_PATH = "polarity_reclass/data/df_ar_ceo_sentences_polarity_reclass_v6.csv"
PANEL_PATH = "esg_e_sg/data/panel_reg_export.csv"

SHARE_COLS = ["Positive_Specific_share_lag", "Negative_Specific_share_lag"]


def build_specific_shares():
    gpt = pd.read_csv(GPT_RESULTS_PATH)
    tier2plus = gpt[gpt["specificity_tier_pred"] >= 2].copy()
    print(f"Of {len(gpt)} reclassified (Corporate, tier>=1) sentences, {len(tier2plus)} are "
          f"tier>=2 (the population these two share variables use).")

    tier2plus["is_pos_specific"] = tier2plus["polarity"] == "positive"
    tier2plus["is_neg_specific"] = tier2plus["polarity"] == "negative"

    panel = pd.read_csv(PANEL_PATH)
    total_by_fy = panel.set_index(["Company", "Year"])["total_sent"]

    pos_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_pos_specific"].sum().rename("n_pos_specific")
    neg_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_neg_specific"].sum().rename("n_neg_specific")

    fy = pd.concat([pos_by_fy, neg_by_fy], axis=1)
    fy.index = fy.index.set_names(["Company", "Year"])
    fy = fy.reset_index()
    fy["Company"] = fy["Company"].replace(NAME_FIXES)

    fy = fy.merge(total_by_fy.rename("total_sent").reset_index(), on=["Company", "Year"], how="right")
    fy[["n_pos_specific", "n_neg_specific"]] = fy[["n_pos_specific", "n_neg_specific"]].fillna(0)
    fy["Positive_Specific_share"] = fy["n_pos_specific"] / fy["total_sent"]
    fy["Negative_Specific_share"] = fy["n_neg_specific"] / fy["total_sent"]

    n_fy_pos = (fy["n_pos_specific"] > 0).sum()
    n_fy_neg = (fy["n_neg_specific"] > 0).sum()
    print(f"Firm-years (of {len(fy)} in the regression panel) with >=1 qualifying "
          f"Positive_Specific sentence: {n_fy_pos}")
    print(f"Firm-years with >=1 qualifying Negative_Specific sentence: {n_fy_neg}  "
          f"<-- SPARSE: interpret any Negative_Specific_share coefficient with caution")

    return fy[["Company", "Year", "Positive_Specific_share", "Negative_Specific_share"]]


def load_panel_with_shares():
    df, meta = load_and_reconstruct()
    shares = build_specific_shares()
    df = df.merge(shares, on=["Company", "Year"], how="left")
    df[["Positive_Specific_share", "Negative_Specific_share"]] = (
        df[["Positive_Specific_share", "Negative_Specific_share"]].fillna(0)
    )
    df = df.sort_values(["Company", "Year"]).reset_index(drop=True)
    for col in ["Positive_Specific_share", "Negative_Specific_share"]:
        df[col + "_lag"] = safe_lag(df, col)
    return df, meta


def fit(df, dv, share_cols, label):
    sample, dropped = build_model_sample(df, share_cols, dv=dv, label=label)
    formula = f"{dv} ~ {' + '.join(share_cols)} + {CTRL_5A_FORMULA}"
    m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
    return m, sample, dropped


def main():
    df, meta = load_panel_with_shares()
    print(f"\nPanel meta: {meta}")
    rows = []

    for dv in ["Q", "ROE"]:
        # ---- PRIMARY: Positive_Specific_share and Negative_Specific_share, jointly ----
        m_joint, sample_joint, dropped = fit(df, dv, SHARE_COLS, f"Joint Pos+Neg specific, {dv}")
        print(f"\n{'=' * 78}\n{dv}: Positive_Specific_share + Negative_Specific_share (joint) -- PRIMARY\n{'=' * 78}")
        print(f"N = {int(m_joint.nobs)}, R-squared = {m_joint.rsquared:.4f}, "
              f"Condition number: {m_joint.condition_number:.3e}")
        print(f"Firms dropped (insufficient obs): {dropped}")
        print(m_joint.summary())
        for var in SHARE_COLS:
            rows.append({
                "dv": dv, "spec": "Joint (Pos + Neg specific)", "term": var,
                "coef": m_joint.params[var], "se": m_joint.bse[var],
                "ci_lo": m_joint.params[var] - Z * m_joint.bse[var],
                "ci_hi": m_joint.params[var] + Z * m_joint.bse[var],
                "p": m_joint.pvalues[var], "n": int(m_joint.nobs),
            })

        # ---- context: each alone ----
        for label, col in [("Positive_Specific", "Positive_Specific_share_lag"),
                            ("Negative_Specific", "Negative_Specific_share_lag")]:
            m_alone, sample_alone, _ = fit(df, dv, [col], f"{label} alone, {dv}")
            rows.append({
                "dv": dv, "spec": f"{label} alone", "term": col,
                "coef": m_alone.params[col], "se": m_alone.bse[col],
                "ci_lo": m_alone.params[col] - Z * m_alone.bse[col],
                "ci_hi": m_alone.params[col] + Z * m_alone.bse[col],
                "p": m_alone.pvalues[col], "n": int(m_alone.nobs),
            })

    out = pd.DataFrame(rows)
    out.to_csv("polarity_reclass/results_specific_share.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"\n{'=' * 78}\nSummary: joint (primary) vs. each alone (context)\n{'=' * 78}")
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: polarity_reclass/results_specific_share.csv")
    print("\nCAVEAT: Negative_Specific_share is built on very few qualifying sentences "
          "(sparse coverage, see above) -- do not over-interpret its coefficient's "
          "significance without checking which firm-years are actually driving it.")


if __name__ == "__main__":
    main()
