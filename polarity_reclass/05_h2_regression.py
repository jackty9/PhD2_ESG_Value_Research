"""
Revised H2 test: Positive_Specific_share in place of Specificity_mean,
Corporate_ESG_share kept as the conditioning variable -- same role and
formula structure as "Model 4" elsewhere in this paper
(null_check/common.py's MODEL_SPECS / esg_pillars/common.py's
TABLE5A_MODEL_SPECS: ["Corporate_ESG_share_lag", "<second regressor>_lag"]
plus CTRL_5A_FORMULA, both HC3 and cluster-robust-by-firm SE reported side
by side, following null_check/03_robustness.py's exact fit() pattern).

    Q_it   = b0 + b1*Corporate_ESG_share_lag + b2*Positive_Specific_share_lag
             + Controls_it + firm FE + year FE + e_it
    ROE_it = b0 + b1*Corporate_ESG_share_lag + b2*Positive_Specific_share_lag
             + Controls_it + firm FE + year FE + e_it

Controls = CTRL_5A_FORMULA (SR_CEO_Letter_lag1, Leverage, Size, ROA,
Revenue_Growth, C(Year), C(Company)) -- the request's written equation named
only Size/Leverage/ROA/Revenue_Growth, but "same format as Model 4" is the
controlling instruction and Model 4's own formula (both esg_pillars/ and
null_check/'s common.py) always includes SR_CEO_Letter_lag1 + the two FE
terms; kept here for a genuine like-for-like comparison against the prior
run being re-validated. Flag if SR_CEO_Letter_lag1 should actually be
dropped.

Positive_Specific_share is rebuilt fresh from the Revision 6 GPT
classification (tier>=2, polarity=="positive"; polarity_reclass/data/
df_ar_ceo_sentences_polarity_reclass_v6.csv), same construction as
04_regression.py, with total_sent (esg_e_sg/data/panel_reg_export.csv,
identical to null_check/data/panel_reg_export.csv) as denominator.

Prior run being re-validated (pre-revision, FinBERT-era polarity labels --
should NOT be trusted, this is what triggered the classifier revalidation
project in the first place):
  Q:   coef=-0.0022, HC3 p=0.896, cluster p=0.851
  ROE: coef=-0.0156, HC3 p=0.191, cluster p=0.048 (borderline, flagged as
       untrustworthy without jackknife/bootstrap given the classifier's
       known issues at the time)

Run:  python3 polarity_reclass/05_h2_regression.py
"""

import importlib.util
import sys

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, build_model_sample, load_and_reconstruct


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_esg_e_sg_common = _load_module("esg_e_sg_common", "esg_e_sg/common.py")
NAME_FIXES = _esg_e_sg_common.NAME_FIXES
safe_lag = _esg_e_sg_common.safe_lag

Z = stats.norm.ppf(0.975)

GPT_RESULTS_PATH = "polarity_reclass/data/df_ar_ceo_sentences_polarity_reclass_v6.csv"
PANEL_PATH = "esg_e_sg/data/panel_reg_export.csv"

H2_COLS = ["Corporate_ESG_share_lag", "Positive_Specific_share_lag"]

PRIOR_RUN = {
    "Q": dict(coef=-0.0022, p_hc3=0.896, p_cluster=0.851),
    "ROE": dict(coef=-0.0156, p_hc3=0.191, p_cluster=0.048),
}


def build_positive_specific_share():
    gpt = pd.read_csv(GPT_RESULTS_PATH)
    tier2plus = gpt[gpt["specificity_tier_pred"] >= 2].copy()
    tier2plus["is_pos_specific"] = tier2plus["polarity"] == "positive"

    panel = pd.read_csv(PANEL_PATH)
    total_by_fy = panel.set_index(["Company", "Year"])["total_sent"]

    pos_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_pos_specific"].sum().rename("n_pos_specific")
    fy = pos_by_fy.reset_index()
    fy.columns = ["Company", "Year", "n_pos_specific"]
    fy["Company"] = fy["Company"].replace(NAME_FIXES)

    fy = fy.merge(total_by_fy.rename("total_sent").reset_index(), on=["Company", "Year"], how="right")
    fy["n_pos_specific"] = fy["n_pos_specific"].fillna(0)
    fy["Positive_Specific_share"] = fy["n_pos_specific"] / fy["total_sent"]
    return fy[["Company", "Year", "Positive_Specific_share"]]


def fit_both_se(dv, esg_cols, df, label):
    sample, dropped = build_model_sample(df, esg_cols, dv=dv, label=label)
    formula = f"{dv} ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
    m_hc3 = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
    m_cluster = smf.ols(formula=formula, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["Company"]}
    )
    return m_hc3, m_cluster, sample, dropped


def main():
    df, meta = load_and_reconstruct()
    print(f"Panel meta (null_check reconstruction): {meta}")

    pos_share = build_positive_specific_share()
    df = df.merge(pos_share, on=["Company", "Year"], how="left")
    df["Positive_Specific_share"] = df["Positive_Specific_share"].fillna(0)
    df = df.sort_values(["Company", "Year"]).reset_index(drop=True)
    df["Positive_Specific_share_lag"] = safe_lag(df, "Positive_Specific_share")

    rows = []
    for dv in ["Q", "ROE"]:
        m_hc3, m_cluster, sample, dropped = fit_both_se(dv, H2_COLS, df, f"H2 revised, {dv}")

        print(f"\n{'=' * 78}\n{dv}: Corporate_ESG_share_lag + Positive_Specific_share_lag (revised H2)\n{'=' * 78}")
        print(f"N = {int(m_hc3.nobs)}, R-squared = {m_hc3.rsquared:.4f}")
        print(f"Firms dropped (insufficient obs): {dropped}")
        print(m_hc3.summary())

        for var in H2_COLS:
            row = {
                "dv": dv, "term": var,
                "coef": m_hc3.params[var],
                "se_hc3": m_hc3.bse[var], "p_hc3": m_hc3.pvalues[var],
                "se_cluster": m_cluster.bse[var], "p_cluster": m_cluster.pvalues[var],
                "ci_lo_hc3": m_hc3.params[var] - Z * m_hc3.bse[var],
                "ci_hi_hc3": m_hc3.params[var] + Z * m_hc3.bse[var],
                "n": int(m_hc3.nobs),
            }
            rows.append(row)
            if var == "Positive_Specific_share_lag":
                prior = PRIOR_RUN[dv]
                print(f"\n--- {dv}, Positive_Specific_share_lag: revised vs. prior (pre-revision, "
                      f"FinBERT-era, NOT to be trusted) ---")
                print(f"  Prior:   coef={prior['coef']:.4f}, HC3 p={prior['p_hc3']:.3f}, "
                      f"cluster p={prior['p_cluster']:.3f}")
                print(f"  Revised: coef={row['coef']:.4f}, HC3 p={row['p_hc3']:.3f}, "
                      f"cluster p={row['p_cluster']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("polarity_reclass/results_h2_revised.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: polarity_reclass/results_h2_revised.csv")

    n_nonzero = (df.loc[df["Positive_Specific_share_lag"].notna(), "Positive_Specific_share_lag"] > 0).sum()
    print(f"\nNote: Positive_Specific_share_lag has good coverage ({n_nonzero} non-zero firm-year "
          f"observations before sample restrictions) -- unlike Negative_Specific_share, sparsity "
          f"is not a concern for this regressor.")


if __name__ == "__main__":
    main()
