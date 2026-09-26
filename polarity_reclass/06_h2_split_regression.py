"""
Splits the revised H2 regressor (Positive_Specific_share = case in
{plain_positive, mitigated_negative}, per polarity=="positive") into its two
component cases, tested separately and jointly against Corporate_ESG_share_lag
-- same Model 4-style spec, CTRL_5A_FORMULA controls/FE, HC3 + cluster-robust
SE, as 05_h2_regression.py.

    PlainPositiveSpecific_share    = share of tier>=2 sentences with
                                      case == "plain_positive" (ordinary good
                                      news -- achievement, investment, growth)
    MitigatedNegativeSpecific_share = share of tier>=2 sentences with
                                      case == "mitigated_negative" (the
                                      reduction-vocabulary-describing-a-
                                      positive-outcome pattern this whole
                                      reclassification project was built to
                                      correctly separate out from FinBERT's
                                      failure mode)

At tier>=2: 1,002 plain_positive vs. 125 mitigated_negative sentences (the
combined Positive_Specific_share regression run in 05_h2_regression.py pools
both; this splits them to see whether either alone is driving that result,
or whether it's plain_positive dominating simply because it's ~8x larger).

Three specs per DV, mirroring the established joint-vs-alone convention
(esg_e_sg/02_regression.py, polarity_reclass/04_regression.py):
  (1) Corporate_ESG_share_lag + PlainPositiveSpecific_share_lag alone
  (2) Corporate_ESG_share_lag + MitigatedNegativeSpecific_share_lag alone
  (3) Corporate_ESG_share_lag + both together (joint)

Run:  python3 polarity_reclass/06_h2_split_regression.py
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

SPECS = [
    ("PlainPositive alone", ["Corporate_ESG_share_lag", "PlainPositiveSpecific_share_lag"]),
    ("MitigatedNegative alone", ["Corporate_ESG_share_lag", "MitigatedNegativeSpecific_share_lag"]),
    ("Both, joint", ["Corporate_ESG_share_lag", "PlainPositiveSpecific_share_lag",
                      "MitigatedNegativeSpecific_share_lag"]),
]


def build_split_shares():
    gpt = pd.read_csv(GPT_RESULTS_PATH)
    tier2plus = gpt[gpt["specificity_tier_pred"] >= 2].copy()
    tier2plus["is_plain_positive"] = tier2plus["case"] == "plain_positive"
    tier2plus["is_mitigated_negative"] = tier2plus["case"] == "mitigated_negative"

    panel = pd.read_csv(PANEL_PATH)
    total_by_fy = panel.set_index(["Company", "Year"])["total_sent"]

    pp_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_plain_positive"].sum().rename("n_plain_positive")
    mn_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_mitigated_negative"].sum().rename("n_mitigated_negative")

    fy = pd.concat([pp_by_fy, mn_by_fy], axis=1)
    fy.index = fy.index.set_names(["Company", "Year"])
    fy = fy.reset_index()
    fy["Company"] = fy["Company"].replace(NAME_FIXES)

    fy = fy.merge(total_by_fy.rename("total_sent").reset_index(), on=["Company", "Year"], how="right")
    fy[["n_plain_positive", "n_mitigated_negative"]] = fy[["n_plain_positive", "n_mitigated_negative"]].fillna(0)
    fy["PlainPositiveSpecific_share"] = fy["n_plain_positive"] / fy["total_sent"]
    fy["MitigatedNegativeSpecific_share"] = fy["n_mitigated_negative"] / fy["total_sent"]

    print(f"Total tier>=2 sentences: plain_positive={int(fy['n_plain_positive'].sum())}, "
          f"mitigated_negative={int(fy['n_mitigated_negative'].sum())}")

    return fy[["Company", "Year", "PlainPositiveSpecific_share", "MitigatedNegativeSpecific_share"]]


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

    split_shares = build_split_shares()
    df = df.merge(split_shares, on=["Company", "Year"], how="left")
    df[["PlainPositiveSpecific_share", "MitigatedNegativeSpecific_share"]] = (
        df[["PlainPositiveSpecific_share", "MitigatedNegativeSpecific_share"]].fillna(0)
    )
    df = df.sort_values(["Company", "Year"]).reset_index(drop=True)
    for col in ["PlainPositiveSpecific_share", "MitigatedNegativeSpecific_share"]:
        df[col + "_lag"] = safe_lag(df, col)

    n_pp_nonzero = (df["PlainPositiveSpecific_share_lag"] > 0).sum()
    n_mn_nonzero = (df["MitigatedNegativeSpecific_share_lag"] > 0).sum()
    print(f"Non-zero firm-year observations (pre-restriction): "
          f"PlainPositiveSpecific_share_lag={n_pp_nonzero}, "
          f"MitigatedNegativeSpecific_share_lag={n_mn_nonzero}")

    rows = []
    for dv in ["Q", "ROE"]:
        for spec_label, esg_cols in SPECS:
            m_hc3, m_cluster, sample, dropped = fit_both_se(dv, esg_cols, df, f"{spec_label}, {dv}")

            print(f"\n{'=' * 78}\n{dv}: {spec_label} -- {' + '.join(esg_cols)}\n{'=' * 78}")
            print(f"N = {int(m_hc3.nobs)}, R-squared = {m_hc3.rsquared:.4f}")
            print(f"Firms dropped (insufficient obs): {dropped}")

            share_terms = [c for c in esg_cols if c != "Corporate_ESG_share_lag"]
            for var in esg_cols:
                rows.append({
                    "dv": dv, "spec": spec_label, "term": var,
                    "coef": m_hc3.params[var],
                    "se_hc3": m_hc3.bse[var], "p_hc3": m_hc3.pvalues[var],
                    "se_cluster": m_cluster.bse[var], "p_cluster": m_cluster.pvalues[var],
                    "ci_lo_hc3": m_hc3.params[var] - Z * m_hc3.bse[var],
                    "ci_hi_hc3": m_hc3.params[var] + Z * m_hc3.bse[var],
                    "n": int(m_hc3.nobs),
                })
            for var in share_terms:
                print(f"  {var}: coef={m_hc3.params[var]:.4f}, HC3 p={m_hc3.pvalues[var]:.3f}, "
                      f"cluster p={m_cluster.pvalues[var]:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("polarity_reclass/results_h2_split.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: polarity_reclass/results_h2_split.csv")


if __name__ == "__main__":
    main()
