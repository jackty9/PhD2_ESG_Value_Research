"""
Core regression: E and S+G-combined together in one model (the "E and SG
combined" request), on Q and ROE, same controls/FE/HC3 as Table 5a. Also
reports each alone for context. Core results only (coef/SE/CI/p/N) -- not
yet the full battery (MDE/TOST/Bayes/Holm-BH) esg_pillars/ has; extend on
request once these are reviewed.

Run:  python3 esg_e_sg/02_regression.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, "esg_e_sg")
from common import CTRL_5A_FORMULA, build_model_sample, load_and_reconstruct

Z = stats.norm.ppf(0.975)


def fit(df, dv, esg_cols, label):
    sample, dropped = build_model_sample(df, esg_cols, dv=dv, label=label)
    formula = f"{dv} ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
    m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
    return m, sample, dropped


def main():
    df, _ = load_and_reconstruct()
    rows = []

    for dv in ["Q", "ROE"]:
        # ---- PRIMARY: E and S+G combined, jointly, one regression ----
        m_joint, sample_joint, dropped = fit(df, dv, ["Corporate_E_share_lag", "Corporate_SG_share_lag"],
                                              f"Joint E+SG, {dv}")
        print(f"\n{'=' * 78}\n{dv}: E + S+G combined (joint model) -- PRIMARY\n{'=' * 78}")
        print(f"N = {int(m_joint.nobs)}, R-squared = {m_joint.rsquared:.4f}, "
              f"Condition number: {m_joint.condition_number:.3e}")
        print(f"Firms dropped (insufficient obs): {dropped}")
        print(m_joint.summary())
        for var in ["Corporate_E_share_lag", "Corporate_SG_share_lag"]:
            rows.append({
                "dv": dv, "spec": "Joint (E + SG combined)", "term": var,
                "coef": m_joint.params[var], "se": m_joint.bse[var],
                "ci_lo": m_joint.params[var] - Z * m_joint.bse[var],
                "ci_hi": m_joint.params[var] + Z * m_joint.bse[var],
                "p": m_joint.pvalues[var], "n": int(m_joint.nobs),
            })

        # ---- context: each alone ----
        for pillar, col in [("E", "Corporate_E_share_lag"), ("SG", "Corporate_SG_share_lag")]:
            m_alone, sample_alone, _ = fit(df, dv, [col], f"{pillar} alone, {dv}")
            rows.append({
                "dv": dv, "spec": f"{pillar} alone", "term": col,
                "coef": m_alone.params[col], "se": m_alone.bse[col],
                "ci_lo": m_alone.params[col] - Z * m_alone.bse[col],
                "ci_hi": m_alone.params[col] + Z * m_alone.bse[col],
                "p": m_alone.pvalues[col], "n": int(m_alone.nobs),
            })

    out = pd.DataFrame(rows)
    out.to_csv("esg_e_sg/results_core.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"\n{'=' * 78}\nSummary: joint (primary) vs. each alone (context)\n{'=' * 78}")
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: esg_e_sg/results_core.csv")


if __name__ == "__main__":
    main()
