"""
Robustness (c) for panel coefficients #1, #3, #4: re-run with the
unrestricted (Corporate+Contextual, i.e. "_allrelevance") construction of
Verified_ESG_share and Specificity_mean substituted for the baseline
Corporate-only, Verified construction -- same models (2, 3, 4), same
controls/FE/HC3, only the ESG-share variable construction changes.

Run:  python3 null_check/11_robustness_unrestricted.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, build_model_sample, load_and_reconstruct, safe_lag


def main():
    df, _ = load_and_reconstruct()
    for col in ["Verified_ESG_share_allrelevance", "Specificity_mean_allrelevance"]:
        df[col + "_lag"] = safe_lag(df, col)

    variants = [
        ("Model 2", ["Corporate_ESG_share_lag"], ["Corporate_ESG_share_lag"]),
        ("Model 3", ["Corporate_ESG_share_lag", "Verified_ESG_share_lag"],
         ["Corporate_ESG_share_lag", "Verified_ESG_share_allrelevance_lag"]),
        ("Model 4", ["Corporate_ESG_share_lag", "Specificity_mean_lag"],
         ["Corporate_ESG_share_lag", "Specificity_mean_allrelevance_lag"]),
    ]

    rows = []
    for label, baseline_cols, unrestricted_cols in variants:
        sample_base, _ = build_model_sample(df, baseline_cols, dv="Q", label=label)
        formula_base = f"Q ~ {' + '.join(baseline_cols)} + {CTRL_5A_FORMULA}"
        m_base = smf.ols(formula=formula_base, data=sample_base).fit(cov_type="HC3")

        sample_unres, _ = build_model_sample(df, unrestricted_cols, dv="Q", label=label + " (unrestricted)")
        formula_unres = f"Q ~ {' + '.join(unrestricted_cols)} + {CTRL_5A_FORMULA}"
        m_unres = smf.ols(formula=formula_unres, data=sample_unres).fit(cov_type="HC3")

        for base_term, unres_term in zip(baseline_cols, unrestricted_cols):
            rows.append({
                "model": label, "term": base_term, "variant": "baseline (Corporate-only, Verified)",
                "coef": m_base.params[base_term], "se": m_base.bse[base_term],
                "p": m_base.pvalues[base_term], "n": int(m_base.nobs),
            })
            rows.append({
                "model": label, "term": base_term, "variant": "unrestricted (Corporate+Contextual)",
                "coef": m_unres.params[unres_term], "se": m_unres.bse[unres_term],
                "p": m_unres.pvalues[unres_term], "n": int(m_unres.nobs),
            })

    results = pd.DataFrame(rows)
    results.to_csv("null_check/results_robustness_unrestricted.csv", index=False)
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: null_check/results_robustness_unrestricted.csv")


if __name__ == "__main__":
    main()
