"""
ROE-DV parallel to 02_analysis.py's full MDE/TOST/BF01 treatment.

02_analysis.py originally scoped Table 5a's Q-based models as the primary
key coefficients and treated ROE as a robustness variant only (see its own
docstring). This session's request treats one ROE coefficient --
Specificity_mean_lag in Model 4 -- as a primary "key coefficient" in its
own right (coefficient #2 of 7), so it gets the same full treatment here,
not just the point-estimate/SE robustness check 03_robustness.py already
ran. All four ROE models are computed (not just Model 4) for consistency
with the Q-side table, even though only Model 4's Specificity_mean_lag was
specifically requested.

Run:  python3 null_check/05_analysis_roe.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, MODEL_SPECS, build_model_sample, load_and_reconstruct
from importlib import import_module

analysis_mod = import_module("02_analysis")


def main():
    df, meta = load_and_reconstruct()

    all_rows = []
    for label, esg_cols in MODEL_SPECS:
        sample, dropped = build_model_sample(df, esg_cols, dv="ROE", label=label)
        formula = f"ROE ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
        m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
        all_rows.extend(analysis_mod.analyze_model("ROE", label, esg_cols, sample, m))

    results = pd.DataFrame(all_rows)
    results.to_csv("null_check/results_roe.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print(results[[
        "model", "term", "coef", "se", "ci_lo", "ci_hi", "p", "n",
        "std_coef", "std_ci_lo", "std_ci_hi",
    ]].round(4).to_string(index=False))
    print()
    print(results[[
        "model", "term", "mde_std_80pct_power", "min_equivalence_bound_std",
        "equivalent_within_0.05SD", "equivalent_within_0.1SD", "equivalent_within_0.2SD",
        "BF01_prior_N(0,0.1)", "BF01_prior_N(0,0.5)",
    ]].round(4).to_string(index=False))

    print(f"\nSaved: null_check/results_roe.csv ({len(results)} rows)")


if __name__ == "__main__":
    main()
