"""
Full MDE/TOST/BF01 treatment for coefficient #7 (the pairwise credible-
optimism vs. cheap-talk comparison at CAR[-1,+1]).

Standardization convention differs from the other six coefficients, and
this is flagged explicitly rather than silently reused: Cluster1_HighSpec
and Cluster0_HighVolLowSpec are dummy/group-membership variables, not
continuous regressors, so there is no natural "SD(x)" to multiply by the
way there is for e.g. Corporate_ESG_share_lag (a continuous share).
Instead this uses the standard Cohen's-d-style convention for a two-group
contrast: the standardized effect is the raw group difference divided by
SD(DV) directly -- treating the full 0-vs-1 group contrast as the
meaningful "one unit," not a fractional SD-of-predictor unit.

Run:  python3 null_check/09_analysis_pairwise.py
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "null_check")
from importlib import import_module
analysis_mod = import_module("02_analysis")

ALPHA = 0.05
POWER = 0.80
Z_CRIT_TWOSIDED = stats.norm.ppf(1 - ALPHA / 2)
Z_POWER = stats.norm.ppf(POWER)

TOST_BOUNDS_SD = [0.05, 0.10, 0.20]
BF_PRIORS_SD = [0.1, 0.5]

SAMPLE_PATH = "null_check/data/panel14_complete_case.csv"


def main():
    sample = pd.read_csv(SAMPLE_PATH)
    import statsmodels.formula.api as smf

    formula = ("Q('CAR[-1,+1]') ~ Cluster1_HighSpec + Cluster0_HighVolLowSpec + "
               "Leverage + Size + ROA + Revenue_Growth + C(Company) + C(Q('Fiscal Year'))")
    m_cluster = smf.ols(formula=formula, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["Company"]}
    )
    tt = m_cluster.t_test("Cluster1_HighSpec - Cluster0_HighVolLowSpec = 0")

    coef = float(np.ravel(tt.effect)[0])
    se = float(np.ravel(tt.sd)[0])
    p = float(np.ravel(tt.pvalue)[0])
    n = int(m_cluster.nobs)
    sd_dv = sample["CAR[-1,+1]"].std(ddof=1)

    ci_lo, ci_hi = coef - Z_CRIT_TWOSIDED * se, coef + Z_CRIT_TWOSIDED * se

    std_coef = coef / sd_dv
    std_se = se / sd_dv

    mde_raw = (Z_CRIT_TWOSIDED + Z_POWER) * se
    mde_std = mde_raw / sd_dv

    tost_results = {f"equivalent_within_{b}SD": analysis_mod.tost_at_bound(std_coef, std_se, b)
                    for b in TOST_BOUNDS_SD}
    min_equiv_bound_std = analysis_mod.tost_min_equivalence_bound(std_coef, std_se)
    bf_results = {f"BF01_prior_N(0,{prior_sd})": analysis_mod.bf01_savage_dickey(std_coef, std_se, prior_sd)
                  for prior_sd in BF_PRIORS_SD}

    row = {
        "dv": "CAR[-1,+1]", "term": "Cluster1_HighSpec - Cluster0_HighVolLowSpec",
        "coef": coef, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p, "n": n,
        "sd_dv": sd_dv, "std_coef": std_coef, "std_se": std_se,
        "std_ci_lo": std_coef - Z_CRIT_TWOSIDED * std_se, "std_ci_hi": std_coef + Z_CRIT_TWOSIDED * std_se,
        "mde_raw_80pct_power": mde_raw, "mde_std_80pct_power": mde_std,
        "min_equivalence_bound_std": min_equiv_bound_std,
        **tost_results, **bf_results,
    }
    results = pd.DataFrame([row])
    results.to_csv("null_check/results_pairwise.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(results[["term", "coef", "se", "ci_lo", "ci_hi", "p", "n",
                    "std_coef", "std_ci_lo", "std_ci_hi"]].round(4).to_string(index=False))
    print()
    print(results[["term", "mde_std_80pct_power", "min_equivalence_bound_std",
                    "equivalent_within_0.05SD", "equivalent_within_0.1SD", "equivalent_within_0.2SD",
                    "BF01_prior_N(0,0.1)", "BF01_prior_N(0,0.5)"]].round(4).to_string(index=False))
    print(f"\nSaved: null_check/results_pairwise.csv")


if __name__ == "__main__":
    main()
