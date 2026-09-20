"""
Full MDE/TOST/BF01 treatment for the event-study coefficients (#5
Specificity, #6 Sentiment, both at CAR[-1,+1], Model 3, Firm+Year FE,
firm-clustered SEs) -- same methodology as 02_analysis.py's Q-side panel
treatment, adapted for this design: a single model (not MODEL_SPECS'
nested sequence) and cluster(firm) SEs rather than HC3, matching what was
actually reported for this design throughout the event-study notebook.

Standardization convention: same as the panel side -- partial standardized
effect (coef * SD(x) / SD(y)) on this model's own complete-case sample
(N=209), using cluster SEs scaled the same way.

Run:  python3 null_check/07_analysis_event.py
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, "null_check")
from importlib import import_module
analysis_mod = import_module("02_analysis")  # reuse bf01_savage_dickey, tost_at_bound, tost_min_equivalence_bound

DATA_PATH = "null_check/data/ceo_letter_regression_sample_209.csv"

ALPHA = 0.05
POWER = 0.80
Z_CRIT_TWOSIDED = stats.norm.ppf(1 - ALPHA / 2)
Z_POWER = stats.norm.ppf(POWER)
Z_ONESIDED_95 = stats.norm.ppf(1 - ALPHA)

TOST_BOUNDS_SD = [0.05, 0.10, 0.20]
BF_PRIORS_SD = [0.1, 0.5]

# Already computed (Step 9.3 of CEO_Letter_Event_Study.ipynb, this same
# session) -- pulled in per instruction rather than re-derived. Pairs-
# cluster bootstrap, N_BOOTSTRAP=2000, seed=42, on this exact CAR[-1,+1]
# Model 3 Firm+Year FE specification.
BOOTSTRAP_REPORTED = {
    "Specificity": {"point": -0.0040, "boot_se": 0.0040, "ci_lo": -0.0131, "ci_hi": 0.0030,
                     "hc3_se": 0.0077, "cluster_se": 0.0036},
    "Sentiment": {"point": -0.0279, "boot_se": 0.0183, "ci_lo": -0.0638, "ci_hi": 0.0084,
                  "hc3_se": 0.0225, "cluster_se": 0.0193},
}


def analyze_term(dv, var, sample, model, se_source):
    coef = model.params[var]
    se = model.bse[var]
    p = model.pvalues[var]
    n = int(model.nobs)
    ci_lo, ci_hi = coef - Z_CRIT_TWOSIDED * se, coef + Z_CRIT_TWOSIDED * se
    sd_dv = sample[dv].std(ddof=1)
    sd_x = sample[var].std(ddof=1)

    std_coef = coef * sd_x / sd_dv
    std_se = se * sd_x / sd_dv

    mde_raw = (Z_CRIT_TWOSIDED + Z_POWER) * se
    mde_std = mde_raw * sd_x / sd_dv

    tost_results = {
        f"equivalent_within_{b}SD": analysis_mod.tost_at_bound(std_coef, std_se, b)
        for b in TOST_BOUNDS_SD
    }
    min_equiv_bound_std = analysis_mod.tost_min_equivalence_bound(std_coef, std_se)

    bf_results = {
        f"BF01_prior_N(0,{prior_sd})": analysis_mod.bf01_savage_dickey(std_coef, std_se, prior_sd)
        for prior_sd in BF_PRIORS_SD
    }

    return {
        "dv": dv, "term": var, "se_source": se_source,
        "coef": coef, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p, "n": n,
        "sd_dv": sd_dv, "sd_predictor": sd_x,
        "std_coef": std_coef, "std_se": std_se,
        "std_ci_lo": std_coef - Z_CRIT_TWOSIDED * std_se,
        "std_ci_hi": std_coef + Z_CRIT_TWOSIDED * std_se,
        "mde_raw_80pct_power": mde_raw, "mde_std_80pct_power": mde_std,
        "min_equivalence_bound_std": min_equiv_bound_std,
        **tost_results, **bf_results,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    formula = "Q('CAR[-1,+1]') ~ Specificity + Sentiment + C(Company) + C(Q('Fiscal Year'))"

    m_cluster = smf.ols(formula=formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["Company"]})
    m_hc3 = smf.ols(formula=formula, data=df).fit(cov_type="HC3")

    rows = []
    for var in ["Specificity", "Sentiment"]:
        rows.append(analyze_term("CAR[-1,+1]", var, df, m_cluster, "cluster(firm) [primary, as reported]"))
        rows.append(analyze_term("CAR[-1,+1]", var, df, m_hc3, "HC3 [comparison]"))

        # Bootstrap SE run through the same standardized/MDE/TOST/BF01
        # machinery, using the already-computed point estimate + bootstrap
        # SE (not re-derived) so the three SE sources are directly
        # comparable on this table, not just listed separately.
        boot = BOOTSTRAP_REPORTED[var]
        sd_dv = df["CAR[-1,+1]"].std(ddof=1)
        sd_x = df[var].std(ddof=1)
        std_coef = boot["point"] * sd_x / sd_dv
        std_se = boot["boot_se"] * sd_x / sd_dv
        mde_raw = (Z_CRIT_TWOSIDED + Z_POWER) * boot["boot_se"]
        mde_std = mde_raw * sd_x / sd_dv
        tost_results = {f"equivalent_within_{b}SD": analysis_mod.tost_at_bound(std_coef, std_se, b)
                         for b in TOST_BOUNDS_SD}
        bf_results = {f"BF01_prior_N(0,{p})": analysis_mod.bf01_savage_dickey(std_coef, std_se, p)
                      for p in BF_PRIORS_SD}
        rows.append({
            "dv": "CAR[-1,+1]", "term": var, "se_source": "pairs-cluster bootstrap [pulled in, Step 9.3]",
            "coef": boot["point"], "se": boot["boot_se"],
            "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "p": np.nan, "n": 209,
            "sd_dv": sd_dv, "sd_predictor": sd_x, "std_coef": std_coef, "std_se": std_se,
            "std_ci_lo": std_coef - Z_CRIT_TWOSIDED * std_se, "std_ci_hi": std_coef + Z_CRIT_TWOSIDED * std_se,
            "mde_raw_80pct_power": mde_raw, "mde_std_80pct_power": mde_std,
            "min_equivalence_bound_std": analysis_mod.tost_min_equivalence_bound(std_coef, std_se),
            **tost_results, **bf_results,
        })

    results = pd.DataFrame(rows)
    results.to_csv("null_check/results_event.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(results[["term", "se_source", "coef", "se", "ci_lo", "ci_hi", "n",
                    "std_coef", "std_ci_lo", "std_ci_hi"]].round(4).to_string(index=False))
    print()
    print(results[["term", "se_source", "mde_std_80pct_power", "min_equivalence_bound_std",
                    "equivalent_within_0.05SD", "equivalent_within_0.1SD", "equivalent_within_0.2SD",
                    "BF01_prior_N(0,0.1)", "BF01_prior_N(0,0.5)"]].round(4).to_string(index=False))
    print(f"\nSaved: null_check/results_event.csv ({len(results)} rows)")


if __name__ == "__main__":
    main()
