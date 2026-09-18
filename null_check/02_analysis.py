"""
Steps 2-5 of the null-result check, for Table 5a (Q as DV) -- the paper's
main hypothesis tests. Scope note (flagged per the user's own point 7):
the request's bracketed placeholders for "key coefficients" were never
filled in. Table 5a's four nested models (ESG_sentence_ratio,
Corporate_ESG_share, Corporate_ESG_share+Verified_ESG_share [[H2, the
headline test]], Corporate_ESG_share+Specificity_mean) are treated here as
the primary key coefficients, since they are the paper's main valuation
test and the ones most discussed in this session. The ROE-DV parallel
(Section 13) is treated as a robustness/alternative-outcome check (step 6)
rather than a second full TOST/Bayes track, to keep scope bounded -- if
that's the wrong read of "key coefficients," say so and this is easy to
extend.

Run:  python3 null_check/02_analysis.py
"""

import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, MODEL_SPECS, build_model_sample, load_and_reconstruct

ALPHA = 0.05
POWER = 0.80
Z_CRIT_TWOSIDED = stats.norm.ppf(1 - ALPHA / 2)   # 1.95996 -- HC3 output uses z, not t, so match that
Z_POWER = stats.norm.ppf(POWER)                    # 0.84162
Z_ONESIDED_95 = stats.norm.ppf(1 - ALPHA)          # 1.64485 -- TOST critical value

TOST_BOUNDS_SD = [0.05, 0.10, 0.20]
BF_PRIORS_SD = [0.1, 0.5]  # standardized-scale Normal(0, prior_sd) priors


def bf01_savage_dickey(d_hat, se_d, prior_sd):
    """
    Approximate BF01 (evidence for H0 relative to H1) via the Savage-Dickey
    density ratio under a normal approximation to the likelihood:
      d_hat | delta ~ N(delta, se_d^2)          [likelihood]
      delta | H1     ~ N(0, prior_sd^2)          [prior under H1]
      delta | H0     =  0                        [point null]
    Marginal likelihoods: L(H0) = dnorm(d_hat; 0, se_d)
                           L(H1) = dnorm(d_hat; 0, sqrt(se_d^2 + prior_sd^2))
    BF01 = L(H0) / L(H1). BF01 > 1 favors the null; BF01 < 1 favors the
    alternative. Standard closed-form device (Wagenmakers et al. 2010;
    Dienes 2014 calculator uses the same approximation) -- not a full
    posterior sampler, deliberately, since the likelihood here is already
    well approximated as normal (OLS coefficient, HC3 SE).
    """
    l_h0 = stats.norm.pdf(d_hat, loc=0, scale=se_d)
    l_h1 = stats.norm.pdf(d_hat, loc=0, scale=np.sqrt(se_d**2 + prior_sd**2))
    return l_h0 / l_h1


def tost_min_equivalence_bound(coef, se):
    """
    Smallest symmetric bound b (raw units) at which TOST equivalence is
    established at alpha=0.05 per one-sided test: equivalence holds iff
    the (1 - 2*alpha) CI (coef +/- z_onesided_95 * se) lies entirely
    within [-b, b], i.e. b >= |coef| + z_onesided_95 * se. The smallest
    such b is exactly |coef| + z_onesided_95 * se -- computed directly,
    not by grid search, though the fixed bounds below are also checked
    for the report table.
    """
    return abs(coef) + Z_ONESIDED_95 * se


def tost_at_bound(coef, se, bound):
    """True/False: is the effect statistically equivalent to zero within +/-bound?"""
    ci_lo = coef - Z_ONESIDED_95 * se
    ci_hi = coef + Z_ONESIDED_95 * se
    return (ci_lo >= -bound) and (ci_hi <= bound)


def analyze_model(dv, label, esg_cols, sample, model):
    rows = []
    sd_dv = sample[dv].std(ddof=1)
    for var in esg_cols:
        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        n = int(model.nobs)
        ci_lo, ci_hi = coef - Z_CRIT_TWOSIDED * se, coef + Z_CRIT_TWOSIDED * se
        sd_x = sample[var].std(ddof=1)

        # Standardized (partial) effect: change in DV, in DV-SDs, per 1-SD
        # change in the predictor, holding the other regressors fixed --
        # the standard way to make an FE-model coefficient comparable
        # across variables measured on different scales.
        std_coef = coef * sd_x / sd_dv
        std_se = se * sd_x / sd_dv

        mde_raw = (Z_CRIT_TWOSIDED + Z_POWER) * se
        mde_std = mde_raw * sd_x / sd_dv

        tost_results = {
            f"equivalent_within_{b}SD": tost_at_bound(std_coef, std_se, b)
            for b in TOST_BOUNDS_SD
        }
        min_equiv_bound_std = tost_min_equivalence_bound(std_coef, std_se)

        bf_results = {
            f"BF01_prior_N(0,{prior_sd})": bf01_savage_dickey(std_coef, std_se, prior_sd)
            for prior_sd in BF_PRIORS_SD
        }

        rows.append({
            "dv": dv,
            "model": label,
            "term": var,
            "coef": coef,
            "se": se,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p": p,
            "n": n,
            "sd_dv": sd_dv,
            "sd_predictor": sd_x,
            "std_coef": std_coef,
            "std_se": std_se,
            "std_ci_lo": std_coef - Z_CRIT_TWOSIDED * std_se,
            "std_ci_hi": std_coef + Z_CRIT_TWOSIDED * std_se,
            "mde_raw_80pct_power": mde_raw,
            "mde_std_80pct_power": mde_std,
            "min_equivalence_bound_std": min_equiv_bound_std,
            **tost_results,
            **bf_results,
        })
    return rows


def main():
    df, meta = load_and_reconstruct()

    all_rows = []
    for label, esg_cols in MODEL_SPECS:
        sample, dropped = build_model_sample(df, esg_cols, dv="Q", label=label)
        formula = f"Q ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
        m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
        all_rows.extend(analyze_model("Q", label, esg_cols, sample, m))

    results = pd.DataFrame(all_rows)
    results.to_csv("null_check/results_q.csv", index=False)

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

    print(f"\nSaved: null_check/results_q.csv ({len(results)} rows)")


if __name__ == "__main__":
    main()
