"""
Fits Corporate_{X}_share_lag + Specificity_{X}_mean_lag jointly, per pillar
(X in {E, SG}), for both Q and ROE -- mirroring Table 5a Model 4's
structure, split by corpus instead of combined. Reports the sentence count
underlying each firm-year's pillar specificity mean
(n_E_corporate_tier1plus / n_SG_corporate_tier1plus, already in the data:
it's exactly the denominator of Specificity_{X}_mean), flags firm-years
below MIN_SENTENCES, and reports both with and without them. Firm-years
with ZERO qualifying sentences in a pillar are never imputed -- their
Specificity_{X}_mean is NaN by construction (see common.py's zero-fill
note: only the SHARE variables are zero-filled, not the specificity mean),
so build_model_sample()'s complete-case filter already drops them; this
script reports exactly how many and why.

Then runs the same battery as esg_pillars/02_analysis.py (std effect, CI,
MDE at 80% power, TOST at +/-0.10/+/-0.20 SD, Holm + BH across all raw
p-values in the battery, BF01 under two priors) on all resulting
coefficients.

Run:  python3 esg_e_sg/03_pillar_specificity.py
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "esg_e_sg")
from common import CTRL_5A_FORMULA, build_model_sample, load_and_reconstruct

MIN_SENTENCES = 5

ALPHA = 0.05
POWER = 0.80
Z_CRIT = stats.norm.ppf(1 - ALPHA / 2)
Z_POWER = stats.norm.ppf(POWER)
Z_ONESIDED_95 = stats.norm.ppf(1 - ALPHA)
TOST_BOUNDS_SD = [0.10, 0.20]
BF_PRIORS_SD = [0.1, 0.5]


def tost_at_bound(coef, se, bound):
    ci_lo, ci_hi = coef - Z_ONESIDED_95 * se, coef + Z_ONESIDED_95 * se
    return (ci_lo >= -bound) and (ci_hi <= bound)


def bf01_savage_dickey(d_hat, se_d, prior_sd):
    l_h0 = stats.norm.pdf(d_hat, loc=0, scale=se_d)
    l_h1 = stats.norm.pdf(d_hat, loc=0, scale=np.sqrt(se_d**2 + prior_sd**2))
    return l_h0 / l_h1


def effect_row(dv, spec, sample_label, var, coef, se, p, n, sample, dv_col):
    sd_dv = sample[dv_col].std(ddof=1)
    sd_x = sample[var].std(ddof=1)
    ci_lo, ci_hi = coef - Z_CRIT * se, coef + Z_CRIT * se
    std_coef = coef * sd_x / sd_dv
    std_se = se * sd_x / sd_dv
    mde_raw = (Z_CRIT + Z_POWER) * se
    mde_std = mde_raw * sd_x / sd_dv
    bounds = {b: tost_at_bound(std_coef, std_se, b) for b in TOST_BOUNDS_SD}
    bf = {f"BF01_N(0,{p_sd})": bf01_savage_dickey(std_coef, std_se, p_sd) for p_sd in BF_PRIORS_SD}
    return {
        "dv": dv, "spec": spec, "sample": sample_label, "term": var,
        "coef": coef, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_raw": p, "n": n,
        "std_coef": std_coef, "std_se": std_se,
        "std_ci_lo": std_coef - Z_CRIT * std_se, "std_ci_hi": std_coef + Z_CRIT * std_se,
        "mde_std_80pct_power": mde_std,
        "equiv_within_0.10SD": bounds[0.10], "equiv_within_0.20SD": bounds[0.20],
        **bf,
    }


def report_sentence_counts(df):
    print(f"{'=' * 78}\nSentence counts underlying each firm-year's pillar specificity mean\n{'=' * 78}")
    for pillar in ["E", "SG"]:
        col = f"n_{pillar}_corporate_tier1plus"
        n_zero = (df[col] == 0).sum()
        n_below_min = ((df[col] > 0) & (df[col] < MIN_SENTENCES)).sum()
        n_valid = (df[col] >= MIN_SENTENCES).sum()
        n_missing = df[col].isna().sum()
        print(f"\n{pillar}: n={col}")
        print(df[col].describe())
        print(f"Zero-sentence firm-years (Specificity_{pillar}_mean is NaN, never imputed, "
              f"dropped from models): {n_zero}")
        print(f"1-{MIN_SENTENCES - 1} sentences (below MIN_SENTENCES={MIN_SENTENCES}, flagged): {n_below_min}")
        print(f"{MIN_SENTENCES}+ sentences: {n_valid}")
        print(f"Missing entirely (firm-year not in the pillar merge at all): {n_missing}")


def fit_and_collect(df, dv, pillar, restrict_min_sentences, rows_out):
    share_col = f"Corporate_{pillar}_share_lag"
    spec_col = f"Specificity_{pillar}_mean_lag"
    count_col = f"n_{pillar}_corporate_tier1plus"

    working = df
    if restrict_min_sentences:
        working = df[(df[count_col].isna()) | (df[count_col] >= MIN_SENTENCES) | (df[count_col] == 0)]
        # NaN/0 rows get dropped anyway by build_model_sample's dropna on spec_col
        # (Specificity_mean is NaN for 0 sentences) -- this filter only removes the
        # 1..MIN_SENTENCES-1 "thin but nonzero" rows from the eligible pool.
        working = df[~((df[count_col] > 0) & (df[count_col] < MIN_SENTENCES))]

    sample, dropped_firms = build_model_sample(working, [share_col, spec_col], dv=dv,
                                                 label=f"{pillar}, {dv}, restrict={restrict_min_sentences}")
    formula = f"{dv} ~ {share_col} + {spec_col} + {CTRL_5A_FORMULA}"
    m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")

    sample_label = f"excl. <{MIN_SENTENCES}-sentence firm-years" if restrict_min_sentences else "full (incl. thin firm-years)"
    for var in [share_col, spec_col]:
        rows_out.append(effect_row(dv, f"{pillar} share+specificity", sample_label, var,
                                    m.params[var], m.bse[var], m.pvalues[var], int(m.nobs), sample, dv))
    return m, sample, dropped_firms


def main():
    df, meta = load_and_reconstruct()
    report_sentence_counts(df)

    rows = []
    fit_log = []
    for dv in ["Q", "ROE"]:
        for pillar in ["E", "SG"]:
            for restrict in [False, True]:
                m, sample, dropped = fit_and_collect(df, dv, pillar, restrict, rows)
                fit_log.append((dv, pillar, restrict, int(m.nobs), dropped, m.condition_number))

    print(f"\n{'=' * 78}\nModel N and condition numbers\n{'=' * 78}")
    for dv, pillar, restrict, n, dropped, cond in fit_log:
        label = f"excl. <{MIN_SENTENCES}" if restrict else "full"
        print(f"{dv} / {pillar} / {label}: N={n}, cond#={cond:.3e}, firms dropped (insuff. obs)={dropped}")

    results = pd.DataFrame(rows)

    # Holm + BH across ALL raw p-values produced in this battery (8 models x
    # 2 terms = 16 tests: 2 pillars x 2 DVs x 2 samples x 2 terms).
    _, p_holm, _, _ = multipletests(results["p_raw"].values, alpha=0.05, method="holm")
    _, p_bh, _, _ = multipletests(results["p_raw"].values, alpha=0.05, method="fdr_bh")
    results["p_holm"] = p_holm
    results["p_bh"] = p_bh

    results.to_csv("esg_e_sg/results_pillar_specificity.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n{'=' * 78}\nFull results\n{'=' * 78}")
    print(results[["dv", "spec", "sample", "term", "coef", "se", "ci_lo", "ci_hi",
                    "p_raw", "p_holm", "p_bh", "n"]].round(4).to_string(index=False))
    print()
    print(results[["dv", "spec", "sample", "term", "std_coef", "mde_std_80pct_power",
                    "equiv_within_0.10SD", "equiv_within_0.20SD",
                    "BF01_N(0,0.1)", "BF01_N(0,0.5)"]].round(4).to_string(index=False))

    print(f"\nSaved: esg_e_sg/results_pillar_specificity.csv ({len(results)} rows)")


if __name__ == "__main__":
    main()
