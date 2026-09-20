"""
Step 1 (per the requested protocol): independently reproduce Table 5a
(Q as DV) and its ROE-DV parallel (Section 13) from the raw exported panel,
and compare against the coefficients already reported in the paper /
conversation. STOP-and-report if any of them don't match -- do not proceed
to the rest of the null-result pipeline on unverified numbers.

Run from the repo root:  python3 null_check/01_reproduce.py
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, MODEL_SPECS, build_model_sample, load_and_reconstruct

# ----------------------------------------------------------------------
# Ground truth: coefficients as already reported back to the user in this
# conversation (Table 5a, Q as DV; and its ROE-DV parallel, Section 13),
# copied verbatim from the notebook output the user pasted, not re-derived.
# Tolerance is loose (abs diff < 5e-4 on coef, < 5e-4 on SE) to allow for
# floating point / pandas version differences, not for a different result.
# ----------------------------------------------------------------------
REPORTED = {
    "Q": {
        ("Model 1", "ESG_sentence_ratio_lag"): dict(coef=0.0290, se=0.0302, p=0.3372, n=183),
        ("Model 2", "Corporate_ESG_share_lag"): dict(coef=0.0310, se=0.0325, p=0.3401, n=183),
        ("Model 3", "Corporate_ESG_share_lag"): dict(coef=0.0680, se=0.0476, p=0.1533, n=183),
        ("Model 3", "Verified_ESG_share_lag"): dict(coef=-0.0746, se=0.0626, p=0.2329, n=183),
        ("Model 4", "Corporate_ESG_share_lag"): dict(coef=0.0305, se=0.0330, p=0.3490, n=183),
        ("Model 4", "Specificity_mean_lag"): dict(coef=0.0008, se=0.0060, p=0.8870, n=183),
    },
    "ROE": {
        ("Model 1", "ESG_sentence_ratio_lag"): dict(coef=0.0062, se=0.0182, p=0.7315, n=183),
        ("Model 2", "Corporate_ESG_share_lag"): dict(coef=0.0047, se=0.0178, p=0.7909, n=183),
        ("Model 3", "Corporate_ESG_share_lag"): dict(coef=0.0391, se=0.0350, p=0.2636, n=183),
        ("Model 3", "Verified_ESG_share_lag"): dict(coef=-0.0695, se=0.0527, p=0.1873, n=183),
        ("Model 4", "Corporate_ESG_share_lag"): dict(coef=0.0076, se=0.0181, p=0.6735, n=183),
        ("Model 4", "Specificity_mean_lag"): dict(coef=-0.0051, se=0.0049, p=0.3033, n=183),
    },
}

COEF_TOL = 5e-4
SE_TOL = 5e-4


def fit_all(df, dv):
    results = {}
    samples = {}
    for label, esg_cols in MODEL_SPECS:
        sample, dropped = build_model_sample(df, esg_cols, dv=dv, label=label)
        formula = f"{dv} ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
        m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
        results[label] = m
        samples[label] = (sample, dropped)
    return results, samples


def compare(dv, models):
    print(f"\n{'=' * 78}\n{dv} as DV -- reproduction check\n{'=' * 78}")
    all_pass = True
    for (label, esg_cols) in MODEL_SPECS:
        m = models[label]
        n_fit = int(m.nobs)
        for var in esg_cols:
            key = (label, var)
            rep = REPORTED[dv].get(key)
            if rep is None:
                continue
            coef, se, p = m.params[var], m.bse[var], m.pvalues[var]
            coef_ok = abs(coef - rep["coef"]) < COEF_TOL
            se_ok = abs(se - rep["se"]) < SE_TOL
            n_ok = n_fit == rep["n"]
            status = "PASS" if (coef_ok and se_ok and n_ok) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"[{status}] {label}: {var}")
            print(f"    reproduced: coef={coef:.4f}  se={se:.4f}  p={p:.4f}  N={n_fit}")
            print(f"    reported:   coef={rep['coef']:.4f}  se={rep['se']:.4f}  "
                  f"p={rep['p']:.4f}  N={rep['n']}")
    return all_pass


def main():
    df, meta = load_and_reconstruct()
    print(f"Raw panel rows (cell 33 export): {meta['n_raw']}")
    print(f"After dropna(subset=['cluster_lag']) (matches notebook's cell 36 filter): "
          f"{meta['n_after_cluster_lag_drop']}")
    expected_n209 = 209
    if meta["n_after_cluster_lag_drop"] != expected_n209:
        print(f"NOTE: expected {expected_n209} rows at this stage (matches "
              f"'Specificity_mean_lag: 184 non-null of 209' and 'ROE descriptive "
              f"stats: count 209' already reported) -- got "
              f"{meta['n_after_cluster_lag_drop']} instead. Proceeding, but flag "
              f"this discrepancy in the report.")

    models_q, samples_q = fit_all(df, "Q")
    models_roe, samples_roe = fit_all(df, "ROE")

    pass_q = compare("Q", models_q)
    pass_roe = compare("ROE", models_roe)

    print(f"\n{'=' * 78}\nOVERALL: {'ALL MODELS REPRODUCED' if (pass_q and pass_roe) else 'REPRODUCTION FAILED -- STOP'}\n{'=' * 78}")

    if not (pass_q and pass_roe):
        sys.exit(1)

    return df, models_q, models_roe, samples_q, samples_roe


if __name__ == "__main__":
    main()
