"""
Step 2: reproduce Table 5a (Q as DV) before doing anything else. Same
ground-truth values and tolerance as null_check/01_reproduce.py (this
script is self-contained, not importing from null_check/, per instruction).

Run:  python3 esg_pillars/01_reproduce.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "esg_pillars")
from common import CTRL_5A_FORMULA, TABLE5A_MODEL_SPECS, build_model_sample, load_and_reconstruct

REPORTED_Q = {
    ("Model 1", "ESG_sentence_ratio_lag"): dict(coef=0.0290, se=0.0302, p=0.3372, n=183),
    ("Model 2", "Corporate_ESG_share_lag"): dict(coef=0.0310, se=0.0325, p=0.3401, n=183),
    ("Model 3", "Corporate_ESG_share_lag"): dict(coef=0.0680, se=0.0476, p=0.1533, n=183),
    ("Model 3", "Verified_ESG_share_lag"): dict(coef=-0.0746, se=0.0626, p=0.2329, n=183),
    ("Model 4", "Corporate_ESG_share_lag"): dict(coef=0.0305, se=0.0330, p=0.3490, n=183),
    ("Model 4", "Specificity_mean_lag"): dict(coef=0.0008, se=0.0060, p=0.8870, n=183),
}
COEF_TOL = 5e-4
SE_TOL = 5e-4


def main():
    df, meta = load_and_reconstruct()
    print(f"Raw panel rows: {meta['n_raw']}")
    print(f"After dropna(subset=['cluster_lag']): {meta['n_after_cluster_lag_drop']} "
          f"(expected 209, matching prior null_check confirmation)")
    print(f"Firm-years with a missing pillar ratio after the E/S/G merge: "
          f"{meta['n_missing_pillar_after_merge']} of {meta['n_before_pillar_merge']}")

    all_pass = True
    print(f"\n{'=' * 78}\nTable 5a (Q as DV) -- reproduction check\n{'=' * 78}")
    for label, esg_cols in TABLE5A_MODEL_SPECS:
        sample, dropped = build_model_sample(df, esg_cols, dv="Q", label=label)
        formula = f"Q ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
        m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
        n_fit = int(m.nobs)
        for var in esg_cols:
            rep = REPORTED_Q[(label, var)]
            coef, se, p = m.params[var], m.bse[var], m.pvalues[var]
            ok = (abs(coef - rep["coef"]) < COEF_TOL and abs(se - rep["se"]) < SE_TOL
                  and n_fit == rep["n"])
            all_pass &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] {label}: {var}  "
                  f"coef={coef:.4f} (rep {rep['coef']:.4f})  "
                  f"se={se:.4f} (rep {rep['se']:.4f})  N={n_fit} (rep {rep['n']})")

    print(f"\n{'OVERALL: REPRODUCED -- proceeding to pillar analysis' if all_pass else 'OVERALL: FAILED -- STOP'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
