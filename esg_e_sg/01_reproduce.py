"""
Reproduce Table 5a (Q as DV) before touching the E/SG-split models.

Run:  python3 esg_e_sg/01_reproduce.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "esg_e_sg")
from common import CTRL_5A_FORMULA, TABLE5A_MODEL_SPECS, build_model_sample, load_and_reconstruct

REPORTED_Q = {
    ("Model 1", "ESG_sentence_ratio_lag"): dict(coef=0.0290, se=0.0302, n=183),
    ("Model 2", "Corporate_ESG_share_lag"): dict(coef=0.0310, se=0.0325, n=183),
    ("Model 3", "Corporate_ESG_share_lag"): dict(coef=0.0680, se=0.0476, n=183),
    ("Model 3", "Verified_ESG_share_lag"): dict(coef=-0.0746, se=0.0626, n=183),
    ("Model 4", "Corporate_ESG_share_lag"): dict(coef=0.0305, se=0.0330, n=183),
    ("Model 4", "Specificity_mean_lag"): dict(coef=0.0008, se=0.0060, n=183),
}
TOL = 5e-4


def main():
    df, meta = load_and_reconstruct()
    print(f"Raw panel rows: {meta['n_raw']}")
    print(f"After dropna(subset=['cluster_lag']): {meta['n_after_cluster_lag_drop']} (expected 209)")
    print(f"Firm-years missing an E/SG measure before zero-fill: "
          f"{meta['n_missing_e_sg_before_zero_fill']} of {meta['n_before_e_sg_merge']} "
          f"(share vars zero-filled per the original notebook's own convention; "
          f"Specificity_*_mean stays NaN when genuinely undefined)")

    all_pass = True
    print(f"\n{'=' * 78}\nTable 5a (Q as DV) -- reproduction check\n{'=' * 78}")
    for label, esg_cols in TABLE5A_MODEL_SPECS:
        sample, _ = build_model_sample(df, esg_cols, dv="Q", label=label)
        formula = f"Q ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
        m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
        n_fit = int(m.nobs)
        for var in esg_cols:
            rep = REPORTED_Q[(label, var)]
            coef, se = m.params[var], m.bse[var]
            ok = abs(coef - rep["coef"]) < TOL and abs(se - rep["se"]) < TOL and n_fit == rep["n"]
            all_pass &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] {label}: {var}  coef={coef:.4f} (rep {rep['coef']:.4f})  "
                  f"se={se:.4f} (rep {rep['se']:.4f})  N={n_fit} (rep {rep['n']})")

    print(f"\n{'REPRODUCED -- proceeding' if all_pass else 'FAILED -- STOP'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
