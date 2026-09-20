"""
Independent reproduction check for coefficient #7 (the pairwise
"credible optimism" [Cluster1_HighSpec] vs. "cheap-talk" [Cluster0_High-
VolLowSpec] comparison at CAR[-1,+1]) before any equivalence/power/Bayes
analysis, per instruction.

Run:  python3 null_check/08_reproduce_pairwise.py
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "null_check")
from common_event import load_and_merge

REPORTED = {"diff": -0.0075, "cluster_se": 0.0116, "n": 209}


def main():
    panel14, meta = load_and_merge()
    print(f"Merged panel: {len(panel14)} rows (event-study reg_sample: {meta['n_reg_sample']})")
    print(f"Missing cluster_3d: {meta['n_missing_cluster']} of {len(panel14)}")

    controls = ["Leverage", "Size", "ROA", "Revenue_Growth"]
    required = ["cluster_3d", "Cluster1_HighSpec", "Cluster0_HighVolLowSpec", "CAR[-1,+1]"] + controls
    sample = panel14.dropna(subset=required).copy()
    sample["Cluster1_HighSpec"] = sample["Cluster1_HighSpec"].astype(int)
    sample["Cluster0_HighVolLowSpec"] = sample["Cluster0_HighVolLowSpec"].astype(int)

    cell_sizes = sample["cluster_3d"].value_counts().sort_index()
    print(f"\nComplete-case N: {len(sample)}")
    print(f"Cell sizes: {cell_sizes.to_dict()}")

    formula = ("Q('CAR[-1,+1]') ~ Cluster1_HighSpec + Cluster0_HighVolLowSpec + "
               "Leverage + Size + ROA + Revenue_Growth + C(Company) + C(Q('Fiscal Year'))")
    m_cluster = smf.ols(formula=formula, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["Company"]}
    )

    tt = m_cluster.t_test("Cluster1_HighSpec - Cluster0_HighVolLowSpec = 0")
    diff = float(np.ravel(tt.effect)[0])
    se = float(np.ravel(tt.sd)[0])
    p = float(np.ravel(tt.pvalue)[0])
    n = int(m_cluster.nobs)

    print(f"\nReproduced: diff={diff:.4f}  cluster_se={se:.4f}  p={p:.4f}  N={n}")
    print(f"Reported:   diff={REPORTED['diff']:.4f}  cluster_se={REPORTED['cluster_se']:.4f}  N={REPORTED['n']}")

    ok = abs(diff - REPORTED["diff"]) < 5e-4 and abs(se - REPORTED["cluster_se"]) < 5e-4 and n == REPORTED["n"]
    print(f"\n{'[PASS]' if ok else '[FAIL]'} coefficient #7 reproduction")
    if not ok:
        sys.exit(1)

    # Save the fitted model artifacts needed by 09_analysis_pairwise.py,
    # so that script doesn't need to re-derive the merge/sample itself.
    sample.to_csv("null_check/data/panel14_complete_case.csv", index=False)
    print(f"\nSaved complete-case sample: null_check/data/panel14_complete_case.csv ({len(sample)} rows)")


if __name__ == "__main__":
    main()
