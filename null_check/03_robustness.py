"""
Step 6: re-run the key Table 5a models under alternative reasonable
specifications, and report whether CIs stay tight around zero or the
conclusion changes. Four variants, each changing exactly one thing from
the baseline (HC3 SE, full N=183 sample, Q as DV, ctrl_5a controls):

  (a) cluster-robust SE by firm instead of HC3 (same sample/formula)
  (b) excl. Progressive (same exclusion convention already used
      elsewhere in the notebook -- Check 1 cell 57, Section 11.5)
  (c) ROE in place of Q as the DV (Section 13's own specification)
  (d) drop ROA from the controls (ROA is never significant in any of
      the notebook's models reported so far; checks whether the ESG
      coefficients depend on its inclusion)

Run:  python3 null_check/03_robustness.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, MODEL_SPECS, build_model_sample, load_and_reconstruct

CTRL_5A_NO_ROA = "SR_CEO_Letter_lag1 + Leverage + Size + Revenue_Growth + C(Year) + C(Company)"


def fit(dv, esg_cols, sample, formula, cov_type="HC3", cov_kwds=None):
    kwargs = {"cov_type": cov_type}
    if cov_kwds is not None:
        kwargs["cov_kwds"] = cov_kwds
    return smf.ols(formula=formula, data=sample).fit(**kwargs)


def main():
    df, _ = load_and_reconstruct()
    rows = []

    for label, esg_cols in MODEL_SPECS:
        # ---- baseline: Q, HC3, full sample, ctrl_5a ----
        sample_q, _ = build_model_sample(df, esg_cols, dv="Q", label=label)
        formula_q = f"Q ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
        m_base = fit("Q", esg_cols, sample_q, formula_q, cov_type="HC3")

        # ---- (a) cluster-robust by firm ----
        m_cluster = fit("Q", esg_cols, sample_q, formula_q,
                         cov_type="cluster", cov_kwds={"groups": sample_q["Company"]})

        # ---- (b) excl. Progressive ----
        sample_excl_prog = sample_q[sample_q["Company"] != "Progressive"].copy()
        m_excl_prog = fit("Q", esg_cols, sample_excl_prog, formula_q, cov_type="HC3")

        # ---- (c) ROE as DV ----
        sample_roe, _ = build_model_sample(df, esg_cols, dv="ROE", label=label)
        formula_roe = f"ROE ~ {' + '.join(esg_cols)} + {CTRL_5A_FORMULA}"
        m_roe = fit("ROE", esg_cols, sample_roe, formula_roe, cov_type="HC3")

        # ---- (d) drop ROA control ----
        sample_no_roa, _ = build_model_sample(
            df, esg_cols, dv="Q", label=label + " (no ROA)"
        )
        # build_model_sample's `required` always includes ROA via
        # CTRL_COLS_5A -- refetch a version that doesn't require ROA to be
        # non-null, so dropping it from the FORMULA also drops it as a
        # completeness requirement (otherwise this "variant" would just
        # silently reuse the same complete-case rows).
        required_no_roa = ["Q"] + esg_cols + ["SR_CEO_Letter_lag1", "Leverage", "Size", "Revenue_Growth"]
        sample_check = df.dropna(subset=required_no_roa)
        counts = sample_check.groupby("Company").size()
        firms_to_drop = set(counts[counts < 3].index) | (set(df["Company"].unique()) - set(counts.index))
        sample_no_roa = sample_check[~sample_check["Company"].isin(firms_to_drop)].copy()
        formula_no_roa = f"Q ~ {' + '.join(esg_cols)} + {CTRL_5A_NO_ROA}"
        m_no_roa = fit("Q", esg_cols, sample_no_roa, formula_no_roa, cov_type="HC3")

        for var in esg_cols:
            for variant_name, m in [
                ("baseline (HC3, N=183)", m_base),
                ("cluster-robust SE by firm", m_cluster),
                ("excl. Progressive", m_excl_prog),
                ("ROE as DV", m_roe),
                ("drop ROA control", m_no_roa),
            ]:
                rows.append({
                    "model": label,
                    "term": var,
                    "variant": variant_name,
                    "coef": m.params[var],
                    "se": m.bse[var],
                    "p": m.pvalues[var],
                    "n": int(m.nobs),
                    "ci_lo": m.params[var] - 1.959964 * m.bse[var],
                    "ci_hi": m.params[var] + 1.959964 * m.bse[var],
                })

    out = pd.DataFrame(rows)
    out.to_csv("null_check/results_robustness.csv", index=False)

    pd.set_option("display.width", 160)
    for label, _ in MODEL_SPECS:
        sub = out[out["model"] == label]
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(sub[["term", "variant", "coef", "se", "ci_lo", "ci_hi", "p", "n"]]
              .round(4).to_string(index=False))

    print(f"\nSaved: null_check/results_robustness.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
