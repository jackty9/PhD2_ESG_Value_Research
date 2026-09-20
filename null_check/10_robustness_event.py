"""
Robustness (b) for the event-study coefficients (#5, #6): re-fit CAR[-1,+1]
Model 3 (Specificity + Sentiment) excluding Cook's-distance-flagged
observations, independently recomputed here (not re-quoting the earlier
Step 9.2 list from memory) via the same OLSInfluence/4-N threshold
methodology already used in that step.

HC3 vs. cluster vs. bootstrap SEs are already fully covered in
results_event.csv (07_analysis_event.py) -- not repeated here.

Run:  python3 null_check/10_robustness_event.py
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

DATA_PATH = "null_check/data/ceo_letter_regression_sample_209.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    formula = "Q('CAR[-1,+1]') ~ Specificity + Sentiment + C(Company) + C(Q('Fiscal Year'))"

    m_base = smf.ols(formula=formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["Company"]})
    influence = OLSInfluence(m_base)
    cooks_d = influence.cooks_distance[0]

    fit_sample = df.loc[m_base.model.data.row_labels].copy()
    fit_sample["cooks_d"] = cooks_d
    threshold = 4 / m_base.nobs
    flagged = fit_sample[fit_sample["cooks_d"] > threshold]

    print(f"Cook's distance threshold (4/N): {threshold:.4f}")
    print(f"Flagged observations: {len(flagged)} of {int(m_base.nobs)}")
    print(flagged[["Company", "Fiscal Year", "cooks_d"]].sort_values("cooks_d", ascending=False).to_string(index=False))

    excl_idx = flagged.index
    sub_excl = df.drop(index=[i for i in excl_idx if i in df.index])
    m_excl = smf.ols(formula=formula, data=sub_excl).fit(
        cov_type="cluster", cov_kwds={"groups": sub_excl["Company"]}
    )

    rows = []
    for term in ["Specificity", "Sentiment"]:
        rows.append({
            "term": term, "variant": "baseline (cluster SE, N=209)",
            "coef": m_base.params[term], "se": m_base.bse[term], "p": m_base.pvalues[term], "n": int(m_base.nobs),
        })
        rows.append({
            "term": term, "variant": f"excl. {len(flagged)} Cook's-distance-flagged obs",
            "coef": m_excl.params[term], "se": m_excl.bse[term], "p": m_excl.pvalues[term], "n": int(m_excl.nobs),
        })
    results = pd.DataFrame(rows)
    results.to_csv("null_check/results_robustness_event.csv", index=False)

    print()
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: null_check/results_robustness_event.csv")


if __name__ == "__main__":
    main()
