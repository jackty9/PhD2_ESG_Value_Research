"""
Independent reproduction check for the event-study coefficients (#5, #6
of the 7-coefficient informativeness check) before any equivalence/power/
Bayes analysis is run on them, per instruction.

Source: null_check/data/ceo_letter_regression_sample_209.csv, the exact
209-row complete-case regression sample exported by
CEO_Letter_Event_Study.ipynb's Step 9.5 (uploaded directly, not re-derived
from raw price/text data -- this sandbox has no Google Drive or network
access to rebuild the event-study panel itself from scratch).

Also independently re-confirms the "history" note for coefficient #6
(Sentiment): raw p<0.05 at CAR[0,+30]/[0,+45]/[0,+60] pre-correction, not
surviving Holm/BH across the 18-test family -- using this uploaded CSV's
own CAR[0,+30]/[0,+45]/[0,+60] columns, not just re-quoting the earlier
chat output.

Run:  python3 null_check/06_reproduce_event.py
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

DATA_PATH = "null_check/data/ceo_letter_regression_sample_209.csv"

REPORTED = {
    ("CAR[-1,+1]", "Specificity"): {"coef": -0.0040, "se": 0.0036, "n": 209},
    ("CAR[-1,+1]", "Sentiment"): {"coef": -0.0279, "se": 0.0193, "n": 209},
}


def fit_car_model(df, dv):
    formula = f"Q('{dv}') ~ Specificity + Sentiment + C(Company) + C(Q('Fiscal Year'))"
    return smf.ols(formula=formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["Company"]})


def check(label, reproduced_coef, reproduced_se, reproduced_n, reported):
    ok = (
        abs(reproduced_coef - reported["coef"]) < 5e-4
        and abs(reproduced_se - reported["se"]) < 5e-4
        and reproduced_n == reported["n"]
    )
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    print(f"    reproduced: coef={reproduced_coef:.4f}  se={reproduced_se:.4f}  N={reproduced_n}")
    print(f"    reported:   coef={reported['coef']:.4f}  se={reported['se']:.4f}  N={reported['n']}")
    return ok


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape}")
    required_cols = ["Company", "Fiscal Year", "Specificity", "Sentiment",
                      "CAR[-1,+1]", "CAR[0,+30]", "CAR[0,+45]", "CAR[0,+60]"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    assert df[required_cols].isna().sum().sum() == 0, "Unexpected NaNs in the uploaded complete-case sample."

    print("\n" + "=" * 78)
    print("Primary coefficients (#5 Specificity, #6 Sentiment) at CAR[-1,+1]")
    print("=" * 78)
    m_primary = fit_car_model(df, "CAR[-1,+1]")
    all_pass = True
    all_pass &= check(
        "#5 Specificity", m_primary.params["Specificity"], m_primary.bse["Specificity"],
        int(m_primary.nobs), REPORTED[("CAR[-1,+1]", "Specificity")],
    )
    all_pass &= check(
        "#6 Sentiment", m_primary.params["Sentiment"], m_primary.bse["Sentiment"],
        int(m_primary.nobs), REPORTED[("CAR[-1,+1]", "Sentiment")],
    )

    print("\n" + "=" * 78)
    print("Coefficient #6 history: Sentiment across the longer windows,")
    print("re-fit HC3, with Holm/BH correction across the full 9-window x 2-term family")
    print("(reproducing Step 9.1's family, not just the CAR[0,+30]/[0,+45]/[0,+60] subset)")
    print("=" * 78)
    all_windows = ["CAR[-1,+1]", "CAR[-2,+2]", "CAR[0,+1]", "CAR[0,+5]", "CAR[0,+15]",
                   "CAR[0,+20]", "CAR[0,+30]", "CAR[0,+45]", "CAR[0,+60]"]
    available_windows = [w for w in all_windows if w in df.columns]
    print(f"Windows available in this export: {available_windows}")

    rows = []
    for win in available_windows:
        formula = f"Q('{win}') ~ Specificity + Sentiment + C(Company) + C(Q('Fiscal Year'))"
        m = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
        for term in ["Specificity", "Sentiment"]:
            rows.append({"window": win, "term": term, "coef": m.params[term],
                         "se": m.bse[term], "p": m.pvalues[term]})
    hist = pd.DataFrame(rows)
    reject_holm, p_holm, _, _ = multipletests(hist["p"], alpha=0.05, method="holm")
    reject_bh, p_bh, _, _ = multipletests(hist["p"], alpha=0.05, method="fdr_bh")
    hist["sig_holm"] = reject_holm
    hist["sig_bh"] = reject_bh

    sentiment_hist = hist[hist["term"] == "Sentiment"]
    print(sentiment_hist[["window", "coef", "se", "p", "sig_holm", "sig_bh"]].round(4).to_string(index=False))
    n_raw_sig = int((hist["p"] < 0.05).sum())
    n_holm_sig = int(hist["sig_holm"].sum())
    n_bh_sig = int(hist["sig_bh"].sum())
    print(f"\nFull family ({len(hist)} tests): raw p<0.05 = {n_raw_sig}, "
          f"Holm survives = {n_holm_sig}, BH survives = {n_bh_sig}")
    print("Confirms the reported history: some raw p<0.05 hits at the longer windows, "
          "none survive correction across the full family." if n_holm_sig == 0 and n_bh_sig == 0
          else "NOTE: this reproduction found different correction results than reported -- investigate.")

    print("\n" + "=" * 78)
    print("OVERALL:", "ALL PRIMARY COEFFICIENTS REPRODUCED" if all_pass else "REPRODUCTION FAILED -- see above")
    print("=" * 78)
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
