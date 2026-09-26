"""
Does the SR x ESGSentiment moderation effect found in the CAR event study
(H4, event-study branch's CEO_Letter_Event_Study.ipynb Step 17.6) replicate
in the annual valuation panel (Q, ROE)?

SR_CEO_Letter_lag1 is already a standalone control in this project's panel
models (CTRL_5A_FORMULA, Models 1-6) -- this test ADDS it as an interaction
alongside its existing main-effect role, not in place of it. This is a more
conservative specification than the CAR model (which was interaction-only,
no separate SR main effect), so a null/fragile result here is not
necessarily inconsistent with the CAR finding, and a robust result here
would be a stronger, more conservative replication than a matching
specification would give.

    Q_it   = b1*Corporate_ESG_share_lag1 + b2*Specificity_mean_lag1
             + b3*SR_CEO_Letter_lag1 + b4*(SR_CEO_Letter_lag1 x Specificity_mean_lag1)
             + b5*ESGSentiment_Corporate_lag1 + b6*(SR_CEO_Letter_lag1 x ESGSentiment_Corporate_lag1)
             + Controls + firm FE + year FE
    ROE_it = same RHS

Controls = Leverage, Size, ROA, Revenue_Growth (CTRL_5A_FORMULA, reused from
null_check/common.py -- SR_CEO_Letter_lag1 is already one of its terms, so
the formula below does not duplicate it, patsy would just dedupe it anyway).

ESGSentiment_Corporate is not pre-lagged in null_check/common.py's
load_and_reconstruct() (only ESG_sentence_ratio, Corporate_ESG_share,
Verified_ESG_share, Specificity_mean are) -- lagged here explicitly with
the same safe_lag() gap-aware convention.

Four steps, report-only:
  1. Cell-size check: SR_CEO_Letter_lag1 (0/1) split in the complete-case
     sample for this specific spec, before fitting anything (same
     discipline as the EU-NFRD and CAR mechanistic tests).
  2. Full 12-coefficient table (6 terms x 2 DVs), HC3 + cluster-robust SE.
  3. Leave-one-firm-out jackknife on any term with cluster p<0.05 (same
     format as CEO_Letter_Event_Study.ipynb Step 17.8 / polarity_reclass/
     07_jackknife_mde.py).
  4. MDE (80% power, alpha=0.05) for the two interaction terms specifically,
     both DVs, raw + SD-standardized, both SE types (same method as Step
     17.11 / 07_jackknife_mde.py).

Verdict per term per DV: null / significant-but-fragile / robust -- same
three-tier language already used for the CAR results.

Run:  python3 sr_moderation_panel/01_regression_jackknife_mde.py
"""

import sys

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, build_model_sample, load_and_reconstruct, safe_lag

TERMS = [
    "Corporate_ESG_share_lag",
    "Specificity_mean_lag",
    "SR_CEO_Letter_lag1",
    "SR_CEO_Letter_lag1:Specificity_mean_lag",
    "ESGSentiment_Corporate_lag",
    "SR_CEO_Letter_lag1:ESGSentiment_Corporate_lag",
]
INTERACTION_TERMS = ["SR_CEO_Letter_lag1:Specificity_mean_lag", "SR_CEO_Letter_lag1:ESGSentiment_Corporate_lag"]
REQUIRED_COLS = ["Corporate_ESG_share_lag", "Specificity_mean_lag", "ESGSentiment_Corporate_lag"]


def build_data():
    df, meta = load_and_reconstruct()
    df["ESGSentiment_Corporate_lag"] = safe_lag(df, "ESGSentiment_Corporate")
    return df, meta


def formula_for(dv):
    return (f"{dv} ~ Corporate_ESG_share_lag + Specificity_mean_lag + SR_CEO_Letter_lag1 "
            f"+ SR_CEO_Letter_lag1:Specificity_mean_lag + ESGSentiment_Corporate_lag "
            f"+ SR_CEO_Letter_lag1:ESGSentiment_Corporate_lag + {CTRL_5A_FORMULA}")


def step1_cell_size(df):
    print(f"{'=' * 78}\nStep 1 -- Cell-size check\n{'=' * 78}")
    results = {}
    for dv in ["Q", "ROE"]:
        sample, dropped = build_model_sample(df, REQUIRED_COLS, dv=dv, label=f"cell-size, {dv}")
        n1 = int((sample["SR_CEO_Letter_lag1"] == 1).sum())
        n0 = int((sample["SR_CEO_Letter_lag1"] == 0).sum())
        print(f"\n{dv}: complete-case N={len(sample)} (firms dropped: {dropped})")
        print(f"  SR_CEO_Letter_lag1=1: {n1} firm-years")
        print(f"  SR_CEO_Letter_lag1=0: {n0} firm-years")
        thin = min(n1, n0) < 50
        print(f"  {'THIN -- smaller group < 50, MDE below must be read carefully' if thin else 'Both groups >= 50'}")
        results[dv] = dict(sample=sample, n1=n1, n0=n0, thin=thin)
    return results


def step2_fit(df):
    print(f"\n{'=' * 78}\nStep 2 -- Full 12-coefficient table\n{'=' * 78}")
    fits = {}
    rows = []
    for dv in ["Q", "ROE"]:
        sample, dropped = build_model_sample(df, REQUIRED_COLS, dv=dv, label=f"fit, {dv}")
        formula = formula_for(dv)
        m_hc3 = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
        m_cluster = smf.ols(formula=formula, data=sample).fit(
            cov_type="cluster", cov_kwds={"groups": sample["Company"]}
        )
        fits[dv] = dict(hc3=m_hc3, cluster=m_cluster, sample=sample, dropped=dropped)

        print(f"\n{dv}: N={int(m_hc3.nobs)}, R-squared={m_hc3.rsquared:.4f}, firms dropped: {dropped}")
        for term in TERMS:
            rows.append({
                "dv": dv, "term": term,
                "coef": m_hc3.params[term],
                "hc3_se": m_hc3.bse[term], "hc3_p": m_hc3.pvalues[term],
                "cluster_se": m_cluster.bse[term], "cluster_p": m_cluster.pvalues[term],
                "n": int(m_hc3.nobs),
            })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(f"\n{'=' * 78}\nTable: all 6 terms x 2 DVs (12 coefficients)\n{'=' * 78}")
    print(out.round(4).to_string(index=False))
    out.to_csv("sr_moderation_panel/results_sr_moderation.csv", index=False)
    print(f"\nSaved: sr_moderation_panel/results_sr_moderation.csv")
    return fits, out


def jackknife_term(dv, term, sample, formula, firms_present):
    m_base_hc3 = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
    m_base_cluster = smf.ols(formula=formula, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["Company"]}
    )
    rows = [{
        "dv": dv, "term": term, "firm_dropped": "None dropped", "n": int(m_base_hc3.nobs),
        "coef": m_base_hc3.params[term], "hc3_p": m_base_hc3.pvalues[term],
        "cluster_p": m_base_cluster.pvalues[term],
    }]
    for firm in firms_present:
        sub = sample[sample["Company"] != firm].copy()
        m_jk_hc3 = smf.ols(formula=formula, data=sub).fit(cov_type="HC3")
        groups_jk = sub.loc[m_jk_hc3.model.data.row_labels, "Company"]
        m_jk_cluster = smf.ols(formula=formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": groups_jk}
        )
        rows.append({
            "dv": dv, "term": term, "firm_dropped": firm, "n": int(m_jk_hc3.nobs),
            "coef": m_jk_hc3.params[term], "hc3_p": m_jk_hc3.pvalues[term],
            "cluster_p": m_jk_cluster.pvalues[term],
        })
    return pd.DataFrame(rows)


def step3_jackknife(fits, out):
    print(f"\n{'=' * 78}\nStep 3 -- Jackknife on any term with cluster p<0.05\n{'=' * 78}")
    sig_terms = out[out["cluster_p"] < 0.05][["dv", "term"]].drop_duplicates()
    if len(sig_terms) == 0:
        print("No term reaches cluster p<0.05 in either DV -- no jackknife needed.")
        return pd.DataFrame()

    all_jk = []
    for _, row in sig_terms.iterrows():
        dv, term = row["dv"], row["term"]
        sample = fits[dv]["sample"]
        formula = formula_for(dv)
        firms_present = sorted(sample["Company"].unique())
        tbl = jackknife_term(dv, term, sample, formula, firms_present)
        all_jk.append(tbl)

        print(f"\n--- {dv}, {term} ---")
        testable = tbl[tbl["firm_dropped"] != "None dropped"]
        n_sig = int((testable["cluster_p"] < 0.05).sum())
        full_coef = tbl.loc[tbl["firm_dropped"] == "None dropped", "coef"].iloc[0]
        full_p = tbl.loc[tbl["firm_dropped"] == "None dropped", "cluster_p"].iloc[0]
        print(f"{n_sig} of {len(testable)} testable leave-one-out fits keep cluster p<0.05 "
              f"(baseline coef={full_coef:.4f}, cluster p={full_p:.4f})")
        losers = testable[testable["cluster_p"] >= 0.05]
        if len(losers) > 0:
            for _, lrow in losers.iterrows():
                shift = (lrow["coef"] - full_coef) / full_coef * 100
                print(f"  DRIVEN BY {lrow['firm_dropped']}: coef {full_coef:.4f} -> {lrow['coef']:.4f} "
                      f"({shift:+.1f}%), cluster p {full_p:.4f} -> {lrow['cluster_p']:.4f}")
        else:
            print("  No single firm's removal loses significance.")

    jk_out = pd.concat(all_jk, ignore_index=True)
    jk_out.to_csv("sr_moderation_panel/results_jackknife.csv", index=False)
    print(f"\nSaved: sr_moderation_panel/results_jackknife.csv")
    return jk_out


def step4_mde(fits):
    print(f"\n{'=' * 78}\nStep 4 -- MDE (80% power, alpha=0.05) for the two interaction terms\n{'=' * 78}")
    z_crit = norm.ppf(1 - 0.05 / 2)
    z_power80 = norm.ppf(0.80)

    rows = []
    for dv in ["Q", "ROE"]:
        sample = fits[dv]["sample"]
        m_hc3 = fits[dv]["hc3"]
        m_cluster = fits[dv]["cluster"]
        sd_dv = sample[dv].std(ddof=1)
        for term in INTERACTION_TERMS:
            hc3_se = m_hc3.bse[term]
            cluster_se = m_cluster.bse[term]
            mde_raw_hc3 = (z_crit + z_power80) * hc3_se
            mde_raw_cluster = (z_crit + z_power80) * cluster_se
            mde_std_hc3 = mde_raw_hc3 / sd_dv
            mde_std_cluster = mde_raw_cluster / sd_dv
            coef = m_hc3.params[term]
            print(f"\n{dv}, {term}: coef={coef:.4f}, SD({dv})={sd_dv:.4f}")
            print(f"  Raw MDE: {mde_raw_hc3:.4f} (HC3), {mde_raw_cluster:.4f} (cluster)")
            print(f"  Standardized MDE: {mde_std_hc3:.2f} SD (HC3), {mde_std_cluster:.2f} SD (cluster)")
            print(f"  |coef|/MDE: {abs(coef)/mde_raw_hc3:.2f}x (HC3), {abs(coef)/mde_raw_cluster:.2f}x (cluster)")
            rows.append({
                "dv": dv, "term": term, "coef": coef, "sd_dv": sd_dv,
                "mde_raw_hc3": mde_raw_hc3, "mde_raw_cluster": mde_raw_cluster,
                "mde_std_hc3": mde_std_hc3, "mde_std_cluster": mde_std_cluster,
            })
    out = pd.DataFrame(rows)
    out.to_csv("sr_moderation_panel/results_mde.csv", index=False)
    print(f"\nSaved: sr_moderation_panel/results_mde.csv")
    return out


def classify(row, jk_out):
    dv, term = row["dv"], row["term"]
    if row["cluster_p"] >= 0.05 and row["hc3_p"] >= 0.05:
        return "null"
    sub_jk = jk_out[(jk_out["dv"] == dv) & (jk_out["term"] == term) & (jk_out["firm_dropped"] != "None dropped")]
    if len(sub_jk) > 0:
        n_losers = int((sub_jk["cluster_p"] >= 0.05).sum())
        if n_losers > 0:
            return f"significant-but-fragile ({n_losers} firm(s) flip it)"
        return "robust (survives jackknife)"
    return "significant, not jackknifed (cluster p>=0.05 in only one SE type or not tested)"


def main():
    df, meta = build_data()
    print(f"Panel meta: {meta}")

    step1_cell_size(df)
    fits, out = step2_fit(df)
    jk_out = step3_jackknife(fits, out)
    mde_out = step4_mde(fits)

    print(f"\n{'=' * 78}\nVerdicts (null / significant-but-fragile / robust)\n{'=' * 78}")
    for _, row in out.iterrows():
        verdict = classify(row, jk_out) if len(jk_out) > 0 else (
            "null" if row["cluster_p"] >= 0.05 else "significant, not yet jackknifed"
        )
        print(f"  {row['dv']:4s} | {row['term']:55s} | coef={row['coef']:+.4f} | "
              f"HC3 p={row['hc3_p']:.3f} | cluster p={row['cluster_p']:.3f} | {verdict}")


if __name__ == "__main__":
    main()
