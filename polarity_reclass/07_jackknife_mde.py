"""
Leave-one-firm-out jackknife + MDE for MitigatedNegativeSpecific_share_lag on
ROE (the "MitigatedNegative alone" spec from 06_h2_split_regression.py:
coef=-0.3267, HC3 p=0.126, cluster p=0.020).

Format matches PHDp2_FinancialData_Regression.ipynb / CEO_Letter_Event_Study.ipynb's
Step 17.8 (leave-one-firm-out jackknife for SR_CEO_Letter_lag1:ESGSentiment,
event-study branch commit fec4e21) and Step 17.11 (MDE, 80% power, alpha=0.05,
raw + SD-standardized, both SE types) exactly -- same loop structure, same
"None dropped" baseline row, same zero-rows-already handling for any firm
absent from the complete-case sample, same MDE formula
(z_crit_twosided + z_power80) * SE, standardized by SD(DV).

Per instruction: if the jackknife comes back clean (no single firm drives
significance), proceed to interpretation; if one or two firms clearly drive
it, stop there and name them -- no wild-cluster bootstrap on a firm-dependent
result.

Run:  python3 polarity_reclass/07_jackknife_mde.py
"""

import importlib.util
import sys

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

sys.path.insert(0, "null_check")
from common import CTRL_5A_FORMULA, build_model_sample, load_and_reconstruct


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_esg_e_sg_common = _load_module("esg_e_sg_common", "esg_e_sg/common.py")
NAME_FIXES = _esg_e_sg_common.NAME_FIXES
safe_lag = _esg_e_sg_common.safe_lag

GPT_RESULTS_PATH = "polarity_reclass/data/df_ar_ceo_sentences_polarity_reclass_v6.csv"
PANEL_PATH = "esg_e_sg/data/panel_reg_export.csv"

DV = "ROE"
ESG_COLS = ["Corporate_ESG_share_lag", "MitigatedNegativeSpecific_share_lag"]
TARGET_TERM = "MitigatedNegativeSpecific_share_lag"


def build_mitigated_negative_share():
    gpt = pd.read_csv(GPT_RESULTS_PATH)
    tier2plus = gpt[gpt["specificity_tier_pred"] >= 2].copy()
    tier2plus["is_mitigated_negative"] = tier2plus["case"] == "mitigated_negative"

    panel = pd.read_csv(PANEL_PATH)
    total_by_fy = panel.set_index(["Company", "Year"])["total_sent"]

    mn_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_mitigated_negative"].sum().rename("n_mitigated_negative")
    fy = mn_by_fy.reset_index()
    fy.columns = ["Company", "Year", "n_mitigated_negative"]
    fy["Company"] = fy["Company"].replace(NAME_FIXES)

    fy = fy.merge(total_by_fy.rename("total_sent").reset_index(), on=["Company", "Year"], how="right")
    fy["n_mitigated_negative"] = fy["n_mitigated_negative"].fillna(0)
    fy["MitigatedNegativeSpecific_share"] = fy["n_mitigated_negative"] / fy["total_sent"]
    return fy[["Company", "Year", "MitigatedNegativeSpecific_share"]]


def build_sample():
    df, meta = load_and_reconstruct()
    mn_share = build_mitigated_negative_share()
    df = df.merge(mn_share, on=["Company", "Year"], how="left")
    df["MitigatedNegativeSpecific_share"] = df["MitigatedNegativeSpecific_share"].fillna(0)
    df = df.sort_values(["Company", "Year"]).reset_index(drop=True)
    df["MitigatedNegativeSpecific_share_lag"] = safe_lag(df, "MitigatedNegativeSpecific_share")

    sample, dropped = build_model_sample(df, ESG_COLS, dv=DV, label=f"{DV}, jackknife base")
    print(f"Base sample: N={len(sample)}, firms dropped for insufficient obs: {dropped}")
    return df, sample


def jackknife(sample, all_firms_22):
    formula = f"{DV} ~ {' + '.join(ESG_COLS)} + {CTRL_5A_FORMULA}"

    m_base_hc3 = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
    m_base_cluster = smf.ols(formula=formula, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["Company"]}
    )

    firms_present = sorted(sample["Company"].unique())
    firms_absent = sorted(set(all_firms_22) - set(firms_present))
    print(f"\nFirms present in this complete-case sample: {len(firms_present)} of {len(all_firms_22)}")
    if firms_absent:
        print(f"Firms with zero rows already (leave-one-out is a no-op for these): {firms_absent}")

    rows = [{
        "firm_dropped": "None dropped", "n": int(m_base_hc3.nobs),
        "coef": m_base_hc3.params[TARGET_TERM],
        "hc3_p": m_base_hc3.pvalues[TARGET_TERM],
        "cluster_p": m_base_cluster.pvalues[TARGET_TERM],
    }]

    for firm in firms_present:
        sub = sample[sample["Company"] != firm].copy()
        m_jk_hc3 = smf.ols(formula=formula, data=sub).fit(cov_type="HC3")
        groups_jk = sub.loc[m_jk_hc3.model.data.row_labels, "Company"]
        m_jk_cluster = smf.ols(formula=formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": groups_jk}
        )
        rows.append({
            "firm_dropped": firm, "n": int(m_jk_hc3.nobs),
            "coef": m_jk_hc3.params[TARGET_TERM],
            "hc3_p": m_jk_hc3.pvalues[TARGET_TERM],
            "cluster_p": m_jk_cluster.pvalues[TARGET_TERM],
        })

    for firm in firms_absent:
        rows.append({
            "firm_dropped": f"{firm} (already 0 rows -- n/a)", "n": int(m_base_hc3.nobs),
            "coef": m_base_hc3.params[TARGET_TERM],
            "hc3_p": m_base_hc3.pvalues[TARGET_TERM],
            "cluster_p": m_base_cluster.pvalues[TARGET_TERM],
        })

    tbl = pd.DataFrame(rows)
    return tbl, m_base_hc3, m_base_cluster, firms_present


def report_jackknife(tbl, firms_present):
    print(f"\n{'=' * 78}\nTable -- leave-one-firm-out jackknife, {TARGET_TERM} on {DV}\n{'=' * 78}")
    pd.set_option("display.width", 160)
    print(tbl.round(4).to_string(index=False))

    testable = tbl[~tbl["firm_dropped"].isin(["None dropped"]) & ~tbl["firm_dropped"].str.contains("n/a")]
    n_sig = int((testable["cluster_p"] < 0.05).sum())
    n_testable = len(testable)
    print(f"\n{n_sig} of {n_testable} testable leave-one-out fits keep cluster p<0.05 "
          f"(of {len(firms_present)} firms actually droppable in this sample)")

    losers = testable[testable["cluster_p"] >= 0.05]
    full_coef = tbl.loc[tbl["firm_dropped"] == "None dropped", "coef"].iloc[0]
    full_cluster_p = tbl.loc[tbl["firm_dropped"] == "None dropped", "cluster_p"].iloc[0]
    if len(losers) > 0:
        print(f"\nFirm(s) whose removal loses significance (cluster p >= 0.05):")
        for _, row in losers.iterrows():
            shift_pct = (row["coef"] - full_coef) / full_coef * 100
            print(f"  {row['firm_dropped']}: coef {full_coef:.4f} -> {row['coef']:.4f} "
                  f"({shift_pct:+.1f}%), cluster p {full_cluster_p:.4f} -> {row['cluster_p']:.4f}")
        return losers
    else:
        print("No single firm's removal loses significance -- result is not driven by any one insurer.")
        return losers


def report_mde(m_hc3, m_cluster, sample):
    z_crit = norm.ppf(1 - 0.05 / 2)
    z_power80 = norm.ppf(0.80)
    sd_dv = sample[DV].std(ddof=1)

    hc3_se = m_hc3.bse[TARGET_TERM]
    cluster_se = m_cluster.bse[TARGET_TERM]

    mde_raw_hc3 = (z_crit + z_power80) * hc3_se
    mde_raw_cluster = (z_crit + z_power80) * cluster_se
    mde_std_hc3 = mde_raw_hc3 / sd_dv
    mde_std_cluster = mde_raw_cluster / sd_dv

    coef = m_hc3.params[TARGET_TERM]

    print(f"\n{'=' * 78}\nMDE (80% power, alpha=0.05) for {TARGET_TERM} on {DV} (N={int(m_hc3.nobs)})\n{'=' * 78}")
    print(f"SD({DV}) in this sample: {sd_dv:.4f}")
    print(f"Coefficient: {coef:.4f}")
    print(f"Raw MDE: {mde_raw_hc3:.4f} (HC3 SE), {mde_raw_cluster:.4f} (cluster SE)")
    print(f"Standardized MDE: {mde_std_hc3:.2f} SD (HC3 SE), {mde_std_cluster:.2f} SD (cluster SE)")
    print(f"|coef| / MDE (raw, HC3): {abs(coef) / mde_raw_hc3:.2f}x -- coefficient relative to "
          f"the smallest true effect this design could detect at 80% power")
    print(f"|coef| / MDE (raw, cluster): {abs(coef) / mde_raw_cluster:.2f}x")

    # Economic magnitude: per 1 percentage point (0.01) of MitigatedNegativeSpecific_share
    delta = 0.01
    roe_change = coef * delta
    print(f"\nEconomic magnitude: a {delta*100:.0f} percentage-point increase in "
          f"MitigatedNegativeSpecific_share (tier>=2, lagged) is associated with a "
          f"{roe_change:+.5f} change in ROE ({roe_change*100:+.3f} percentage points of ROE, "
          f"{roe_change*10000:+.1f} basis points), holding Corporate_ESG_share and controls fixed.")
    print(f"For reference, mean ROE in this sample is {sample[DV].mean()*100:.1f}% "
          f"(SD {sd_dv*100:.1f} pp).")
    print("\nNo Hope-Hu-Lu (or other external literature) benchmark value is available in this "
          "repository/session to compare this magnitude against -- supply the specific benchmark "
          "figure if you want that comparison made explicitly.")

    return dict(mde_raw_hc3=mde_raw_hc3, mde_raw_cluster=mde_raw_cluster,
                mde_std_hc3=mde_std_hc3, mde_std_cluster=mde_std_cluster,
                sd_dv=sd_dv, coef=coef)


def main():
    all_firms_22 = None  # filled in from the panel below
    df, sample = build_sample()

    panel = pd.read_csv(PANEL_PATH)
    all_firms_22 = sorted(panel["Company"].unique())

    tbl, m_base_hc3, m_base_cluster, firms_present = jackknife(sample, all_firms_22)
    losers = report_jackknife(tbl, firms_present)

    tbl.to_csv("polarity_reclass/results_jackknife_mitigated_negative_roe.csv", index=False)
    print(f"\nSaved: polarity_reclass/results_jackknife_mitigated_negative_roe.csv")

    mde_stats = report_mde(m_base_hc3, m_base_cluster, sample)

    print(f"\n{'=' * 78}\nDecision\n{'=' * 78}")
    if len(losers) == 0:
        print("Jackknife is clean -- proceed to the wild-cluster bootstrap if you want a further check.")
    elif len(losers) <= 2:
        print(f"Result is driven by {len(losers)} firm(s): {list(losers['firm_dropped'])}. "
              f"STOPPING HERE per instruction -- do not run the bootstrap on a firm-dependent result.")
    else:
        print(f"Result loses significance under {len(losers)} of {len(firms_present)} firm exclusions -- "
              f"broadly fragile, not just one or two firms. STOPPING HERE.")


if __name__ == "__main__":
    main()
