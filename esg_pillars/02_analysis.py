"""
Steps 3-7: fit the six pillar-alone models (E/S/G x Q/ROE) as the primary
analysis, the two joint three-pillar models as a labeled secondary check,
report correlations/VIF, standardized effects/CI/MDE, Holm + BH correction
across the six primary tests, and TOST at +/-0.10/+/-0.20 SD.

Run:  python3 esg_pillars/02_analysis.py
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, "esg_pillars")
from common import CTRL_5A_FORMULA, CTRL_COLS_5A, PILLARS, build_model_sample, load_and_reconstruct

ALPHA = 0.05
POWER = 0.80
Z_CRIT = stats.norm.ppf(1 - ALPHA / 2)      # 1.95996, matches HC3's z-tests
Z_POWER = stats.norm.ppf(POWER)             # 0.84162
Z_ONESIDED_95 = stats.norm.ppf(1 - ALPHA)   # 1.64485, TOST critical value

TOST_BOUNDS_SD = [0.10, 0.20]


def tost_at_bound(coef, se, bound):
    ci_lo = coef - Z_ONESIDED_95 * se
    ci_hi = coef + Z_ONESIDED_95 * se
    return (ci_lo >= -bound) and (ci_hi <= bound)


def classify(coef, se, p, bounds_result):
    """significant / informative null / inconclusive, per instruction step 6."""
    if p < 0.05:
        return "Significant"
    if bounds_result[0.10]:
        return "Informative null (equivalent within ±0.10 SD)"
    if bounds_result[0.20]:
        return "Informative null (equivalent within ±0.20 SD only)"
    return "Inconclusive"


def fit_pillar_alone(df, dv, pillar):
    lag_col = f"{pillar}_ratio_lag"
    sample, dropped = build_model_sample(df, [lag_col], dv=dv, label=f"{pillar} alone, {dv}")
    formula = f"{dv} ~ {lag_col} + {CTRL_5A_FORMULA}"
    m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
    return m, sample, lag_col


def fit_joint(df, dv):
    lag_cols = [f"{p}_ratio_lag" for p in PILLARS]
    sample, dropped = build_model_sample(df, lag_cols, dv=dv, label=f"joint, {dv}")
    formula = f"{dv} ~ {' + '.join(lag_cols)} + {CTRL_5A_FORMULA}"
    m = smf.ols(formula=formula, data=sample).fit(cov_type="HC3")
    return m, sample, lag_cols


def effect_row(dv, spec_label, var, coef, se, p, n, sample, dv_col):
    sd_dv = sample[dv_col].std(ddof=1)
    sd_x = sample[var].std(ddof=1)
    ci_lo, ci_hi = coef - Z_CRIT * se, coef + Z_CRIT * se
    std_coef = coef * sd_x / sd_dv
    std_se = se * sd_x / sd_dv
    mde_raw = (Z_CRIT + Z_POWER) * se
    mde_std = mde_raw * sd_x / sd_dv
    bounds_result = {b: tost_at_bound(std_coef, std_se, b) for b in TOST_BOUNDS_SD}
    return {
        "dv": dv, "spec": spec_label, "term": var,
        "coef": coef, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_raw": p, "n": n,
        "sd_dv": sd_dv, "sd_predictor": sd_x,
        "std_coef": std_coef, "std_se": std_se,
        "std_ci_lo": std_coef - Z_CRIT * std_se, "std_ci_hi": std_coef + Z_CRIT * std_se,
        "mde_raw_80pct_power": mde_raw, "mde_std_80pct_power": mde_std,
        "equiv_within_0.10SD": bounds_result[0.10],
        "equiv_within_0.20SD": bounds_result[0.20],
        "classification": classify(coef, se, p, bounds_result),
    }


def main():
    df, meta = load_and_reconstruct()

    # ---- pairwise correlations among E/S/G (lagged, on the full merged panel) ----
    corr_cols = [f"{p}_ratio_lag" for p in PILLARS]
    corr = df[corr_cols].corr()
    print("Pairwise correlations among E/S/G (lagged) ratios, full merged panel:")
    print(corr.round(3).to_string())
    corr.to_csv("esg_pillars/results_pillar_correlations.csv")

    # ==================================================================
    # PRIMARY: six pillar-alone models
    # ==================================================================
    primary_rows = []
    primary_models = {}
    for dv in ["Q", "ROE"]:
        for pillar in PILLARS:
            m, sample, lag_col = fit_pillar_alone(df, dv, pillar)
            primary_models[(dv, pillar)] = (m, sample, lag_col)
            row = effect_row(dv, f"{pillar} alone", lag_col,
                              m.params[lag_col], m.bse[lag_col], m.pvalues[lag_col],
                              int(m.nobs), sample, dv)
            primary_rows.append(row)

    primary = pd.DataFrame(primary_rows)

    # Step 5: Holm and BH correction across the six primary tests
    raw_p = primary["p_raw"].values
    _, p_holm, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
    _, p_bh, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
    primary["p_holm"] = p_holm
    primary["p_bh"] = p_bh

    primary.to_csv("esg_pillars/results_primary.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print(f"\n{'=' * 78}\nPRIMARY: six pillar-alone models\n{'=' * 78}")
    print(primary[["dv", "spec", "coef", "se", "ci_lo", "ci_hi", "p_raw", "p_holm", "p_bh", "n"]]
          .round(4).to_string(index=False))
    print()
    print(primary[["dv", "spec", "std_coef", "std_ci_lo", "std_ci_hi",
                    "mde_std_80pct_power", "equiv_within_0.10SD", "equiv_within_0.20SD",
                    "classification"]].round(4).to_string(index=False))

    # ==================================================================
    # SECONDARY (labeled): joint three-pillar model, one per outcome
    # ==================================================================
    secondary_rows = []
    vif_rows = []
    for dv in ["Q", "ROE"]:
        m, sample, lag_cols = fit_joint(df, dv)
        for var in lag_cols:
            pillar = var[0]
            row = effect_row(dv, "Joint (all 3 pillars)", var,
                              m.params[var], m.bse[var], m.pvalues[var],
                              int(m.nobs), sample, dv)
            secondary_rows.append(row)

        # VIF among the three pillar lags + controls, on this joint model's
        # own complete-case sample (FE dummies excluded from the VIF matrix,
        # same reasoning as null_check: categorical FE mechanically inflates
        # VIF regardless of real collinearity between the pillars).
        vif_cols = lag_cols + CTRL_COLS_5A
        X_vif = sample[vif_cols].copy()
        X_vif.insert(0, "const", 1.0)
        for i, col in enumerate(X_vif.columns):
            if col == "const":
                continue
            vif_rows.append({"dv": dv, "variable": col,
                              "VIF": variance_inflation_factor(X_vif.values, i)})

    secondary = pd.DataFrame(secondary_rows)
    secondary.to_csv("esg_pillars/results_secondary_joint.csv", index=False)

    vif = pd.DataFrame(vif_rows)
    vif.to_csv("esg_pillars/results_vif.csv", index=False)

    print(f"\n{'=' * 78}\nSECONDARY (labeled): joint three-pillar model\n{'=' * 78}")
    print(secondary[["dv", "term", "coef", "se", "ci_lo", "ci_hi", "p_raw", "n"]]
          .round(4).to_string(index=False))
    print("\nVIF, joint models (FE dummies excluded):")
    print(vif.round(3).to_string(index=False))

    # ---- compare: where does the joint model differ from the pillar-alone models? ----
    print(f"\n{'=' * 78}\nComparison: pillar-alone vs. joint coefficient for the same pillar/DV\n{'=' * 78}")
    for dv in ["Q", "ROE"]:
        for pillar in PILLARS:
            alone = primary[(primary["dv"] == dv) & (primary["spec"] == f"{pillar} alone")].iloc[0]
            joint = secondary[(secondary["dv"] == dv) & (secondary["term"] == f"{pillar}_ratio_lag")].iloc[0]
            flip = np.sign(alone["coef"]) != np.sign(joint["coef"])
            sig_change = (alone["p_raw"] < 0.05) != (joint["p_raw"] < 0.05)
            print(f"{dv} / {pillar}: alone coef={alone['coef']:.4f} (p={alone['p_raw']:.3f})  "
                  f"joint coef={joint['coef']:.4f} (p={joint['p_raw']:.3f})  "
                  f"{'SIGN FLIP' if flip else ''} {'SIG. CHANGE' if sig_change else ''}")

    print(f"\nSaved: results_primary.csv, results_secondary_joint.csv, "
          f"results_pillar_correlations.csv, results_vif.csv")


if __name__ == "__main__":
    main()
