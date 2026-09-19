"""
Assembles esg_e_sg/report.md from results_core.csv and
results_pillar_specificity.csv (both already written). No new estimation.

Run:  python3 esg_e_sg/04_report.py
"""

import pandas as pd

core = pd.read_csv("esg_e_sg/results_core.csv")
spec = pd.read_csv("esg_e_sg/results_pillar_specificity.csv")

MIN_SENTENCES = 5


def classify(row):
    if row["p_raw"] < 0.05:
        return "Significant (raw)" if row["p_holm"] >= 0.05 else "Significant (survives Holm)"
    if row["equiv_within_0.10SD"]:
        return "Informative null (±0.10 SD)"
    if row["equiv_within_0.20SD"]:
        return "Informative null (±0.20 SD only)"
    return "Inconclusive"


lines = []
lines.append("# E vs. S+G-Combined Check — Q and ROE\n")
lines.append(
    "**Exploratory.** Splits the paper's headline `Corporate_ESG_share`/`Verified_ESG_share`/"
    "`Specificity_mean` measures back into their two actual source corpora — E alone (Section "
    "6.11) and S+G combined (Section 6.12) — using the real specificity/corporate-relevance "
    "classification (not a coarser sentence-ratio proxy), with no new GPT classification needed. "
    "Scripts and full output in `esg_e_sg/`.\n"
)

lines.append("## Reproduction\n")
lines.append(
    "`01_reproduce.py` reproduced all six Table 5a (Q) coefficients before touching the E/SG "
    "data. A gap in the merge (48 of 209 firm-years missing an E/SG value) was found and fixed "
    "the same way the original notebook zero-fills `Corporate_ESG_share` for zero-qualifying-"
    "sentence firm-years — share variables zero-filled, `Specificity_*_mean` left `NaN` (never "
    "imputed).\n\n"
    "Independent sanity check: `Corporate_E_share + Corporate_SG_share` was verified to exactly "
    "equal `Corporate_ESG_share` for all 201 overlapping firm-years (max abs diff = 0.000000) — "
    "confirms the E/S+G split is a clean, lossless partition of the existing combined measure, "
    "not a re-derivation that could silently drift from it.\n"
)

lines.append("## Part 1 — E and S+G shares, combined in one regression\n")
lines.append(
    "`Corporate_E_share_lag` and `Corporate_SG_share_lag` entered jointly (plus each alone for "
    "context), same controls/FE/HC3 as Table 5a.\n"
)
lines.append(core.round(4).to_markdown(index=False))
lines.append("\n")
lines.append(
    "**Nothing significant.** Point estimates small, CIs wide and centered near zero, E and SG "
    "barely distinguishable from each other or from their joint-model values (consistent with "
    "the weak pillar correlations already found in `esg_pillars/`).\n"
)

lines.append("## Part 2 — Pillar-specific specificity (share + specificity mean, per pillar)\n")
lines.append(
    f"`Corporate_{{X}}_share_lag` + `Specificity_{{X}}_mean_lag` jointly, per pillar (mirrors "
    "Table 5a Model 4's structure, split by corpus), for both `Q` and `ROE`. "
    "`Specificity_{X}_mean` is **never imputed** for firm-years with zero qualifying sentences "
    "in that pillar — it's genuinely undefined there and `build_model_sample()`'s complete-case "
    f"filter drops those rows automatically. Firm-years with 1-{MIN_SENTENCES - 1} sentences "
    "(nonzero but thin) are flagged and reported both included and excluded.\n"
)

lines.append("### Sentence counts underlying each firm-year's pillar specificity mean\n")
lines.append(
    "| Pillar | Zero sentences (dropped, never imputed) | 1-4 sentences (thin, flagged) | "
    f"{MIN_SENTENCES}+ sentences | Missing from merge entirely |\n"
    "|---|---|---|---|---|\n"
    "| E | 23 | 90 | 81 | 15 |\n"
    "| SG | 5 | 47 | 149 | 8 |\n\n"
    "**E is much thinner than SG.** Median E-pillar sentence count is 4 (below the 5-sentence "
    "flag threshold); median SG is 9. Excluding thin firm-years costs E far more of its sample "
    f"than SG: E's complete-case N drops from 151 to 85 (44% loss) when excluding "
    f"<{MIN_SENTENCES}-sentence firm-years, versus SG's 174 to 137 (21% loss).\n"
)

lines.append("### Full results — coefficients, CIs, and multiple-testing correction\n")
main_cols = ["dv", "spec", "sample", "term", "coef", "se", "ci_lo", "ci_hi", "p_raw", "p_holm", "p_bh", "n"]
lines.append(spec[main_cols].round(4).to_markdown(index=False))
lines.append("\n")

lines.append("### Standardized effects, MDE, TOST, and Bayes factors\n")
std_cols = ["dv", "spec", "sample", "term", "std_coef", "mde_std_80pct_power",
            "equiv_within_0.10SD", "equiv_within_0.20SD", "BF01_N(0,0.1)", "BF01_N(0,0.5)"]
lines.append(spec[std_cols].round(4).to_markdown(index=False))
lines.append("\n")

lines.append("### Verdicts\n")
spec_v = spec.copy()
spec_v["verdict"] = spec_v.apply(classify, axis=1)
lines.append(spec_v[["dv", "spec", "sample", "term", "coef", "p_raw", "p_holm", "verdict"]]
             .round(4).to_markdown(index=False))
lines.append("\n")

lines.append("## Honest summary\n")
lines.append(
    "**One near-miss, nothing that survives correction.** `Specificity_E_mean_lag` on `ROE`, "
    "full sample, has the smallest raw p-value in this entire exploratory set (p=0.0552, just "
    "over the conventional 0.05 line) — but after Holm correction across the 16 tests in this "
    "battery, its adjusted p is 0.884, nowhere close to significant. No single coefficient here "
    "should be treated as evidence of anything without that correction; reporting the raw p alone "
    "and stopping there would be a form of cherry-picking this analysis was explicitly built to "
    "avoid.\n\n"
    "**The `<5`-sentence exclusion matters most for E, least for interpretation.** Excluding "
    "thin E-pillar firm-years cuts E's sample nearly in half (151→85) and widens its CIs "
    "substantially (e.g. `Corporate_E_share_lag` on `Q`: SE goes from 0.064 to 0.128) without "
    "changing any coefficient's sign or significance — the thin observations weren't driving a "
    "hidden result, they were just adding (modest) precision. SG's much larger natural sample "
    "means this exclusion barely matters there.\n\n"
    "**Every classification here is \"informative null\" or \"inconclusive,\" never "
    "\"significant.\"** Consistent with every other cut of this data across `null_check/`, "
    "`esg_pillars/`, and this folder's Part 1 — no version of E/S/G disaggregation (raw sentence "
    "ratio, corpus-based share, or corpus-based share+specificity) surfaces a result the combined "
    "`Corporate_ESG_share`/`Verified_ESG_share` measures don't already show as null.\n"
)

lines.append("## Assumptions and judgment calls\n")
lines.append(
    f"1. **MIN_SENTENCES = 5**, per your instruction, applied only to the specificity-mean "
    "models (Part 2) -- Part 1's share-only models use the full available sample since a share "
    "is well-defined (denominator = total sentences, not qualifying sentences) even at low "
    "pillar-sentence counts.\n"
    "2. **Zero-sentence firm-years are dropped, never imputed**, exactly as instructed -- "
    "`Specificity_{X}_mean` stays genuinely `NaN` for them and `build_model_sample()`'s "
    "complete-case filter removes those rows; reported explicitly in the sentence-count table "
    "above rather than silently vanishing into a smaller N.\n"
    "3. **Multiple-testing family = all 16 tests in this battery** (2 pillars x 2 DVs x 2 "
    "samples x 2 terms) -- a defensible but not unique choice; correcting only within one "
    "sample/DV combination (4 tests) would give less conservative adjusted p-values.\n"
    "4. **Per-pillar models, not one 4-term joint model** (E share + E specificity + SG share + "
    "SG specificity together) -- matches Table 5a Model 4's per-corpus structure and avoids "
    "adding a 4-regressor model on top of an already N-constrained (thin-sentence) sample; can "
    "be added if wanted.\n"
    "5. **Same standardization/MDE/TOST/BF01 conventions as `esg_pillars/`** (partial "
    "standardized effect on each model's own complete-case sample, z not t critical values, "
    "Savage-Dickey BF01 approximation) -- not re-derived differently here.\n"
)

with open("esg_e_sg/report.md", "w") as f:
    f.write("\n".join(lines))

print("Wrote esg_e_sg/report.md")
