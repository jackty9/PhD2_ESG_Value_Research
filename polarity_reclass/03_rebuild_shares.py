"""
Run this AFTER both:
  (a) 01_classify_batch.py has completed and produced
      df_ar_ceo_sentences_polarity_reclass.csv (the full GPT-relabeled
      Corporate/tier>=1 population), and
  (b) polarity_reclass/data/validation_sample_BLIND.csv has been hand-coded
      (the 'manual_case' column filled in for all 100 rows).

NOT runnable yet -- both inputs are placeholders/pending real execution.
This script will refuse to run (raises FileNotFoundError) until they exist,
rather than silently producing output from missing data.

Computes:
  1. Cohen's kappa: GPT case vs. manual case (the real validation metric)
  2. Cohen's kappa: GPT polarity vs. FinBERT sent_label (quantifies how much
     actually changed, to rule out this being relabeling noise)
  3. Rebuilds Positive_Specific_share / Negative_Specific_share using GPT
     labels, restricted to tier>=2 (the threshold the two regressors use --
     NOT lowered to tier>=1 even though reclassification covered tier>=1)
  4. Coverage check for both (firm-years with >=1 qualifying sentence),
     same format as the original FinBERT-based check, to confirm whether
     Negative_Specific_share's sparsity problem persists.

Run:  python3 polarity_reclass/03_rebuild_shares.py
"""

import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

GPT_RESULTS_PATH = "polarity_reclass/data/df_ar_ceo_sentences_polarity_reclass.csv"
MANUAL_SAMPLE_PATH = "polarity_reclass/data/validation_sample_BLIND.csv"
FINBERT_KEY_PATH = "polarity_reclass/data/validation_sample_finbert_key.csv"


def check_inputs_exist():
    missing = []
    if not os.path.exists(GPT_RESULTS_PATH):
        missing.append(GPT_RESULTS_PATH)
    if not os.path.exists(MANUAL_SAMPLE_PATH):
        missing.append(MANUAL_SAMPLE_PATH)
    if missing:
        raise FileNotFoundError(
            f"Required input(s) not found: {missing}. This script cannot run until "
            f"01_classify_batch.py has produced GPT results and the validation sample "
            f"has been hand-coded (manual_case column filled in). Nothing was computed."
        )
    manual = pd.read_csv(MANUAL_SAMPLE_PATH)
    if manual["manual_case"].isna().all() or (manual["manual_case"] == "").all():
        raise ValueError(
            f"{MANUAL_SAMPLE_PATH} exists but 'manual_case' column is entirely empty -- "
            f"hand-coding has not been done yet. Nothing was computed."
        )
    n_uncoded = (manual["manual_case"].isna() | (manual["manual_case"] == "")).sum()
    if n_uncoded > 0:
        raise ValueError(
            f"{n_uncoded} of {len(manual)} rows in the validation sample have no manual_case "
            f"value -- hand-coding is incomplete. Finish coding all rows before running kappa."
        )


def compute_kappa_stats():
    gpt = pd.read_csv(GPT_RESULTS_PATH)
    manual = pd.read_csv(MANUAL_SAMPLE_PATH)
    finbert_key = pd.read_csv(FINBERT_KEY_PATH)

    val = manual.merge(
        gpt[["polarity_sentence_id", "case", "polarity"]], on="polarity_sentence_id", how="left"
    )
    val = val.merge(finbert_key, on="polarity_sentence_id", how="left")

    n_missing_gpt = val["case"].isna().sum()
    if n_missing_gpt > 0:
        print(f"WARNING: {n_missing_gpt} validation-sample sentences have no GPT classification "
              f"(id mismatch or failed request) -- excluded from kappa.")
    val_complete = val.dropna(subset=["case", "manual_case"])

    kappa_gpt_manual = cohen_kappa_score(val_complete["case"], val_complete["manual_case"])
    print(f"\nCohen's kappa, GPT case vs. manual case (n={len(val_complete)}): {kappa_gpt_manual:.4f}")
    print("Reference: this project's specificity classifier achieved kappa 0.885 (relevance), "
          "0.851 (Corporate/Contextual), 0.803 (specificity_tier, weighted) after 4 rounds of "
          "refinement. This is the standard this reclassification is held to (target >=0.80).")

    # GPT polarity vs FinBERT (three-way: positive/negative/neutral), on the
    # SAME validation-sample rows, so this is a like-for-like comparison, not
    # the full 3,994-row population.
    val_complete_fb = val.dropna(subset=["polarity", "finbert_label"])
    kappa_gpt_finbert = cohen_kappa_score(val_complete_fb["polarity"], val_complete_fb["finbert_label"])
    print(f"\nCohen's kappa, GPT polarity vs. FinBERT sent_label (same {len(val_complete_fb)} "
          f"validation-sample sentences): {kappa_gpt_finbert:.4f}")
    print("Low kappa here is EXPECTED and not itself a problem -- it quantifies how much the "
          "labels changed, which is the whole point of this reclassification. High kappa vs. "
          "manual (above) combined with low-to-moderate kappa vs. FinBERT (here) is the signature "
          "of 'fixed a real problem,' not 'relabeling noise.' If kappa vs. manual is also low, "
          "that's a different, worse signal: the GPT reclassifier doesn't match human judgment "
          "either, and the fix didn't work.")

    return val_complete


def rebuild_shares():
    gpt = pd.read_csv(GPT_RESULTS_PATH)
    tier2plus = gpt[gpt["specificity_tier_pred"] >= 2].copy()
    print(f"\nOf the {len(gpt)} reclassified (Corporate, tier>=1) sentences, "
          f"{len(tier2plus)} are tier>=2 (the population these two share variables use).")

    tier2plus["is_pos_specific_new"] = tier2plus["polarity"] == "positive"
    tier2plus["is_neg_specific_new"] = tier2plus["polarity"] == "negative"

    # Need total_sentences per firm-year as denominator -- reuse the same
    # source as esg_e_sg/02_regression.py's construction (total sentence
    # count per firm-year from the full, unrestricted corpus, not just this
    # tier>=1 population). Adjust this path if your Drive layout differs.
    full_corpus = pd.read_csv("esg_e_sg/data/df_esg_combined_specificity.csv")  # placeholder;
    # swap for the actual full (non-ESG-restricted) per-firm-year sentence
    # count source when running this for real, matching Positive_Specific_share's
    # original denominator convention (share of ALL sentences, not just ESG ones).
    total_by_fy = full_corpus.groupby(["Company Name", "Year"]).size().rename("total_sentences")

    pos_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_pos_specific_new"].sum().rename("n_pos_specific_new")
    neg_by_fy = tier2plus.groupby(["Company Name", "Year"])["is_neg_specific_new"].sum().rename("n_neg_specific_new")

    fy = pd.concat([total_by_fy, pos_by_fy, neg_by_fy], axis=1).reset_index()
    fy[["n_pos_specific_new", "n_neg_specific_new"]] = fy[["n_pos_specific_new", "n_neg_specific_new"]].fillna(0)
    fy["Positive_Specific_share_new"] = fy["n_pos_specific_new"] / fy["total_sentences"]
    fy["Negative_Specific_share_new"] = fy["n_neg_specific_new"] / fy["total_sentences"]

    n_fy_total = fy.shape[0]
    n_fy_pos = (fy["n_pos_specific_new"] > 0).sum()
    n_fy_neg = (fy["n_neg_specific_new"] > 0).sum()

    print(f"\n=== New coverage check (GPT-relabeled, tier>=2 only) ===")
    print(f"Total firm-years: {n_fy_total}")
    print(f"Firm-years with >=1 qualifying Positive_Specific sentence: {n_fy_pos}")
    print(f"Firm-years with >=1 qualifying Negative_Specific sentence: {n_fy_neg}")
    print(f"\nFor comparison, the ORIGINAL FinBERT-based coverage (from the earlier audit) was: "
          f"186 firm-years positive, 26 firm-years negative (of 236 total).")

    fy.to_csv("polarity_reclass/data/pos_neg_specific_fy_RECLASSIFIED.csv", index=False)
    print("\nSaved: polarity_reclass/data/pos_neg_specific_fy_RECLASSIFIED.csv")
    return fy


def main():
    check_inputs_exist()
    compute_kappa_stats()
    rebuild_shares()


if __name__ == "__main__":
    main()
