"""
Builds the held-out manual-annotation sample for validating the GPT polarity
reclassifier, matching the rigor convention already established for the
specificity classifier (100-sentence pilot, 4-round refinement).

This script only SELECTS sentences and exports a blank annotation sheet --
it does not produce any labels itself. Real Cohen's kappa requires a human
(you, or a research assistant) to actually fill in the "manual_case" column
by reading each sentence, blind to both the FinBERT and GPT labels.

Design: 100 sentences total, stratified to deliberately oversample the
FinBERT-negative stratum, since that's the stratum the known failure mode
(reduction/decline vocabulary misread as bad news) concentrates in, and
it's rare (67 of 3,994) -- a pure random sample would draw only ~2 of them
at n=100, too few to say anything about the specific failure this
reclassification exists to fix.

  - All 67 sentences FinBERT currently labels "negative" (full census of
    the stratum most likely to contain the known error).
  - 20 random FinBERT-"positive" sentences (checks the reclassifier doesn't
    introduce new errors in the other direction).
  - 13 random FinBERT-"neutral" sentences (fills to 100, checks the
    ambiguous_scale case against FinBERT's own neutral calls).

Run:  python3 polarity_reclass/02_validation_sample.py
"""

import pandas as pd

SOURCE_PATH = "esg_e_sg/data/df_esg_combined_specificity.csv"
SEED = 42
N_POSITIVE_SAMPLE = 20
N_NEUTRAL_SAMPLE = 13


def main():
    df = pd.read_csv(SOURCE_PATH)
    pop = df[
        (df["corporate_relevance_pred"] == "Corporate") & (df["specificity_tier_pred"] >= 1)
    ].copy().reset_index(drop=True)
    pop["polarity_sentence_id"] = [f"pol_{i}" for i in range(len(pop))]

    print(f"Population (Corporate, tier>=1): {len(pop)}")
    print(pop["sent_label"].value_counts())

    neg_all = pop[pop["sent_label"] == "negative"].copy()
    pos_sample = pop[pop["sent_label"] == "positive"].sample(n=N_POSITIVE_SAMPLE, random_state=SEED)
    neu_sample = pop[pop["sent_label"] == "neutral"].sample(n=N_NEUTRAL_SAMPLE, random_state=SEED)

    sample = pd.concat([neg_all, pos_sample, neu_sample], ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle order

    print(f"\nValidation sample: {len(sample)} sentences "
          f"({len(neg_all)} negative, {len(pos_sample)} positive, {len(neu_sample)} neutral, by FinBERT label)")

    # Blind annotation sheet: sentence text + metadata only, no FinBERT label,
    # so manual coding isn't anchored on the label it's meant to check.
    annotation_sheet = sample[
        ["polarity_sentence_id", "Company Name", "Year", "Sentence", "specificity_tier_pred"]
    ].copy()
    annotation_sheet["manual_case"] = ""  # to be filled in by hand:
    # negative_event / mitigated_negative / ambiguous_scale
    annotation_sheet["manual_notes"] = ""  # optional free text

    annotation_sheet.to_csv("polarity_reclass/data/validation_sample_BLIND.csv", index=False)

    # Separate answer key (FinBERT label only, for later comparison -- do
    # NOT open this file while hand-coding, to keep the annotation blind).
    answer_key = sample[["polarity_sentence_id", "sent_label"]].rename(
        columns={"sent_label": "finbert_label"}
    )
    answer_key.to_csv("polarity_reclass/data/validation_sample_finbert_key.csv", index=False)

    print("\nSaved:")
    print("  polarity_reclass/data/validation_sample_BLIND.csv "
          "-- hand-code the 'manual_case' column with negative_event/mitigated_negative/"
          "ambiguous_scale, blind (do not look at the FinBERT key file while doing this)")
    print("  polarity_reclass/data/validation_sample_finbert_key.csv "
          "-- FinBERT's original label per sentence, for later comparison only")
    print("\nAfter both manual coding AND the GPT batch classification (01_classify_batch.py) "
          "are done, run 03_rebuild_shares.py to compute kappa (GPT vs. manual, GPT vs. FinBERT) "
          "and rebuild the two share variables.")


if __name__ == "__main__":
    main()
