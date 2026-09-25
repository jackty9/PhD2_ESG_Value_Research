# polarity_reclass

GPT-based polarity reclassification for Corporate-relevant, specificity
tier>=1 sentences, replacing FinBERT's `sent_label` for this population.
Motivated by a documented FinBERT failure mode (keys on reduction/decline
vocabulary without distinguishing a negative event from a negative outcome
being successfully mitigated -- see this project's spot-check audit: 4 of
10 random tier>=2/negative sentences were clear mislabels).

**Status: prompt and pipeline are built; classification has not been run.**
This requires an authenticated OpenAI client in Colab (no API access in
this sandbox) and genuine human manual annotation for validation (cannot be
fabricated). Steps 1 (population count) and the validation-sample selection
have been run for real; everything downstream of the actual GPT
classification is written but blocked on you running it.

## Population

3,994 sentences (Corporate-relevant, specificity tier>=1), as of the
current `esg_e_sg/data/df_esg_combined_specificity.csv` snapshot.
Re-verify this count before submitting the batch job, in case the
underlying corpus file has changed since.

Tier breakdown: tier 1 = 2,523, tier 2 = 940, tier 3 = 531.
Current (FinBERT) polarity breakdown: positive = 2,072, neutral = 1,855,
negative = 67.

## Contents

- `00_prompt_design.py` -- the locked `POLARITY_SYSTEM_PROMPT` and JSON
  schema. Three-case taxonomy (`negative_event` / `mitigated_negative` /
  `ambiguous_scale`), with the three exact sentences from this project's
  own spot-check audit as few-shot anchors (Chubb 2018 GHG-reduction,
  Progressive 2022 engagement-index, Travelers 2012 catastrophe-loss).
  Run standalone to print prompt length and schema enums (no API call).
- `01_classify_batch.py` -- Batch API submission script, mirroring
  Section 6.11/6.12's exact pattern (write JSONL, submit, poll, download,
  parse, merge back). **Must be run in Colab** with `client`/`MODEL`
  already set up (same as Section 6.10.1) and `POLARITY_SYSTEM_PROMPT`/
  `polarity_schema` imported from `00_prompt_design.py`. The actual
  `submit_polarity_batch_job(...)` call is commented out by default --
  uncomment when ready to actually fire the job (can take up to 24h to
  complete).
- `02_validation_sample.py` -- **already run.** Selects the 100-sentence
  hand-coding sample (all 67 current FinBERT-negative sentences, plus 20
  random positive and 13 random neutral, oversampling the rare stratum
  the known failure mode concentrates in). Outputs:
  - `data/validation_sample_BLIND.csv` -- hand-code the `manual_case`
    column (`negative_event` / `mitigated_negative` / `ambiguous_scale`)
    for all 100 rows, blind to both FinBERT's and GPT's labels.
  - `data/validation_sample_finbert_key.csv` -- FinBERT's original label,
    kept separate so it doesn't bias hand-coding.
- `03_rebuild_shares.py` -- **not runnable yet.** Refuses to run
  (raises an explicit error) until (a) the GPT batch job has completed and
  produced `data/df_ar_ceo_sentences_polarity_reclass.csv`, and (b) the
  validation sample has been fully hand-coded. Once both exist, computes:
  1. Cohen's kappa, GPT case vs. manual case (the real validation metric;
     target >=0.80, matching the specificity classifier's own standard --
     kappa 0.885/0.851/0.803 across its three fields after 4 refinement
     rounds).
  2. Cohen's kappa, GPT polarity vs. FinBERT `sent_label` (quantifies how
     much actually changed -- low kappa here is expected and desired, not
     a problem, as long as kappa-vs-manual is high).
  3. Rebuilds `Positive_Specific_share`/`Negative_Specific_share` from the
     GPT labels, restricted to tier>=2 (the threshold the two regressors
     actually use -- NOT lowered to tier>=1 even though reclassification
     covered the broader tier>=1 population).
  4. New coverage check (firm-years with >=1 qualifying sentence, same
     format as the original audit), to confirm or rule out whether
     `Negative_Specific_share`'s sparsity persists.

## Run order

```
python3 polarity_reclass/00_prompt_design.py       # sanity-check the prompt (no API call)
python3 polarity_reclass/02_validation_sample.py    # already run -- re-run only if you want a new sample
# 1. Hand-code data/validation_sample_BLIND.csv (manual_case column, all 100 rows)
# 2. In Colab: import 00_prompt_design, run 01_classify_batch.py's cells (submit, wait, download, merge)
python3 polarity_reclass/03_rebuild_shares.py       # after both of the above are complete
```

## Known dependency

`03_rebuild_shares.py` imports `sklearn.metrics.cohen_kappa_score` --
standard in Colab (already used elsewhere in this project for KMeans
clustering), not installed in this sandbox.

## Decision pending

Whether `Negative_Specific_share` remains too sparse to test after
reclassification is an open question this folder is built to answer, not
assume. If it does, per instruction, no regression should be run on it --
report the new coverage check only and decide from there.
