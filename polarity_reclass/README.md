# polarity_reclass

GPT-based polarity reclassification for Corporate-relevant, specificity
tier>=1 sentences, replacing FinBERT's `sent_label` for this population.
Motivated by a documented FinBERT failure mode (keys on reduction/decline
vocabulary without distinguishing a negative event from a negative outcome
being successfully mitigated -- see this project's spot-check audit: 4 of
10 random tier>=2/negative sentences were clear mislabels).

**Status: Revision 3 (4-case schema, tightened Case 2 qualifying-word whitelist).**
Revision 1 (3-case) was run and produced `df_ar_ceo_sentences_polarity_reclass.csv`
(200 batch requests, 3,994/3,994 classified). A spot-check of 15 random `mitigated_negative`
sentences found 14 of 15 were plain positive statements with no negative/reduction vocabulary --
Case 2 had become a generic positive catch-all. Case 4 (`plain_positive`) was added in Revision 2
to fix this and re-run, producing `df_ar_ceo_sentences_polarity_reclass_v2.csv`. A fresh
15-sentence spot-check of the corrected `mitigated_negative` bucket was then disputed: independent
review found 4 of 15 still misclassified, in two patterns -- (1) vague progress/care language
("gained ground," "moved up," "progress," "paying attention to") accepted as reduction vocabulary
with no specific negative quantity actually named, and (2) admitted-gap/deferred-commitment
sentences (a measurement method still under development, an unaddressed gap) misclassified as
positive mitigations. Revision 3 adds an explicit whitelist of qualifying negative/reduction
words/phrases for Case 2, an explicit non-qualifying list of vague-progress words that must not
justify it, and a rule routing admitted-gap/deferred-commitment sentences to Case 3
(`ambiguous_scale`). The notebook's Section 6.13 now uses task name
`polarity_reclass_tier1plus_v3` and exports to `df_ar_ceo_sentences_polarity_reclass_v3.csv`,
keeping Revisions 1 and 2's output on Drive untouched for the audit trail. Re-run pending.
Coverage numbers/firm-year CSV are NOT to be trusted until a fresh spot-check of the v3
`mitigated_negative` bucket AND Cohen's kappa vs. manual annotation both pass.
Re-running requires an authenticated OpenAI client in Colab (no API access in
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
  schema. Four-case taxonomy (`negative_event` / `mitigated_negative` /
  `ambiguous_scale` / `plain_positive`), Revision 3: Case 2 now requires an
  explicit qualifying negative/reduction word or phrase from a whitelist
  (reduce, cut, limit, avoid, decline, protect-against-harm, etc.), with a
  matching non-qualifying list (progress, gained ground, moved up, momentum,
  etc.) that must not be used to justify it, and admitted-gap/deferred-
  commitment sentences now route to Case 3. Few-shot anchors: the three
  Revision 1 sentences (Chubb 2018 GHG-reduction, Progressive 2022
  engagement-index, Travelers 2012 catastrophe-loss), the Revision 2 Case 4
  anchor (Prudential 2018 capital-deployment), two new valid Case 2 anchors
  (Allianz 2013 "despite considerable losses," Prudential 2020 "protect
  their health and safety"), two counter-example anchors (Progressive 2021
  "gained ground," Allianz 2016 "moved up" -- both plain_positive, not
  mitigated_negative), and one new ambiguous_scale anchor (Allstate 2023
  Scope 3 measurement-not-yet-developed sentence).
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
