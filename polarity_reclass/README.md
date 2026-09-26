# polarity_reclass

GPT-based polarity reclassification for Corporate-relevant, specificity
tier>=1 sentences, replacing FinBERT's `sent_label` for this population.
Motivated by a documented FinBERT failure mode (keys on reduction/decline
vocabulary without distinguishing a negative event from a negative outcome
being successfully mitigated -- see this project's spot-check audit: 4 of
10 random tier>=2/negative sentences were clear mislabels).

**Status: Revision 5 (4-case schema, Case 2 harm-scope fix + 4 ambiguous_scale patterns).**
Revision 1 (3-case) was run and produced `df_ar_ceo_sentences_polarity_reclass.csv` (200 batch
requests, 3,994/3,994 classified). A spot-check of 15 random `mitigated_negative` sentences found
14 of 15 were plain positive statements with no negative/reduction vocabulary -- Case 2 had become
a generic positive catch-all. Case 4 (`plain_positive`) was added in Revision 2 to fix this and
re-run, producing `df_ar_ceo_sentences_polarity_reclass_v2.csv`. The REAL validation metric was
then run for the first time: Cohen's kappa against 100 manually-coded sentences came back at
**0.7368**, below the 0.80 target. A Revision 3 attempt tightened Case 2's qualifying vocabulary
based on a disputed spot-check claim, but the real kappa confusion matrix showed Case 2 was
already strong (90.5% recall on mitigated_negative, 92.9% on plain_positive) -- Revision 3 was
solving the wrong problem and was **reverted**. The actual bottleneck was Case 3
(`ambiguous_scale`): only 17.6% recall (3 of 17) against manual coding. Revision 4 added three new
Case 3 anchors and re-ran the validation-only pilot (6.13.2b): kappa improved to **0.7667**
(4-case) / 0.7517 (3-way polarity), and ambiguous_scale recall nearly tripled to 47.1% -- still
below target. Reading all 17 disagreement sentences from that pilot found five distinct root
causes: (1) the new pattern (c) [bare scale/fact] over-triggered onto performance metrics and
established practices that ARE the achievement (a 95.6% survey favorability score, an annual
pay-equity review); (2) a real structural gap in Case 2 -- qualifying reduction words fired
regardless of what was being reduced, wrongly pulling job cuts, a leadership-team headcount trim,
and a narrowed charitable-program scope into `mitigated_negative`; (3) pattern (a)
[rhetorical/no-outcome] was too narrow, missing conditionals ("if X, then Y") and generic
statements of business principle; (4) a new pattern (d) was needed for personal/emotional
reflection with no company action described; (5) a narrow, real piece of the reverted Revision 3
-- "despite adversity, delivered a good outcome" -- was reinstated with a single anchor, since the
real manual annotation confirmed this specific pattern does matter (distinct from Revision 3's
broader, unnecessary whitelist changes). Revision 5 fixes all five with real corpus sentences as
anchors. A sixth issue -- inconsistent over/under-triggering of the vague forward-commitment
pattern (b) -- is treated as residual noise for now, not chased further. The notebook's Section
6.13 now uses task name `polarity_reclass_tier1plus_v5` and exports to
`df_ar_ceo_sentences_polarity_reclass_v5.csv`, keeping Revisions 1, 2, and 4's output on Drive
untouched for the audit trail (Revision 3's output, if any was produced, should not be trusted or
reused).

**Validation-only pilot check (notebook Step 6.13.2b)** classifies just the 100-sentence
manually-coded validation sample synchronously (mirroring the 6.10 pilot pattern, not the Batch
API) and reports kappa immediately -- before committing to a full Batch API re-run over all 3,994
sentences. Only proceed to the full re-run if that pilot clears kappa >= 0.80. Coverage
numbers/firm-year CSV are NOT to be trusted until the full re-run's Cohen's kappa (6.13.10) also
passes.
Re-running requires an authenticated OpenAI client in Colab (no API access in
this sandbox) and genuine human manual annotation for validation (cannot be
fabricated). Steps 1 (population count), the validation-sample selection, and two full
validation-only pilot runs (Revisions 2 and 4) have been done for real; everything downstream of
the Revision 5 classification is written but blocked on you running it.

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
  `ambiguous_scale` / `plain_positive`), Revision 5: Case 2 now requires the
  reduced quantity to be a genuine risk/harm (not headcount/jobs/program
  scope), plus a "despite adversity, delivered a good outcome" anchor
  (Allianz 2020); Case 3 gains a fourth pattern (personal/emotional
  reflection with no company action described) and broadens pattern (a) to
  cover conditionals and generic statements of principle, with counter-
  examples for the reduction-whitelist gap (Talanx 2016 job cuts, AIG 2015
  headcount trim, Progressive 2021 narrowed program scope); Case 4 gains two
  counter-examples distinguishing a performance metric that IS the
  achievement from a bare neutral scale fact (MS&AD 2021 survey score,
  Allstate 2020 equity review). Revision 4's four Case 3 anchors (Daiichi
  2020, MS&AD 2022, Tokio Marine 2021, Ping An 2021) and the original
  Revision 1/2 anchors (Chubb 2018 GHG-reduction, Progressive 2022
  engagement-index, Travelers 2012 catastrophe-loss, Prudential 2018
  capital-deployment) are unchanged. All anchors are real corpus sentences.
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
# 1. Hand-code data/validation_sample_BLIND.csv (manual_case column, all 100 rows) -- already done
# 2. In Colab: notebook Step 6.13.2b -- validation-only synchronous pilot check (100 sentences).
#    Only proceed to the full batch re-run if this clears kappa >= 0.80.
# 3. In Colab: import 00_prompt_design, run 01_classify_batch.py's cells (submit, wait, download, merge)
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
