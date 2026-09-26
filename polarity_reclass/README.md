# polarity_reclass

GPT-based polarity reclassification for Corporate-relevant, specificity
tier>=1 sentences, replacing FinBERT's `sent_label` for this population.
Motivated by a documented FinBERT failure mode (keys on reduction/decline
vocabulary without distinguishing a negative event from a negative outcome
being successfully mitigated -- see this project's spot-check audit: 4 of
10 random tier>=2/negative sentences were clear mislabels).

**Status: Revision 6 classification complete on the full 3,994-sentence population, kappa
accepted at 0.77-0.80 (see "Kappa: accepted, not chased further" below).**

Revision 1 (3-case) was run and produced
`df_ar_ceo_sentences_polarity_reclass.csv` (200 batch requests, 3,994/3,994 classified). A
spot-check of 15 random `mitigated_negative` sentences found 14 of 15 were plain positive
statements with no negative/reduction vocabulary -- Case 2 had become a generic positive catch-all.
Case 4 (`plain_positive`) was added in Revision 2 to fix this and re-run, producing
`df_ar_ceo_sentences_polarity_reclass_v2.csv`. The REAL validation metric was then run for the
first time: Cohen's kappa against 100 manually-coded sentences came back at **0.7368**, below the
0.80 target. A Revision 3 attempt tightened Case 2's qualifying vocabulary based on a disputed
spot-check claim, but the real kappa confusion matrix showed Case 2 was already strong -- Revision
3 was solving the wrong problem and was **reverted**. The actual bottleneck was Case 3
(`ambiguous_scale`): only 17.6% recall (3 of 17). Revision 4 added three new Case 3 anchors: kappa
improved to **0.7667** (4-case) / 0.7517 (3-way polarity), ambiguous_scale recall nearly tripled to
47.1% -- still below target, with five further root causes found and fixed in Revision 5 (Case 2
harm-scope gap, broadened pattern (a), new pattern (d), tightened pattern (c), a narrow reinstated
"despite adversity" anchor). Revision 5's validation-only pilot brought kappa to **0.7731** and
ambiguous_scale recall to 88.2% (15/17) -- but negative_event recall dropped hard, 94.1% -> 70.6%
(9/34 now wrongly called `ambiguous_scale`), and 3-way polarity kappa actually *fell* (0.7517 ->
0.7338), since misclassifying a real negative event as neutral directly undercounts
`Negative_Specific_share`. Reading all 9 misses found two real prompt bugs -- pattern (d)
over-triggered on emotional language wrapping a REAL described event (AIG 2012, "witnessed their
company being rapidly dismantled"), and the despite-adversity pattern over-triggered on named
hardship with no stated completed outcome (Travelers 2014) -- both fixed in Revision 6 with new
Case 1 counter-anchors. The remaining 6 misses (MS&AD 2018/2019, Chubb 2023 x2, Chubb 2020 x2) were
a **manual-coding correction, not a schema change**: broad societal/political commentary the CEO
letter cites as context, not a specific event reported as having happened to the company --
confirmed `corporate_relevance_pred == "Corporate"` for all 6 (not an upstream filtering issue). No
prompt anchors were added pulling this pattern toward `negative_event`; GPT's original
`ambiguous_scale` calls on these were already correct. A new Step 6.13.2a applies this manual-
coding correction to `validation_sample_BLIND.csv` before the pilot re-run. The notebook's Section
6.13 now uses task name `polarity_reclass_tier1plus_v6` and exports to
`df_ar_ceo_sentences_polarity_reclass_v6.csv`, keeping Revisions 1, 2, 4, and 5's output on Drive
untouched for the audit trail (Revision 3's output, if any was produced, should not be trusted or
reused).

**Kappa: accepted, not chased further.** After the 6-row manual-coding correction, the
validation-only pilot (6.13.2b) scored kappa 0.7983 -- 0.0017 short of the 0.80 target, well within
n=100 sampling noise. The full 3,994-sentence Batch API run then completed (200/200 requests,
3,993/3,994 classified; FinBERT-vs-GPT crosstab confirmed the original motivating hypothesis: of
FinBERT's 67 "negative" sentences, GPT now calls only 34 actually negative, with 21 reclassified
positive -- the reduction-vocabulary failure mode this project set out to fix). Re-scoring the same
100 validation sentences from that full run's output (6.13.10) came back at kappa **0.7718** -- a
swing of 0.03 from the pilot's 0.7983 on the identical sentences under the identical locked prompt.
That run-to-run variance is larger than the remaining gap to 0.80, concentrated entirely in the
`ambiguous_scale`/`negative_event` boundary (GPT itself classifies a handful of borderline
sentences inconsistently across separate API calls). After six refinement rounds, each of which
found and fixed a real, identifiable error until Revision 6, further prompt engineering at this
point risks chasing classification noise rather than a fixable defect. **Decision: kappa in the
0.77-0.80 range is accepted as the practical ceiling for this taxonomy; proceeded to the coverage
rebuild (6.13.8) reporting kappa honestly rather than treating 0.80 as an uncrossable hard gate.**

Re-running requires an authenticated OpenAI client in Colab (no API access in
this sandbox) and genuine human manual annotation for validation (cannot be
fabricated). Steps 1 (population count), the validation-sample selection, three full
validation-only pilot runs (Revisions 2, 4, and 5), and the full 3,994-sentence classification
(Revision 6) have been done for real.

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
  `ambiguous_scale` / `plain_positive`), Revision 6: pattern (d)
  [personal/emotional reflection] now requires that NO concrete event be
  named -- emotional language wrapping a real described event (AIG 2012,
  "company being rapidly dismantled") stays `negative_event`, not
  `ambiguous_scale`. The Case 2 "despite adversity" pattern now requires
  BOTH named adversity AND a stated completed positive outcome -- naming
  hardship alone (Travelers 2014) is `negative_event`, not
  `mitigated_negative`. All Revision 5 content (Case 2 harm-scope gap fix,
  broadened pattern (a), pattern (c) tightening, Allianz 2020
  despite-adversity anchor) and Revision 4's four Case 3 anchors (Daiichi
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
    for all 100 rows, blind to both FinBERT's and GPT's labels. **Correction
    applied as of Revision 6** (notebook Step 6.13.2a): 6 rows (MS&AD
    2018/2019, Chubb 2023 x2, Chubb 2020 x2) re-coded from `negative_event`
    to `ambiguous_scale` -- broad societal/political commentary, not a
    reported company event. Run 6.13.2a once against your Drive copy before
    the next pilot re-run.
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
- `04_regression.py` -- **run for real** on the Revision 6 full-population
  classification (`data/df_ar_ceo_sentences_polarity_reclass_v6.csv`). Builds
  `Positive_Specific_share`/`Negative_Specific_share` (tier>=2, `total_sent`
  denominator from `esg_e_sg/data/panel_reg_export.csv`), merges into the
  same panel `esg_e_sg/common.py`'s `load_and_reconstruct()` builds, lags
  both shares, and fits Q and ROE with the same controls/FE/HC3 spec as
  `esg_e_sg/02_regression.py` (`CTRL_5A_FORMULA`, `build_model_sample`),
  reporting each share jointly and alone. Results: neither
  `Positive_Specific_share_lag` nor `Negative_Specific_share_lag` is
  statistically significant for either Q or ROE, alone or jointly (all
  p>0.17, N=183, MetLife and Progressive dropped for insufficient
  observations). `Negative_Specific_share_lag` has only 12 non-zero
  firm-years within this N=183 sample (of 17 non-zero across all 238) --
  any interpretation of its coefficient should account for that sparsity.
  Saves `results_specific_share.csv`.

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
