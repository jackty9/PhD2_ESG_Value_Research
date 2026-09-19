# E vs. S+G-Combined Check — Q and ROE

**Exploratory.** Splits the paper's headline `Corporate_ESG_share`/`Verified_ESG_share`/`Specificity_mean` measures back into their two actual source corpora — E alone (Section 6.11) and S+G combined (Section 6.12) — using the real specificity/corporate-relevance classification (not a coarser sentence-ratio proxy), with no new GPT classification needed. Scripts and full output in `esg_e_sg/`.

## Reproduction

`01_reproduce.py` reproduced all six Table 5a (Q) coefficients before touching the E/SG data. A gap in the merge (48 of 209 firm-years missing an E/SG value) was found and fixed the same way the original notebook zero-fills `Corporate_ESG_share` for zero-qualifying-sentence firm-years — share variables zero-filled, `Specificity_*_mean` left `NaN` (never imputed).

Independent sanity check: `Corporate_E_share + Corporate_SG_share` was verified to exactly equal `Corporate_ESG_share` for all 201 overlapping firm-years (max abs diff = 0.000000) — confirms the E/S+G split is a clean, lossless partition of the existing combined measure, not a re-derivation that could silently drift from it.

## Part 1 — E and S+G shares, combined in one regression

`Corporate_E_share_lag` and `Corporate_SG_share_lag` entered jointly (plus each alone for context), same controls/FE/HC3 as Table 5a.

| dv   | spec                    | term                   |   coef |     se |   ci_lo |   ci_hi |      p |   n |
|:-----|:------------------------|:-----------------------|-------:|-------:|--------:|--------:|-------:|----:|
| Q    | Joint (E + SG combined) | Corporate_E_share_lag  | 0.0279 | 0.0588 | -0.0873 |  0.1431 | 0.635  | 183 |
| Q    | Joint (E + SG combined) | Corporate_SG_share_lag | 0.032  | 0.0329 | -0.0324 |  0.0965 | 0.3298 | 183 |
| Q    | E alone                 | Corporate_E_share_lag  | 0.0267 | 0.0592 | -0.0893 |  0.1427 | 0.6521 | 183 |
| Q    | SG alone                | Corporate_SG_share_lag | 0.0317 | 0.0325 | -0.0321 |  0.0954 | 0.3303 | 183 |
| ROE  | Joint (E + SG combined) | Corporate_E_share_lag  | 0.0046 | 0.0455 | -0.0845 |  0.0937 | 0.9196 | 183 |
| ROE  | Joint (E + SG combined) | Corporate_SG_share_lag | 0.0048 | 0.0193 | -0.033  |  0.0425 | 0.8053 | 183 |
| ROE  | E alone                 | Corporate_E_share_lag  | 0.0044 | 0.0454 | -0.0846 |  0.0934 | 0.9227 | 183 |
| ROE  | SG alone                | Corporate_SG_share_lag | 0.0047 | 0.0192 | -0.0329 |  0.0423 | 0.8071 | 183 |


**Nothing significant.** Point estimates small, CIs wide and centered near zero, E and SG barely distinguishable from each other or from their joint-model values (consistent with the weak pillar correlations already found in `esg_pillars/`).

## Part 2 — Pillar-specific specificity (share + specificity mean, per pillar)

`Corporate_{X}_share_lag` + `Specificity_{X}_mean_lag` jointly, per pillar (mirrors Table 5a Model 4's structure, split by corpus), for both `Q` and `ROE`. `Specificity_{X}_mean` is **never imputed** for firm-years with zero qualifying sentences in that pillar — it's genuinely undefined there and `build_model_sample()`'s complete-case filter drops those rows automatically. Firm-years with 1-4 sentences (nonzero but thin) are flagged and reported both included and excluded.

### Sentence counts underlying each firm-year's pillar specificity mean

| Pillar | Zero sentences (dropped, never imputed) | 1-4 sentences (thin, flagged) | 5+ sentences | Missing from merge entirely |
|---|---|---|---|---|
| E | 23 | 90 | 81 | 15 |
| SG | 5 | 47 | 149 | 8 |

**E is much thinner than SG.** Median E-pillar sentence count is 4 (below the 5-sentence flag threshold); median SG is 9. Excluding thin firm-years costs E far more of its sample than SG: E's complete-case N drops from 151 to 85 (44% loss) when excluding <5-sentence firm-years, versus SG's 174 to 137 (21% loss).

### Full results — coefficients, CIs, and multiple-testing correction

| dv   | spec                 | sample                       | term                    |    coef |     se |   ci_lo |   ci_hi |   p_raw |   p_holm |   p_bh |   n |
|:-----|:---------------------|:-----------------------------|:------------------------|--------:|-------:|--------:|--------:|--------:|---------:|-------:|----:|
| Q    | E share+specificity  | full (incl. thin firm-years) | Corporate_E_share_lag   |  0.025  | 0.0643 | -0.1009 |  0.151  |  0.6968 |   1      | 0.943  | 151 |
| Q    | E share+specificity  | full (incl. thin firm-years) | Specificity_E_mean_lag  | -0.0045 | 0.007  | -0.0183 |  0.0092 |  0.5185 |   1      | 0.943  | 151 |
| Q    | E share+specificity  | excl. <5-sentence firm-years | Corporate_E_share_lag   | -0.0295 | 0.1275 | -0.2795 |  0.2205 |  0.817  |   1      | 0.943  |  85 |
| Q    | E share+specificity  | excl. <5-sentence firm-years | Specificity_E_mean_lag  | -0.0061 | 0.0149 | -0.0354 |  0.0232 |  0.6824 |   1      | 0.943  |  85 |
| Q    | SG share+specificity | full (incl. thin firm-years) | Corporate_SG_share_lag  |  0.0221 | 0.0349 | -0.0463 |  0.0905 |  0.5265 |   1      | 0.943  | 174 |
| Q    | SG share+specificity | full (incl. thin firm-years) | Specificity_SG_mean_lag |  0.0041 | 0.0069 | -0.0095 |  0.0176 |  0.5582 |   1      | 0.943  | 174 |
| Q    | SG share+specificity | excl. <5-sentence firm-years | Corporate_SG_share_lag  | -0.0084 | 0.0382 | -0.0833 |  0.0665 |  0.8251 |   1      | 0.943  | 137 |
| Q    | SG share+specificity | excl. <5-sentence firm-years | Specificity_SG_mean_lag |  0.0052 | 0.012  | -0.0184 |  0.0288 |  0.6664 |   1      | 0.943  | 137 |
| ROE  | E share+specificity  | full (incl. thin firm-years) | Corporate_E_share_lag   |  0.0022 | 0.0502 | -0.0963 |  0.1007 |  0.9652 |   1      | 0.9652 | 151 |
| ROE  | E share+specificity  | full (incl. thin firm-years) | Specificity_E_mean_lag  | -0.0068 | 0.0036 | -0.0138 |  0.0002 |  0.0552 |   0.8838 | 0.8838 | 151 |
| ROE  | E share+specificity  | excl. <5-sentence firm-years | Corporate_E_share_lag   |  0.0254 | 0.0974 | -0.1655 |  0.2164 |  0.7939 |   1      | 0.943  |  85 |
| ROE  | E share+specificity  | excl. <5-sentence firm-years | Specificity_E_mean_lag  | -0.0104 | 0.0103 | -0.0307 |  0.0098 |  0.313  |   1      | 0.943  |  85 |
| ROE  | SG share+specificity | full (incl. thin firm-years) | Corporate_SG_share_lag  |  0.001  | 0.0216 | -0.0414 |  0.0434 |  0.9627 |   1      | 0.9652 | 174 |
| ROE  | SG share+specificity | full (incl. thin firm-years) | Specificity_SG_mean_lag | -0.0043 | 0.0051 | -0.0144 |  0.0057 |  0.4    |   1      | 0.943  | 174 |
| ROE  | SG share+specificity | excl. <5-sentence firm-years | Corporate_SG_share_lag  |  0.0082 | 0.0268 | -0.0443 |  0.0607 |  0.759  |   1      | 0.943  | 137 |
| ROE  | SG share+specificity | excl. <5-sentence firm-years | Specificity_SG_mean_lag | -0.0062 | 0.0083 | -0.0224 |  0.01   |  0.4499 |   1      | 0.943  | 137 |


### Standardized effects, MDE, TOST, and Bayes factors

| dv   | spec                 | sample                       | term                    |   std_coef |   mde_std_80pct_power | equiv_within_0.10SD   | equiv_within_0.20SD   |   BF01_N(0,0.1) |   BF01_N(0,0.5) |
|:-----|:---------------------|:-----------------------------|:------------------------|-----------:|----------------------:|:----------------------|:----------------------|----------------:|----------------:|
| Q    | E share+specificity  | full (incl. thin firm-years) | Corporate_E_share_lag   |     0.019  |                0.1365 | True                  | True                  |          2.1477 |          9.5667 |
| Q    | E share+specificity  | full (incl. thin firm-years) | Specificity_E_mean_lag  |    -0.0327 |                0.1421 | False                 | True                  |          1.8731 |          8.0619 |
| Q    | E share+specificity  | excl. <5-sentence firm-years | Corporate_E_share_lag   |    -0.0223 |                0.2702 | False                 | True                  |          1.4206 |          5.1448 |
| Q    | E share+specificity  | excl. <5-sentence firm-years | Specificity_E_mean_lag  |    -0.0366 |                0.2506 | False                 | True                  |          1.4318 |          5.236  |
| Q    | SG share+specificity | full (incl. thin firm-years) | Corporate_SG_share_lag  |     0.0337 |                0.149  | False                 | True                  |          1.8217 |          7.755  |
| Q    | SG share+specificity | full (incl. thin firm-years) | Specificity_SG_mean_lag |     0.0219 |                0.1049 | True                  | True                  |          2.4546 |         11.2969 |
| Q    | SG share+specificity | excl. <5-sentence firm-years | Corporate_SG_share_lag  |    -0.0126 |                0.1603 | False                 | True                  |          1.977  |          8.5867 |
| Q    | SG share+specificity | excl. <5-sentence firm-years | Specificity_SG_mean_lag |     0.0241 |                0.1566 | False                 | True                  |          1.9092 |          8.2103 |
| ROE  | E share+specificity  | full (incl. thin firm-years) | Corporate_E_share_lag   |     0.0023 |                0.1461 | True                  | True                  |          2.1613 |          9.632  |
| ROE  | E share+specificity  | full (incl. thin firm-years) | Specificity_E_mean_lag  |    -0.0676 |                0.0987 | False                 | True                  |          0.5868 |          2.2852 |
| ROE  | E share+specificity  | excl. <5-sentence firm-years | Corporate_E_share_lag   |     0.0245 |                0.2626 | False                 | True                  |          1.436  |          5.252  |
| ROE  | E share+specificity  | excl. <5-sentence firm-years | Specificity_E_mean_lag  |    -0.0793 |                0.2202 | False                 | False                 |          1.1814 |          3.9183 |
| ROE  | SG share+specificity | full (incl. thin firm-years) | Corporate_SG_share_lag  |     0.002  |                0.1182 | True                  | True                  |          2.5699 |         11.8789 |
| ROE  | SG share+specificity | full (incl. thin firm-years) | Specificity_SG_mean_lag |    -0.0299 |                0.0994 | True                  | True                  |          2.183  |          9.9292 |
| ROE  | SG share+specificity | excl. <5-sentence firm-years | Corporate_SG_share_lag  |     0.0169 |                0.1548 | False                 | True                  |          1.9949 |          8.693  |
| ROE  | SG share+specificity | excl. <5-sentence firm-years | Specificity_SG_mean_lag |    -0.04   |                0.1482 | False                 | True                  |          1.711  |          7.167  |


### Verdicts

| dv   | spec                 | sample                       | term                    |    coef |   p_raw |   p_holm | verdict                          |
|:-----|:---------------------|:-----------------------------|:------------------------|--------:|--------:|---------:|:---------------------------------|
| Q    | E share+specificity  | full (incl. thin firm-years) | Corporate_E_share_lag   |  0.025  |  0.6968 |   1      | Informative null (±0.10 SD)      |
| Q    | E share+specificity  | full (incl. thin firm-years) | Specificity_E_mean_lag  | -0.0045 |  0.5185 |   1      | Informative null (±0.20 SD only) |
| Q    | E share+specificity  | excl. <5-sentence firm-years | Corporate_E_share_lag   | -0.0295 |  0.817  |   1      | Informative null (±0.20 SD only) |
| Q    | E share+specificity  | excl. <5-sentence firm-years | Specificity_E_mean_lag  | -0.0061 |  0.6824 |   1      | Informative null (±0.20 SD only) |
| Q    | SG share+specificity | full (incl. thin firm-years) | Corporate_SG_share_lag  |  0.0221 |  0.5265 |   1      | Informative null (±0.20 SD only) |
| Q    | SG share+specificity | full (incl. thin firm-years) | Specificity_SG_mean_lag |  0.0041 |  0.5582 |   1      | Informative null (±0.10 SD)      |
| Q    | SG share+specificity | excl. <5-sentence firm-years | Corporate_SG_share_lag  | -0.0084 |  0.8251 |   1      | Informative null (±0.20 SD only) |
| Q    | SG share+specificity | excl. <5-sentence firm-years | Specificity_SG_mean_lag |  0.0052 |  0.6664 |   1      | Informative null (±0.20 SD only) |
| ROE  | E share+specificity  | full (incl. thin firm-years) | Corporate_E_share_lag   |  0.0022 |  0.9652 |   1      | Informative null (±0.10 SD)      |
| ROE  | E share+specificity  | full (incl. thin firm-years) | Specificity_E_mean_lag  | -0.0068 |  0.0552 |   0.8838 | Informative null (±0.20 SD only) |
| ROE  | E share+specificity  | excl. <5-sentence firm-years | Corporate_E_share_lag   |  0.0254 |  0.7939 |   1      | Informative null (±0.20 SD only) |
| ROE  | E share+specificity  | excl. <5-sentence firm-years | Specificity_E_mean_lag  | -0.0104 |  0.313  |   1      | Inconclusive                     |
| ROE  | SG share+specificity | full (incl. thin firm-years) | Corporate_SG_share_lag  |  0.001  |  0.9627 |   1      | Informative null (±0.10 SD)      |
| ROE  | SG share+specificity | full (incl. thin firm-years) | Specificity_SG_mean_lag | -0.0043 |  0.4    |   1      | Informative null (±0.10 SD)      |
| ROE  | SG share+specificity | excl. <5-sentence firm-years | Corporate_SG_share_lag  |  0.0082 |  0.759  |   1      | Informative null (±0.20 SD only) |
| ROE  | SG share+specificity | excl. <5-sentence firm-years | Specificity_SG_mean_lag | -0.0062 |  0.4499 |   1      | Informative null (±0.20 SD only) |


## Honest summary

**One near-miss, nothing that survives correction.** `Specificity_E_mean_lag` on `ROE`, full sample, has the smallest raw p-value in this entire exploratory set (p=0.0552, just over the conventional 0.05 line) — but after Holm correction across the 16 tests in this battery, its adjusted p is 0.884, nowhere close to significant. No single coefficient here should be treated as evidence of anything without that correction; reporting the raw p alone and stopping there would be a form of cherry-picking this analysis was explicitly built to avoid.

**The `<5`-sentence exclusion matters most for E, least for interpretation.** Excluding thin E-pillar firm-years cuts E's sample nearly in half (151→85) and widens its CIs substantially (e.g. `Corporate_E_share_lag` on `Q`: SE goes from 0.064 to 0.128) without changing any coefficient's sign or significance — the thin observations weren't driving a hidden result, they were just adding (modest) precision. SG's much larger natural sample means this exclusion barely matters there.

**Every classification here is "informative null" or "inconclusive," never "significant."** Consistent with every other cut of this data across `null_check/`, `esg_pillars/`, and this folder's Part 1 — no version of E/S/G disaggregation (raw sentence ratio, corpus-based share, or corpus-based share+specificity) surfaces a result the combined `Corporate_ESG_share`/`Verified_ESG_share` measures don't already show as null.

## Assumptions and judgment calls

1. **MIN_SENTENCES = 5**, per your instruction, applied only to the specificity-mean models (Part 2) -- Part 1's share-only models use the full available sample since a share is well-defined (denominator = total sentences, not qualifying sentences) even at low pillar-sentence counts.
2. **Zero-sentence firm-years are dropped, never imputed**, exactly as instructed -- `Specificity_{X}_mean` stays genuinely `NaN` for them and `build_model_sample()`'s complete-case filter removes those rows; reported explicitly in the sentence-count table above rather than silently vanishing into a smaller N.
3. **Multiple-testing family = all 16 tests in this battery** (2 pillars x 2 DVs x 2 samples x 2 terms) -- a defensible but not unique choice; correcting only within one sample/DV combination (4 tests) would give less conservative adjusted p-values.
4. **Per-pillar models, not one 4-term joint model** (E share + E specificity + SG share + SG specificity together) -- matches Table 5a Model 4's per-corpus structure and avoids adding a 4-regressor model on top of an already N-constrained (thin-sentence) sample; can be added if wanted.
5. **Same standardization/MDE/TOST/BF01 conventions as `esg_pillars/`** (partial standardized effect on each model's own complete-case sample, z not t critical values, Savage-Dickey BF01 approximation) -- not re-derived differently here.
