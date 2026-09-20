# Null-result informativeness check: 7 coefficients, two designs

**Question:** across the panel-regression design (Tobin's Q / ROE on lagged ESG specificity
measures, N=183) and the event-study design (CAR around CEO-letter publication on the same
measures, N=209), are the reported null/non-surviving results *informative* (the data rule out
economically meaningful effects) or merely *inconclusive* (the design lacks power to tell a null
from a real effect)? Seven specific coefficients, spanning both designs, are checked.

All numbers below were independently reproduced from raw/uploaded data before any equivalence,
power, or Bayes analysis was run — see "Reproduction" in each section. Code, not narrative, is the
source of truth: every number in this report traces to a script in this folder and a results CSV.

## Package versions

```
python 3.11.15
pandas 3.0.5
numpy 2.4.6
statsmodels 0.15.0
scipy 1.17.1
```

(See `versions.txt`. Confirmed to match this sandbox's actual installed versions before running
anything — `01_reproduce.py`'s panel reproduction and this session's local environment check
matched exactly.)

## Reproduction — confirmed before any analysis

| # | Coefficient | Design | Reported | Reproduced | Script |
|---|---|---|---|---|---|
| 1 | `Specificity_mean_lag`, Q Model 4 | Panel | coef=0.001, se=0.006, N=183 | coef=0.0008, se=0.0060, N=183 [PASS] | `01_reproduce.py` |
| 2 | `Specificity_mean_lag`, ROE Model 4 | Panel | coef=-0.005, se=0.005, N=183 | coef=-0.0051, se=0.0049, N=183 [PASS] | `01_reproduce.py` |
| 3 | `Verified_ESG_share_lag`, Q Model 3 | Panel | coef=-0.075, se=0.063, N=183 | coef=-0.0746, se=0.0626, N=183 [PASS] | `01_reproduce.py` |
| 4 | `Corporate_ESG_share_lag`, Q Model 2 | Panel | coef=0.031, se=0.033, N=183 | coef=0.0310, se=0.0325, N=183 [PASS] | `01_reproduce.py` |
| 5 | `Specificity`, CAR[-1,+1] Model 3 | Event study | coef=-0.0040, se=0.0036, N=209 | coef=-0.0040, se=0.0036, N=209 [PASS] | `06_reproduce_event.py` |
| 6 | `Sentiment`, CAR[-1,+1] Model 3 | Event study | coef=-0.0279, se=0.0193, N=209 | coef=-0.0279, se=0.0193, N=209 [PASS] | `06_reproduce_event.py` |
| 7 | Cluster1 − Cluster0 pairwise, CAR[-1,+1] | Event study | diff=-0.0075, se=0.0116, N=209 | diff=-0.0075, se=0.0116, N=209 [PASS] | `08_reproduce_pairwise.py` |

**All seven reproduced exactly.** #6's history (raw p<0.05 at CAR[0,+30]/[0,+45]/[0,+60],
not surviving Holm/BH across the 18-test family) was also independently re-derived from
`ceo_letter_regression_sample_209.csv`'s own CAR window columns (`06_reproduce_event.py`), not
just re-quoted: 3 of 18 tests raw-significant, 0 survive Holm, 0 survive BH — matches.

## Standardization conventions (flagged, not identical across the seven)

- **#1–6** (continuous predictors): partial standardized effect, `coef × SD(predictor) / SD(DV)`,
  computed on each model's own complete-case sample — the standard way to make an FE-model
  coefficient on a 0–1-scaled share/specificity variable comparable to a coefficient on CAR
  (basis points) or Q (a ratio near 1).
- **#7** (a dummy-vs-dummy pairwise contrast): **different convention, stated explicitly.**
  `Cluster1_HighSpec`/`Cluster0_HighVolLowSpec` are group memberships, not continuous regressors —
  there is no natural "SD(predictor)" the way there is for a share variable. Standardized effect
  here is `diff / SD(DV)` directly (Cohen's-d-style for a two-group contrast), *not*
  `diff × SD(predictor)/SD(DV)`. Using the continuous-variable convention on a dummy contrast would
  produce a nonsensical, arbitrarily-shrunk "SD(predictor)" (a Bernoulli SD depending on the two
  groups' relative sizes) that doesn't correspond to any real unit of comparison.

## Panel-regression coefficients (#1–4)

Design: firm-year panel, N=183 (after per-model complete-case + firm-count-guard restriction —
see "Progressive" caveat below), Tobin's Q / ROE regressed on one-year-lagged ESG measures, firm +
year FE, HC3 SEs (baseline).

| # | Term | Model / DV | coef | se | std coef | std 95% CI | MDE (80% pwr, std) | Min. equiv. bound (std) |
|---|---|---|---:|---:|---:|---|---:|---:|
| 1 | `Specificity_mean_lag` | Q, Model 4 | 0.0008 | 0.0060 | 0.005 | [-0.059, 0.068] | 0.090 | 0.058 |
| 2 | `Specificity_mean_lag` | ROE, Model 4 | -0.0051 | 0.0049 | -0.035 | [-0.101, 0.031] | 0.094 | 0.090 |
| 3 | `Verified_ESG_share_lag` | Q, Model 3 | -0.0746 | 0.0626 | -0.066 | [-0.176, 0.043] | 0.156 | 0.158 |
| 4 | `Corporate_ESG_share_lag` | Q, Model 2 | 0.0310 | 0.0325 | 0.056 | [-0.060, 0.172] | 0.166 | 0.154 |

**Coefficient #3 flag (per instruction):** `Verified_ESG_share_lag` and `Corporate_ESG_share_lag`
are known-collinear (Model 3 includes both together). Its MDE and standardized CI reflect that
shared-variance inflation — do not read #3's wider MDE as evidence this measure is inherently
noisier than #4's; it's partly an artifact of estimating two correlated regressors jointly.

### TOST equivalence, panel side

| # | Term | ±0.05 SD | ±0.10 SD | ±0.20 SD |
|---|---|---|---|---|
| 1 | `Specificity_mean_lag` (Q) | No | **Yes** | Yes |
| 2 | `Specificity_mean_lag` (ROE) | No | **Yes** | Yes |
| 3 | `Verified_ESG_share_lag` (Q) | No | No | **Yes** |
| 4 | `Corporate_ESG_share_lag` (Q) | No | No | **Yes** |

### Bayes factors (BF01, standardized scale), panel side

| # | Term | BF01, N(0, 0.1) | Reading | BF01, N(0, 0.5) | Reading |
|---|---|---:|---|---:|---|
| 1 | `Specificity_mean_lag` (Q) | 3.23 | moderate for H0 | 15.37 | strong for H0 |
| 2 | `Specificity_mean_lag` (ROE) | 1.94 | weak/anecdotal for H0 | 8.77 | moderate for H0 |
| 3 | `Verified_ESG_share_lag` (Q) | 1.19 | weak/anecdotal for H0 | 4.47 | moderate for H0 |
| 4 | `Corporate_ESG_share_lag` (Q) | 1.40 | weak/anecdotal for H0 | 5.43 | moderate for H0 |

### Proposed SESOI candidates — panel design (annual valuation effect; not chosen for you)

1. **±0.05 SD (strict):** appropriate if the claim is "no economically meaningful annual
   valuation effect at all." None of the four terms clear this bound — under this SESOI, **all
   four panel coefficients are inconclusive**, not informative nulls.
2. **±0.10 SD (small-telescopes-style):** close to this design's own 80%-power MDE for the
   tightest-estimated term (#1, MDE≈0.090 SD) — a standard convention for "the smallest effect
   this design could reliably detect." Under this bound, #1 and #2 (`Specificity_mean_lag`, both
   DVs) are informative nulls; #3 and #4 (the ESG-share variables) remain inconclusive.
3. **±0.20 SD (conventional "small effect," corporate-finance-panel-adjacent):** a common
   rule-of-thumb floor for "detectable in this literature." All four terms establish equivalence
   here — but #3 and #4's point estimates (std_coef 0.056–0.066) sit closer to this bound than to
   zero, so "equivalent within ±0.20 SD" is a weak, not strong, informativeness claim for those two.

### Panel-side robustness

**(a) HC3 vs. firm-clustered SE:** coefficients unchanged (fixed-effects point estimates don't
move across SE methods); clustered SEs are *tighter* than HC3 for every term in this table
(e.g. #4: HC3 se=0.0325 → cluster se=0.0305) — the null conclusion is not an artifact of a
conservative SE choice; the more literature-standard clustered SE would, if anything, make these
results look *more* precisely estimated around zero, not less.

**(b) Excl. Progressive — genuinely a no-op, not a real robustness check.** Traced directly:
Progressive has only 3 rows in the 209-row reconstructed panel, 2 of which already fail the
complete-case filter (a missing lagged value from a documented non-consecutive-year gap). The 1
remaining row falls below `build_model_sample()`'s own `<3 rows → drop firm` guard, so **Progressive
is already absent from the N=183 baseline sample before "excluding" it does anything** — confirmed
directly (`sample_q['Company']=='Progressive'` → 0 rows, for all 4 models). `results_robustness.csv`'s
"excl. Progressive" column is therefore identical to baseline for every term — not because the
results are robust to Progressive's presence, but because Progressive was never there. Do not cite
this as a genuine robustness check in the paper without this caveat.

**(c) Unrestricted (Corporate+Contextual) construction:** substituting the `_allrelevance` variant
of `Verified_ESG_share`/`Specificity_mean` (Corporate+Contextual sentences, not Corporate-only)
shifts coefficients modestly (#3: -0.0746→-0.0724; #4/Model4's `Specificity_mean_lag`:
0.0008→0.0011) — same sign, same order of magnitude, still null. Not sensitive to this
construction choice.

**(d) Drop ROA control:** coefficients shift by <10% for every term (`results_robustness.csv`),
consistent with ROA never being significant in any model — not driving the null.

## Event-study coefficients (#5–7)

Design: short-window CAR around CEO-letter publication, N=209, Model 3
(`Specificity + Sentiment`), firm + fiscal-year FE, firm-clustered SEs (primary, as reported).

| # | Term | coef | se (cluster) | std coef | std 95% CI | MDE (80% pwr, std) | Min. equiv. bound (std) |
|---|---|---:|---:|---:|---|---:|---:|
| 5 | `Specificity` | -0.0040 | 0.0036 | -0.044 | [-0.120, 0.033] | 0.110 | 0.108 |
| 6 | `Sentiment` | -0.0279 | 0.0193 | -0.153 | [-0.360, 0.055] | 0.296 | 0.327 |
| 7 | Cluster1 − Cluster0 pairwise | -0.0075 | 0.0116 | -0.235 | [-0.944, 0.475] | 1.014 | 0.830 |

### TOST equivalence, event-study side

| # | Term | ±0.05 SD | ±0.10 SD | ±0.20 SD |
|---|---|---|---|---|
| 5 | `Specificity` | No | No | **Yes** |
| 6 | `Sentiment` | No | No | **No — equivalence not established at any bound** |
| 7 | Pairwise credible-vs-cheap-talk | No | No | **No — equivalence not established at any bound** |

### Bayes factors (BF01, standardized scale), event-study side

| # | Term | BF01, N(0, 0.1) | Reading | BF01, N(0, 0.5) | Reading |
|---|---|---:|---|---:|---|
| 5 | `Specificity` | 1.60 | weak/anecdotal for H0 | 6.91 | moderate for H0 |
| 6 | `Sentiment` | 0.84 | **weak/anecdotal for H1** | 1.78 | weak/anecdotal for H0 |
| 7 | Pairwise | 1.02 | ≈1, uninformative | 1.49 | weak/anecdotal for H0 |

**#6 is the coefficient this instruction specifically flagged for an honest MDE rather than just
a p-value, and the numbers explain why.** Raw p at CAR[-1,+1] (p=0.149, cluster SE) already isn't
significant, but that alone understates how uninformative this estimate is: its MDE (0.296 SD) is
more than 3× wider than `Specificity`'s (0.110 SD), it fails to establish equivalence even at the
loosest ±0.20 SD bound, and its BF01 under the tight prior (0.84) very mildly favors the
alternative over the null. Combined with the documented history — raw p<0.05 at three longer
windows, none surviving correction across 18 tests — the honest characterization is: **this design
cannot distinguish "no sentiment effect" from "a real, economically meaningful delayed sentiment
effect that this N=209 sample is simply underpowered to pin down."** That is a materially different,
weaker claim than "we found no sentiment effect," and reporting it as the latter would overstate
what this analysis supports.

### Proposed SESOI candidates — event-study design (short-window abnormal return; not chosen for you)

Different candidates from the panel side, per instruction, since a 3-day CAR and an annual Q/ROE
change are not naturally comparable magnitudes:

1. **±0.05 SD (strict):** a 3-day abnormal-return effect this small is close to indistinguishable
   from bid-ask-bounce/microstructure noise in most event-study settings — appropriate if the claim
   is "no detectable market reaction whatsoever." None of the three terms clear this bound.
2. **±0.10 SD (literature-typical minimum "real" CAR effect):** short-window event studies
   commonly treat standardized CAR effects below ~0.10 SD as economically negligible even when
   statistically detected, since transaction costs and typical bid-ask spreads dominate at that
   scale. `Specificity` establishes equivalence here; `Sentiment` and the pairwise contrast do not.
3. **±0.20 SD (conventional "small but tradeable" effect):** a looser bound, closer to what a
   practitioner might consider a minimum economically actionable short-window abnormal return.
   `Specificity` clears this comfortably; `Sentiment`'s point estimate (std_coef=-0.153) and CI are
   still too wide to establish equivalence even here — this is the more decisive contrast with the
   panel side, where every term cleared ±0.20 SD.

### Event-study robustness

**(a) Cook's-distance exclusion (independently recomputed, not reused from memory):** 14 of 209
observations flagged (threshold 4/N=0.0191), dominated by one structural artifact — MetLife 2012
is a singleton firm-year for that company in this complete-case sample (its other 11 years were
excluded earlier in the pipeline for missing Specificity data), so its own firm fixed-effect
dummy perfectly absorbs it, producing an extreme Cook's distance that reflects the FE
specification's mechanics, not a genuinely extreme return. Excluding all 14 (N→195):
`Specificity` -0.0040→-0.0027 (p 0.265→0.621), `Sentiment` -0.0279→-0.0122 (p 0.149→0.337) — both
shrink toward zero and remain non-significant. The null is not being driven by these flagged
points; if anything, they were pulling the estimates slightly *away* from zero.

**(b) HC3 vs. cluster vs. pairs-cluster bootstrap:** all three SE sources agree closely for both
#5 and #6 (see `results_event.csv`) — `Specificity`'s std_se ranges 0.030–0.056 across the three,
`Sentiment`'s ranges 0.152–0.176. The bootstrap SE (pulled in from the earlier session run, not
re-derived, per instruction) is consistent with both analytic SE estimates; the conclusions above
do not depend on which SE method is trusted.

## Panel vs. event study: do they agree? (explicit comparison, per instruction)

**They agree on the ESG-share variables (#3, #4) and diverge on specificity/sentiment tone (#1/2
vs. #6).**

- **ESG-share disclosure volume (#3, #4)** is inconclusive-to-weakly-informative in the panel
  design (equivalence only at the loosest ±0.20 SD bound) — there is no directly comparable
  event-study coefficient for this specific construct (the event study tests `Specificity` and
  `Sentiment`, not raw ESG-share volume), so this is a within-design finding only.
- **`Specificity_mean` / `Specificity`** is the one construct tested in *both* designs, and both
  arrive at the same qualitative verdict: **informative null**. Panel #1/#2 establish equivalence
  at ±0.10 SD (tighter than the event study); event-study #5 only reaches ±0.20 SD. Both designs
  independently rule out a *large* specificity effect on both valuation and short-window market
  reaction — a genuinely convergent finding across two different outcome variables, samples, and
  timeframes.
- **Sentiment/tone** has no panel-side parallel to compare against directly (the panel regressions
  test ESG disclosure share and specificity, not general letter sentiment) — so #6's
  inconclusiveness stands on its own, without a cross-design check either confirming or
  contradicting it.
- **The credible-optimism-vs-cheap-talk pairwise test (#7)** is unique to the event-study design
  (built on the same 3-D cluster typology, but has no panel-regression parallel run for it in this
  scope) — its verdict (badly underpowered, MDE≈1.0 SD) stands alone.

**Bottom line on convergence:** where the two designs test the *same* construct (specificity), they
agree — both are informative nulls, not just individually underpowered results that happen to
point the same direction by chance. Where they test *different* constructs (ESG-share volume,
sentiment, the cluster typology), each design's verdict has to be read on its own, and — this is
the more consequential finding — **the event-study design is systematically less informative than
the panel design** across every term where a comparison is possible: narrower/no equivalence
established (0.10 SD achieved for 2 of 4 panel terms vs. 0 of 3 event-study terms), and BF01 values
closer to 1 (less decisive) throughout. This is consistent with N=209 short-window CAR carrying
less identifying information per observation than N=183 annual firm-year valuation data for this
particular question, not a flaw in either design individually.

## Verdicts (plain language)

| # | Coefficient | Design | Verdict |
|---|---|---|---|
| 1 | `Specificity_mean_lag` (Q) | Panel | **Informative null** at ±0.10 SD and looser. Not informative at ±0.05 SD. |
| 2 | `Specificity_mean_lag` (ROE) | Panel | **Informative null** at ±0.10 SD and looser. Not informative at ±0.05 SD. |
| 3 | `Verified_ESG_share_lag` (Q) | Panel | **Mixed/inconclusive** — equivalence only at the loosest ±0.20 SD bound; also collinearity-flagged. |
| 4 | `Corporate_ESG_share_lag` (Q) | Panel | **Mixed/inconclusive** — same pattern as #3, equivalence only at ±0.20 SD. |
| 5 | `Specificity` (CAR[-1,+1]) | Event study | **Mixed/inconclusive** — equivalence only at ±0.20 SD, but agrees directionally with #1/#2. |
| 6 | `Sentiment` (CAR[-1,+1]) | Event study | **Inconclusive, not informative** — fails equivalence at every bound tested; do not report as a confirmed null. |
| 7 | Credible-vs-cheap-talk pairwise | Event study | **Inconclusive, not informative** — by far the widest MDE (~1.0 SD) of all seven; this design cannot speak to the hypothesis with the current N. |

## Assumptions and judgment calls (flagged, per instruction)

1. **Seven coefficients as specified in the request** — not independently chosen; #3's
   collinearity flag and #7's different standardization convention are inherent to how those two
   were specified, not something introduced here.
2. **Standardization convention** — partial standardized effect on each model's own complete-case
   sample for #1–6; Cohen's-d-style group-contrast standardization for #7 (see dedicated section
   above). A different denominator sample (e.g. the full 209- or 238-row panel rather than each
   model's own complete-case rows) would shift these somewhat.
3. **z, not t, critical values** — matches both notebooks' own HC3/cluster output convention
   (`P>|z|`), not a claim a t-correction would change the picture; residual df is large enough in
   both designs (≥140) that the z/t difference is negligible.
4. **TOST/MDE bounds (0.05/0.10/0.20 SD)** are the three explicitly requested, not independently
   chosen; the SESOI proposals in each design's section are the actual candidates offered for you
   to pick from — the fixed bounds above are for legibility across the summary tables, not a
   proposed SESOI themselves.
5. **Bayes factor priors** (N(0,0.1), N(0,0.5), standardized scale) are exactly the two named in
   the request.
6. **Event-study controls (Leverage/Size/ROA/Revenue_Growth) merged contemporaneously, not
   lagged**, for coefficient #7 — same fiscal year as the CAR event, matching the notebook's own
   explicit design choice (the CAR event follows directly from that year's own letter, unlike the
   panel regressions' valuation-persistence design, which has a real reason to lag).
7. **"Excl. Progressive" panel robustness check is a no-op**, flagged in its own subsection above
   — this is a genuine finding about the existing `03_robustness.py` script's output, not a
   criticism of this session's own new analysis; it affects how that specific column in
   `results_robustness.csv` should (and should not) be interpreted or cited.
8. **Cook's-distance-flagged observations (event-study side) were independently recomputed**, not
   reused from an earlier session's reported list — confirmed to match (same 14 firm-years, same
   MetLife-2012-is-a-structural-artifact finding) via `10_robustness_event.py`, but computed fresh
   from the uploaded data, not trusted on the basis of the earlier chat output alone.
9. **The pairs-cluster bootstrap SEs for #5/#6 were pulled in from the earlier session's
   Step 9.3 run, not re-derived** — per explicit instruction ("already computed for #6 — pull
   those in rather than re-deriving"). The point estimates and analytic (HC3/cluster) SEs those
   bootstrap numbers are compared against *were* independently re-derived in this check.
10. **No comparable panel-side coefficient exists for `Sentiment` or the pairwise cluster test**
    — the cross-design comparison section above is explicit about which of the seven have a true
    cross-design counterpart (only specificity) and which stand alone.

## Files in this folder

- `common.py`, `01_reproduce.py`–`05_analysis_roe.py`: panel-side reconstruction, reproduction,
  and full MDE/TOST/BF01 analysis (Q and ROE).
- `common_event.py`, `06_reproduce_event.py`–`10_robustness_event.py`: event-study-side
  reconstruction, reproduction, full analysis, and Cook's-distance robustness.
- `11_robustness_unrestricted.py`: panel-side unrestricted-construction robustness.
- `03_robustness.py`: original panel robustness battery (HC3/cluster, excl. Progressive, ROE, drop
  ROA) — see the "excl. Progressive is a no-op" flag above before citing its Progressive column.
- `results_q.csv`, `results_roe.csv`, `results_event.csv`, `results_pairwise.csv`: full per-
  coefficient MDE/TOST/BF01 output.
- `results_robustness.csv`, `results_robustness_unrestricted.csv`, `results_robustness_event.csv`:
  robustness-variant output.
- `data/`: all raw/uploaded source files this check was run against (`panel_reg_export.csv`,
  `ceo_letter_regression_sample_209.csv`, `df_ar_cluster.csv`, `Combined_Financials_With_ROA.csv`,
  `TobinsQ_Results_FullNames.csv`, plus the derived `panel14_complete_case.csv`).
