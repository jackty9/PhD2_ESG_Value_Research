# P2 Revision Spec: ESG Specificity Classification & NFRD Content Analysis

**Purpose of this document:** brief for extending the existing ESG-BERT/FinBERT notebook. This is an
**additional stage inserted into the existing pipeline**, not a rebuild. Before writing any new code,
inventory the current notebook's structure (data loading, ESG-BERT/FinBERT calls, firm-year aggregation,
clustering, regression specs) and confirm where each new piece below plugs in.

---

## 0. Inputs already available

- Sentence-level corpus (`df_ar_ceo_sentences_finbert_senti.csv` or equivalent): `Company Name`, `Year`,
  `Sentence`, `prob_env`/`prob_soc`/`prob_gov`, `env_flag`/`soc_flag`/`gov_flag`, `any_esg_flag`,
  `sent_pos`/`sent_neg`/`sent_neu`, `sent_label`.
- 22 firms, 238 firm-years (2012–2023), matches the existing GFFC/P2 regression sample.
- Existing firm-year regression panel (Tobin's Q, controls: Size, ROA, Leverage, Revenue Growth,
  `SR_CEO_Letter` indicator) — **not in this notebook yet if separate; confirm join key is `Company Name`+`Year`.**
- Existing three-group NFRD regulatory classification (already used in P2 Table 7 / GFFC Appendix Table 7):
  - **EU-NFRD** (treated): Allianz, AXA, Assicurazioni Generali, Munich Re, Talanx
  - **Non-EU-European** (placebo): Chubb, Swiss Re, Zurich
  - **Non-European** (reference): all remaining firms
- Existing ESG narrative pattern clusters (Substantive / Symbolic / Minimal), built on
  (ESG ratio, ESG sentiment) via k-means.

---

## 1. Specificity classification (new stage)

Insert **after** ESG-BERT dimension flagging, **before** firm-year aggregation. Applies to every sentence
where `any_esg_flag == 1` (start with `env_flag == 1` only if time-constrained — this is where contamination
concentrates; extend to S/G once validated).

### 1.1 Taxonomy (four-way, ordinal)

| Tier | Label | Definition |
|---|---|---|
| 0 | **False positive** | ESG-BERT flagged it, but the sentence isn't genuinely ESG content. Test: delete the trigger word(s) — if no ESG meaning survives, it's a false positive. |
| 1 | **Vague / boilerplate** | Genuinely ESG in topic and intentional in framing, but no metric, named program, or verifiable claim. |
| 2 | **Named / unquantified** | Names a real program, partnership, framework, or event, but no number attached. |
| 3 | **Quantified** | Contains a specific, verifiable figure (%, currency, count, target year with a number attached). |

Separately, tag **Corporate vs. Contextual**:
- **Corporate**: the firm's own action, policy, commitment, or performance.
- **Contextual**: external conditions (industry statistics, macro/climate trends, catastrophe events) with no firm actor.

A sentence gets both tags independently — e.g., a Contextual sentence can still be Tier 3 (quantified industry
loss statistic) or Tier 1 (vague industry-outlook language).

### 1.2 Decision rules (resolved from manual coding — encode these explicitly in the classification prompt)

1. **Delete-the-trigger test** (relevance): remove the ESG buzzword (sustainable, stakeholder, society,
   resilient, green, environmentally friendly) from the sentence. If no ESG content remains, it's Tier 0.
   If real ESG content remains — even if vague — it is genuine (Tier 1+), not Tier 0.
2. **Author-intent test** (secondary, for ambiguous cases): is the ESG reading the one the author plainly
   intended, or is it a word colliding with a non-ESG meaning (e.g., "sustainable pricing" = underwriting
   cycle; "sustainability" defined in-sentence as shareholder value; "Renewal Agenda" ≠ renewable energy)?
3. **Generic stakeholder-inclusion rule**: sentences naming "society"/"stakeholders" as an intended
   beneficiary default to **Tier 1 genuine** (not Tier 0) **only when that beneficiary language is the
   sentence's main clause or point** (e.g., "we will contribute to realizing a safe and secure society").
   When it's appended to an unrelated business claim (e.g., a competitive-differentiation statement that
   happens to end in "creating value for all stakeholders"), it does not earn the default — apply the
   delete-the-trigger test normally instead. *(Revised after the pilot run's relevance recall came back
   at 0.20 on false positives — the original unconditional version of this rule was catching cases where
   the stakeholder language was incidental, not the sentence's point.)*
4. **Mixed-clause sentences**: this rule governs which tag (Corporate vs. Contextual) to assign when both
   a corporate clause and a contextual clause are present, not whether ESG content exists at all — code
   the Corporate/Contextual tag by the sentence's dominant/main clause, not a subordinate one (e.g., a
   contextual disaster mention attached to a corporate metric, like the ESR ~200% example — tag by the
   metric's actor). **If the sentence's dominant clause is NOT itself an ESG claim** (e.g., a plain
   financial-performance statement) **and the only ESG-relevant content is in a subordinate clause** (e.g.,
   a named or quantified catastrophe mention), that subordinate content should still be scored normally
   as Contextual — do not default to Tier 0 just because the grammatically dominant clause is non-ESG. A
   quantified contextual figure (e.g., a named catastrophe's dollar loss) is still Tier 3 even when it
   sits in a subordinate clause of an otherwise non-ESG sentence. This applies even when the contextual
   ESG content is unnamed/generic (e.g., "several natural disasters" with no event name) or appears
   alongside other unrelated subordinate clauses in a list — the presence of **any** genuine contextual
   ESG content, however minor or generic relative to the dominant clause, means the sentence should be
   tagged Contextual with the appropriate (likely low) specificity tier, not defaulted to Tier 0. *(Revised
   twice now. First pass after the pilot run — three sentences like "we posted a good profit despite
   Typhoon Jebi" and one quantified case ("$175 million related to estimated hurricane claims") were being
   zeroed to Tier 0 because the profit/earnings clause was treated as erasing the genuine catastrophe
   content in the subordinate clause, contradicting the axis-independence principle in Section 1.1 that a
   Contextual sentence can be Tier 3. Second pass — a spot-check found the first-pass fix only reliably
   triggered when the contextual content was strong/specific (a named storm alone, a dollar figure); two
   sentences with weaker signal — "several massive natural disasters" (no event name) and "Storm Sandy"
   buried in a list alongside two unrelated non-ESG clauses — were still defaulting to Tier 0, so the rule
   now explicitly covers generic/unnamed and crowded-out contextual content too.)*
5. **Proper-noun collisions**: applies **only** to literal proper-noun collisions — an ESG word embedded
   in a company/subsidiary/product name (e.g., "Power Sustainable" as a division name). It does **not**
   apply to generic reputation claims, values statements, or vague self-praise — those should go through
   the normal delete-the-trigger and specificity-tier logic instead, not get defaulted to Tier 0 via this
   rule. Where a genuine proper-noun collision is present: Tier 0 regardless of surrounding content,
   unless real ESG content independently survives the delete-test elsewhere in the sentence. The presence
   of an unrelated quantified figure elsewhere in the sentence does **not** rescue a proper-noun collision
   from Tier 0 — the number must itself quantify a genuine ESG claim, not just any business metric (e.g.,
   fundraising totals, revenue figures) that happens to share the sentence. *(Revised twice now — first
   for the Power Sustainable $4.2B fundraising figure, then to stop the rule being invoked as a generic
   "vague content" excuse on sentences with no actual proper-noun collision, e.g. a garbled corporate-
   citizenship sentence and an "international ESG ratings" reputation claim.)*
6. **Named business lines** count as Tier 2 even without a formal "initiative" or "alliance" label — e.g.
   a sentence naming specific underwriting or investment activity (carbon capture, hydrogen, EV charging,
   renewable energy infrastructure) is Tier 2 even with no explicit program name, because the activity
   itself is specific and checkable. This came out of manually reviewing the tier ground truth — the
   original named-pattern rules only caught formally-branded programs and missed named-but-unbranded
   business activity.

**Note on the excluded temporal-ambiguity rule**: an earlier draft of this list included a placeholder
rule 4, "Present-perfect temporal ambiguity — not in scope for this round." It has been removed rather
than renumbered around, since it was never applied (temporal orientation is out of scope for this round)
and its presence only offset the numbering of the rules that are actually in use.

### 1.3 Few-shot examples for the classification prompt

Use these as few-shot examples — they are manually validated (κ = 0.645 on relevance, n=30 blind check;
κ = 1.00 on a second 20-sentence blind check across all three axes).

**Tier 0 — False positive**
- *"Travelers' sustainability – our ability to maintain our industry-leading position and maximize
  shareholder value over the long term – depends on the successful execution of our financial strategy."*
  → FP: sentence defines "sustainability" as shareholder value in its own text.
- *"Our industry now finds itself at a turning point, as we are experiencing a return to an environment
  characterised by more sustainable pricing."* → FP: "sustainable pricing" is underwriting-cycle vocabulary.
- *"As we implement our Renewal Agenda 2.0, we will move up a gear."* → FP: likely lexical near-miss on
  "renewable"; no environmental content.
- *"Both Sagard and Power Sustainable launched new investment vehicles and accelerated external funding,
  raising a total of $4.2 billion."* → FP: "Sustainable" is part of a subsidiary's proper name; the
  fundraising total is an unrelated business metric that does not rescue it (Rule 5).
- *"We will speed up the integration of service platforms, and shape the Company's operation and services
  to be more integrated, intelligent and ecological."* → FP: "ecological" means service-ecosystem
  integration here, not environment — no ESG content survives the delete-test.
- *"Imerys, a producer of mineral-based specialties for industry; Lafarge, which produces cement,
  aggregates and concrete; Total, in the oil, gas and alternative energy industry; SGS, engaged in
  testing, inspection and certification; Pernod Ricard, a leader in wines and spirits; and GDF Suez, a
  provider of electricity, natural gas, and energy and environmental services."* → FP: a portfolio-
  company/entity list; industry descriptors trigger the flag, not genuine ESG content.
- *"Depressed commodity prices China's slowing growth continues to impact all natural resource- and
  export-dependent countries around the world, particularly throughout Asia and Latin America."* → FP:
  macro/commodity commentary; "natural resource" is a sector term here, not environmental content.
- *"U.S. growth is on a more solid footing and more broad-based — construction, manufacturing
  (particularly auto), energy and housing — and there is less impact from recent government fiscal
  contraction."* → FP: macro-economic commentary listing sectors; "energy" is a sector category, not
  environmental content.
- *"To respond promptly to environmental flux, we ourselves have to become a resilient and sustainable
  group."* → FP: "environmental flux" means changing business conditions, not ecology; both trigger words
  carry the entire ESG signal with nothing surviving the delete-test.
- *"Jay S. Fishman Chairman and Chief Executive Officer INDUSTRY UNDERWRITING ENVIRONMENT — IMPACT OF
  CATASTROPHES Metric 2012 2011 2010 2009 2008 Earned Premiums $22,357 $22,090 $21,432 $21,418 $21,579
  Total Revenues $25,740 $25,446 $25,112 $24,680 $24,477 ... Return On Equity 9.8% 5.7% 12.1% 13.5% 11.4%
  ... HISTORICAL MILESTONES AND COMPANY HERITAGE: 1853:"* → FP: a financial-results table under a header
  using "ENVIRONMENT" in its business sense; dense numeric content does not make it ESG disclosure — treat
  tables/headers like this as Tier 0 regardless of number density.

*(The six examples above were added after the first pilot run's relevance recall came back at 0.20 on
false positives; they cover patterns the original four-example set did not — see the classification
prompt in `PHDp2_CEOLetters_AnnualReports_TextAnalysis.ipynb` Section 6.10.2 for the same set with the
model's own short_reason field alongside.)*

**Tier 1 — Vague / boilerplate (Corporate)**
- *"...enhanced social well-being in serving the overall national development plan... advanced steady
  operations to support modernization of corporate governance."* → real topics, no metric or verifiable outcome.
- *"Chubb has outperformed while navigating a world of constant and challenging risk events, some due to
  nature, others man-made."* → real events named, but no benchmark/figure for the company claim itself.
- *"Being people-oriented, we embarked on building a reliable team."* → near-content-free.

**Tier 2 — Named / unquantified (Corporate)**
- *"We are working to increase flood resilience in at-risk communities through our innovative flood
  resilience alliance."* → named program, no metric.
- *"The Group has joined the Net-Zero Insurance Alliance (NZIA)... as the first Japanese member."* → named
  external initiative, verifiable membership, no quantification.

**Tier 3 — Quantified (Corporate)**
- *"At the end of 2019, we achieved the first of two goals by reducing emissions 22% from a 2016 baseline,
  exceeding our goal of 20% by 2025."* → named baseline, target, and outcome.
- *"...by quadrupling our green investments targets, by divesting $3 billion of our assets from coal
  producers, and by committing to stop investing in new coal and oil sands producers..."* → multiple
  quantified, named commitments.

**Contextual — concrete**
- *"170 storms of $1 billion or greater over the last 10 years, worth $1.2T of economic loss."* (Allstate)
- *"$13 billion of insured losses for European flooding was the costliest disaster on record."* (AIG)

**Contextual — vague**
- *"The industry's social status is expected to be further elevated, the market further expanded..."*
- *"With climate change very likely linked to rising sea levels and increased rainfall, the impact from
  flooding... could be even more severe in the future."*

**Boundary / mixed (code by dominant clause)**
- *"...the insurance market as a whole experienced historic high insurance claims due to hurricanes...
  our Group's ESR maintained about 200%..."* → dominant clause is the firm's own metric → Corporate, Tier 3.

### 1.4 Classifier implementation

- Use GPT (per the original validation architecture: ESG-BERT↔GPT already validated in GFFC at
  κ=0.458/77.4% agreement; this is a second, distinct validation — Human↔GPT on the new tags).
- Output per sentence: `corporate_relevance` (Corporate/Contextual), `specificity_tier` (0–3).
- **Pilot first**: run on the ~100 sentences already hand-coded across the prior audit rounds
  (available in the audit CSVs) as a sanity check before scaling.
- **Validation**: stratified random sample of 300–500 ESG-flagged sentences, independently hand-coded
  (blind — no model output visible during coding), compared to GPT output via accuracy, Cohen's κ, macro-F1.
  Report per-axis κ (relevance, specificity), not just an aggregate.
- Scope: E-flagged sentences first (highest documented contamination — 30% contextual, 14% false-positive
  in prior random-sample audits); extend to S/G only after E is validated and time permits.

---

## 2. Firm-year aggregation (new variables)

For each firm-year, using `n_sentences` = total CEO-letter sentences:

```
ESGProminence          = N(any_esg_flag==1) / n_sentences                         [existing — keep as-is]
CorporateESGShare      = N(corporate_relevance=="Corporate", tier>0) / n_sentences
VerifiedESGShare       = N(tier in {2,3}) / n_sentences
SpecificityMean        = mean(tier) over genuine ESG sentences (tier>0)
ESGSentiment_All       = (PositiveShare - NegativeShare) over all any_esg_flag==1 sentences   [existing]
ESGSentiment_Corporate = (PositiveShare - NegativeShare) over Corporate, tier>0 sentences     [new]
```

Join to the existing regression panel on `Company Name` + `Year`. Apply the same one-year lag convention
already used for `ESG_Narrative*` terms in the valuation models.

---

## 3. Re-cluster narrative patterns (new)

Re-run the existing k-means clustering (currently 2-D: ESG ratio × sentiment) as a 3-D version adding
`SpecificityMean`. Report as a companion table to the existing Table 4:

- Old cluster label → New cluster label, firm-year counts (crosstab).
- Characterize new clusters on all three dimensions (ratio, sentiment, specificity mean), same format as
  the existing pattern-description table.

*Prior heuristic pilot (regex-based, not GPT-validated) found the old "high ratio/high sentiment"
("substantive"-like) cluster had the **lowest** mean specificity of the three groups once specificity was
added, while the old "moderate ratio/very high sentiment" ("symbolic"-like) cluster had the **highest**.
Treat this as a directional prior to check against, not a result to reproduce — it was built on a crude
regex heuristic, not the validated classifier.*

---

## 4. Regression specifications

### 4.1 Nested valuation sequence (new headline table)

Run as three nested models, same panel/FE structure as existing Table 5:

```
Q_it = β1 ESGProminence_it-1        + Controls_it-1 + FirmFE + YearFE + ε   [replicates existing baseline]
Q_it = β1 CorporateESGShare_it-1    + Controls_it-1 + FirmFE + YearFE + ε
Q_it = β1 CorporateESGShare_it-1 + β2 VerifiedESGShare_it-1 + Controls_it-1 + FirmFE + YearFE + ε   [H2 — headline test]
```

Report side by side in one table so the reader sees exactly what changes at each step.

### 4.2 NFRD content-shift interaction (new — companion to existing Table 7)

Same three-group structure and specification pattern as the existing NFRD valuation-interaction table
(firm FE + explicit post-2017 indicator in place of year FE, per the existing collinearity fix):

```
VerifiedESGShare_it = β1(EU-NFRD_i × Post2017_t) + β2(Non-EU-European_i × Post2017_t)
                       + Controls_it + FirmFE + Post2017_t + ε
```

Run three times — dependent variable = `VerifiedESGShare`, then `CorporateESGShare`, then
`ESGProminence` (raw) — to show whether NFRD shifted disclosure *quality*, separate from the existing
result that it didn't shift valuation. Use identical group definitions and estimator (cluster-robust SEs
by firm) as the existing Table 7 for direct comparability.

**Caution to carry into the writeup:** EU-NFRD treated group is only 5 firms, placebo group only 3 —
same small-sample caveat already noted for the valuation version of this test. Pair the regression with
a descriptive plot (VerifiedESGShare by group, pre/post 2017) rather than relying on the coefficient alone.

---

## 5. Sequencing

1. Inventory existing notebook structure; confirm join keys and lag conventions before adding anything.
2. Implement Section 1 (classification) on the ~100 already-hand-coded sentences as a pilot check.
3. Scale to full E-flagged corpus; run 300–500 sentence validation (Section 1.4).
4. Compute Section 2 firm-year variables.
5. Re-run Section 3 clustering.
6. Run Section 4.1 nested valuation sequence — requires the existing Tobin's Q/controls panel.
7. Run Section 4.2 NFRD content-shift interaction.
8. Only after 6–7: revisit whether Market-to-Book/ROE robustness (separate, lower-priority workstream) is
   still worth adding.
