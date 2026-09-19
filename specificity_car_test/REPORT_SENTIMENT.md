# Specificity, sentiment, and abnormal returns

## Result

The extended analysis finds no statistically detectable association between CEO-letter specificity, overall FinBERT sentiment, ESG-specific sentiment, or their interactions and either signed or absolute three-day CAR under the preferred firm and fiscal-year fixed-effects specification with HC3 inference.

For signed `CAR[-1,+1]`, the joint specificity + overall-sentiment model gives specificity β = −0.0199 (SE 0.0207, p = 0.341) and overall sentiment β = −0.0086 (SE 0.0468, p = 0.855). Replacing overall sentiment with ESG sentiment gives specificity β = −0.0243 (SE 0.0215, p = 0.267) and ESG sentiment β = 0.0205 (SE 0.0244, p = 0.406).

For `|CAR[-1,+1]|`, the corresponding joint coefficients are also nonsignificant: specificity β = 0.0079 (p = 0.607) and overall sentiment β = 0.0202 (p = 0.531); with ESG sentiment, specificity β = 0.0107 (p = 0.505) and ESG sentiment β = −0.0002 (p = 0.988).

Neither interaction is supported. For signed CAR, specificity × overall sentiment has β = −0.0971 (p = 0.412), and specificity × ESG sentiment has β = −0.0296 (p = 0.692). For absolute CAR, their p-values are 0.763 and 0.884. Thus the data do not show that specificity systematically amplifies positive or negative tone.

Alternative windows `[−2,+2]`, `[0,+1]`, `[0,+5]`, and `[0,+20]` do not produce HC3-significant coefficients in the preferred joint or interaction models. A few six-cluster wild-bootstrap p-values are small while HC3 results are not, including a zero p-value for one `[0,+5]` coefficient. With only six within-firm clusters and 64 possible Rademacher sign patterns, these coarse and conflicting bootstrap results are treated as unstable sensitivity signals, not confirmatory evidence.

## Merge and variables

The event panel has 84 unique insurer-years. Sixty-five match the sentence-level narrative outputs; 62 contain primary specificity, overall sentiment, and ESG sentiment. The pooled regressions use N=62 across seven firms. Fixed-effects regressions remove MetLife's single eligible observation and use N=61 across six firms.

Primary `Specificity_mean` is the validated paper measure: mean tier among Corporate ESG sentences with tier ≥1. Tier 0 represents a false positive, so it is not relabeled “generic” or included in the primary mean. `Specificity_all_flagged_mean` records the proposed 0–3 mean over every classified ESG-flagged sentence as a robustness variable. The re-aggregated primary measure agrees with the prior regression panel to numerical precision.

`Overall_sentiment` equals the share of all CEO-letter sentences labeled positive minus the share labeled negative. `ESG_sentiment` uses the same formula among sentences with `any_esg_flag == 1`. Both use the existing FinBERT discrete labels.

The longer `[0,+5]` and `[0,+20]` CARs use the saved event-study alpha and beta and the same archived adjusted stock and URTH prices. They contain 6 and 21 observed sessions, including the adjusted event day.

## Influence and data quality

AIG FY2019 remains the most influential primary-window observation. Excluding the most influential observation or all March–April 2020 publication events materially attenuates several estimates but does not create HC3-significant primary associations.

The specificity source contains 62 exact duplicate rows and the FinBERT source contains 125 exact duplicate rows. These duplicates were already present in the inputs underlying the validated paper measure. They were retained for comparability and explicitly reported rather than silently deleted. A future data-cleaning revision should investigate their origin and rerun the analysis after a documented deduplication rule.

The conclusions are associational. Annual-report publication events may bundle other value-relevant information, and publication dates have not been independently authenticated against original filings and timestamps.
