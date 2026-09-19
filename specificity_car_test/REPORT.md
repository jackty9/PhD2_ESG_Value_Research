# Specificity and abnormal equity returns — US insurers

This analysis uses existing outputs; it does not rerun the classifier or modify event-study construction.
The preferred specification was fixed before estimation: firm and fiscal-year fixed effects. Publication dates determine the COVID exclusion, while fiscal years define year fixed effects.

## Measurement and provenance

`Specificity_mean` is the mean ordinal specificity tier (1 vague, 2 named/unquantified, 3 quantified) among **Corporate** ESG sentences with tier >= 1, combining E and S/G classifications. Contextual sentences and tier-0 false positives are excluded. Undefined means remain missing. This is the existing measure, not the distinct `Verified_ESG_share` or the all-relevance measure.

Source: `specificity-pipeline-3d-regression` commit `2428ae52895ffba97d280f5d776c9e88b376bfc9`, `null_check/data/panel_reg_export.csv`. All 238 specificity values (including missingness), sentence counts, and corporate sentiment agree with the original Drive `df_ar_cluster.csv`. The latter also confirms narrative coverage, so missing years are not manufactured by this integration. No valuation-model lag is carried over: each letter's own fiscal year is joined to its publication event.

CAR source: the supplied Drive `ceo_letter_event_study_panel_US.csv`, snapshotted without alteration. Event-study code at commit `6693c1d60844c31c76dc6a129cb40863dc50b05c` sets `PRICE_END = "2024-07-31"`; every saved stock series and URTH ends 2024-07-30. Raw prices independently reproduce all 84 panels' alpha, beta, estimation counts and three CAR windows within 1e-10. This checks calculations against saved inputs, not the historical accuracy of publication dates or vendor prices.

## Merge and sample

Event rows: 84; specificity rows: 238; matches: 65; event-only rows: 19; specificity-only rows: 173 (outside the seven target firms); duplicate keys: 0. Three matched rows have undefined specificity. Pooled regression N = 62. All 84 events remain in the merged CSV with eligibility and exclusion reasons. Full unmatched keys and exclusions are saved separately.

MetLife contributes only FY2012 after missing specificity is excluded. This singleton is removed explicitly from firm-FE and firm/year-FE models: it contributes no within-firm slope information and has leverage one. Main FE N = 61 and **six**, not seven, firm clusters. Model-specific removals are saved.

Specificity mean = 1.439526, sample SD = 0.229556. Standardized versions use this fixed pooled-sample mean and SD (ddof=1), also in sensitivities. CARs are decimal returns; multiply coefficients by 100 for percentage points.

## Descriptive correlations

| outcome      | method   |   N |   correlation |   p_value_iid_descriptive |
|:-------------|:---------|----:|--------------:|--------------------------:|
| CAR[-1,+1]   | Pearson  |  62 |    -0.0561536 |                  0.664653 |
| CAR[-1,+1]   | Spearman |  62 |    -0.0998249 |                  0.44014  |
| |CAR|[-1,+1] | Pearson  |  62 |    -0.0201593 |                  0.876411 |
| |CAR|[-1,+1] | Spearman |  62 |    -0.085233  |                  0.51011  |
| CAR[-2,+2]   | Pearson  |  62 |    -0.125364  |                  0.331617 |
| CAR[-2,+2]   | Spearman |  62 |    -0.152799  |                  0.235766 |
| CAR[0,+1]    | Pearson  |  62 |    -0.136393  |                  0.290487 |
| CAR[0,+1]    | Spearman |  62 |    -0.132361  |                  0.305118 |

Correlation p-values assume independent observations and are descriptive, not panel-robust inference.

## Main models: HC3 inference

| outcome      | specification   |   N |   firms |   coefficient |       SE |   p_value |   CI95_low |   CI95_high |
|:-------------|:----------------|----:|--------:|--------------:|---------:|----------:|-----------:|------------:|
| CAR[-1,+1]   | pooled          |  62 |       7 |     -0.005757 | 0.008000 |  0.474527 |  -0.021759 |    0.010245 |
| CAR[-1,+1]   | firm_FE         |  61 |       6 |     -0.019322 | 0.015040 |  0.204372 |  -0.049475 |    0.010831 |
| CAR[-1,+1]   | firm_year_FE    |  61 |       6 |     -0.021111 | 0.021455 |  0.330649 |  -0.064380 |    0.022158 |
| |CAR|[-1,+1] | pooled          |  62 |       7 |     -0.001586 | 0.006678 |  0.813063 |  -0.014944 |    0.011772 |
| |CAR|[-1,+1] | firm_FE         |  61 |       6 |      0.008786 | 0.012064 |  0.469577 |  -0.015401 |    0.032973 |
| |CAR|[-1,+1] | firm_year_FE    |  61 |       6 |      0.010716 | 0.015366 |  0.489344 |  -0.020274 |    0.041705 |
| CAR[-2,+2]   | firm_year_FE    |  61 |       6 |     -0.034089 | 0.028431 |  0.237086 |  -0.091424 |    0.023247 |
| CAR[0,+1]    | firm_year_FE    |  61 |       6 |     -0.015991 | 0.015601 |  0.311083 |  -0.047452 |    0.015471 |

Neither primary outcome shows a statistically detectable association in pooled, firm-FE, or firm/year-FE models. This is not evidence that the true association is zero: the confidence intervals are wide.

In the preferred model, the signed-CAR coefficient is −0.021111 (HC3 SE 0.021455, p=0.330649, 95% CI [−0.064380, 0.022158]). The absolute-CAR coefficient is +0.010716 (SE 0.015366, p=0.489344, CI [−0.020274, 0.041705]). A one-SD specificity increase corresponds to approximately −0.485 percentage points of signed CAR and +0.246 percentage points of absolute CAR; both intervals include zero. Magnitude's coefficient changes from slightly negative pooled to positive with firm effects, so its sign is not stable across specifications.

Alternative signed windows remain negative and nonsignificant under HC3. The five-day window has a coarse wild-cluster p-value of 0.0625, contrasting with HC3 p=0.2371; report this sensitivity without treating it as reliable confirmatory evidence.

AIG FY2019 is the most influential observation in both preferred models. Removing it reduces the signed coefficient to −0.005833 (p=0.6789) and changes the magnitude coefficient to −0.000786 (p=0.9372). Removing all March–April 2020 events gives −0.009814 (p=0.4929) and −0.000239 (p=0.9813), respectively. The magnitude estimate's positive sign and much of its size depend on these observations; the absence of conventional statistical evidence persists. The signed estimate remains negative but is materially attenuated. Existing length/tone controls likewise do not produce a statistically detectable association.

## Prespecified sensitivities: HC3 inference

| outcome      | scenario                 |   N |   firms |   coefficient |       SE |   p_value |   CI95_low |   CI95_high |
|:-------------|:-------------------------|----:|--------:|--------------:|---------:|----------:|-----------:|------------:|
| CAR[-1,+1]   | exclude_most_influential |  60 |       6 |     -0.005833 | 0.013992 |  0.678875 |  -0.034071 |    0.022404 |
| CAR[-1,+1]   | exclude_March_April_2020 |  55 |       6 |     -0.009814 | 0.014172 |  0.492853 |  -0.038504 |    0.018876 |
| CAR[-1,+1]   | existing_text_controls   |  61 |       6 |     -0.031283 | 0.024102 |  0.201551 |  -0.079957 |    0.017391 |
| |CAR|[-1,+1] | exclude_most_influential |  60 |       6 |     -0.000786 | 0.009919 |  0.937193 |  -0.020803 |    0.019231 |
| |CAR|[-1,+1] | exclude_March_April_2020 |  55 |       6 |     -0.000239 | 0.010124 |  0.981301 |  -0.020734 |    0.020256 |
| |CAR|[-1,+1] | existing_text_controls   |  61 |       6 |      0.011524 | 0.018696 |  0.541044 |  -0.026234 |    0.049283 |

## Wild-cluster inference

| outcome      | specification   | scenario                 |   N |   firms |   p_value | method                                           |   weight_patterns | warnings                                                               |
|:-------------|:----------------|:-------------------------|----:|--------:|----------:|:-------------------------------------------------|------------------:|:-----------------------------------------------------------------------|
| CAR[-1,+1]   | pooled          | main                     |  62 |       7 |  0.890625 | Null-imposed WCR11; Rademacher; full enumeration |               128 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| CAR[-1,+1]   | firm_FE         | main                     |  61 |       6 |  0.531250 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| CAR[-1,+1]   | firm_year_FE    | main                     |  61 |       6 |  0.906250 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| |CAR|[-1,+1] | pooled          | main                     |  62 |       7 |  0.859375 | Null-imposed WCR11; Rademacher; full enumeration |               128 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| |CAR|[-1,+1] | firm_FE         | main                     |  61 |       6 |  0.812500 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| |CAR|[-1,+1] | firm_year_FE    | main                     |  61 |       6 |  0.750000 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| CAR[-2,+2]   | firm_year_FE    | main                     |  61 |       6 |  0.062500 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| CAR[0,+1]    | firm_year_FE    | main                     |  61 |       6 |  0.375000 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| CAR[-1,+1]   | firm_year_FE    | exclude_most_influential |  60 |       6 |  0.843750 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| CAR[-1,+1]   | firm_year_FE    | exclude_March_April_2020 |  55 |       6 |  0.687500 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| CAR[-1,+1]   | firm_year_FE    | existing_text_controls   |  61 |       6 |  0.843750 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| |CAR|[-1,+1] | firm_year_FE    | exclude_most_influential |  60 |       6 |  0.968750 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| |CAR|[-1,+1] | firm_year_FE    | exclude_March_April_2020 |  55 |       6 |  0.906250 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |
| |CAR|[-1,+1] | firm_year_FE    | existing_text_controls   |  61 |       6 |  0.843750 | Null-imposed WCR11; Rademacher; full enumeration |                64 | 2^G < the number of boot iterations, setting full_enumeration to True. |

HC3 uses Student-t reference with OLS residual degrees of freedom. Firm-clustered CR1 uses the finite-sample correction and t(G−1); it is a sensitivity check, with only seven pooled clusters and six main FE clusters. HC3 does not address within-firm dependence or contemporaneous cross-firm dependence. Wild-cluster tests impose a zero-specificity-coefficient null, use WCR11 and enumerate Rademacher weights (128 patterns for seven firms; 64 for six); their p-values are coarse, and they do not eliminate few-cluster limitations. Standardized models have the same null tests. No bootstrap confidence intervals are claimed.

## Influence

| Company                            |   Fiscal Year | outcome      |   studentized_external |   cooks_distance |   leverage |   specificity_dfbeta |
|:-----------------------------------|--------------:|:-------------|-----------------------:|-----------------:|-----------:|---------------------:|
| American International Group (AIG) |          2019 | |CAR|[-1,+1] |               5.480547 |         0.392502 |   0.282665 |             1.304562 |
| American International Group (AIG) |          2019 | CAR[-1,+1]   |              -5.240817 |         0.372195 |   0.282665 |            -1.247497 |
| Chubb                              |          2019 | CAR[-1,+1]   |               2.786412 |         0.122673 |   0.247632 |             0.076125 |
| Prudential Financial, Inc.         |          2022 | |CAR|[-1,+1] |               2.374199 |         0.104725 |   0.270328 |            -0.250242 |
| The Progressive Corporation        |          2019 | |CAR|[-1,+1] |              -1.976061 |         0.102210 |   0.334658 |            -0.088183 |
| Prudential Financial, Inc.         |          2021 | |CAR|[-1,+1] |              -1.784829 |         0.070145 |   0.294032 |            -0.378865 |
| Prudential Financial, Inc.         |          2022 | CAR[-1,+1]   |              -1.885721 |         0.069083 |   0.270328 |             0.198756 |
| The Progressive Corporation        |          2021 | |CAR|[-1,+1] |               1.532156 |         0.064227 |   0.336829 |            -0.111246 |
| American International Group (AIG) |          2020 | CAR[-1,+1]   |              -1.676654 |         0.062062 |   0.292849 |            -0.309922 |
| The Progressive Corporation        |          2022 | CAR[-1,+1]   |               1.370238 |         0.052250 |   0.338249 |             0.118096 |

Influence diagnostics use conventional OLS studentization, separately from robust inference. The exclusion sensitivity removes the observation with the largest Cook’s distance in each outcome’s preferred model. COVID sensitivity separately removes publication events from March 1 through April 30, 2020 inclusive. No observation is winsorized or removed from the primary pooled sample for being an outlier. The control sensitivity adds existing log sentence count and corporate ESG sentiment; it adjusts jointly for document length and tone. Sentiment is itself part of narrative content, so this is a sensitivity specification, not an identified causal adjustment.

## Data-quality limitations

- 4 publication dates shift to the next observed URTH session. All tested event windows have complete returns.
- URTH data begin January 12, 2012, despite the January 2011 requested start. Paired estimation counts were checked directly.
- Publication dates are taken as supplied; their source documents and time-of-day have not been independently authenticated. Annual-report announcements can include other value-relevant news.
- Chubb calculations reproduce using CB only. ACE became Chubb Limited and its shares began trading as CB on January 15, 2016, per the [issuer announcement](https://investors.chubb.com/News--Events/news/news-details/2016/ACE-Completes-Acquisition-of-Chubb-Adopts-Chubb-Name-and-Launches-New-Chubb-Brand-01-14-2016/default.aspx). This establishes legal/ticker continuity, but does not independently authenticate every pre-2016 Yahoo historical quote. The upstream notebook itself asks for a historical-price spot-check; that limitation remains.
- The upstream event code drops missing firm prices before calculating returns and sums abnormal returns with default missing-value handling. The independent audit checks a common benchmark calendar, no forward filling, and complete windows; no effect on these supplied CARs was found.
- The regression sample is small and unbalanced, with limited generalizability. Multiple outcomes/specifications are reported as prespecified; no claim rests on a selected minimum p-value. No multiplicity-adjusted confirmatory claim is made.

Interpret results only as associations between narrative specificity and the direction or magnitude of abnormal equity returns surrounding publication. Signed CAR and absolute CAR answer different questions.
