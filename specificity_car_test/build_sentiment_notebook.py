from pathlib import Path
import json, sys
import nbformat as nbf
root=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook(); md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
nb.metadata.kernelspec={'display_name':'Specificity CAR analysis','language':'python','name':'specificity-car'}
nb.cells=[
 md('# CEO-letter specificity, sentiment, and abnormal returns\n\nThis extends the validated CAR analysis with overall FinBERT sentiment, ESG-specific sentiment, interactions, and `[0,+5]`/`[0,+20]` CAR windows. It uses frozen classifier and event-study outputs and makes no causal claim.'),
 code("from pathlib import Path\nimport sys, pandas as pd\nfrom IPython.display import display, Markdown\nROOT=Path.cwd(); sys.path.insert(0,str(ROOT))\nfrom specificity_car_test.analysis_sentiment import main"),
 md('## Merge and definitions\n\nPrimary specificity is the existing Corporate tier≥1 mean. Overall sentiment is positive share minus negative share across all sentences. ESG sentiment applies the same formula to ESG-flagged sentences. Tier 0 is a false positive and is not redefined as generic.'),
 code("audit, descriptive, correlations, results = main()\ndisplay(pd.DataFrame([audit]))\ndisplay(descriptive)\ndisplay(correlations)"),
 md('## Preferred firm and fiscal-year fixed-effects results\n\nHC3 is the main inference. Firm-clustered and wild-cluster results are sensitivity checks because there are only six within-firm clusters.'),
 code("preferred=results[(results.fixed_effects=='firm_year_FE')&(results.covariance=='HC3')&results.model.isin(['M3_specificity_overall','M3b_specificity_ESG','M4_interaction_overall','M4b_interaction_ESG'])]\ndisplay(preferred)\ndisplay(pd.read_csv(ROOT/'specificity_car_test/results_sentiment/wild_cluster_bootstrap.csv'))"),
 md('## Influence and COVID sensitivities'),
 code("display(pd.read_csv(ROOT/'specificity_car_test/results_sentiment/influence_diagnostics.csv').sort_values('cooks_distance',ascending=False).head(20))\ndisplay(pd.read_csv(ROOT/'specificity_car_test/results_sentiment/influence_COVID_sensitivities.csv'))"),
 md('## Interpretation'),
 code("display(Markdown((ROOT/'specificity_car_test/REPORT_SENTIMENT.md').read_text()))"),
 code("assert audit['event_rows']==84 and audit['regression_N']==62\nassert len(results)==468\nassert results[['coefficient','SE','p_value','CI95_low','CI95_high']].notna().all().all()\nprint('Validated extended analysis outputs.')")]
nbf.validate(nb); nbf.write(nb,root/'CEO_Letter_Specificity_Sentiment_CAR_Test.ipynb')
print('Built extended notebook.')
