"""Aggregate frozen sentence-level outputs to firm-year inputs for the extended test."""
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Users/jackyeetan/Library/CloudStorage/GoogleDrive-jackty92@gmail.com/My Drive/phd/Data')
OUT = ROOT / 'specificity_car_test' / 'inputs'

spec_path = DATA / 'df_ar_ceo_sentences_esg_combined_specificity.csv'
sent_path = DATA / 'df_ar_ceo_sentences_finbert_senti.csv'
spec = pd.read_csv(spec_path)
sent = pd.read_csv(sent_path)

assert sent['sent_label'].isin(['positive','negative','neutral']).all()
spec_exact_duplicates = int(spec.duplicated().sum())
sent_exact_duplicates = int(sent.duplicated().sum())

spec['specificity_tier_pred'] = pd.to_numeric(spec['specificity_tier_pred'], errors='raise')
spec['primary_tier'] = spec['specificity_tier_pred'].where(
    spec['corporate_relevance_pred'].eq('Corporate') & spec['specificity_tier_pred'].ge(1))
spec_fy = spec.groupby(['Company Name','Year']).agg(
    Specificity_mean=('primary_tier','mean'),
    Specificity_all_flagged_mean=('specificity_tier_pred','mean'),
    specificity_sentences_classified=('specificity_tier_pred','size'),
    specificity_primary_sentences=('primary_tier','count'),
).reset_index()

sent['is_positive'] = sent['sent_label'].eq('positive').astype(int)
sent['is_negative'] = sent['sent_label'].eq('negative').astype(int)
overall = sent.groupby(['Company Name','Year']).agg(
    total_sentences=('sent_label','size'), positive_share=('is_positive','mean'),
    negative_share=('is_negative','mean')).reset_index()
overall['Overall_sentiment'] = overall.positive_share - overall.negative_share
esg = sent.loc[sent.any_esg_flag.eq(1)].groupby(['Company Name','Year']).agg(
    ESG_sentences=('sent_label','size'), ESG_positive_share=('is_positive','mean'),
    ESG_negative_share=('is_negative','mean')).reset_index()
esg['ESG_sentiment'] = esg.ESG_positive_share - esg.ESG_negative_share
fy = overall.merge(esg, on=['Company Name','Year'], how='left', validate='one_to_one')
fy = fy.merge(spec_fy, on=['Company Name','Year'], how='outer', validate='one_to_one', indicator=True)
fy.to_csv(OUT/'specificity_sentiment_firmyear.csv', index=False)

audit = {
    'specificity_sentence_rows': len(spec), 'finbert_sentence_rows': len(sent),
    'specificity_exact_duplicate_rows': spec_exact_duplicates,
    'finbert_exact_duplicate_rows': sent_exact_duplicates,
    'specificity_firmyears': len(spec_fy), 'sentiment_firmyears': len(overall),
    'firmyear_merge_counts': fy['_merge'].value_counts().to_dict(),
    'source_sha256': {
        spec_path.name: hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        sent_path.name: hashlib.sha256(sent_path.read_bytes()).hexdigest(),
    },
    'sentiment_definition': 'share(sent_label==positive) - share(sent_label==negative)',
    'ESG_sentiment_population': 'any_esg_flag == 1',
    'primary_specificity_definition': 'mean tier among Corporate sentences with tier >= 1',
    'robustness_specificity_definition': 'mean tier 0-3 over all classified ESG-flagged sentences',
}
(OUT/'specificity_sentiment_source_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(audit)
