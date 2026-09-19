"""Extended specificity, FinBERT sentiment, interaction, and CAR-window analysis."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'specificity_car_test'; IN=BASE/'inputs'; OUT=BASE/'results_sentiment'
OUT.mkdir(exist_ok=True)
MAP={'AIG':'American International Group (AIG)','Chubb':'Chubb','MET':'MetLife, Inc.',
     'Prudential Financials':'Prudential Financial, Inc.','Allstate':'The Allstate Corporation',
     'Progressive':'The Progressive Corporation','Travelers':'The Travelers Companies, Inc.'}
TICK={'American International Group (AIG)':'AIG','Chubb':'CB','MetLife, Inc.':'MET',
      'Prudential Financial, Inc.':'PRU','The Allstate Corporation':'ALL',
      'The Progressive Corporation':'PGR','The Travelers Companies, Inc.':'TRV'}
OUTCOMES=['CAR[-1,+1]','|CAR|[-1,+1]','CAR[-2,+2]','CAR[0,+1]','CAR[0,+5]','CAR[0,+20]']

def extend_car(event):
    bench=pd.read_csv(IN/'raw_prices/URTH.csv',header=[0,1],index_col=0)['Close'].iloc[:,0]
    bench.index=pd.to_datetime(bench.index); cal=bench.index; br=bench.pct_change(fill_method=None)
    cache={}
    for company,ticker in TICK.items():
        s=pd.read_csv(IN/f'raw_prices/{ticker}.csv',header=[0,1],index_col=0)['Close'].iloc[:,0]
        s.index=pd.to_datetime(s.index); cache[company]=s.reindex(cal).pct_change(fill_method=None)
    vals=[]
    for _,r in event.iterrows():
        d=pd.Timestamp(r['Event Date (adjusted)']); pos=cal.get_loc(d); fr=cache[r.Company]
        row={}
        for name,hi in [('CAR[0,+5]',5),('CAR[0,+20]',20)]:
            idx=cal[pos:pos+hi+1]; ar=fr.loc[idx]-(r.Alpha+r.Beta*br.loc[idx])
            assert len(idx)==hi+1 and ar.notna().all(); row[name]=ar.sum()
        vals.append(row)
    return event.join(pd.DataFrame(vals,index=event.index))

def singletons(d,spec):
    if spec=='pooled': return d.copy(),[]
    x=d.copy(); dropped=[]; effects=['firm']+(['year'] if spec=='firm_year_FE' else [])
    while True:
        bad=pd.Series(False,index=x.index)
        for e in effects: bad |= x.groupby(e)[e].transform('size').eq(1)
        if not bad.any(): return x,dropped
        dropped += x.loc[bad,['Company','Fiscal Year']].to_dict('records'); x=x.loc[~bad].copy()

def main():
    event=extend_car(pd.read_csv(IN/'ceo_letter_event_study_panel_US.csv'))
    fy=pd.read_csv(IN/'specificity_sentiment_firmyear.csv').rename(columns={'Year':'Fiscal Year'})
    fy['source_company']=fy['Company Name']; fy['Company']=fy['Company Name'].replace(MAP)
    target=fy[fy.Company.isin(TICK)].drop(columns=['_merge'], errors='ignore').copy()
    assert not event.duplicated(['Company','Fiscal Year']).any() and not target.duplicated(['Company','Fiscal Year']).any()
    merged=event.merge(target.drop(columns='Company Name'),on=['Company','Fiscal Year'],how='left',indicator=True,validate='one_to_one')
    required=['Specificity_mean','Overall_sentiment','ESG_sentiment']
    merged['exclusion_reason']=np.select([merged._merge.ne('both'),merged.Specificity_mean.isna(),merged.Overall_sentiment.isna(),merged.ESG_sentiment.isna()],
      ['missing firm-year narrative output','specificity undefined','overall sentiment missing','ESG sentiment missing'],default='')
    merged['regression_eligible']=merged.exclusion_reason.eq('')
    merged['Event Date (adjusted)']=pd.to_datetime(merged['Event Date (adjusted)'])
    merged['COVID_period']=merged['Event Date (adjusted)'].between('2020-03-01','2020-04-30')
    sample=merged.loc[merged.regression_eligible].copy(); sample['firm']=sample.Company; sample['year']=sample['Fiscal Year'].astype(int)
    for v in ['Specificity_mean','Overall_sentiment','ESG_sentiment']:
        sample[v+'_z']=(sample[v]-sample[v].mean())/sample[v].std(ddof=1)
    merged.to_csv(ROOT/'specificity_sentiment_car_US_merged.csv',index=False)
    audit={'event_rows':len(event),'specificity_sentiment_firmyears':len(fy),'matched':int((merged._merge=='both').sum()),
           'missing_specificity_years':merged.loc[merged.Specificity_mean.isna(),['Company','Fiscal Year']].to_dict('records'),
           'missing_overall_sentiment_years':merged.loc[merged.Overall_sentiment.isna(),['Company','Fiscal Year']].to_dict('records'),
           'missing_ESG_sentiment_years':merged.loc[merged.ESG_sentiment.isna(),['Company','Fiscal Year']].to_dict('records'),
           'duplicate_event_keys':0,'duplicate_narrative_keys':0,'regression_N':len(sample),'firms':sample.firm.nunique()}
    (OUT/'merge_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    desc=sample[OUTCOMES+['Specificity_mean','Specificity_all_flagged_mean','Overall_sentiment','ESG_sentiment']].describe().T
    desc.to_csv(OUT/'table1_descriptive_statistics.csv')
    corr=sample[OUTCOMES+['Specificity_mean','Overall_sentiment','ESG_sentiment']].corr(method='pearson')
    corr.to_csv(OUT/'table2_correlation_matrix.csv')
    specs={'M1_specificity':'Specificity_mean','M2_overall_sentiment':'Overall_sentiment',
           'M2b_ESG_sentiment':'ESG_sentiment','M3_specificity_overall':'Specificity_mean + Overall_sentiment',
           'M3b_specificity_ESG':'Specificity_mean + ESG_sentiment',
           'M4_interaction_overall':'Specificity_mean * Overall_sentiment',
           'M4b_interaction_ESG':'Specificity_mean * ESG_sentiment'}
    rows=[]; infl=[]; wild=[]
    for outcome in OUTCOMES:
      for model_name,rhs in specs.items():
       for fe in ['pooled','firm_FE','firm_year_FE']:
        d,drop=singletons(sample,fe); d=d.copy(); d['y']=d[outcome]
        formula='y ~ '+rhs+(' + C(firm)' if fe!='pooled' else '')+(' + C(year)' if fe=='firm_year_FE' else '')
        m=smf.ols(formula,d).fit(); groups=pd.Categorical(d.firm).codes
        covs={'HC3':m.get_robustcov_results(cov_type='HC3',use_t=True),
              'firm_cluster_CR1':m.get_robustcov_results(cov_type='cluster',groups=groups,use_correction=True,df_correction=True,use_t=True)}
        terms=[x for x in m.model.exog_names if x in ['Specificity_mean','Overall_sentiment','ESG_sentiment','Specificity_mean:Overall_sentiment','Specificity_mean:ESG_sentiment']]
        for cov,res in covs.items():
         for term in terms:
          j=m.model.exog_names.index(term); ci=res.conf_int()[j]
          rows.append({'outcome':outcome,'model':model_name,'fixed_effects':fe,'covariance':cov,'term':term,'N':len(d),'firms':d.firm.nunique(),
                       'coefficient':res.params[j],'SE':res.bse[j],'p_value':res.pvalues[j],'CI95_low':ci[0],'CI95_high':ci[1],'R_squared':m.rsquared})
        if fe=='firm_year_FE' and model_name in ['M3_specificity_overall','M3b_specificity_ESG','M4_interaction_overall','M4b_interaction_ESG']:
         for term in terms:
          with warnings.catch_warnings(record=True):
           b=wildboottest(m.model,B=9999,cluster=groups,param=term,weights_type='rademacher',impose_null=True,bootstrap_type='11',seed=20260919,parallel=False,show=False)
          wild.append({'outcome':outcome,'model':model_name,'term':term,'N':len(d),'firms':d.firm.nunique(),'p_value':float(b['p-value'].iloc[0])})
        if fe=='firm_year_FE' and model_name in ['M3_specificity_overall','M3b_specificity_ESG'] and outcome in OUTCOMES[:2]:
         z=m.get_influence(); q=d[['Company','Fiscal Year']].copy(); q['outcome']=outcome;q['model']=model_name
         q['studentized_residual']=z.resid_studentized_external;q['cooks_distance']=z.cooks_distance[0];q['leverage']=z.hat_matrix_diag; infl.append(q)
    results=pd.DataFrame(rows); results.to_csv(ROOT/'specificity_sentiment_car_regression_results.csv',index=False)
    pd.DataFrame(wild).to_csv(OUT/'wild_cluster_bootstrap.csv',index=False)
    influence=pd.concat(infl); influence.to_csv(OUT/'influence_diagnostics.csv',index=False)
    # Requested exclusion sensitivities for preferred joint models and primary outcomes.
    sens=[]
    for outcome in OUTCOMES[:2]:
     for model_name,rhs in [('M3_specificity_overall',specs['M3_specificity_overall']),('M3b_specificity_ESG',specs['M3b_specificity_ESG'])]:
      base=influence[(influence.outcome==outcome)&(influence.model==model_name)]; worst=base.loc[base.cooks_distance.idxmax()]
      scenarios={'exclude_most_influential':~((sample.Company==worst.Company)&(sample['Fiscal Year']==worst['Fiscal Year'])),
                 'exclude_COVID_period':~sample.COVID_period}
      for scenario,mask in scenarios.items():
       d,_=singletons(sample.loc[mask], 'firm_year_FE'); d=d.copy();d['y']=d[outcome]
       m=smf.ols('y ~ '+rhs+' + C(firm) + C(year)',d).fit(); h=m.get_robustcov_results(cov_type='HC3',use_t=True)
       for term in [x for x in m.model.exog_names if x in required]:
        j=m.model.exog_names.index(term);ci=h.conf_int()[j];sens.append({'outcome':outcome,'model':model_name,'scenario':scenario,'term':term,'N':len(d),'coefficient':h.params[j],'SE':h.bse[j],'p_value':h.pvalues[j],'CI95_low':ci[0],'CI95_high':ci[1]})
    pd.DataFrame(sens).to_csv(OUT/'influence_COVID_sensitivities.csv',index=False)
    print(json.dumps(audit,indent=2)); print(results.shape); return audit,desc,corr,results
if __name__=='__main__': main()
