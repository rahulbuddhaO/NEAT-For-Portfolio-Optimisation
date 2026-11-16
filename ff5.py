import numpy as np, pandas as pd, statsmodels.api as sm
F="/mnt/user-data/uploads/Downloads/F-F_Research_Data_5_Factors_2x3_daily_CSV/F-F_Research_Data_5_Factors_2x3_daily.CSV"
ff=pd.read_csv(F,skiprows=3,index_col=0); ff=ff[pd.to_numeric(ff.index,errors='coerce').notna()]
ff.index=pd.to_datetime(ff.index.astype(str).str.strip(),format='%Y%m%d'); ff=ff.astype(float)/100
raw=pd.read_csv('/mnt/user-data/uploads/neat_runs/daily_returns_by_draw.csv.gz',index_col=0,parse_dates=True)
S=sorted({c.split('|')[0] for c in raw.columns}); D=sorted({c.split('|')[1] for c in raw.columns})
ens=pd.DataFrame({s: raw[[f'{s}|{d}' for d in D]].mean(axis=1) for s in S})
df=ens.join(ff,how='inner'); print('days matched',len(df),'of',len(ens))
X=sm.add_constant(df[['Mkt-RF','SMB','HML','RMW','CMA']])
def reg(y):
    m=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    return m
rows=[]
for s in S:
    m=reg(df[s]-df['RF'])
    rows.append(dict(Strategy=s,alpha_ann=252*m.params['const'],t=m.tvalues['const'],p_two=m.pvalues['const'],
        p_one=(m.pvalues['const']/2 if m.params['const']>0 else 1-m.pvalues['const']/2),
        MKT=m.params['Mkt-RF'],SMB=m.params['SMB'],HML=m.params['HML'],RMW=m.params['RMW'],CMA=m.params['CMA'],R2=m.rsquared))
R=pd.DataFrame(rows).set_index('Strategy')
def holm(p):
    p=np.asarray(p); o=np.argsort(p); n=len(p); r=np.empty(n); run=0
    for k,i in enumerate(o): run=max(run,(n-k)*p[i]); r[i]=min(1,run)
    return r
def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); r=np.empty(n); prev=1
    for k in range(n-1,-1,-1): i=o[k]; prev=min(prev,p[i]*n/(k+1)); r[i]=prev
    return r
R['p_BH']=bh(R.p_two); R['p_Holm']=holm(R.p_two)
R.sort_values('alpha_ann',ascending=False).to_csv('/tmp/claude-0/ff/ff5_all.csv')
pd.set_option('display.width',200)
print(R.sort_values('alpha_ann',ascending=False).round(3).head(20).to_string())
print('benchmarks:\n',R.loc[['EqualWeight','Markowitz']].round(3).to_string())
print('n positive alpha',int((R.alpha_ann>0).sum()),'of',len(R),'| sig raw 5%:',int((R.p_two<.05).sum()),'| BH:',int((R.p_BH<.05).sum()),'| Holm:',int((R.p_Holm<.05).sum()))
cfg=['PRICE_ONLY','CORR_ALONE','SENT_ONLY','CORR_ONLY','PRICE_SENT','CORR_SENT','PRICE_CORR_SENT','FULL']
al=['NEAT','NSGA2','CMAES','GA','FIREFLY','DE','ABC','PSO','GWO','GPTREE']
A=pd.DataFrame({c:[R.loc[f'{a}_{c}','alpha_ann'] for a in al] for c in cfg},index=al)
print((A*100).round(2).to_string())
print('NEAT highest alpha per cfg:', (A.idxmax()=='NEAT').to_dict())
# long-short NEAT minus best algo / minus EW
for a,b in [('NEAT_PRICE_ONLY','EqualWeight'),('NEAT_PRICE_ONLY','Markowitz'),('NEAT_PRICE_ONLY','NSGA2_PRICE_ONLY')]:
    m=reg(df[a]-df[b]); print(f'{a} minus {b}: alpha {252*m.params["const"]*100:.2f}%/yr t={m.tvalues["const"]:.2f} p={m.pvalues["const"]:.3f}')
# per-draw robustness for NEAT price
for d in D:
    y=raw[f'NEAT_PRICE_ONLY|{d}'].reindex(df.index)-df['RF']; m=reg(y); print(d,'NEAT price alpha',round(252*m.params['const']*100,2),'p',round(m.pvalues['const'],3))
