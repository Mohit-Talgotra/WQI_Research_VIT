"""Composition check: is the pooled TDS/WQI trend an artifact of which stations were sampled when?
Balanced panel = stations with >=1 observation in BOTH 2022-23 and 2024-25. Compares pooled monthly-mean MK trend, yearly means, and a within-station (fixed-effect) slope."""
import numpy as np, pandas as pd, pymannkendall as mk, statsmodels.formula.api as smf
from pathlib import Path
R = Path(__file__).resolve().parents[2]
m = pd.read_csv(R/'src/data/monthly_wqi_parameter_dataset.csv')
m['Date'] = pd.to_datetime(dict(year=m.Year.astype(int), month=m.Month.astype(int), day=1)); m['dy'] = m.Year + (m.Month-1)/12
m['Station'] = m.Block+'|'+m.Location
early = set(m[m.Year<=2023].Station); late = set(m[m.Year>=2024].Station); bal = early & late
rows = []
for par in ['TDS_mean','Hardness_mean','Chloride_mean','Fluoride_mean','WQI_mean']:
    for nm, sub in [('all', m), ('balanced', m[m.Station.isin(bal)])]:
        s = sub.dropna(subset=[par]); ms = s.groupby('Date')[par].mean().sort_index()
        r = mk.original_test(ms); ym = s.groupby('Year')[par].mean().round(1).to_dict()
        fe = smf.ols(f'{par} ~ dy + C(Station)', data=s).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s.Station)[0]})
        rows.append(dict(param=par, panel=nm, n_stations=s.Station.nunique(), n_obs=len(s), mk_trend=r.trend, mk_p=r.p, fe_slope_per_year=fe.params['dy'], fe_p=fe.pvalues['dy'], yearly_means=ym))
o = pd.DataFrame(rows); pd.set_option('display.width', 250, 'display.max_colwidth', 80); print(o.to_string(index=False))
o.to_csv(R/'reproducibility_results/balanced_panel_trend.csv', index=False)
