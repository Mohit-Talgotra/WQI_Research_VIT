"""Leave-one-observation-out masking validation of the three imputation methods (linear, cubic spline, Kalman smoother)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pathlib import Path
R = Path(__file__).resolve().parents[2]
m = pd.read_csv(R/'src/data/monthly_wqi_parameter_dataset.csv')
m['t'] = (m.Year.astype(int)-2022)*12+m.Month.astype(int); m['Station'] = m.Block+'|'+m.Location
P = ['pH_mean','TDS_mean','Hardness_mean','Chloride_mean','Fluoride_mean','Alkalinity_mean','Sulphate_mean','Nitrate_mean']
rows = []
for st, g in m.groupby('Station'):
    g = g.sort_values('t')
    for par in P:
        s = g[['t', par]].dropna()
        if len(s) < 5: continue
        t = s.t.to_numpy(); y = s[par].to_numpy()
        for k in range(1, len(t)-1):                      # interior points only
            keep = np.arange(len(t)) != k
            tt, yy = t[keep], y[keep]; tk = t[k]
            lin = np.interp(tk, tt, yy)
            spl = float(CubicSpline(tt, yy)(tk)) if len(tt) >= 4 else np.nan
            grid = np.arange(t.min(), t.max()+1); z = np.full(len(grid), np.nan); z[tt-t.min()] = yy
            try:
                kal = float(UnobservedComponents(z, 'llevel').fit(disp=False).smoothed_state[0][tk-t.min()])
            except Exception: kal = np.nan
            rows.append(dict(Station=st, param=par, gap=int(t[k+1]-t[k-1]), y=y[k], linear=lin, spline=spl, kalman=kal, mean=yy.mean()))
D = pd.DataFrame(rows); D.to_csv(R/'reproducibility_results/imputation_masking_predictions.csv', index=False)
out = []
for par, g in D.groupby('param'):
    g = g.dropna(); sd = g.y.std()
    out.append(dict(param=par, n=len(g), **{f'nrmse_{c}': np.sqrt(((g.y-g[c])**2).mean())/sd for c in ['linear','spline','kalman','mean']}))
o = pd.DataFrame(out).round(3); print(o.to_string(index=False))
g = D.dropna(); g['gapbin'] = pd.cut(g.gap, [0,4,8,100], labels=['<=4','5-8','>8'])
print(g.groupby('gapbin').apply(lambda x: pd.Series({c: np.sqrt(((x.y-x[c])**2/x.groupby('param').y.transform('var')).mean()) for c in ['linear','spline','kalman','mean']}|{'n':len(x)})).round(3))
o.to_csv(R/'reproducibility_results/imputation_masking_summary.csv', index=False)
