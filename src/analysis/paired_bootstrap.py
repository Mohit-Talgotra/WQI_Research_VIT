"""Paired location-block bootstrap of RMSE difference vs persistence (2025 holdout), incl. hierarchical model."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(42); NB = 5000
a = pd.read_csv(ROOT/'src/data/forecasting_parameters/holdout_predictions_detail.csv')
g = pd.read_csv(ROOT/'src/data/forecasting_gam/gam_holdout_predictions.csv')
h = pd.read_csv(ROOT/'reproducibility_results/hierarchical_state_space/hierarchical_holdout_predictions.csv'); h['model'] = 'Hierarchical'
df = pd.concat([a, g, h], ignore_index=True)
df['se'] = (df.Actual_WQI-df.Predicted_WQI)**2
key = ['Block','Location','Date']
w = df.pivot_table(index=key, columns='model', values='se')
w = w.dropna()
print('paired rows', len(w), 'models', list(w.columns))
locs = w.reset_index().Location.values
ul = np.unique(locs); idx = {l: np.where(locs==l)[0] for l in ul}
M = w.to_numpy(); cols = list(w.columns); p = cols.index('Persistence')
boot = np.empty((NB, len(cols)))
for i in range(NB):
    s = np.concatenate([idx[l] for l in rng.choice(ul, len(ul))])
    m = M[s].mean(0); boot[i] = np.sqrt(m) - np.sqrt(m[p])
pt = np.sqrt(M.mean(0))
rows = []
for j, c in enumerate(cols):
    if c == 'Persistence': continue
    d = pt[j]-pt[p]; lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
    rows.append(dict(model=c, rmse=pt[j], delta_vs_persistence=d, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo>0 or hi<0)))
out = pd.DataFrame(rows).round(3); print('persistence rmse', pt[p].round(3), 'locations', len(ul)); print(out.to_string())
out.to_csv(ROOT/'reproducibility_results/holdout_paired_bootstrap_vs_persistence.csv', index=False)
