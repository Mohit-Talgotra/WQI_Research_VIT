"""Held-out-station spatio-temporal forecast: predict WQI of station L in 2025 using ONLY other stations' pre-2025... history
(plus the other stations' observations in months before the target month), coordinates, block and calendar features. L's own history is never used."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
ROOT = Path(__file__).resolve().parents[2]
m = pd.read_csv(ROOT/'src/data/monthly_wqi_dataset.csv').dropna(subset=['WQI_mean'])
m['Date'] = pd.to_datetime(dict(year=m.Year.astype(int), month=m.Month.astype(int), day=1))
g = pd.read_csv(ROOT/'src/data/locations_geocoded_98.csv')[['Block','Location','Latitude','Longitude']]
# block-name spelling differs between files; join on Location+Block fallback to Location
g['Block'] = g.Block.replace({'Ayodhiyapattanam':'Ayothiyapattinam'})
m = m.merge(g, on=['Block','Location'], how='left')
if m.Latitude.isna().any(): m = m.drop(columns=['Latitude','Longitude']).merge(g.drop_duplicates('Location')[['Location','Latitude','Longitude']], on='Location', how='left')
print('rows', len(m), 'missing coords', m.Latitude.isna().sum())
m['t'] = (m.Date.dt.year-2022)*12+m.Date.dt.month
CUT = (2025-2022)*12+1
def feats(row, pool):
    """features for target (row) from pool of OTHER stations' observations strictly before row.t"""
    p = pool[(pool.Location != row.Location) & (pool.t < row.t)]
    if p.empty: return None
    w = 1/((p.Latitude-row.Latitude)**2+(p.Longitude-row.Longitude)**2+1e-4)
    rec = p[p.t >= row.t-3]
    r = rec if len(rec) else p
    wr = 1/((r.Latitude-row.Latitude)**2+(r.Longitude-row.Longitude)**2+1e-4)
    return dict(idw_all=np.average(p.WQI_mean, weights=w), idw_recent=np.average(r.WQI_mean, weights=wr),
                recent_mean=r.WQI_mean.mean(), blk_recent=(r[r.Block==row.Block].WQI_mean.mean() if (r.Block==row.Block).any() else r.WQI_mean.mean()),
                lat=row.Latitude, lon=row.Longitude, month=row.Month, t=row.t)
rows = []
for i, r in m.iterrows():
    f = feats(r, m[m.t < CUT]) if r.t < CUT else feats(r, m[m.t < r.t])
    if f: rows.append(dict(f, Location=r.Location, tt=r.t, y=r.WQI_mean))
D = pd.DataFrame(rows)
F = ['idw_all','idw_recent','recent_mean','blk_recent','lat','lon','month','t']
# NOTE: for each held-out L, train rows are other stations' pre-2025 rows whose features were built without L's data in the pool
test = D[D.tt >= CUT]; res = []; preds = []
for L in test.Location.unique():
    te = test[test.Location == L]
    tr = D[(D.tt < CUT) & (D.Location != L)]
    # rebuild train features excluding L entirely from the pool
    pool = m[(m.Location != L)]
    trr = [dict(feats(r, pool[pool.t < CUT]), y=r.WQI_mean) for _, r in m[(m.t < CUT) & (m.Location != L)].iterrows() if feats(r, pool[pool.t < CUT]) is not None]
    tr = pd.DataFrame(trr)
    tef = []
    for _, r in m[(m.Location == L) & (m.t >= CUT)].iterrows():
        f = feats(r, pool[pool.t < r.t])
        if f: tef.append(dict(f, y=r.WQI_mean))
    if not tef: continue
    te = pd.DataFrame(tef)
    out = dict(Location=L, y=te.y.values)
    out['IDW_recent'] = te.idw_recent.values; out['MeanOthers'] = te.recent_mean.values
    out['TrainMean'] = np.full(len(te), tr.y.mean())
    for nm, mod in [('Ridge', Ridge(1.0)), ('GB', GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=2)), ('RF', RandomForestRegressor(200, min_samples_leaf=3, random_state=42))]:
        out[nm] = mod.fit(tr[F], tr.y).predict(te[F])
    preds.append(pd.DataFrame(out))
P = pd.concat([pd.DataFrame({k: (v if hasattr(v, '__len__') and not isinstance(v, str) else [v]*len(pd.DataFrame(p.Location).index)) for k, v in {}.items()}) for p in []] or preds, ignore_index=True)
P = pd.concat(preds, ignore_index=True)
print('held-out rows', len(P), 'stations', P.Location.nunique())
locs = P.Location.values; ul = np.unique(locs); idx = {l: np.where(locs == l)[0] for l in ul}
rng = np.random.default_rng(42); tab = []
persist_ref = 16.454
for c in ['TrainMean','MeanOthers','IDW_recent','Ridge','GB','RF']:
    se = (P.y-P[c]).to_numpy()**2
    b = [np.sqrt(se[np.concatenate([idx[l] for l in rng.choice(ul, len(ul))])].mean()) for _ in range(2000)]
    tab.append(dict(model=c, rmse=np.sqrt(se.mean()), ci_lo=np.percentile(b,2.5), ci_hi=np.percentile(b,97.5), mae=np.abs(P.y-P[c]).mean()))
T = pd.DataFrame(tab).round(2); print(T.to_string(index=False)); print('own-history persistence RMSE on 63 rows (reference):', persist_ref)
T.to_csv(ROOT/'reproducibility_results/heldout_station_forecast.csv', index=False)
P.explode(['y','IDW_recent','MeanOthers','TrainMean','Ridge','GB','RF']).to_csv(ROOT/'reproducibility_results/heldout_station_forecast_predictions.csv', index=False)
