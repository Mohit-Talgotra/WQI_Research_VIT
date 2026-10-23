"""Per-parameter 2025-holdout accuracy of the parameter-first CatBoost models vs. persistence / location mean (original units, plus RMSE normalised by the holdout SD)."""
import sys, types
for _m in ('xgboost','dotenv'):
    try: __import__(_m)
    except ImportError: sys.modules[_m]=types.SimpleNamespace(XGBRegressor=object, load_dotenv=lambda *a, **k: None)  # unused here
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R))
import src.forecasting.parameter_forecaster as pf
from sklearn.metrics import r2_score
pm = pf.load_parameter_monthly_data()
slopes = pf.compute_sens_slopes(pm[pm['Date'] < pf.HOLDOUT_START])
filled = pf.kalman_fill_monthly_dataset(pm)
ds = pf.attach_sens_slopes(pf.build_parameter_feature_dataset(pm, filled), slopes)
fc = pf.get_parameter_feature_columns() + pf.get_sens_slope_feature_columns()
tr = ds[ds.Date < pf.HOLDOUT_START]; te = ds[ds.Date >= pf.HOLDOUT_START]
pred, _ = pf.fit_predict_parameter_models(tr, te, fc, model_params=pf.DEFAULT_PARAMETER_MODEL_PARAMS)
rows = []
for p in pf.PARAMETER_COLS:
    y = te[f'{p}_target'].to_numpy(); lagc = [c for c in te.columns if c.startswith(p) and c.endswith('lag_1')]
    lm = tr.groupby(['Block','Location'])[f'{p}_target'].mean()
    loc = np.array([lm.get((r.Block, r.Location), tr[f'{p}_target'].mean()) for r in te.itertuples()])
    pers = te[lagc[0]].fillna(tr[f'{p}_target'].mean()).to_numpy() if lagc else np.full(len(te), np.nan)
    f = lambda a: np.sqrt(np.nanmean((y-a)**2))
    rows.append(dict(parameter=p, n=len(y), sd=y.std(), rmse_model=f(pred[p].to_numpy()), rmse_persistence=f(pers), rmse_locmean=f(loc), r2_model=r2_score(y, pred[p]), r2_persistence=r2_score(y, pers) if lagc else np.nan))
o = pd.DataFrame(rows); o['nrmse_model'] = o.rmse_model/o.sd; o['nrmse_persistence'] = o.rmse_persistence/o.sd
print(o.round(3).to_string(index=False)); o.round(4).to_csv(R/'reproducibility_results/per_parameter_holdout_accuracy.csv', index=False)
