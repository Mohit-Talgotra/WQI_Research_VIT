"""
Track B: Sparsity-Density Simulation Experiment (Reviewer Concern 2.5).
Evaluates forecasting performance (Persistence vs. ML) across varying observational
density levels (15%, 25%, 40%, 60%, 80%, 100%) using synthetic panels generated
from empirical location level, seasonality, and variance parameters.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
PARAMETER_DATASET_PATH = ROOT / 'src' / 'data' / 'monthly_wqi_parameter_dataset.csv'
OUT_DIR = ROOT / 'reproducibility_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_DATE = pd.Timestamp('2025-01-01')
RNG = np.random.default_rng(42)

PARAMETER_STANDARDS = {
    'pH_mean': {'Sn': 8.5, 'Videal': 7.0},
    'TDS_mean': {'Sn': 500.0, 'Videal': 0.0},
    'Hardness_mean': {'Sn': 200.0, 'Videal': 0.0},
    'Chloride_mean': {'Sn': 250.0, 'Videal': 0.0},
    'Fluoride_mean': {'Sn': 1.0, 'Videal': 0.0},
    'Alkalinity_mean': {'Sn': 200.0, 'Videal': 0.0},
    'Sulphate_mean': {'Sn': 200.0, 'Videal': 0.0},
    'Nitrate_mean': {'Sn': 45.0, 'Videal': 0.0},
}
K = 1.0 / sum(1.0 / spec['Sn'] for spec in PARAMETER_STANDARDS.values())
WEIGHTS = {name: K / spec['Sn'] for name, spec in PARAMETER_STANDARDS.items()}
PARAMS = list(PARAMETER_STANDARDS.keys())


def calc_wqi(row: pd.Series) -> float:
    total = 0.0
    for param, spec in PARAMETER_STANDARDS.items():
        v = float(row[param])
        qn = ((v - spec['Videal']) / (spec['Sn'] - spec['Videal'])) * 100.0
        total += float(np.clip(qn, 0.0, 300.0)) * WEIGHTS[param]
    return total


def simulate_dense_panel(df_real: pd.DataFrame) -> pd.DataFrame:
    """Generate dense monthly panel (2022-01 to 2025-04) for all locations."""
    dates = pd.date_range('2022-01-01', '2025-04-01', freq='MS')
    locations = df_real[['Block', 'Location']].drop_duplicates()

    # Precompute per-location parameter profiles
    loc_profiles = {}
    for (block, loc), g in df_real.groupby(['Block', 'Location']):
        loc_profiles[(block, loc)] = {
            p: {
                'mean': float(g[p].mean()),
                'std': float(g[p].std()) if len(g) > 1 and not np.isnan(g[p].std()) and g[p].std() > 0 else 0.1 * float(g[p].mean() if g[p].mean() > 0 else 1.0),
            }
            for p in PARAMS
        }

    rows = []
    for _, loc_row in locations.iterrows():
        b, l = loc_row['Block'], loc_row['Location']
        prof = loc_profiles.get((b, l))
        for d in dates:
            r = {'Block': b, 'Location': l, 'Date': d, 'Month': d.month, 'Year': d.year}
            # Add seasonal variation (monsoon / post-monsoon factor)
            seasonal_factor = 1.0 + 0.08 * np.sin(2 * np.pi * d.month / 12)
            for p in PARAMS:
                m = prof[p]['mean'] * seasonal_factor
                s = prof[p]['std']
                val = max(0.01, RNG.normal(m, s * 0.5))
                r[p] = val
            r['WQI'] = calc_wqi(pd.Series(r))
            rows.append(r)

    return pd.DataFrame(rows).sort_values(['Block', 'Location', 'Date']).reset_index(drop=True)


def build_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(['Block', 'Location', 'Date']).copy()
    panel['WQI_lag1'] = panel.groupby('Location')['WQI'].shift(1)
    panel['WQI_lag2'] = panel.groupby('Location')['WQI'].shift(2)
    panel['WQI_lag3'] = panel.groupby('Location')['WQI'].shift(3)
    return panel.dropna(subset=['WQI_lag1']).reset_index(drop=True)


def run_experiment(coverage_levels: list[float] = [0.15, 0.25, 0.40, 0.60, 0.80, 1.00]):
    print("=" * 70)
    print("SPARSITY-DENSITY SIMULATION EXPERIMENT (Track B, Concern 2.5)")
    print("=" * 70)

    df_real = pd.read_csv(PARAMETER_DATASET_PATH, parse_dates=['Date'])
    dense = simulate_dense_panel(df_real)
    print(f"Generated dense synthetic panel: {len(dense)} rows across {dense['Location'].nunique()} locations.")

    results = []

    for cov in coverage_levels:
        train_pool = dense[dense['Date'] < HOLDOUT_DATE].copy()
        holdout_pool = dense[dense['Date'] >= HOLDOUT_DATE].copy()

        # Thin train pool to target coverage
        if cov < 1.0:
            mask = pd.Series(RNG.random(len(train_pool)) < cov, index=train_pool.index)
            # Ensure minimum history per location
            for loc in train_pool['Location'].unique():
                loc_idx = train_pool[train_pool['Location'] == loc].index
                if mask.loc[loc_idx].sum() < 3:
                    mask.loc[loc_idx[:3]] = True
            train_sub = train_pool[mask].copy()
        else:
            train_sub = train_pool.copy()

        # Build features with simple forward-lag lookup
        train_feat = build_lag_features(train_sub)
        holdout_feat = build_lag_features(holdout_pool)

        if len(train_feat) < 50 or len(holdout_feat) == 0:
            continue

        # 1. Persistence baseline: pred = WQI_lag1
        persistence_preds = holdout_feat['WQI_lag1'].to_numpy()
        holdout_actuals = holdout_feat['WQI'].to_numpy()
        persistence_rmse = float(np.sqrt(mean_squared_error(holdout_actuals, persistence_preds)))

        # 2. ML model (CatBoost)
        feat_cols = ['WQI_lag1', 'WQI_lag2', 'WQI_lag3', 'Month']
        # Forward fill lag2 and lag3 if missing
        X_train = train_feat[feat_cols].ffill().bfill()
        y_train = train_feat['WQI']
        X_holdout = holdout_feat[feat_cols].ffill().bfill()

        model = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.03, verbose=False, random_seed=42)
        model.fit(X_train, y_train)
        ml_preds = model.predict(X_holdout)
        ml_rmse = float(np.sqrt(mean_squared_error(holdout_actuals, ml_preds)))

        advantage = persistence_rmse - ml_rmse  # positive means ML beats persistence
        winner = "ML" if ml_rmse < persistence_rmse else "Persistence"

        results.append({
            'Coverage_Level_Pct': int(cov * 100),
            'N_Train_Rows': len(train_sub),
            'N_Holdout_Rows': len(holdout_feat),
            'Persistence_RMSE': round(persistence_rmse, 3),
            'CatBoost_ML_RMSE': round(ml_rmse, 3),
            'ML_Advantage_RMSE': round(advantage, 3),
            'Winner': winner,
        })

    res_df = pd.DataFrame(results)
    out_file = OUT_DIR / 'simulation_sparsity_density_curve.csv'
    res_df.to_csv(out_file, index=False)
    print(res_df.to_string(index=False))
    print(f"\nSaved simulation curve to {out_file}")


if __name__ == '__main__':
    run_experiment()
