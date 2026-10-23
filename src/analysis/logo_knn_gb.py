"""
LOGO (Leave-One-Group-Out) cross-validation for KNN and Gradient Boosting.

Reads final_df_for_analysis.csv (the 42-row place×year dataset produced by
aggreate_train2.py / aggregate_train.py) and runs LOGO-CV for:
  - KNN (k=5)
  - Gradient Boosting (sklearn, default params)

Saves per-fold details and a summary to:
  src/results/rf_results/fold_details_KNN.csv
  src/results/rf_results/fold_details_GradientBoosting.csv
  src/results/rf_results/logo_model_comparison.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[2]

# Use the rf_results version of the file (it's the canonical one)
DATA_PATH = ROOT / "src" / "results" / "rf_results" / "final_df_for_analysis.csv"
OUT_DIR   = ROOT / "src" / "results" / "rf_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH.name}")
print(f"Columns: {list(df.columns)}")
print(f"Years: {sorted(df['Year'].unique())}")
print(f"Places: {df['Place'].nunique()}")

# Feature matrix: everything except WQI, Place, Year_Str
drop_cols = ["WQI", "Place"]
if "Year_Str" in df.columns:
    drop_cols.append("Year_Str")

X = df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())
y = df["WQI"].astype(float)
groups = df["Place"].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# ── LOGO evaluation helper ─────────────────────────────────────────────────
def run_logo(estimator, X, y, groups, name):
    logo = LeaveOneGroupOut()
    fold_rows = []
    all_preds, all_actuals = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        test_group = groups[test_idx[0]]

        if len(y_tr) == 0 or len(y_te) == 0:
            continue

        estimator.fit(X_tr, y_tr)
        y_pred = estimator.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae  = float(mean_absolute_error(y_te, y_pred))
        r2   = float(r2_score(y_te, y_pred)) if len(y_te) >= 2 else float("nan")

        fold_rows.append({
            "fold":       fold_idx,
            "test_group": test_group,
            "n_train":    len(y_tr),
            "n_test":     len(y_te),
            "rmse":       rmse,
            "mae":        mae,
            "r2":         r2,
        })

        all_preds.extend(y_pred)
        all_actuals.extend(y_te)

    fold_df = pd.DataFrame(fold_rows)

    # Pooled metrics (from raw predictions, not mean-of-folds)
    pooled_rmse = float(np.sqrt(mean_squared_error(all_actuals, all_preds)))
    pooled_r2   = float(r2_score(all_actuals, all_preds))
    pooled_mae  = float(mean_absolute_error(all_actuals, all_preds))

    summary = {
        "model":         name,
        "mean_rmse":     fold_df["rmse"].mean(),
        "std_rmse":      fold_df["rmse"].std(),
        "mean_mae":      fold_df["mae"].mean(),
        "std_mae":       fold_df["mae"].std(),
        "mean_r2":       fold_df["r2"].mean(),
        "std_r2":        fold_df["r2"].std(),
        "pooled_rmse":   pooled_rmse,
        "pooled_mae":    pooled_mae,
        "pooled_r2":     pooled_r2,
        "n_folds":       len(fold_df),
    }

    print(f"\n{name} LOGO results:")
    print(f"  Mean RMSE : {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
    print(f"  Mean MAE  : {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
    print(f"  Mean R²   : {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
    print(f"  Pooled RMSE: {pooled_rmse:.4f}  |  Pooled R²: {pooled_r2:.4f}")

    return fold_df, summary

# ── Run models ────────────────────────────────────────────────────────────
models = {
    "KNN": KNeighborsRegressor(n_neighbors=5, weights="uniform"),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}

all_summaries = []
for model_name, estimator in models.items():
    fold_df, summary = run_logo(estimator, X, y, groups, model_name)

    fold_out = OUT_DIR / f"fold_details_{model_name}.csv"
    fold_df.to_csv(fold_out, index=False)
    print(f"  Saved fold details -> {fold_out.name}")

    all_summaries.append(summary)

comparison_df = pd.DataFrame(all_summaries)
comp_out = OUT_DIR / "logo_model_comparison.csv"
comparison_df.to_csv(comp_out, index=False)
print(f"\n\nAll model comparison saved -> {comp_out.name}")
print("\n" + comparison_df[["model","mean_rmse","mean_mae","mean_r2","pooled_rmse","pooled_r2"]].to_string(index=False))

# ── Bootstrap SHAP stability check ─────────────────────────────────────────
shap_path = ROOT / "src" / "results" / "rf_results" / "shap_values.csv"
if shap_path.exists():
    shap_df = pd.read_csv(shap_path)
    feat_cols = [c for c in shap_df.columns if c not in ["WQI_actual", "Place"]]
    vals = shap_df[feat_cols].to_numpy()
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(500):
        idx = rng.choice(len(vals), size=len(vals), replace=True)
        boot_means.append(np.abs(vals[idx]).mean(axis=0))
    boot_means = np.array(boot_means)
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)
    shap_ci_df = pd.DataFrame({
        'feature': feat_cols,
        'mean_abs_shap': np.abs(vals).mean(axis=0),
        'ci_2.5': ci_lo,
        'ci_97.5': ci_hi,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    out_shap_ci = ROOT / "reproducibility_results" / "shap_bootstrap_ci.csv"
    shap_ci_df.to_csv(out_shap_ci, index=False)
    print(f"\nSaved SHAP bootstrap CIs -> {out_shap_ci}")
    print(shap_ci_df.to_string(index=False))

