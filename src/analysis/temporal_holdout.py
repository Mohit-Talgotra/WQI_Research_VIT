"""
Temporal holdout analysis — reproduces Table 4.

Train set  : rows where Year ∈ {2022, 2023}
Test set   : rows where Year == 2024

Models evaluated: Gradient Boosting, KNN, Random Forest, Linear Regression

Outputs:
  src/results/temporal_results/temporal_holdout_comparison.csv
  src/results/temporal_results/temporal_holdout_predictions.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "src" / "results" / "rf_results" / "final_df_for_analysis.csv"
OUT_DIR   = ROOT / "src" / "results" / "temporal_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")
print(f"Year distribution:\n{df['Year'].value_counts().sort_index()}")

# ── Feature prep ─────────────────────────────────────────────────────────
drop_cols = ["WQI", "Place"]
if "Year_Str" in df.columns:
    drop_cols.append("Year_Str")

feature_cols = [c for c in df.columns if c not in drop_cols]

# Ensure Year is integer
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)

train_mask = df["Year"].isin([2022, 2023])
test_mask  = df["Year"] == 2024

X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
col_means = X_all.mean()
X_all = X_all.fillna(col_means)

X_train = X_all[train_mask]
X_test  = X_all[test_mask]
y_train = df.loc[train_mask, "WQI"].astype(float)
y_test  = df.loc[test_mask,  "WQI"].astype(float)

# KNN needs scaled features (distance-based); other models use the raw values above.
knn_scaler = StandardScaler()
X_train_knn = knn_scaler.fit_transform(X_train)
X_test_knn  = knn_scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"Train years: {sorted(df.loc[train_mask, 'Year'].unique())}")
print(f"Test year : {sorted(df.loc[test_mask,  'Year'].unique())}")

# ── Models ────────────────────────────────────────────────────────────────
models = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
    "KNN":              KNeighborsRegressor(n_neighbors=3, weights="distance"),
    "RandomForest":     RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "LinearRegression": LinearRegression(),
}

# ── Evaluate ─────────────────────────────────────────────────────────────
rows = []
pred_rows = []

for name, model in models.items():
    Xtr, Xte = (X_train_knn, X_test_knn) if name == "KNN" else (X_train, X_test)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2,
                 "n_train": len(X_train), "n_test": len(X_test)})

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

    for place, actual, pred in zip(
        df.loc[test_mask, "Place"].values, y_test.values, y_pred
    ):
        pred_rows.append({
            "model": name, "Place": place, "Year": 2024,
            "Actual_WQI": actual, "Predicted_WQI": pred,
            "Residual": actual - pred,
        })

# ── Save ─────────────────────────────────────────────────────────────────
summary_df  = pd.DataFrame(rows)
pred_df     = pd.DataFrame(pred_rows)

summary_out = OUT_DIR / "temporal_holdout_comparison.csv"
pred_out    = OUT_DIR / "temporal_holdout_predictions.csv"

summary_df.to_csv(summary_out, index=False)
pred_df.to_csv(pred_out, index=False)

print(f"\n\nResults saved:")
print(f"  {summary_out.name}")
print(f"  {pred_out.name}")
print("\n" + summary_df[["model","rmse","mae","r2"]].to_string(index=False))
