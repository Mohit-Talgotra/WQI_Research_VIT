import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

load_dotenv()

DATASET_PATH = Path(os.environ["FORECAST_DATASET_PATH"])
MODEL_PATH   = Path(os.environ.get("FORECAST_MODEL_PATH", "data/wqi_forecaster.joblib"))
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("forecasting_results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "Block_encoded", "Location_encoded",
    "Year", "Month_sin", "Month_cos",
    "WQI_lag_1", "WQI_lag_2", "WQI_lag_3",
    "WQI_rolling_mean_2", "WQI_rolling_mean_3",
]
TARGET_COL = "WQI_target"

# Minimum samples a location must have to be included in CV evaluation
# Locations below this still contribute to training, just not evaluated
MIN_SAMPLES_FOR_EVAL = 3

print("=" * 70)
print("WQI FORECASTING MODEL — TRAINING & EVALUATION")
print("=" * 70)

# Load data
df = pd.read_csv(DATASET_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Block", "Location", "Date"]).reset_index(drop=True)

print(f"\n  Dataset rows     : {len(df):,}")
print(f"  Locations        : {df['Location'].nunique()}")
print(f"  Blocks           : {sorted(df['Block'].unique())}")
print(f"  Feature columns  : {FEATURE_COLS}")
print(f"  Target           : {TARGET_COL}")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()
groups = df["Location"].values

print(f"\n  Rows with any NaN feature : {X.isna().any(axis=1).sum()}")
print(f"  WQI target range          : {y.min():.1f} – {y.max():.1f}")

# Define models to compare
models = {
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "GradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ]),
}

# Leave-One-Group-Out cross validation
# Each fold holds out one location entirely — tests whether the model
# can predict WQI for a location it has never seen during training
print("\n" + "=" * 70)
print("LEAVE-ONE-LOCATION-OUT CROSS VALIDATION")
print("=" * 70)

logo = LeaveOneGroupOut()
all_results = {}

for model_name, pipeline in models.items():
    print(f"\n  Evaluating: {model_name}")

    fold_metrics = []
    all_preds    = []
    all_actuals  = []
    all_locs     = []
    all_dates    = []

    loc_counts = pd.Series(groups).value_counts()

    for train_idx, test_idx in logo.split(X, y, groups):
        test_loc = groups[test_idx[0]]

        # Skip evaluation for locations with too few samples
        if loc_counts[test_loc] < MIN_SAMPLES_FOR_EVAL:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds) if len(y_test) >= 2 else np.nan

        fold_metrics.append({
            "location" : test_loc,
            "block"    : df.iloc[test_idx[0]]["Block"],
            "n_test"   : len(y_test),
            "rmse"     : rmse,
            "mae"      : mae,
            "r2"       : r2,
        })

        all_preds.extend(preds)
        all_actuals.extend(y_test.values)
        all_locs.extend([test_loc] * len(y_test))
        all_dates.extend(df.iloc[test_idx]["Date"].values)

    fold_df = pd.DataFrame(fold_metrics)
    all_results[model_name] = {
        "fold_df"     : fold_df,
        "all_preds"   : all_preds,
        "all_actuals" : all_actuals,
        "all_locs"    : all_locs,
        "all_dates"   : all_dates,
    }

    print(f"    Folds evaluated : {len(fold_df)}")
    print(f"    Mean RMSE       : {fold_df['rmse'].mean():.3f} ± {fold_df['rmse'].std():.3f}")
    print(f"    Mean MAE        : {fold_df['mae'].mean():.3f} ± {fold_df['mae'].std():.3f}")
    print(f"    Mean R²         : {fold_df['r2'].mean():.3f} ± {fold_df['r2'].std():.3f}")
    print(f"\n    Per-location results:")
    print(fold_df[["location", "block", "n_test", "rmse", "mae", "r2"]]
          .sort_values("rmse")
          .to_string(index=False))

    fold_df.to_csv(RESULTS_DIR / f"cv_results_{model_name}.csv", index=False)

# Select best model
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

summary = []
for model_name, res in all_results.items():
    fd = res["fold_df"]
    summary.append({
        "model"     : model_name,
        "mean_rmse" : fd["rmse"].mean(),
        "std_rmse"  : fd["rmse"].std(),
        "mean_mae"  : fd["mae"].mean(),
        "mean_r2"   : fd["r2"].mean(),
    })

summary_df = pd.DataFrame(summary).sort_values("mean_rmse")
print(summary_df.to_string(index=False))

best_model_name = summary_df.iloc[0]["model"]
best_pipeline   = models[best_model_name]
print(f"\n  Best model: {best_model_name}")

# Train final model on ALL data
print("\n" + "=" * 70)
print("TRAINING FINAL MODEL ON ALL DATA")
print("=" * 70)

best_pipeline.fit(X, y)

# Save model + metadata
joblib.dump(best_pipeline, MODEL_PATH)
print(f"  Model saved: {MODEL_PATH}")

# Save feature column order (needed for prediction)
meta = {
    "feature_cols" : FEATURE_COLS,
    "target_col"   : TARGET_COL,
    "best_model"   : best_model_name,
    "n_locations"  : int(df["Location"].nunique()),
    "n_train_rows" : int(len(df)),
    "wqi_min"      : float(y.min()),
    "wqi_max"      : float(y.max()),
}
with open(MODEL_PATH.parent / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata saved: {MODEL_PATH.parent / 'model_meta.json'}")

# Plots
print("\nGENERATING PLOTS")

best_res     = all_results[best_model_name]
preds_arr    = np.array(best_res["all_preds"])
actuals_arr  = np.array(best_res["all_actuals"])
locs_arr     = np.array(best_res["all_locs"])
dates_arr    = pd.to_datetime(best_res["all_dates"])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"WQI Forecasting — {best_model_name} (LOGO CV)", fontsize=14, fontweight="bold")

# A: Predicted vs Actual
axes[0].scatter(actuals_arr, preds_arr, alpha=0.6, edgecolors="k", linewidth=0.4)
mn = min(actuals_arr.min(), preds_arr.min())
mx = max(actuals_arr.max(), preds_arr.max())
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect")
axes[0].set_xlabel("Actual WQI")
axes[0].set_ylabel("Predicted WQI")
axes[0].set_title("A: Predicted vs Actual")
axes[0].legend()
axes[0].grid(alpha=0.3)

# B: Residuals
residuals = actuals_arr - preds_arr
axes[1].scatter(preds_arr, residuals, alpha=0.6, edgecolors="k", linewidth=0.4)
axes[1].axhline(0, color="r", linestyle="--", lw=2)
axes[1].set_xlabel("Predicted WQI")
axes[1].set_ylabel("Residual (Actual − Predicted)")
axes[1].set_title("B: Residuals")
axes[1].grid(alpha=0.3)

# C: RMSE per location (sorted)
fold_df = all_results[best_model_name]["fold_df"].sort_values("rmse", ascending=True)
colors  = ["steelblue" if r <= 10 else "coral" for r in fold_df["rmse"]]
axes[2].barh(fold_df["location"], fold_df["rmse"], color=colors, edgecolor="black", linewidth=0.4)
axes[2].axvline(fold_df["rmse"].mean(), color="red", linestyle="--", lw=1.5,
                label=f"Mean RMSE={fold_df['rmse'].mean():.2f}")
axes[2].set_xlabel("RMSE")
axes[2].set_title("C: RMSE per Location")
axes[2].legend(fontsize=9)
axes[2].tick_params(axis="y", labelsize=7)
axes[2].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "forecasting_evaluation.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: forecasting_evaluation.png")

# Prediction function
print("\n" + "=" * 70)
print("PREDICTION INTERFACE")
print("=" * 70)

# Load encoder maps
enc_block    = pd.read_csv(DATASET_PATH.parent / "encoder_block.csv")
enc_location = pd.read_csv(DATASET_PATH.parent / "encoder_location.csv")

block_to_enc    = dict(zip(enc_block["Block"],    enc_block["Block_encoded"]))
location_to_enc = dict(zip(enc_location["Location"], enc_location["Location_encoded"]))

def predict_wqi(location: str, target_date: str, block: str = None) -> dict:
    """
    Predict WQI for a location at a future date.

    Parameters
    ----------
    location    : Gram panchayat name (must match training data exactly)
    target_date : Date string e.g. "2025-06-01"
    block       : Block name (optional, auto-detected if not given)

    Returns
    -------
    dict with predicted WQI and input features used
    """
    target_date = pd.Timestamp(target_date).replace(day=1)  # normalize to month start

    # Auto-detect block if not provided
    if block is None:
        match = df[df["Location"] == location]
        if match.empty:
            raise ValueError(f"Location '{location}' not found in training data. "
                             f"Available: {sorted(df['Location'].unique())}")
        block = match["Block"].iloc[0]

    # Encode
    if location not in location_to_enc:
        raise ValueError(f"Location '{location}' not in encoder. Check spelling.")
    if block not in block_to_enc:
        raise ValueError(f"Block '{block}' not in encoder.")

    loc_enc   = location_to_enc[location]
    block_enc = block_to_enc[block]

    # Get historical WQI for this location to build lag features
    loc_history = (
        df[df["Location"] == location]
        .sort_values("Date")[["Date", "WQI_target"]]
        .set_index("Date")["WQI_target"]
    )

    def get_lag(months_back: int) -> float:
        lag_date = target_date - pd.DateOffset(months=months_back)
        # Find closest available historical point at or before lag_date
        available = loc_history[loc_history.index <= lag_date]
        if available.empty:
            return np.nan
        # Use exact month if available, otherwise nearest
        if lag_date in available.index:
            return float(available[lag_date])
        return float(available.iloc[-1])

    lag1 = get_lag(1)
    lag2 = get_lag(2)
    lag3 = get_lag(3)

    rolling2 = np.nanmean([v for v in [lag1, lag2] if not np.isnan(v)]) if not (np.isnan(lag1) and np.isnan(lag2)) else np.nan
    rolling3 = np.nanmean([v for v in [lag1, lag2, lag3] if not np.isnan(v)]) if not all(np.isnan(v) for v in [lag1, lag2, lag3]) else np.nan

    month     = target_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    features = pd.DataFrame([{
        "Block_encoded"      : block_enc,
        "Location_encoded"   : loc_enc,
        "Year"               : target_date.year,
        "Month_sin"          : month_sin,
        "Month_cos"          : month_cos,
        "WQI_lag_1"          : lag1,
        "WQI_lag_2"          : lag2,
        "WQI_lag_3"          : lag3,
        "WQI_rolling_mean_2" : rolling2,
        "WQI_rolling_mean_3" : rolling3,
    }])[FEATURE_COLS]

    predicted_wqi = float(best_pipeline.predict(features)[0])

    # WQI classification
    if predicted_wqi <= 25:
        category = "Excellent"
    elif predicted_wqi <= 50:
        category = "Good"
    elif predicted_wqi <= 75:
        category = "Poor"
    elif predicted_wqi <= 100:
        category = "Very Poor"
    else:
        category = "Unsuitable"

    return {
        "location"      : location,
        "block"         : block,
        "target_date"   : str(target_date.date()),
        "predicted_wqi" : round(predicted_wqi, 2),
        "category"      : category,
        "lag_1_wqi"     : round(lag1, 2) if not np.isnan(lag1) else None,
        "lag_2_wqi"     : round(lag2, 2) if not np.isnan(lag2) else None,
        "lag_3_wqi"     : round(lag3, 2) if not np.isnan(lag3) else None,
    }


# Demo predictions
print("\n  Sample predictions (using trained model):\n")

demo_locations = df.groupby("Location").size().nlargest(5).index.tolist()

for loc in demo_locations:
    block = df[df["Location"] == loc]["Block"].iloc[0]
    result = predict_wqi(loc, "2025-06-01", block=block)
    print(f"  {loc} ({block})")
    print(f"    → Predicted WQI: {result['predicted_wqi']}  [{result['category']}]")
    print(f"    → Lags used: t-1={result['lag_1_wqi']}, t-2={result['lag_2_wqi']}, t-3={result['lag_3_wqi']}\n")

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to: {RESULTS_DIR}/")
print(f"Model saved to  : {MODEL_PATH}")
print(f"\nTo predict WQI for any location:")
print(f"  result = predict_wqi('Valaiyakaranur', '2025-09-01', block='Ayodhiyapattanam')")
print(f"  print(result)")