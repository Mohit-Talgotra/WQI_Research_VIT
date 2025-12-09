import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, RandomizedSearchCV
import shap
import xgboost as xgb
from catboost import CatBoostRegressor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path(os.environ["RESULTS_FILE_PATH"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment
load_dotenv()
file = os.environ.get("WQI_AGGREGATE_FILE_PATH")
if not file:
    logger.error("WQI_AGGREGATE_FILE_PATH not set in .env file")
    sys.exit(1)

if not Path(file).exists():
    logger.error(f"File not found: {file}")
    sys.exit(1)

logger.info(f"Loading data from: {file}")

# Define sheets to load
sheets = {
    "2022-2023 WQI Calculation": "2022-2023",
    "2023-2024 WQI Calculation": "2023-2024",
    "2024-2025 WQI Calculation": "2024-2025"
}

# Load and concatenate data from multiple sheets
dfs = []
for sheet_name, year in sheets.items():
    try:
        tmp_df = pd.read_excel(file, sheet_name=sheet_name)
        tmp_df["Year"] = year
        dfs.append(tmp_df)
        logger.info(f"Loaded {len(tmp_df)} rows from sheet: {sheet_name}")
    except Exception as e:
        logger.error(f"Failed to load sheet {sheet_name}: {e}")
        sys.exit(1)

df = pd.concat(dfs, ignore_index=True)

# Clean column names and data
df.columns = df.columns.str.strip()

# Validate required columns exist
required_cols = ["Place", "Parameter", "Mean Value (Vn)"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    logger.error(f"Missing required columns: {missing_cols}")
    sys.exit(1)

df["Place"] = df["Place"].astype(str).str.strip()
df["Parameter"] = df["Parameter"].astype(str).str.strip()
df["Year"] = df["Year"].astype(str).str.strip()

# Clean numeric values
df["Mean Value (Vn)"] = (
    df["Mean Value (Vn)"]
    .astype(str)
    .str.replace(",", "")
    .str.strip()
    .replace("", np.nan)
)
df["Mean Value (Vn)"] = pd.to_numeric(df["Mean Value (Vn)"], errors="coerce")

logger.info(f"Total records after cleaning: {len(df)}")
logger.info(f"Unique places: {df['Place'].nunique()}")
logger.info(f"Unique parameters: {df['Parameter'].nunique()}")

# Pivot to create feature matrix
pivot_df = df.pivot_table(
    index=["Year", "Place"],
    columns="Parameter",
    values="Mean Value (Vn)",
    aggfunc="first"
).reset_index()

logger.info(f"Pivot table shape: {pivot_df.shape}")

# Load WQI target values
try:
    wqi_raw = pd.read_excel(file, sheet_name="Final WQI Data- 3 Years", header=1)
except Exception as e:
    logger.error(f"Failed to load WQI data: {e}")
    sys.exit(1)

wqi_raw.columns = wqi_raw.columns.str.strip()

# Validate Place column exists in WQI data
if "Place" not in wqi_raw.columns:
    logger.error("'Place' column not found in WQI data")
    sys.exit(1)

# Extract year columns
year_cols = [c for c in wqi_raw.columns if str(c).startswith("202")]
if not year_cols:
    logger.error("No year columns found in WQI data")
    sys.exit(1)

logger.info(f"Found year columns: {year_cols}")

# Reshape WQI data to long format
wqi_long = wqi_raw.melt(
    id_vars=["Place"],
    value_vars=year_cols,
    var_name="Year",
    value_name="WQI"
)
wqi_long["Place"] = wqi_long["Place"].astype(str).str.strip()
wqi_long["Year"] = wqi_long["Year"].astype(str).str.strip()

# Merge features with target
final_df = pivot_df.merge(wqi_long, on=["Year", "Place"], how="left")

# Check for duplicates
duplicates = final_df.duplicated(subset=["Year", "Place"]).sum()
if duplicates > 0:
    logger.warning(f"Found {duplicates} duplicate (Year, Place) combinations - keeping first")
    final_df = final_df.drop_duplicates(subset=["Year", "Place"], keep="first")

# Remove rows without WQI values
initial_len = len(final_df)
final_df = final_df.dropna(subset=["WQI"]).reset_index(drop=True)
logger.info(f"Removed {initial_len - len(final_df)} rows with missing WQI values")

if len(final_df) == 0:
    logger.error("No valid data remaining after merge")
    sys.exit(1)

# Extract year as integer (more robust)
final_df["Year_Str"] = final_df["Year"]
try:
    final_df["Year"] = final_df["Year"].str.extract(r"(\d{4})")[0].astype(int)
except Exception as e:
    logger.error(f"Failed to extract year from format: {e}")
    sys.exit(1)

logger.info(f"Final dataset: {len(final_df)} samples across {final_df['Place'].nunique()} places")

# Data quality checks
logger.info("\n=== Data Quality Checks ===")
logger.info(f"WQI range: [{final_df['WQI'].min():.2f}, {final_df['WQI'].max():.2f}]")
logger.info(f"WQI mean ± std: {final_df['WQI'].mean():.2f} ± {final_df['WQI'].std():.2f}")

# Check for outliers in WQI
q1, q3 = final_df["WQI"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = final_df[(final_df["WQI"] < q1 - 3*iqr) | (final_df["WQI"] > q3 + 3*iqr)]
if len(outliers) > 0:
    logger.warning(f"Found {len(outliers)} potential WQI outliers (3*IQR rule)")

# Prepare features and target
X = final_df.drop(columns=["WQI", "Place", "Year_Str"])
X = X.apply(pd.to_numeric, errors="coerce")

# Check missing values per feature
missing_pct = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 50]
if len(high_missing) > 0:
    logger.warning(f"Features with >50% missing values:\n{high_missing}")

# Impute missing values with mean
X = X.fillna(X.mean())

# Check for features with zero variance
zero_var_features = X.columns[X.std() == 0].tolist()
if zero_var_features:
    logger.warning(f"Removing {len(zero_var_features)} zero-variance features: {zero_var_features}")
    X = X.drop(columns=zero_var_features)

y = final_df["WQI"].astype(float)
groups = final_df["Place"].values

logger.info(f"\nFeature matrix: {X.shape}")
logger.info(f"Features: {list(X.columns)}")
logger.info(f"Target variable: {len(y)} samples")
logger.info(f"Groups (places): {len(np.unique(groups))}")

# Check minimum samples per group
group_counts = pd.Series(groups).value_counts()
min_samples = group_counts.min()
if min_samples < 2:
    logger.warning(f"Some groups have only {min_samples} sample(s) - CV may be unstable")

# Save processed data for analysis
final_df.to_csv(RESULTS_DIR / "final_df_for_analysis.csv", index=False)
logger.info(f"Saved processed data to {RESULTS_DIR / 'final_df_for_analysis.csv'}")

# Model Evaluation Function with Detailed Metrics
def evaluate_model(estimator, X, y, groups, name):
    logo = LeaveOneGroupOut()
    
    fold_results = []
    all_predictions = []
    all_actuals = []
    all_groups_test = []
    
    logger.info(f"\nEvaluating {name}...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_group = groups[test_idx[0]]
        
        if len(y_train) == 0 or len(y_test) == 0:
            logger.warning(f"Skipping fold {fold_idx} due to empty train/test set")
            continue
        
        # Fit and predict
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # R² requires at least 2 samples
        r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan
        
        fold_results.append({
            "fold": fold_idx,
            "test_group": test_group,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # Store predictions for later analysis
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        all_groups_test.extend([test_group] * len(y_test))
    
    # Aggregate results
    fold_df = pd.DataFrame(fold_results)
    
    summary = {
        "model": name,
        "mean_rmse": fold_df["rmse"].mean(),
        "std_rmse": fold_df["rmse"].std(),
        "mean_mae": fold_df["mae"].mean(),
        "std_mae": fold_df["mae"].std(),
        "mean_r2": fold_df["r2"].mean(),
        "std_r2": fold_df["r2"].std(),
        "n_folds": len(fold_df)
    }
    
    logger.info(f"  RMSE: {summary['mean_rmse']:.3f} ± {summary['std_rmse']:.3f}")
    logger.info(f"  MAE:  {summary['mean_mae']:.3f} ± {summary['std_mae']:.3f}")
    logger.info(f"  R²:   {summary['mean_r2']:.3f} ± {summary['std_r2']:.3f}")
    
    return summary, fold_df, all_predictions, all_actuals, all_groups_test


# Model Training and Comparison
RANDOM_STATE = 42
results = []
fold_details = {}
prediction_details = {}

logger.info("\n" + "="*80)
logger.info("MODEL TRAINING AND EVALUATION")
logger.info("="*80)

# 1. Random Forest Baseline
logger.info("\n1. Training Random Forest baseline...")
rf_model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
res_rf, fold_rf, pred_rf, actual_rf, groups_rf = evaluate_model(
    rf_model, X, y, groups, "RandomForest_baseline"
)
results.append(res_rf)
fold_details["RandomForest_baseline"] = fold_rf
prediction_details["RandomForest_baseline"] = (pred_rf, actual_rf, groups_rf)

# 2. XGBoost Default (if available)
if xgb is not None:
    logger.info("\n2. Training XGBoost default...")
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    res_xgb, fold_xgb, pred_xgb, actual_xgb, groups_xgb = evaluate_model(
        xgb_model, X, y, groups, "XGBoost_default"
    )
    results.append(res_xgb)
    fold_details["XGBoost_default"] = fold_xgb
    prediction_details["XGBoost_default"] = (pred_xgb, actual_xgb, groups_xgb)
else:
    logger.warning("XGBoost not available - skipping")

# 3. CatBoost Default (if available)
if CatBoostRegressor is not None:
    logger.info("\n3. Training CatBoost default...")
    cb_model = CatBoostRegressor(
        verbose=0,
        random_state=RANDOM_STATE
    )
    res_cb, fold_cb, pred_cb, actual_cb, groups_cb = evaluate_model(
        cb_model, X, y, groups, "CatBoost_default"
    )
    results.append(res_cb)
    fold_details["CatBoost_default"] = fold_cb
    prediction_details["CatBoost_default"] = (pred_cb, actual_cb, groups_cb)
else:
    logger.warning("CatBoost not available - skipping")


# Hyperparameter Tuning with GroupKFold (Fixes Data Leakage)
logger.info("\n" + "="*80)
logger.info("HYPERPARAMETER TUNING (with GroupKFold to prevent leakage)")
logger.info("="*80)

# Use GroupKFold instead of regular CV to respect group structure
group_cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

tuning_results = {}

# 4. XGBoost Tuning
if xgb is not None:
    logger.info("\n4. Tuning XGBoost...")
    xgb_param_space = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
        xgb_param_space,
        n_iter=10,
        scoring="neg_mean_squared_error",
        cv=group_cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1
    )
    
    xgb_search.fit(X, y, groups=groups)
    logger.info(f"  Best params: {xgb_search.best_params_}")
    logger.info(f"  Best CV score: {-xgb_search.best_score_:.3f} (MSE)")
    
    # Store best params for later use
    tuning_results["xgboost"] = xgb_search.best_params_
    
    # Evaluate tuned model with LOGO
    res_xgb_tuned, fold_xgb_tuned, pred_xgb_t, actual_xgb_t, groups_xgb_t = evaluate_model(
        xgb_search.best_estimator_, X, y, groups, "XGBoost_tuned"
    )
    results.append(res_xgb_tuned)
    fold_details["XGBoost_tuned"] = fold_xgb_tuned
    prediction_details["XGBoost_tuned"] = (pred_xgb_t, actual_xgb_t, groups_xgb_t)

# 5. CatBoost Tuning
if CatBoostRegressor is not None:
    logger.info("\n5. Tuning CatBoost...")
    cb_param_space = {
        "iterations": [200, 500, 800],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5]
    }
    
    cb_search = RandomizedSearchCV(
        CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
        cb_param_space,
        n_iter=10,
        scoring="neg_mean_squared_error",
        cv=group_cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1
    )
    
    cb_search.fit(X, y, groups=groups)
    logger.info(f"  Best params: {cb_search.best_params_}")
    logger.info(f"  Best CV score: {-cb_search.best_score_:.3f} (MSE)")
    
    # Store best params
    tuning_results["catboost"] = cb_search.best_params_
    
    # Evaluate tuned model
    res_cb_tuned, fold_cb_tuned, pred_cb_t, actual_cb_t, groups_cb_t = evaluate_model(
        cb_search.best_estimator_, X, y, groups, "CatBoost_tuned"
    )
    results.append(res_cb_tuned)
    fold_details["CatBoost_tuned"] = fold_cb_tuned
    prediction_details["CatBoost_tuned"] = (pred_cb_t, actual_cb_t, groups_cb_t)


# Save Results and Select Best Model
logger.info("\n" + "="*80)
logger.info("RESULTS SUMMARY")
logger.info("="*80)

# Save model comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("mean_rmse")
results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
logger.info(f"\nModel comparison saved to {RESULTS_DIR / 'model_comparison.csv'}")
logger.info("\n" + results_df.to_string(index=False))

# Save fold-level details for each model
for model_name, fold_df in fold_details.items():
    fold_df.to_csv(RESULTS_DIR / f"fold_details_{model_name}.csv", index=False)

# Select best model based on RMSE
best_model_name = results_df.iloc[0]["model"]
logger.info(f"\n🏆 Best model: {best_model_name}")
logger.info(f"   RMSE: {results_df.iloc[0]['mean_rmse']:.3f} ± {results_df.iloc[0]['std_rmse']:.3f}")
logger.info(f"   R²: {results_df.iloc[0]['mean_r2']:.3f} ± {results_df.iloc[0]['std_r2']:.3f}")

# Train final model on all data using best parameters
logger.info(f"\nTraining final {best_model_name} model on all data...")

if "XGBoost" in best_model_name and xgb is not None:
    if "tuned" in best_model_name and "xgboost" in tuning_results:
        final_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **tuning_results["xgboost"]
        )
    else:
        final_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
elif "CatBoost" in best_model_name and CatBoostRegressor is not None:
    if "tuned" in best_model_name and "catboost" in tuning_results:
        final_model = CatBoostRegressor(
            random_state=RANDOM_STATE,
            verbose=0,
            **tuning_results["catboost"]
        )
    else:
        final_model = CatBoostRegressor(
            random_state=RANDOM_STATE,
            verbose=0
        )
else:
    final_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

final_model.fit(X, y)
joblib.dump(final_model, RESULTS_DIR / "best_model.joblib")
logger.info(f"Best model saved to {RESULTS_DIR / 'best_model.joblib'}")


# SHAP Analysis for Model Interpretability
logger.info("\n" + "="*80)
logger.info("SHAP ANALYSIS")
logger.info("="*80)

try:
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    
    logger.info("Generating SHAP visualizations...")
    
    # 1. SHAP Summary Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance - {best_model_name}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: shap_summary_bar.png")
    
    # 2. SHAP Beeswarm Plot (Feature Effects)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Feature Effects - {best_model_name}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: shap_beeswarm.png")
    
    # 3. Force plots for top 3 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    
    logger.info("\nGenerating force plots for top 3 features:")
    for rank, feat_idx in enumerate(top_features_idx, 1):
        feature_name = X.columns[feat_idx]
        
        # Pick sample with median WQI for more representative visualization
        median_idx = (y - y.median()).abs().argmin()
        
        shap_vector = shap_values[median_idx]
        feature_values = X.iloc[median_idx]
        
        fig = shap.force_plot(
            explainer.expected_value,
            shap_vector,
            feature_values,
            matplotlib=True,
            show=False
        )
        
        # Clean feature name for filename
        clean_name = feature_name.replace(" ", "_").replace("/", "_")
        outname = f"shap_force_TOP{rank}_{clean_name}.png"
        fig.savefig(RESULTS_DIR / outname, bbox_inches="tight", dpi=200)
        plt.close(fig)
        logger.info(f"  {rank}. {feature_name} (saved: {outname})")
    
    # 4. Save SHAP values for further analysis
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df["WQI_actual"] = y.values
    shap_df["Place"] = groups
    shap_df.to_csv(RESULTS_DIR / "shap_values.csv", index=False)
    logger.info(f"\n  SHAP values saved to: shap_values.csv")
    
except Exception as e:
    logger.error(f"SHAP analysis failed: {e}")


# Prediction Analysis and Residual Plots
logger.info("\n" + "="*80)
logger.info("PREDICTION ANALYSIS")
logger.info("="*80)

# Analyze predictions from best model
if best_model_name in prediction_details:
    preds, actuals, test_groups = prediction_details[best_model_name]
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        "Actual": actuals,
        "Predicted": preds,
        "Place": test_groups,
        "Residual": np.array(actuals) - np.array(preds)
    })
    pred_df.to_csv(RESULTS_DIR / "predictions_best_model.csv", index=False)
    logger.info(f"Predictions saved to: predictions_best_model.csv")
    
    # Residual analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs Actual
    axes[0].scatter(actuals, preds, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(min(actuals), min(preds))
    max_val = max(max(actuals), max(preds))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual WQI', fontsize=12)
    axes[0].set_ylabel('Predicted WQI', fontsize=12)
    axes[0].set_title(f'Predicted vs Actual - {best_model_name}', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residual plot
    axes[1].scatter(preds, pred_df["Residual"], alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted WQI', fontsize=12)
    axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prediction_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: prediction_analysis.png")
    
    # Per-place error analysis
    place_errors = pred_df.groupby("Place")["Residual"].agg(['mean', 'std', 'count'])
    place_errors.columns = ['Mean_Residual', 'Std_Residual', 'N_Samples']
    place_errors = place_errors.sort_values('Mean_Residual', ascending=False)
    place_errors.to_csv(RESULTS_DIR / "per_place_errors.csv")
    logger.info("Saved: per_place_errors.csv")
    
    logger.info("\nPlaces with largest prediction errors:")
    logger.info(place_errors.head(5).to_string())


# WQI Trend Visualizations
logger.info("\n" + "="*80)
logger.info("GENERATING WQI TREND VISUALIZATIONS")
logger.info("="*80)

# 1. WQI trends over time (line plot)
plt.figure(figsize=(14, 7))
for place, grp in final_df.groupby("Place"):
    plt.plot(grp["Year"], grp["WQI"], marker="o", label=place, linewidth=2, markersize=6)

plt.xlabel("Year", fontsize=13)
plt.ylabel("WQI", fontsize=13)
plt.title("WQI Trends by Location (2022-2024)", fontsize=14, pad=15)
plt.xticks(sorted(final_df["Year"].unique()))
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "wqi_trends.png", dpi=200, bbox_inches="tight")
plt.close()
logger.info("Saved: wqi_trends.png")

# 2. WQI heatmap
heat = final_df.pivot(index="Place", columns="Year", values="WQI")
plt.figure(figsize=(10, max(8, len(heat) * 0.5)))
sns.heatmap(
    heat,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    cbar_kws={"label": "WQI"},
    linewidths=0.5,
    linecolor='gray'
)
plt.title("WQI Heatmap by Location and Year", fontsize=14, pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Place", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "wqi_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
logger.info("Saved: wqi_heatmap.png")

# 3. Distribution of WQI across years
plt.figure(figsize=(10, 6))
final_df.boxplot(column="WQI", by="Year", figsize=(10, 6), patch_artist=True)
plt.suptitle("")  # Remove default title
plt.title("WQI Distribution by Year", fontsize=14, pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("WQI", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "wqi_distribution_by_year.png", dpi=200, bbox_inches="tight")
plt.close()
logger.info("Saved: wqi_distribution_by_year.png")

# 4. Summary statistics table
summary_stats = final_df.groupby("Year")["WQI"].agg([
    ('Count', 'count'),
    ('Mean', 'mean'),
    ('Std', 'std'),
    ('Min', 'min'),
    ('25%', lambda x: x.quantile(0.25)),
    ('Median', 'median'),
    ('75%', lambda x: x.quantile(0.75)),
    ('Max', 'max')
]).round(2)
summary_stats.to_csv(RESULTS_DIR / "wqi_summary_statistics.csv")
logger.info("Saved: wqi_summary_statistics.csv")

logger.info("\nWQI Summary Statistics by Year:")
logger.info("\n" + summary_stats.to_string())


# Feature Correlation Analysis
logger.info("\n" + "="*80)
logger.info("FEATURE CORRELATION ANALYSIS")
logger.info("="*80)

# Correlation with WQI
feature_correlations = X.corrwith(y).sort_values(ascending=False)
feature_correlations.to_csv(RESULTS_DIR / "feature_correlations_with_wqi.csv", header=["Correlation"])
logger.info("Saved: feature_correlations_with_wqi.csv")

logger.info("\nTop 10 features correlated with WQI:")
logger.info(feature_correlations.head(10).to_string())

# Correlation heatmap for top features
if len(X.columns) > 3:
    top_n = min(15, len(X.columns))
    top_features = feature_correlations.abs().nlargest(top_n).index.tolist()
    
    plt.figure(figsize=(12, 10))
    corr_matrix = X[top_features].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"}
    )
    plt.title(f"Feature Correlation Matrix (Top {top_n} Features)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_correlation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: feature_correlation_matrix.png")


# Final Summary Report
logger.info("\n" + "="*80)
logger.info("ANALYSIS COMPLETE - SUMMARY")
logger.info("="*80)

logger.info(f"""
    Dataset Summary:
   - Total samples: {len(final_df)}
   - Unique locations: {final_df['Place'].nunique()}
   - Years covered: {sorted(final_df['Year'].unique())}
   - Features: {len(X.columns)}

    Best Model: {best_model_name}
   - Mean RMSE: {results_df.iloc[0]['mean_rmse']:.3f} ± {results_df.iloc[0]['std_rmse']:.3f}
   - Mean MAE: {results_df.iloc[0]['mean_mae']:.3f} ± {results_df.iloc[0]['std_mae']:.3f}
   - Mean R²: {results_df.iloc[0]['mean_r2']:.3f} ± {results_df.iloc[0]['std_r2']:.3f}

All outputs saved to: {RESULTS_DIR.absolute()}
""")

logger.info("="*80)
logger.info("Analysis completed successfully!")
logger.info("="*80)