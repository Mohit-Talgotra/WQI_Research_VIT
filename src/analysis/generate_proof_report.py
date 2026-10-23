import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

# Load raw monthly data
monthly_path = r"c:\Mohit\Repositories\WQI_Research_VIT\src\data\monthly_wqi_parameter_interpolated_spline.csv"
df_monthly = pd.read_csv(monthly_path)

print("==========================================================================")
print("PROOF PART 1: DATASET VERIFICATION")
print("==========================================================================")
print(f"Dataset path: {monthly_path}")
print(f"Total Rows: {len(df_monthly)}")
print(f"Total Unique Locations: {df_monthly['Location'].nunique()}")
print(f"Date Range: {df_monthly['Date'].min()} to {df_monthly['Date'].max()}")

# --------------------------------------------------------------------------
# PROOF PART 2: SURROGATE SPATIAL CLASSIFICATION (LOGO CV)
# --------------------------------------------------------------------------
bins = [-np.inf, 100, 200, np.inf]
labels = [0, 1, 2] # 0: Good (<100), 1: Poor (100-200), 2: Very Poor (>200)
df_monthly['BIS_Class'] = pd.cut(df_monthly['WQI_mean'], bins=bins, labels=labels).astype(int)

cheap_features = ['pH_mean', 'TDS_mean', 'Hardness_mean', 'Chloride_mean']
df_cls = df_monthly.dropna(subset=['BIS_Class', 'Location'] + cheap_features).copy()

X_cls = df_cls[cheap_features]
y_cls = df_cls['BIS_Class']
groups_cls = df_cls['Location']

logo = LeaveOneGroupOut()
y_true, y_pred = [], []

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

for train_idx, test_idx in logo.split(X_cls, y_cls, groups_cls):
    rf.fit(X_cls.iloc[train_idx], y_cls.iloc[train_idx])
    pred = rf.predict(X_cls.iloc[test_idx])
    y_true.extend(y_cls.iloc[test_idx])
    y_pred.extend(pred)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

print("\n==========================================================================")
print("PROOF PART 2: SPATIAL SURROGATE CLASSIFICATION LOGO RESULTS")
print("==========================================================================")
print(f"Features used (ONLY 4 cheap parameters): {cheap_features}")
print(f"Features EXCLUDED (expensive lab parameters): ['Fluoride', 'Nitrate', 'Sulphate', 'Alkalinity']")
print(f"Total Evaluated Samples: {len(y_true)} across {len(np.unique(groups_cls))} held-out locations")
print(f"Overall Accuracy: {acc*100:.2f}% ({np.sum(np.array(y_true) == np.array(y_pred))}/{len(y_true)} correct predictions)")
print(f"Weighted F1-Score: {f1:.4f}")
print("\nConfusion Matrix (Rows = Actual BIS Class, Columns = Predicted BIS Class):")
print("                 Pred Good (<100)  Pred Poor (100-200)  Pred Very Poor (>200)")
print(f"Actual Good        {cm[0,0]:14d}  {cm[0,1]:19d}  {cm[0,2]:20d}")
print(f"Actual Poor        {cm[1,0]:14d}  {cm[1,1]:19d}  {cm[1,2]:20d}")
print(f"Actual Very Poor   {cm[2,0]:14d}  {cm[2,1]:19d}  {cm[2,2]:20d}")

print("\nDetailed Per-Class Performance:")
print(classification_report(y_true, y_pred, target_names=['Good (<100)', 'Poor (100-200)', 'Very Poor (>200)']))

# --------------------------------------------------------------------------
# PROOF PART 3: MONTHLY 1-MONTH-AHEAD FORECASTING (2024 HOLDOUT)
# --------------------------------------------------------------------------
df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])
df_monthly = df_monthly.sort_values(['Location', 'Date']).reset_index(drop=True)

for lag in [1, 2, 3]:
    df_monthly[f'WQI_lag_{lag}'] = df_monthly.groupby('Location')['WQI_mean'].shift(lag)
    df_monthly[f'F_lag_{lag}'] = df_monthly.groupby('Location')['Fluoride_mean'].shift(lag)
    df_monthly[f'TDS_lag_{lag}'] = df_monthly.groupby('Location')['TDS_mean'].shift(lag)

df_forecast = df_monthly.dropna(subset=['WQI_mean', 'WQI_lag_1', 'WQI_lag_2', 'WQI_lag_3']).copy()

train_df = df_forecast[df_forecast['Date'] < '2024-01-01']
test_df = df_forecast[(df_forecast['Date'] >= '2024-01-01') & (df_forecast['Date'] < '2025-01-01')]

feature_cols = ['WQI_lag_1', 'WQI_lag_2', 'WQI_lag_3', 'F_lag_1', 'TDS_lag_1', 'pH_mean', 'Hardness_mean', 'Chloride_mean', 'Alkalinity_mean']

X_train = train_df[feature_cols]
y_train = train_df['WQI_mean']
X_test = test_df[feature_cols]
y_test = test_df['WQI_mean']

print("==========================================================================")
print("PROOF PART 3: TEMPORAL HOLDOUT FORECASTING RESULTS (2024 TEST SET)")
print("==========================================================================")
print(f"Training Period: 2022-01-01 to 2023-12-01 ({len(train_df)} monthly samples)")
print(f"Testing Period: 2024-01-01 to 2024-12-01 ({len(test_df)} monthly samples)")

models = {
    'Linear Regression': LinearRegression(),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:22s} -> R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Persistence
persist_preds = test_df['WQI_lag_1']
r2_p = r2_score(y_test, persist_preds)
rmse_p = root_mean_squared_error(y_test, persist_preds)
mae_p = mean_absolute_error(y_test, persist_preds)
print(f"{'Persistence (Lag-1)':22s} -> R2: {r2_p:.4f}, RMSE: {rmse_p:.2f}, MAE: {mae_p:.2f}")
print("==========================================================================")
