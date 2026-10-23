import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut

out_dir = r"c:\Mohit\Repositories\WQI_Research_VIT\reproducibility_results\monthly_results"
os.makedirs(out_dir, exist_ok=True)

print("--- Running Monthly Forecasting and Spatial Classification Pipeline ---")

# ==========================================
# PART 1: 1-Month-Ahead Monthly Forecasting
# ==========================================
monthly_path = r"c:\Mohit\Repositories\WQI_Research_VIT\src\data\monthly_wqi_parameter_interpolated_spline.csv"
df_monthly = pd.read_csv(monthly_path)
df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])
df_monthly = df_monthly.sort_values(['Location', 'Date']).reset_index(drop=True)

# Build monthly lag features
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

reg_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingRegressor(random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
}

forecast_results = []

for name, model in reg_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    forecast_results.append({'Model': name, 'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 3)})

# Add Persistence Baseline
persist_preds = test_df['WQI_lag_1']
r2_p = r2_score(y_test, persist_preds)
rmse_p = root_mean_squared_error(y_test, persist_preds)
mae_p = mean_absolute_error(y_test, persist_preds)
forecast_results.append({'Model': 'Persistence Baseline (Lag-1)', 'RMSE': round(rmse_p, 2), 'MAE': round(mae_p, 2), 'R2': round(r2_p, 3)})

df_forecast_res = pd.DataFrame(forecast_results)
df_forecast_res.to_csv(os.path.join(out_dir, "monthly_forecasting_2024_holdout.csv"), index=False)
print("\n[Monthly 2024 Holdout Forecasting Results]")
print(df_forecast_res.to_string(index=False))

# ==========================================
# PART 2: Spatial BIS Hazard Classification (LOGO CV)
# ==========================================
# Class 0: Good (< 100), Class 1: Poor (100 - 200), Class 2: Very Poor (> 200)
bins = [-np.inf, 100, 200, np.inf]
labels = [0, 1, 2]
df_monthly['BIS_Class'] = pd.cut(df_monthly['WQI_mean'], bins=bins, labels=labels).astype(int)

class_features = ['pH_mean', 'TDS_mean', 'Hardness_mean', 'Chloride_mean', 'Fluoride_mean', 'Alkalinity_mean', 'Sulphate_mean', 'Nitrate_mean']
df_cls = df_monthly.dropna(subset=['BIS_Class', 'Location'] + class_features).copy()

X_cls = df_cls[class_features]
y_cls = df_cls['BIS_Class']
groups_cls = df_cls['Location']

cls_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

cls_results = []
logo = LeaveOneGroupOut()

for name, model in cls_models.items():
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in logo.split(X_cls, y_cls, groups_cls):
        model.fit(X_cls.iloc[train_idx], y_cls.iloc[train_idx])
        pred = model.predict(X_cls.iloc[test_idx])
        y_true_all.extend(y_cls.iloc[test_idx])
        y_pred_all.extend(pred)
    
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='weighted')
    prec = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    
    cls_results.append({
        'Model': name,
        'Accuracy (%)': round(acc * 100, 2),
        'Weighted F1': round(f1, 3),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3)
    })

df_cls_res = pd.DataFrame(cls_results)
df_cls_res.to_csv(os.path.join(out_dir, "monthly_spatial_classification_logo.csv"), index=False)
print("\n[Monthly Spatial BIS Hazard Classification (LOGO CV across 98 Locations)]")
print(df_cls_res.to_string(index=False))

print("\n--- Pipeline Completed Successfully ---")
