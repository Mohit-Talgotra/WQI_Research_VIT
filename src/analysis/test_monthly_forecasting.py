import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

# Load monthly dataset
monthly_df = pd.read_csv(r"c:\Mohit\Repositories\WQI_Research_VIT\src\data\monthly_wqi_parameter_interpolated_spline.csv")
monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
monthly_df = monthly_df.sort_values(['Location', 'Date']).reset_index(drop=True)

# Create 1-month, 2-month, 3-month lag features for WQI and key parameters
for lag in [1, 2, 3]:
    monthly_df[f'WQI_lag_{lag}'] = monthly_df.groupby('Location')['WQI_mean'].shift(lag)
    monthly_df[f'F_lag_{lag}'] = monthly_df.groupby('Location')['Fluoride_mean'].shift(lag)
    monthly_df[f'TDS_lag_{lag}'] = monthly_df.groupby('Location')['TDS_mean'].shift(lag)

df_forecast = monthly_df.dropna(subset=['WQI_mean', 'WQI_lag_1', 'WQI_lag_2', 'WQI_lag_3']).copy()

# Temporal split: Train on 2022-2023 monthly data, Test on 2024 monthly data
train_df = df_forecast[df_forecast['Date'] < '2024-01-01']
test_df = df_forecast[(df_forecast['Date'] >= '2024-01-01') & (df_forecast['Date'] < '2025-01-01')]

feature_cols = ['WQI_lag_1', 'WQI_lag_2', 'WQI_lag_3', 'F_lag_1', 'TDS_lag_1', 'pH_mean', 'Hardness_mean', 'Chloride_mean']

X_train = train_df[feature_cols]
y_train = train_df['WQI_mean']
X_test = test_df[feature_cols]
y_test = test_df['WQI_mean']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# Persistence baseline (predict WQI_lag_1)
y_pred_persist = test_df['WQI_lag_1']

print("=== Monthly 1-Month-Ahead WQI Forecasting (Train 2022-2023 -> Test 2024) ===")
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

for name, preds in [('Random Forest', y_pred_rf), ('Gradient Boosting', y_pred_gb), ('Persistence (Lag-1)', y_pred_persist)]:
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:20s} | R2: {r2:7.4f} | RMSE: {rmse:7.4f} | MAE: {mae:7.4f}")
