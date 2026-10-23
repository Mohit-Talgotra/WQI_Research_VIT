import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# Output directories
t3_dir = r"c:\Mohit\Repositories\WQI_Research_VIT\reproducibility_results\table3_logo_cv"
t4_dir = r"c:\Mohit\Repositories\WQI_Research_VIT\reproducibility_results\table4_temporal_holdout"
os.makedirs(t3_dir, exist_ok=True)
os.makedirs(t4_dir, exist_ok=True)

# Load 14-location 42-row Salem block dataset
df_42 = pd.read_csv(r"c:\Mohit\Repositories\WQI_Research_VIT\src\results\rf_results\final_df_for_analysis.csv")

features = ['Cl', 'F', 'Nitrates', 'Sulphates', 'TDS', 'Total Alakalinity', 'Total Hardness', 'pH']
X = df_42[features]
y = df_42['WQI']
groups = df_42['Place']

logo = LeaveOneGroupOut()

print("==========================================================================")
print("PART 1: VERIFYING TABLE 3 (KNN & MEAN BASELINE LOGO CV)")
print("==========================================================================")

# Test KNN variations (k=3, k=5, k=7, uniform vs distance, scaled vs unscaled)
knn_results = []
for k in [1, 3, 5, 7]:
    for weights in ['uniform', 'distance']:
        for scale in [False, True]:
            y_true, y_pred = [], []
            fold_rmse_list = []
            
            for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
                
                if scale:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)
                
                knn = KNeighborsRegressor(n_neighbors=k, weights=weights)
                knn.fit(X_tr, y_tr)
                pred = knn.predict(X_te)
                
                y_true.extend(y_te)
                y_pred.extend(pred)
                fold_rmse_list.append(root_mean_squared_error(y_te, pred))
            
            mean_rmse = np.mean(fold_rmse_list)
            pooled_rmse = root_mean_squared_error(y_true, y_pred)
            pooled_r2 = r2_score(y_true, y_pred)
            
            knn_results.append({
                'k': k, 'weights': weights, 'scaled': scale,
                'mean_rmse': round(mean_rmse, 2),
                'pooled_rmse': round(pooled_rmse, 2),
                'pooled_r2': round(pooled_r2, 3)
            })

df_knn_sweep = pd.DataFrame(knn_results)
print("KNN Parameter Sweep Results on Table 3 LOGO:")
print(df_knn_sweep.to_string(index=False))

# Mean Baseline LOGO
y_true_mb, y_pred_mb = [], []
fold_rmse_mb = []
for train_idx, test_idx in logo.split(X, y, groups):
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    pred = np.full(len(y_te), y_tr.mean())
    y_true_mb.extend(y_te)
    y_pred_mb.extend(pred)
    fold_rmse_mb.append(root_mean_squared_error(y_te, pred))

mb_mean_rmse = np.mean(fold_rmse_mb)
mb_pooled_rmse = root_mean_squared_error(y_true_mb, y_pred_mb)
mb_pooled_r2 = r2_score(y_true_mb, y_pred_mb)

print(f"\nMean Baseline LOGO CV -> Mean RMSE: {mb_mean_rmse:.2f}, Pooled RMSE: {mb_pooled_rmse:.2f}, Pooled R2: {mb_pooled_r2:.3f}")

print("\n==========================================================================")
print("PART 2: VERIFYING TABLE 4 (KNN & MEAN BASELINE 2024 TEMPORAL HOLDOUT)")
print("==========================================================================")

train_t4 = df_42[df_42['Year'].isin([2022, 2023])]
test_t4 = df_42[df_42['Year'] == 2024]

X_tr_t4, y_tr_t4 = train_t4[features], train_t4['WQI']
X_te_t4, y_te_t4 = test_t4[features], test_t4['WQI']

# Mean Baseline 2024 Holdout
mb_pred_t4 = np.full(len(y_te_t4), y_tr_t4.mean())
mb_rmse_t4 = root_mean_squared_error(y_te_t4, mb_pred_t4)
mb_mae_t4 = mean_absolute_error(y_te_t4, mb_pred_t4)
mb_r2_t4 = r2_score(y_te_t4, mb_pred_t4)

print(f"Mean Baseline 2024 Holdout -> RMSE: {mb_rmse_t4:.2f}, MAE: {mb_mae_t4:.2f}, R2: {mb_r2_t4:.2f}")

# KNN 2024 Holdout Sweep
t4_knn_results = []
for k in [1, 3, 5, 7]:
    for weights in ['uniform', 'distance']:
        for scale in [False, True]:
            X_tr, X_te = X_tr_t4.copy(), X_te_t4.copy()
            if scale:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
            
            knn = KNeighborsRegressor(n_neighbors=k, weights=weights)
            knn.fit(X_tr, y_tr_t4)
            pred = knn.predict(X_te)
            
            rmse = root_mean_squared_error(y_te_t4, pred)
            mae = mean_absolute_error(y_te_t4, pred)
            r2 = r2_score(y_te_t4, pred)
            
            t4_knn_results.append({
                'k': k, 'weights': weights, 'scaled': scale,
                'rmse': round(rmse, 2), 'mae': round(mae, 2), 'r2': round(r2, 2)
            })

df_t4_knn = pd.DataFrame(t4_knn_results)
print("\nKNN Parameter Sweep Results on Table 4 (2024 Holdout):")
print(df_t4_knn.to_string(index=False))
