import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

# Load monthly dataset
monthly_df = pd.read_csv(r"c:\Mohit\Repositories\WQI_Research_VIT\src\data\monthly_wqi_parameter_interpolated_spline.csv")
print("Monthly dataset shape:", monthly_df.shape)

# Feature mapping
param_cols = ['pH_mean', 'TDS_mean', 'Hardness_mean', 'Chloride_mean', 'Alkalinity_mean', 'Sulphate_mean', 'Nitrate_mean']
target_col = 'Fluoride_mean'
group_col = 'Location'

# Drop NAs
df_f = monthly_df.dropna(subset=[target_col, group_col] + param_cols).copy()
X_f = df_f[param_cols]
y_f = df_f[target_col]
groups_f = df_f[group_col]

logo = LeaveOneGroupOut()
y_true_f, y_pred_f = [], []

for train_idx, test_idx in logo.split(X_f, y_f, groups_f):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_f.iloc[train_idx], y_f.iloc[train_idx])
    pred = rf.predict(X_f.iloc[test_idx])
    y_true_f.extend(y_f.iloc[test_idx])
    y_pred_f.extend(pred)

r2_f = r2_score(y_true_f, y_pred_f)
rmse_f = root_mean_squared_error(y_true_f, y_pred_f)
print(f"\n=======================================================")
print(f"Direction 2: Fluoride Surrogate Modeling (LOGO CV on Monthly Data)")
print(f"Total samples: {len(y_true_f)}, Locations: {len(np.unique(groups_f))}")
print(f"Fluoride Prediction R2: {r2_f:.4f}, RMSE: {rmse_f:.4f}")
print(f"=======================================================")

# Direction 3: BIS Classification (Good < 100, Poor 100-200, Very Poor > 200)
annual_df = pd.read_csv(r"c:\Mohit\Repositories\WQI_Research_VIT\src\results\rf_results\final_df_for_analysis.csv")
bins = [-np.inf, 100, 200, np.inf]
labels = [0, 1, 2] # Good, Poor, Very Poor
annual_df['WQI_class'] = pd.cut(annual_df['WQI'], bins=bins, labels=labels)

features = ['Cl', 'F', 'Nitrates', 'Sulphates', 'TDS', 'Total Alakalinity', 'Total Hardness', 'pH']
X_c = annual_df[features]
y_c = annual_df['WQI_class'].astype(int)
groups_c = annual_df['Place']

y_true_c, y_pred_c = [], []
for train_idx, test_idx in logo.split(X_c, y_c, groups_c):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_c.iloc[train_idx], y_c.iloc[train_idx])
    pred = clf.predict(X_c.iloc[test_idx])
    y_true_c.extend(y_c.iloc[test_idx])
    y_pred_c.extend(pred)

acc = accuracy_score(y_true_c, y_pred_c)
f1 = f1_score(y_true_c, y_pred_c, average='weighted')
print(f"\n=======================================================")
print(f"Direction 3: BIS Water Quality Classification (LOGO CV on Annual Data)")
print(f"Classification Accuracy: {acc:.4f}, Weighted F1-score: {f1:.4f}")
print("Class distribution:", pd.Series(y_c).value_counts().to_dict())
print(f"=======================================================")
