import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

df_monthly = pd.read_csv(r"c:\Mohit\Repositories\WQI_Research_VIT\src\data\monthly_wqi_parameter_interpolated_spline.csv")

# Create BIS class
bins = [-np.inf, 100, 200, np.inf]
labels = [0, 1, 2] # Good, Poor, Very Poor
df_monthly['BIS_Class'] = pd.cut(df_monthly['WQI_mean'], bins=bins, labels=labels).astype(int)

# CHEAP field parameters ONLY (pH, TDS, Hardness, Chloride) - excluding Fluoride, Nitrates, Sulphates!
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

print(f"--- Surrogate Classification (Predicting BIS Class using ONLY 4 Cheap Field Parameters) ---")
print(f"Features used: {cheap_features}")
print(f"Excluded expensive lab params: Fluoride, Nitrates, Sulphates, Alkalinity")
print(f"Accuracy across 98 held-out locations: {acc*100:.2f}%")
print(f"Weighted F1-score: {f1:.4f}")
