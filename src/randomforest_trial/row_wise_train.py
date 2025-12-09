import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

df = pd.read_csv(os.environ["WQI_CALCULATED_DATA_FILE_PATH"])

df = df.dropna(subset=["WQI"])

columns_to_drop = ["Place", "Parameter", "S.No"]
df = df.drop(columns=columns_to_drop, errors="ignore")

df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(int)

zero_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(columns=zero_var_cols, errors="ignore")

print("Dropped zero variance columns:", zero_var_cols)

X = df.drop(columns=["WQI"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

y = df["WQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("RandomForest RMSE:", rmse)
print("RandomForest R²:", r2)

importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Parameter")
plt.title("RandomForest Feature Importance (Per-Record WQI Model)")
plt.show()