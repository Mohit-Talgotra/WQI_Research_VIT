import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

df = pd.read_csv(os.environ["CLEANED_DATA_FILE_PATH"])

standards = {
    "pH (NA)": {"Sn": 8.5, "Videal": 7},
    "TDS (mg/l)": {"Sn": 500, "Videal": 0},
    "Total Hardness (As CaCO3) (mg/l)": {"Sn": 200, "Videal": 0},
    "Chloride (as Cl) (mg/l)": {"Sn": 250, "Videal": 0},
    "Fluoride (as F) (mg/l)": {"Sn": 1.0, "Videal": 0},
    "Total Alkalinity (as Calcium Carbonate) (mg/l)": {"Sn": 200, "Videal": 0},
    "Sulphate (as SO4) (mg/l)": {"Sn": 200, "Videal": 0},
    "Nitrate (as NO3) (mg/l)": {"Sn": 45, "Videal": 0}
}

K = 1 / sum([1 / standards[p]["Sn"] for p in standards])
weights = {p: K / standards[p]["Sn"] for p in standards}

def calc_qn(param, value, Sn, Videal):
    if param == "pH":
        qn = ((value - Videal) / (Sn - Videal)) * 100
    else:
        qn = (value / Sn) * 100
        
    qn = min(max(qn, 0), 300)
    return qn

def calc_wqi(row):
    wqi_sum = 0
    for param in standards:
        qn = calc_qn(param, row[param], standards[param]["Sn"], standards[param]["Videal"])
        wn = weights[param]
        wqi_sum += qn * wn
    return wqi_sum

df["WQI"] = df.apply(calc_wqi, axis=1)

X = df.drop(columns=["WQI"])
X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean())

y = df["WQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=800,
    max_depth=None,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("RMSE after Qn cap:", rmse)
print("R² after Qn cap:", r2)