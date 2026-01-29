import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

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

K = 1 / sum(1 / standards[p]["Sn"] for p in standards)
weights = {p: K / standards[p]["Sn"] for p in standards}


def calc_qn(value, Sn, Videal):
    qn = ((value - Videal) / (Sn - Videal)) * 100
    qn = min(qn, 300)
    return max(qn, 0)


def calc_wqi(row):
    total = 0
    for p in standards:
        qn = calc_qn(row[p], standards[p]["Sn"], standards[p]["Videal"])
        total += qn * weights[p]
    return total

def map_season(month):
    if month in [3, 4, 5]:
        return "Summer"
    if month in [6, 7, 8, 9]:
        return "Monsoon"
    return "Winter"

def build_quarterly_dataset(df):
    df["Year"] = df["Sample Collection date"].dt.year
    df["Quarter"] = df["Sample Collection date"].dt.to_period("Q").astype(str)

    grouped = (
        df
        .groupby(["Village", "Year", "Quarter"])[list(standards.keys())]
        .mean()
        .reset_index()
        .rename(columns={"Village": "Place"})
    )

    grouped["WQI"] = grouped.apply(calc_wqi, axis=1)
    return grouped


def build_seasonal_dataset(df):
    df["Year"] = df["Sample Collection date"].dt.year
    df["Season"] = df["Sample Collection date"].dt.month.map(map_season)

    grouped = (
        df
        .groupby(["Village", "Year", "Season"])[list(standards.keys())]
        .mean()
        .reset_index()
        .rename(columns={"Village": "Place"})
    )

    grouped["WQI"] = grouped.apply(calc_wqi, axis=1)
    return grouped

def evaluate_logo_model(df, feature_cols, name):
    X = df[feature_cols]
    y = df["WQI"]
    groups = df["Place"].values

    logo = LeaveOneGroupOut()

    rmses, maes, r2s = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
        maes.append(mean_absolute_error(y_te, preds))
        if len(y_te) >= 2:
            r2s.append(r2_score(y_te, preds))

    print("\n" + "=" * 70)
    print(f"{name} RESULTS (LOGO CV)")
    print("=" * 70)
    print(f"RMSE : {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    print(f"MAE  : {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    if r2s:
        print(f"R²   : {np.mean(r2s):.3f}")
    print("=" * 70)

def main():
    data_path = os.environ["WQI_CALCULATED_DATA_FILE_PATH"]

    df = pd.read_csv(data_path)
    df["Sample Collection date"] = pd.to_datetime(
        df["Sample Collection date"], errors="coerce"
    )
    df = df.dropna(subset=["Sample Collection date"])

    print(f"Loaded {len(df)} raw records")
    
    quarterly_df = build_quarterly_dataset(df)
    quarterly_df.to_csv("quarterly_wqi_dataset.csv", index=False)

    quarterly_features = list(standards.keys())
    evaluate_logo_model(
        quarterly_df,
        quarterly_features,
        "QUARTERLY WQI PREDICTION"
    )
    seasonal_df = build_seasonal_dataset(df)

    seasonal_df["Season_encoded"] = seasonal_df["Season"].map(
        {"Winter": 0, "Summer": 1, "Monsoon": 2}
    )

    seasonal_df.to_csv("seasonal_wqi_dataset.csv", index=False)

    seasonal_features = list(standards.keys()) + ["Season_encoded"]
    evaluate_logo_model(
        seasonal_df,
        seasonal_features,
        "SEASONAL WQI PREDICTION"
    )


if __name__ == "__main__":
    main()