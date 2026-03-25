import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

OUTPUT_DIR = Path("quarterly_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    df["Quarter_num"] = df["Sample Collection date"].dt.to_period("Q").astype(int)

    grouped = (
        df
        .groupby(["Village", "Year", "Quarter", "Quarter_num"])[list(standards.keys())]
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
    all_preds = []
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
        
        fold_df = df.iloc[test_idx].copy()
        fold_df["Predicted_WQI"] = preds
        all_preds.append(fold_df)

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
    predictions_df = pd.concat(all_preds, ignore_index=True)
    return predictions_df

def plot_quarterly_predictions(pred_df, locations_per_fig=3):
    pred_df = pred_df.sort_values(["Place", "Year", "Quarter"])

    places = pred_df["Place"].unique()
    n_figs = math.ceil(len(places) / locations_per_fig)

    for i in range(n_figs):
        subset_places = places[i*locations_per_fig:(i+1)*locations_per_fig]
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10.colors  # 10 distinct colors

        for idx, place in enumerate(subset_places):
            p_df = pred_df[pred_df["Place"] == place]
            
            p_df = p_df.sort_values("Quarter_num")
            
            p_df["Quarter_gap"] = p_df["Quarter_num"].diff() > 1
            
            p_df = p_df.reset_index(drop=True)

            gap_pos = np.where(p_df["Quarter_gap"].values)[0]

            segments = []
            start = 0

            for g in gap_pos:
                segments.append(p_df.iloc[start:g])
                start = g

            segments.append(p_df.iloc[start:])
            
            color = colors[idx % len(colors)]
            
            for seg in segments:
                ax.plot(
                    seg["Quarter_num"],
                    seg["WQI"],
                    linestyle="--",
                    marker="o",
                    color=color,
                    label=f"{place} (Actual)" if seg is segments[0] else None
                )

                ax.plot(
                    seg["Quarter_num"],
                    seg["Predicted_WQI"],
                    linestyle="-",
                    marker="x",
                    color=color,
                    label=f"{place} (Predicted)" if seg is segments[0] else None
                )
        
        ax.set_xlabel("Quarter (time order)")
        ax.set_ylabel("WQI")
        ax.set_title("Quarterly WQI: Actual vs Predicted")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        xticks = sorted(p_df["Quarter_num"].unique())
        xtick_labels = (
            p_df.drop_duplicates("Quarter_num")
                .set_index("Quarter_num")
                .loc[xticks]["Quarter"]
                .tolist()
        )

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45)
        
        plt.tight_layout()
        out_file = OUTPUT_DIR / f"quarterly_wqi_actual_vs_predicted_{i+1}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        

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
    quarterly_preds = evaluate_logo_model(
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
    
    plot_quarterly_predictions(quarterly_preds, locations_per_fig=3)

if __name__ == "__main__":
    main()