import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
import shap

try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

load_dotenv()
file = os.environ.get("WQI_AGGREGATE_FILE_PATH")
if not file:
    print("set WQI_AGGREGATE_FILE_PATH in .env")
    sys.exit(1)

sheets = {
    "2022-2023 WQI Calculation": "2022-2023",
    "2023-2024 WQI Calculation": "2023-2024",
    "2024-2025 WQI Calculation": "2024-2025"
}

dfs = []
for sheet_name, year in sheets.items():
    tmp = pd.read_excel(file, sheet_name=sheet_name)
    tmp["Year"] = year
    dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()
df["Place"] = df["Place"].astype(str).str.strip()
df["Parameter"] = df["Parameter"].astype(str).str.strip()
df["Year"] = df["Year"].astype(str).str.strip()
df["Mean Value (Vn)"] = df["Mean Value (Vn)"].astype(str).str.replace(",", "").str.strip().replace("", np.nan)
df["Mean Value (Vn)"] = pd.to_numeric(df["Mean Value (Vn)"], errors="coerce")

pivot_df = df.pivot_table(index=["Year","Place"], columns="Parameter", values="Mean Value (Vn)", aggfunc="first").reset_index()

wqi_raw = pd.read_excel(file, sheet_name="Final WQI Data- 3 Years", header=1)
wqi_raw.columns = wqi_raw.columns.str.strip()
year_cols = [c for c in wqi_raw.columns if str(c).startswith("202")]
wqi_long = wqi_raw.melt(id_vars=["Place"], value_vars=year_cols, var_name="Year", value_name="WQI")
wqi_long["Place"] = wqi_long["Place"].astype(str).str.strip()
wqi_long["Year"] = wqi_long["Year"].astype(str).str.strip()

final_df = pivot_df.merge(wqi_long, on=["Year","Place"], how="left")
final_df = final_df.dropna(subset=["WQI"]).reset_index(drop=True)
final_df["Year"] = final_df["Year"].str.extract(r"(\d{4})").astype(int)

X = final_df.drop(columns=["WQI","Place"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())
y = final_df["WQI"].astype(float)
groups = final_df["Place"].values

results = []
def evaluate_model(estimator, X, y, groups, name):
    logo = LeaveOneGroupOut()
    rmses = []
    r2s = []
    for train_idx, test_idx in logo.split(X,y,groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(y_tr)==0 or len(y_te)==0:
            continue
        estimator.fit(X_tr, y_tr)
        pred = estimator.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, pred)))
        if len(y_te) >= 2:
            r2s.append(r2_score(y_te, pred))
    return {"model": name, "mean_rmse": np.mean(rmses) if rmses else np.nan, "std_rmse": np.std(rmses) if rmses else np.nan, "mean_r2": np.mean(r2s) if r2s else np.nan, "std_r2": np.std(r2s) if r2s else np.nan}

rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42)
res_rf = evaluate_model(rf, X, y, groups, "RandomForest")
results.append(res_rf)

if xgb is not None:
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    res_xgb = evaluate_model(xgb_model, X, y, groups, "XGBoost_default")
    results.append(res_xgb)

if CatBoostRegressor is not None:
    cb = CatBoostRegressor(verbose=0, random_state=42)
    res_cb = evaluate_model(cb, X, y, groups, "CatBoost_default")
    results.append(res_cb)

param_spaces = []
if xgb is not None:
    param_spaces.append(("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1), {
        "n_estimators": [100,300,500],
        "max_depth": [3,5,8],
        "learning_rate": [0.01,0.05,0.1],
        "subsample": [0.6,0.8,1.0]
    }))
if CatBoostRegressor is not None:
    param_spaces.append(("catboost", CatBoostRegressor(random_state=42, verbose=0), {
        "iterations": [200,500],
        "depth": [4,6,8],
        "learning_rate": [0.01,0.05,0.1]
    }))

for name, estimator, space in param_spaces:
    rs = RandomizedSearchCV(estimator, space, n_iter=6, scoring="neg_mean_squared_error", cv=3, random_state=42, n_jobs=1)
    rs.fit(X, y)
    best = rs.best_estimator_
    res_best = evaluate_model(best, X, y, groups, f"{name}_tuned")
    results.append(res_best)

res_df = pd.DataFrame(results)
res_df.to_csv("../results/model_comparison.csv", index=False)
best_row = res_df.sort_values("mean_rmse").iloc[0]
best_name = best_row["model"]

if best_name.startswith("xgboost") and xgb is not None:
    best_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, n_estimators=300, max_depth=5, learning_rate=0.05)
    best_model.fit(X,y)
elif best_name.startswith("catboost") and CatBoostRegressor is not None:
    best_model = CatBoostRegressor(random_state=42, verbose=0, iterations=500, depth=6, learning_rate=0.05)
    best_model.fit(X,y)
else:
    best_model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
    best_model.fit(X,y)

joblib.dump(best_model, "../results/best_model.joblib")

explainer = None
if xgb is not None and isinstance(best_model, xgb.XGBRegressor):
    explainer = shap.TreeExplainer(best_model)
elif CatBoostRegressor is not None and isinstance(best_model, CatBoostRegressor):
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X)
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("../results/shap_summary.png", dpi=200)
plt.close()

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("../results/shap_beeswarm.png", dpi=200)
plt.close()

top_feat_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-3:]
for i, feat_idx in enumerate(reversed(top_feat_idx)):
    sample_idx = int(np.argmax(np.abs(shap_values[:, feat_idx])))
    shap_vect = shap_values[sample_idx]
    feat_values = X.iloc[sample_idx]
    
    fig = shap.force_plot(
        explainer.expected_value,
        shap_vect,
        feat_values,
        matplotlib=True,
        show=False
    )

    outname = f"shap_force_TOP{i+1}_{ X.columns[feat_idx] }.png"
    fig.savefig("../results/" + outname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {outname} for sample {sample_idx} (feature {X.columns[feat_idx]})")


final_df.to_csv("final_df_for_analysis.csv", index=False)

plt.figure(figsize=(12,7))
for place, grp in final_df.groupby("Place"):
    plt.plot(grp["Year"], grp["WQI"], marker="o", label=place)
plt.xlabel("Year")
plt.ylabel("WQI")
plt.title("WQI trend per Place (one line per place)")
plt.xticks(sorted(final_df["Year"].unique()))
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", ncol=1, fontsize="small")
plt.tight_layout()
plt.savefig("../results/wqi_trends.png", dpi=200)
plt.close()

heat = final_df.pivot(index="Place", columns="Year", values="WQI")
plt.figure(figsize=(8,10))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label":"WQI"})
plt.title("WQI heatmap (Place vs Year)")
plt.tight_layout()
plt.savefig("../results/wqi_heatmap.png", dpi=200)
plt.close()

print("Done. Outputs: model_comparison.csv, best_model.joblib, shap_*.png, wqi_trends.png, wqi_heatmap.png, final_df_for_analysis.csv, wqi_map.html (if coords provided)")