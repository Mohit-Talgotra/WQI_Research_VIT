import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

load_dotenv()

MONTHLY_PATH   = Path(os.environ["MONTHLY_WQI_PATH"])
OUTPUT_PATH    = Path(os.environ.get("FORECAST_DATASET_PATH", "data/forecast_dataset.csv"))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Minimum number of monthly data points a location must have to be included.
# Locations below this can't form even one lag-1 training sample reliably.
MIN_MONTHS = 2

# How many lag months to use as features.
# Small given the limited data (mean 5.8 months/location).
# After fixing bad dates this may increase — adjust if needed.
LAG_MONTHS = [1, 2, 3]

# Rolling window sizes for summary features
ROLLING_WINDOWS = [2, 3]

print("=" * 70)
print("BUILDING FORECAST DATASET")
print("=" * 70)

# Load monthly data
monthly = pd.read_csv(MONTHLY_PATH)
monthly["Date"] = pd.to_datetime(monthly["Date"])
monthly = monthly.sort_values(["Block", "Location", "Date"]).reset_index(drop=True)

print(f"  Monthly rows loaded  : {len(monthly):,}")
print(f"  Unique locations     : {monthly['Location'].nunique()}")
print(f"  Blocks               : {sorted(monthly['Block'].unique())}")
print(f"  Date range           : {monthly['Date'].min().date()} → {monthly['Date'].max().date()}")

# Drop Veerapandi — only 1 year of data, useless for lag features
# veerapandi_rows = (monthly["Block"] == "Veerapandi").sum()
# monthly = monthly[monthly["Block"] != "Veerapandi"].copy()
# print(f"\n  Dropped Veerapandi   : {veerapandi_rows} rows (single year, no lag possible)")

# Drop locations with too few data points
loc_counts = monthly.groupby("Location")["WQI_mean"].count()
sparse_locs = loc_counts[loc_counts < MIN_MONTHS].index.tolist()
monthly = monthly[~monthly["Location"].isin(sparse_locs)].copy()
print(f"  Dropped {len(sparse_locs)} sparse locations (< {MIN_MONTHS} months): {sparse_locs}")
print(f"  Remaining rows       : {len(monthly):,}")
print(f"  Remaining locations  : {monthly['Location'].nunique()}")

# Build a complete monthly date spine per location
# This makes gaps explicit (NaN) rather than hidden, which is important
# so lag features don't accidentally bridge across gaps.
print("\nBUILDING DATE SPINE (making gaps explicit)")

all_dates = pd.date_range(
    start=monthly["Date"].min(),
    end=monthly["Date"].max(),
    freq="MS"   # Month Start
)

locations = monthly[["Block", "Location"]].drop_duplicates()
spine = locations.assign(key=1).merge(
    pd.DataFrame({"Date": all_dates, "key": 1}),
    on="key"
).drop(columns="key")

monthly_full = spine.merge(
    monthly[["Block", "Location", "Date", "WQI_mean", "n_records"]],
    on=["Block", "Location", "Date"],
    how="left"
)

monthly_full = monthly_full.sort_values(["Block", "Location", "Date"]).reset_index(drop=True)
total_slots = len(monthly_full)
filled_slots = monthly_full["WQI_mean"].notna().sum()
print(f"  Total date slots     : {total_slots:,}  ({filled_slots:,} filled, {total_slots - filled_slots:,} gaps)")

# Build lag and rolling features per location
print("\nBUILDING LAG AND ROLLING FEATURES")

feature_rows = []

for (block, location), grp in monthly_full.groupby(["Block", "Location"]):
    grp = grp.sort_values("Date").copy()
    wqi = grp["WQI_mean"].values
    dates = grp["Date"].values
    n = len(grp)

    for i in range(n):
        row_date   = dates[i]
        row_target = wqi[i]

        # Target must exist
        if pd.isna(row_target):
            continue

        # Build lag features — if any lag is missing (gap), mark as NaN
        # The model will handle NaN lags via imputation later
        lags = {}
        for lag in LAG_MONTHS:
            lag_idx = i - lag
            lags[f"WQI_lag_{lag}"] = wqi[lag_idx] if lag_idx >= 0 else np.nan

        # Rolling means — only over available (non-NaN) lag values
        for w in ROLLING_WINDOWS:
            lag_vals = [wqi[i - l] for l in range(1, w + 1) if (i - l) >= 0]
            lag_vals = [v for v in lag_vals if not pd.isna(v)]
            lags[f"WQI_rolling_mean_{w}"] = np.mean(lag_vals) if lag_vals else np.nan

        # Time features (cyclical encoding for month — helps model learn seasonality)
        month_num = pd.Timestamp(row_date).month
        year_num  = pd.Timestamp(row_date).year

        feature_rows.append({
            "Block"           : block,
            "Location"        : location,
            "Date"            : row_date,
            "Year"            : year_num,
            "Month"           : month_num,
            # Cyclical month encoding so December (12) and January (1) are close
            "Month_sin"       : np.sin(2 * np.pi * month_num / 12),
            "Month_cos"       : np.cos(2 * np.pi * month_num / 12),
            **lags,
            "n_records"       : grp["n_records"].iloc[i],
            "WQI_target"      : row_target,
        })

df = pd.DataFrame(feature_rows)
print(f"  Total rows before lag filtering : {len(df):,}")

# Require at least lag_1 to be present
# Rows where ALL lags are NaN (i.e. first data point for a location with no
# prior history) are not useful for training
all_lags_null = df[[f"WQI_lag_{l}" for l in LAG_MONTHS]].isna().all(axis=1)
df = df[~all_lags_null].copy()
print(f"  Rows after removing all-null-lag rows : {len(df):,}")

# Encode categorical features
print("\nENCODING CATEGORICAL FEATURES")

le_block    = LabelEncoder()
le_location = LabelEncoder()

df["Block_encoded"]    = le_block.fit_transform(df["Block"])
df["Location_encoded"] = le_location.fit_transform(df["Location"])

print(f"  Blocks    : {dict(zip(le_block.classes_, le_block.transform(le_block.classes_)))}")
print(f"  Locations : {df['Location'].nunique()} unique → encoded 0–{df['Location_encoded'].max()}")

# Save encoder mappings so we can decode predictions later
block_map    = pd.DataFrame({"Block": le_block.classes_,
                              "Block_encoded": le_block.transform(le_block.classes_)})
location_map = pd.DataFrame({"Location": le_location.classes_,
                              "Location_encoded": le_location.transform(le_location.classes_)})
block_map.to_csv(OUTPUT_PATH.parent / "encoder_block.csv", index=False)
location_map.to_csv(OUTPUT_PATH.parent / "encoder_location.csv", index=False)

# Final feature set
feature_cols = (
    ["Block_encoded", "Location_encoded", "Year", "Month_sin", "Month_cos"]
    + [f"WQI_lag_{l}"           for l in LAG_MONTHS]
    + [f"WQI_rolling_mean_{w}"  for w in ROLLING_WINDOWS]
)

target_col = "WQI_target"

print(f"\nFINAL DATASET")
print(f"  Rows         : {len(df):,}")
print(f"  Feature cols : {feature_cols}")
print(f"  Target       : {target_col}")
print(f"  WQI range    : {df[target_col].min():.1f} – {df[target_col].max():.1f}")
print(f"  Rows with any NaN lag : {df[feature_cols].isna().any(axis=1).sum()}")
print(f"  (NaN lags will be median-imputed during model training)")

# Coverage after feature building
print("\nROWS PER LOCATION (training samples available):")
loc_sample_counts = df.groupby(["Block", "Location"]).size().reset_index(name="n_samples")
print(loc_sample_counts.to_string(index=False))
print(f"\n  Mean samples/location : {loc_sample_counts['n_samples'].mean():.1f}")
print(f"  Min                   : {loc_sample_counts['n_samples'].min()}")
print(f"  Max                   : {loc_sample_counts['n_samples'].max()}")

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Saved: {OUTPUT_PATH.parent / 'encoder_block.csv'}")
print(f"Saved: {OUTPUT_PATH.parent / 'encoder_location.csv'}")
print("\n" + "=" * 70)
print("FORECAST DATASET COMPLETE")
print("=" * 70)