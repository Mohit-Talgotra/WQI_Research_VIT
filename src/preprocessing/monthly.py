import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Config
DATA_ROOT   = Path(os.environ["DATA_ROOT"])
OUTPUT_PATH = Path(os.environ.get("MONTHLY_WQI_PATH", "data/monthly_wqi_dataset.csv"))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SKIP_PREFIXES = ("~", ".", "identifier")
SKIP_SUFFIXES = (".tmp", ".bak")

# WQI standards (exact same as wqi_calc.py)
STANDARDS = {
    "pH (NA)":                                          {"Sn": 8.5,  "Videal": 7},
    "TDS (mg/l)":                                       {"Sn": 500,  "Videal": 0},
    "Total Hardness (As CaCO3) (mg/l)":                 {"Sn": 200,  "Videal": 0},
    "Chloride (as Cl) (mg/l)":                          {"Sn": 250,  "Videal": 0},
    "Fluoride (as F) (mg/l)":                           {"Sn": 1.0,  "Videal": 0},
    "Total Alkalinity (as Calcium Carbonate) (mg/l)":   {"Sn": 200,  "Videal": 0},
    "Sulphate (as SO4) (mg/l)":                         {"Sn": 200,  "Videal": 0},
    "Nitrate (as NO3) (mg/l)":                          {"Sn": 45,   "Videal": 0},
}

K = 1 / sum(1 / STANDARDS[p]["Sn"] for p in STANDARDS)
WEIGHTS = {p: K / STANDARDS[p]["Sn"] for p in STANDARDS}

PARAMETER_OUTPUT_NAMES = {
    "pH (NA)": "pH_mean",
    "TDS (mg/l)": "TDS_mean",
    "Total Hardness (As CaCO3) (mg/l)": "Hardness_mean",
    "Chloride (as Cl) (mg/l)": "Chloride_mean",
    "Fluoride (as F) (mg/l)": "Fluoride_mean",
    "Total Alkalinity (as Calcium Carbonate) (mg/l)": "Alkalinity_mean",
    "Sulphate (as SO4) (mg/l)": "Sulphate_mean",
    "Nitrate (as NO3) (mg/l)": "Nitrate_mean",
}

def calc_qn(value: float, Sn: float, Videal: float) -> float:
    qn = ((value - Videal) / (Sn - Videal)) * 100
    return float(np.clip(qn, 0, 300))

def calc_wqi(row: pd.Series) -> float:
    total = 0.0
    for param, spec in STANDARDS.items():
        val = row.get(param, np.nan)
        if pd.isna(val):
            return np.nan
        total += calc_qn(float(val), spec["Sn"], spec["Videal"]) * WEIGHTS[param]
    return total


def should_skip(name: str) -> bool:
    n = name.lower()
    return any(n.startswith(p) for p in SKIP_PREFIXES) or \
           any(n.endswith(s) for s in SKIP_SUFFIXES)

def extract_year(folder_name: str):
    m = re.search(r"(\d{4})", folder_name)
    return int(m.group(1)) if m else None

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s*\(\d{4}-\d{4}\)", "", c).strip() for c in df.columns]
    return df


print("=" * 70)
print("LOADING RAW DATA")
print("=" * 70)

all_dfs = []
errors  = []

for block_dir in sorted(DATA_ROOT.iterdir()):
    if not block_dir.is_dir() or should_skip(block_dir.name):
        continue

    for year_dir in sorted(block_dir.iterdir()):
        if not year_dir.is_dir() or should_skip(year_dir.name):
            continue
        folder_year = extract_year(year_dir.name)
        if folder_year is None:
            continue

        for excel_file in sorted(year_dir.iterdir()):
            if should_skip(excel_file.name):
                continue
            if excel_file.suffix.lower() not in (".xlsx", ".xls", ".xlsm"):
                continue

            try:
                xls = pd.ExcelFile(excel_file, engine="openpyxl")
                if len(xls.sheet_names) < 2:
                    continue

                df = pd.read_excel(xls, sheet_name=xls.sheet_names[1], engine="openpyxl")
                # Handle files where the real header is in row 0 as data (not as df columns)
                # Detected when columns are all "Unnamed: X"
                if all("Unnamed" in str(c) for c in df.columns):
                    df.columns = df.iloc[0].astype(str).str.strip()
                    df = df.iloc[1:].reset_index(drop=True)
                
                df = normalize_cols(df)

                # Drop non-data rows (header bleed, totals)
                sno_col = next(
                    (c for c in df.columns if c.strip().lower() in ("s.no.", "s. no.", "s.no")),
                    None
                )
                if sno_col:
                    df = df[pd.to_numeric(df[sno_col], errors="coerce").fillna(0) > 0]

                if df.empty:
                    continue

                df["_block"]       = block_dir.name
                df["_folder_year"] = folder_year
                df["_file"]        = excel_file.name
                all_dfs.append(df)

            except Exception as e:
                errors.append((str(excel_file), str(e)))

if not all_dfs:
    raise RuntimeError("No data loaded — check DATA_ROOT")

raw = pd.concat(all_dfs, ignore_index=True)
raw.columns = [c.strip() for c in raw.columns]
print(f"  Loaded {len(raw):,} records from {len(all_dfs)} files")
if errors:
    print(f"  {len(errors)} files failed to load")


print("\nPARSING DATES")
date_col = "Sample Collection date"
# Try standard parsing first, then explicitly try DD-MM-YYYY with hyphens
raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)

# Catch remaining unparsed dates that use DD-MM-YYYY hyphen format
still_null = raw[date_col].isna() & raw["Sample Collection date"].notna()
if still_null.sum() > 0:
    raw.loc[still_null, date_col] = pd.to_datetime(
        raw.loc[still_null, "Sample Collection date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

bad_date    = raw[date_col].isna()
placeholder = (~bad_date) & (raw[date_col].dt.year < 2000)

# Folders are named "2022-2023" so both years are valid for that folder
raw["_folder_year2"] = raw["_folder_year"] + 1
wrong_year = (
    (~bad_date) &
    (~placeholder) &
    (raw[date_col].dt.year != raw["_folder_year"]) &
    (raw[date_col].dt.year != raw["_folder_year2"])
)

raw["_skip_reason"] = ""
raw.loc[bad_date,    "_skip_reason"] = "unparseable_date"
raw.loc[placeholder, "_skip_reason"] = "placeholder_date"
raw.loc[wrong_year,  "_skip_reason"] = "wrong_year"

print(f"  Unparseable dates        : {bad_date.sum()} — skipped")
print(f"  Placeholder dates (<2000): {placeholder.sum()} — skipped")
print(f"  Date-year != folder-year : {wrong_year.sum()} — skipped")
print(f"  Clean records            : {(raw['_skip_reason'] == '').sum():,}")

# Only assign year/month for clean records; bad ones stay NA and are excluded from groupby
clean_mask = raw["_skip_reason"] == ""
raw["_year"]  = np.nan
raw["_month"] = np.nan
raw.loc[clean_mask, "_year"]  = raw.loc[clean_mask, date_col].dt.year.astype(int)
raw.loc[clean_mask, "_month"] = raw.loc[clean_mask, date_col].dt.month.astype(int)


location_col = next(
    (c for c in raw.columns if c.lower() in ("gram panchayat", "village", "place")),
    None
)
if location_col is None:
    raise ValueError("Cannot find location column")
print(f"\n  Location column: '{location_col}'")


print("\nCONVERTING PARAMETERS TO NUMERIC")
missing_params = [p for p in STANDARDS if p not in raw.columns]
if missing_params:
    print(f"  WARNING — missing parameter columns: {missing_params}")
    print("  These will be treated as NaN in WQI calculation")

for param in STANDARDS:
    if param in raw.columns:
        raw[param] = pd.to_numeric(raw[param], errors="coerce")


print("\nCALCULATING PER-RECORD WQI")
raw["WQI"] = raw.apply(calc_wqi, axis=1)

wqi_null = raw["WQI"].isna().sum()
wqi_ok   = raw["WQI"].notna().sum()
print(f"  WQI calculated   : {wqi_ok:,} records")
print(f"  WQI skipped (NaN): {wqi_null:,} records (missing parameter values)")


print("\nAGGREGATING TO MONTHLY")

# Group by block + location + year + month
group_cols = ["_block", location_col, "_year", "_month"]

# Only aggregate records with valid dates AND valid WQI
# Bad date records (_skip_reason != "") are simply not included — not deleted
clean_for_agg = raw[(raw["_skip_reason"] == "") & raw["WQI"].notna()]
print(f"  Records used in aggregation: {len(clean_for_agg):,}  "
      f"(skipped {len(raw) - len(clean_for_agg):,} bad/null records)")

monthly = (
    clean_for_agg
       .groupby(group_cols)
       .agg(
           WQI_mean        = ("WQI",  "mean"),
           WQI_std         = ("WQI",  "std"),
           n_records       = ("WQI",  "count"),
           n_unique_dates  = (date_col, "nunique"),
           # Also aggregate raw parameters with stable output names
           **{PARAMETER_OUTPUT_NAMES[p]: (p, "mean")
              for p in STANDARDS if p in raw.columns}
       )
       .reset_index()
)

monthly = monthly.rename(columns={
    "_block":       "Block",
    location_col:   "Location",
    "_year":        "Year",
    "_month":       "Month",
})

# Add a proper date column (first day of each month) for easy time-series use
# Cast to int first to avoid float formatting like "2022.0-5.0-01"
monthly["Year"]  = monthly["Year"].astype(int)
monthly["Month"] = monthly["Month"].astype(int)
monthly["Date"]  = pd.to_datetime(
    monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2) + "-01",
    format="%Y-%m-%d"
)

# Sort cleanly
monthly = monthly.sort_values(["Block", "Location", "Date"]).reset_index(drop=True)

print(f"  Monthly rows created : {len(monthly):,}")
print(f"  Unique locations     : {monthly['Location'].nunique()}")
print(f"  Blocks               : {sorted(monthly['Block'].unique())}")
print(f"  Date range           : {monthly['Date'].min().date()} → {monthly['Date'].max().date()}")


print("\nCOVERAGE SUMMARY (months of data per location per year)")
coverage = monthly.groupby(["Block", "Location", "Year"]).size().reset_index(name="months_with_data")
piv = coverage.pivot_table(index=["Block","Location"], columns="Year", values="months_with_data", fill_value=0)
print(piv.to_string())

total_ts_points = len(monthly)
n_locs = monthly["Location"].nunique()
print(f"\n  Total pooled time-series points : {total_ts_points}")
print(f"  Mean months per location        : {total_ts_points / n_locs:.1f}")

# Warn about sparse locations
total_per_loc = monthly.groupby("Location").size()
sparse = total_per_loc[total_per_loc < 6]
if len(sparse):
    print(f"\n  WARNING — {len(sparse)} locations have fewer than 6 monthly data points total:")
    print(f"  {list(sparse.index)}")
    print("  Consider excluding these from the forecasting model.")

# Save
monthly.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print("\n" + "=" * 70)
print("MONTHLY AGGREGATION COMPLETE")
print("=" * 70)