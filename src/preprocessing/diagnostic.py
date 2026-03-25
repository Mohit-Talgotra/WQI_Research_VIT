import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Config
DATA_ROOT = os.environ.get("DATA_ROOT")
if not DATA_ROOT:
    raise ValueError("Add DATA_ROOT='path/to/Water quality data- 5 blocks' to your .env file")

DATA_ROOT = Path(DATA_ROOT)
OUTPUT_DIR = Path("diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)

# Files to skip (WSL artifacts, temp files, identifier files, etc.)
SKIP_PREFIXES = ("~", ".", "identifier")
SKIP_SUFFIXES = (".tmp", ".bak")

# Helpers
def should_skip(filename: str) -> bool:
    name = filename.lower()
    if any(name.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(name.endswith(s) for s in SKIP_SUFFIXES):
        return True
    return False


def extract_year_from_folder(folder_name: str):
    """Pull the first 4-digit year out of a folder name like '2022-2023' or '2022- 2023'."""
    match = re.search(r"(\d{4})", folder_name)
    return int(match.group(1)) if match else None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and year suffixes from column names."""
    df.columns = [re.sub(r"\s*\(\d{4}-\d{4}\)", "", c).strip() for c in df.columns]
    return df


# Load all data
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

all_dfs = []
load_errors = []

for block_dir in sorted(DATA_ROOT.iterdir()):
    if not block_dir.is_dir() or should_skip(block_dir.name):
        continue
    block_name = block_dir.name

    for year_dir in sorted(block_dir.iterdir()):
        if not year_dir.is_dir() or should_skip(year_dir.name):
            continue
        year = extract_year_from_folder(year_dir.name)
        if year is None:
            print(f"  WARN: Could not parse year from folder '{year_dir.name}', skipping")
            continue

        for excel_file in sorted(year_dir.iterdir()):
            if should_skip(excel_file.name):
                continue
            if excel_file.suffix.lower() not in (".xlsx", ".xls", ".xlsm"):
                continue

            try:
                xls = pd.ExcelFile(excel_file, engine="openpyxl")
                if len(xls.sheet_names) < 2:
                    print(f"  WARN: Less than 2 sheets in {excel_file.name}, skipping")
                    continue

                df = pd.read_excel(xls, sheet_name=xls.sheet_names[1], engine="openpyxl")

                if all("Unnamed" in str(c) for c in df.columns):
                    df.columns = df.iloc[0].astype(str).str.strip()
                    df = df.iloc[1:].reset_index(drop=True)
                df = normalize_columns(df)

                # Drop rows where S.No is not a positive integer
                sno_col = next(
                    (c for c in df.columns if c.strip().lower() in ("s.no.", "s. no.", "s.no")),
                    None
                )
                if sno_col:
                    df = df[pd.to_numeric(df[sno_col], errors="coerce").fillna(0) > 0]

                if df.empty:
                    continue

                df["_block"]       = block_name
                df["_year_folder"] = year
                df["_source_file"] = excel_file.name

                all_dfs.append(df)
                print(f"  OK  {block_name} / {year_dir.name} / {excel_file.name}  ({len(df)} rows)")

            except Exception as e:
                load_errors.append((str(excel_file), str(e)))
                print(f"  ERR {excel_file.name}: {e}")

if not all_dfs:
    raise RuntimeError("No data loaded. Check DATA_ROOT and folder structure.")

combined = pd.concat(all_dfs, ignore_index=True)
combined.columns = [c.strip() for c in combined.columns]

print(f"\n  Loaded {len(combined)} total records from {len(all_dfs)} files")
if load_errors:
    print(f"  {len(load_errors)} files failed — see diagnostics/load_errors.txt")
    with open(OUTPUT_DIR / "load_errors.txt", "w") as f:
        for path, err in load_errors:
            f.write(f"{path}\n  → {err}\n\n")

# Detect key columns
print("\n  Sample of columns found:", list(combined.columns[:20]))

date_col = next((c for c in combined.columns if "collection date" in c.lower()), None)
if date_col is None:
    raise ValueError("Could not find 'Sample Collection date' column. Check column names printed above.")

location_col = next(
    (c for c in combined.columns if c.lower() in ("village", "gram panchayat", "place")),
    None
)
if location_col is None:
    raise ValueError("Could not find a location column (Village / Gram panchayat / Place).")

print(f"\n  Date column     : '{date_col}'")
print(f"  Location column : '{location_col}'")

# Parse dates
combined[date_col] = pd.to_datetime(combined[date_col], errors="coerce", dayfirst=True)

still_null = combined[date_col].isna() & combined["Sample Collection date"].notna()
if still_null.sum() > 0:
    combined.loc[still_null, date_col] = pd.to_datetime(
        combined.loc[still_null, "Sample Collection date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

before = len(combined)
combined = combined.dropna(subset=[date_col])
dropped = before - len(combined)
if dropped:
    print(f"  Dropped {dropped} rows with unparseable dates")

combined["_year"]  = combined[date_col].dt.year.astype(int)
combined["_month"] = combined[date_col].dt.month.astype(int)
combined["_year_folder"] = combined["_year_folder"].astype(int)

# Warn if date year doesn't match folder year (placeholder dates in some files)
wrong_year = (
    (combined["_year"] != combined["_year_folder"]) &
    (combined["_year"] != combined["_year_folder"] + 1)
)
if wrong_year.sum() > 0:
    print(f"\n  WARN: {wrong_year.sum()} records have date-year != folder-year or folder-year+1")
    print("        These are genuinely mismatched and will be flagged.")

# 1. Basic counts
print("\n" + "=" * 70)
print("1. BASIC COUNTS")
print("=" * 70)
print(f"  Total records    : {len(combined)}")
print(f"  Blocks           : {sorted(combined['_block'].unique())}")
print(f"  Years            : {sorted(combined['_year'].unique())}")
print(f"  Unique locations : {combined[location_col].nunique()}")
print(f"  Date range       : {combined[date_col].min().date()} → {combined[date_col].max().date()}")

# 2. Records per block per year
print("\n" + "=" * 70)
print("2. RECORDS PER BLOCK PER YEAR")
print("=" * 70)
block_year = combined.groupby(["_block", "_year"]).size().unstack(fill_value=0)
print(block_year.to_string())

# 3. Samples per location per year
print("\n" + "=" * 70)
print("3. SAMPLES PER LOCATION PER YEAR")
print("=" * 70)
sply = combined.groupby(["_block", location_col, "_year"]).size().reset_index(name="n_samples")
for block, grp in sply.groupby("_block"):
    print(f"\n  Block: {block}")
    pivot = grp.pivot_table(index=location_col, columns="_year", values="n_samples", fill_value=0)
    print(pivot.to_string())
print(f"\n  Overall mean samples/location/year : {sply['n_samples'].mean():.1f}")
print(f"  Min : {sply['n_samples'].min()}    Max : {sply['n_samples'].max()}")

# 4. Unique dates per location per year
print("\n" + "=" * 70)
print("4. UNIQUE COLLECTION DATES PER LOCATION PER YEAR")
print("=" * 70)
udply = (
    combined.groupby(["_block", location_col, "_year"])[date_col]
            .nunique()
            .reset_index(name="unique_dates")
)
for block, grp in udply.groupby("_block"):
    print(f"\n  Block: {block}")
    pivot = grp.pivot_table(index=location_col, columns="_year", values="unique_dates", fill_value=0)
    print(pivot.to_string())
print(f"\n  Overall mean unique dates/location/year : {udply['unique_dates'].mean():.1f}")

# 5. Duplicate dates
print("\n" + "=" * 70)
print("5. DUPLICATE DATES (same location + same date)")
print("=" * 70)
dup_counts = (
    combined.groupby([location_col, date_col])
            .size()
            .reset_index(name="records_on_same_date")
)
multi = dup_counts[dup_counts["records_on_same_date"] > 1]
print(f"  Location+date combos with >1 record : {len(multi)}")
if len(multi):
    print(f"  Max records on a single date        : {multi['records_on_same_date'].max()}")
    print(f"  Mean when duplicated                : {multi['records_on_same_date'].mean():.1f}")
    print(f"\n  Top 10 most duplicated dates:")
    print(multi.sort_values("records_on_same_date", ascending=False).head(10).to_string(index=False))

# 6. Gap analysis
print("\n" + "=" * 70)
print("6. GAP ANALYSIS (days between consecutive dates per location)")
print("=" * 70)
gaps = []
for (block, loc), grp in combined.groupby(["_block", location_col]):
    dates_sorted = grp[date_col].sort_values().unique()
    if len(dates_sorted) > 1:
        diffs = pd.Series(dates_sorted).diff().dt.days.dropna()
        for d in diffs:
            gaps.append({"block": block, "location": loc, "gap_days": d})

gap_df = pd.DataFrame(gaps)
if len(gap_df):
    print(f"  Median gap : {gap_df['gap_days'].median():.0f} days")
    print(f"  Mean gap   : {gap_df['gap_days'].mean():.1f} days")
    print(f"  Min gap    : {gap_df['gap_days'].min():.0f} days")
    print(f"  Max gap    : {gap_df['gap_days'].max():.0f} days")

    bins   = [0,   1,   7,   30,   90,   180,   365, 99999]
    labels = ["same day", "≤1 week", "≤1 month", "≤1 quarter", "≤6 months", "≤1 year", ">1 year"]
    gap_df["bucket"] = pd.cut(gap_df["gap_days"], bins=bins, labels=labels)
    print(f"\n  Gap distribution:")
    print(gap_df["bucket"].value_counts().sort_index().to_string())
    print(f"\n  Median gap per block:")
    print(gap_df.groupby("block")["gap_days"].median().to_string())

# 7. Monthly density
print("\n" + "=" * 70)
print("7. MONTHLY COLLECTION DENSITY (records per block per month)")
print("=" * 70)
monthly_block = combined.groupby(["_block", "_month"]).size().unstack(fill_value=0)
print(monthly_block.to_string())

# 8. Feasibility verdict
print("\n" + "=" * 70)
print("8. TIME SERIES FORECASTING FEASIBILITY VERDICT")
print("=" * 70)
mean_unique = udply["unique_dates"].mean()
median_gap  = gap_df["gap_days"].median() if len(gap_df) else None
total_locs  = combined[location_col].nunique()

if median_gap is not None:
    if median_gap <= 1:
        res = "DAILY / SUB-DAILY  →  aggregate to weekly or monthly before modelling"
    elif median_gap <= 10:
        res = "ROUGHLY WEEKLY     →  monthly aggregation recommended"
    elif median_gap <= 35:
        res = "ROUGHLY MONTHLY    →  can model at monthly granularity"
    elif median_gap <= 100:
        res = "ROUGHLY QUARTERLY  →  aggregate to quarterly"
    else:
        res = "SPARSE             →  seasonal aggregation only"
    print(f"  Effective resolution          : {res}")

print(f"  Total unique locations        : {total_locs}")
print(f"  Mean unique dates/loc/year    : {mean_unique:.1f}")

q_points = total_locs * 3 * 4
m_points = total_locs * 3 * 12
print(f"  Projected pooled TS points:")
print(f"    Quarterly (4/year × 3 yrs)  : ~{q_points}")
print(f"    Monthly   (12/year × 3 yrs) : ~{m_points}")

if q_points >= 200:
    print(f"\n  ✓ SUFFICIENT for a global pooled time-series model")
elif q_points >= 80:
    print(f"\n  ~ MARGINAL — pooled quarterly model should work but will be limited")
else:
    print(f"\n  ✗ SPARSE — consider seasonal aggregation only")

# 9. Save plots
print("\n" + "=" * 70)
print("9. SAVING PLOTS & CSVs")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Temporal Diagnostic Report — All 5 Blocks", fontsize=16, fontweight="bold")

# A: records per block per year
block_year.T.plot(kind="bar", ax=axes[0, 0], edgecolor="black")
axes[0, 0].set_title("A: Records per Block per Year")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Records")
axes[0, 0].legend(fontsize=7)
axes[0, 0].tick_params(axis="x", rotation=0)
axes[0, 0].grid(alpha=0.3, axis="y")

# B: monthly density heatmap
sns.heatmap(monthly_block, ax=axes[0, 1], cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5)
axes[0, 1].set_title("B: Records per Month per Block")
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Block")

# C: gap histogram
if len(gap_df):
    clipped = gap_df["gap_days"].clip(upper=400)
    axes[0, 2].hist(clipped, bins=40, edgecolor="black", color="steelblue", alpha=0.8)
    axes[0, 2].axvline(gap_df["gap_days"].median(), color="red", linestyle="--",
                       label=f"Median = {gap_df['gap_days'].median():.0f} days")
    axes[0, 2].set_title("C: Gap Between Consecutive Dates")
    axes[0, 2].set_xlabel("Days")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend()

# D: unique dates per location per year box plot
blocks = sorted(udply["_block"].unique())
data_box = [udply[udply["_block"] == b]["unique_dates"].values for b in blocks]
axes[1, 0].boxplot(data_box, labels=blocks)
axes[1, 0].set_title("D: Unique Dates/Location/Year by Block")
axes[1, 0].set_ylabel("Unique dates")
axes[1, 0].tick_params(axis="x", rotation=20)
axes[1, 0].grid(alpha=0.3)

# E: duplicate records distribution
if len(multi):
    dup_hist = dup_counts["records_on_same_date"].value_counts().sort_index()
    axes[1, 1].bar(dup_hist.index.astype(str), dup_hist.values, color="coral", edgecolor="black")
    axes[1, 1].set_title("E: Records Sharing Same Location + Date")
    axes[1, 1].set_xlabel("Records on same date")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(alpha=0.3, axis="y")
else:
    axes[1, 1].text(0.5, 0.5, "No duplicate dates found", ha="center", va="center", fontsize=13)
    axes[1, 1].set_title("E: Duplicate Date Check")

# F: total records per location histogram
loc_counts = combined.groupby(location_col).size()
axes[1, 2].hist(loc_counts.values, bins=20, edgecolor="black", color="mediumseagreen", alpha=0.8)
axes[1, 2].set_title("F: Total Records per Location (all years)")
axes[1, 2].set_xlabel("Total records")
axes[1, 2].set_ylabel("Number of locations")
axes[1, 2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plot_path = OUTPUT_DIR / "temporal_diagnostic.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Plot saved : {plot_path}")

sply.to_csv(OUTPUT_DIR / "samples_per_location_per_year.csv", index=False)
udply.to_csv(OUTPUT_DIR / "unique_dates_per_location_per_year.csv", index=False)
if len(gap_df):
    gap_df.to_csv(OUTPUT_DIR / "gap_analysis.csv", index=False)
print(f"  CSVs saved : {OUTPUT_DIR}/")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)