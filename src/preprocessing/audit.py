import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT  = Path(os.environ["DATA_ROOT"])
AUDIT_DIR  = Path("diagnostics/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SKIP_PREFIXES = ("~", ".", "identifier")
SKIP_SUFFIXES = (".tmp", ".bak")

# Physical plausibility limits — values outside these are almost certainly
# data entry errors (extra zero, decimal in wrong place, etc.)
PARAM_LIMITS = {
    "pH (NA)":                                          (0,    14),
    "TDS (mg/l)":                                       (0,  5000),
    "Total Hardness (As CaCO3) (mg/l)":                 (0,  2000),
    "Chloride (as Cl) (mg/l)":                          (0,  2000),
    "Fluoride (as F) (mg/l)":                           (0,    20),   # >20 is almost impossible naturally
    "Total Alkalinity (as Calcium Carbonate) (mg/l)":   (0,  2000),
    "Sulphate (as SO4) (mg/l)":                         (0,  2000),
    "Nitrate (as NO3) (mg/l)":                          (0,   500),
}

# Helpers
def should_skip(name: str) -> bool:
    n = name.lower()
    return any(n.startswith(p) for p in SKIP_PREFIXES) or \
           any(n.endswith(s)   for s in SKIP_SUFFIXES)

def extract_year(folder_name: str):
    m = re.search(r"(\d{4})", folder_name)
    return int(m.group(1)) if m else None

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s*\(\d{4}-\d{4}\)", "", c).strip() for c in df.columns]
    return df

# Load raw data
print("=" * 70)
print("LOADING ALL RAW DATA")
print("=" * 70)

all_dfs = []

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
                if all("Unnamed" in str(c) for c in df.columns):
                    df.columns = df.iloc[0].astype(str).str.strip()
                    df = df.iloc[1:].reset_index(drop=True)
                
                df = normalize_cols(df)

                sno_col = next(
                    (c for c in df.columns if c.strip().lower() in ("s.no.", "s. no.", "s.no")),
                    None
                )
                if sno_col:
                    df = df[pd.to_numeric(df[sno_col], errors="coerce").fillna(0) > 0]

                if df.empty:
                    continue

                # Track provenance so you know exactly which file to fix
                df["_block"]       = block_dir.name
                df["_folder_year"] = folder_year
                df["_source_file"] = str(excel_file.relative_to(DATA_ROOT))
                all_dfs.append(df)

            except Exception as e:
                print(f"  LOAD ERROR: {excel_file.name} — {e}")

raw = pd.concat(all_dfs, ignore_index=True)
raw.columns = [c.strip() for c in raw.columns]
print(f"  Total records loaded: {len(raw):,}\n")

date_col     = "Sample Collection date"
location_col = next(
    (c for c in raw.columns if c.lower() in ("gram panchayat", "village", "place")),
    None
)

# Keep a copy of the original raw date string for display
raw["_raw_date_str"] = raw[date_col].astype(str)

# Audit 1: Unparseable dates
parsed = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)

still_null = parsed.isna() & raw[date_col].notna()
if still_null.sum() > 0:
    parsed.loc[still_null] = pd.to_datetime(
        raw.loc[still_null, date_col],
        format="%d-%m-%Y",
        errors="coerce"
    )
unparseable_mask = parsed.isna()

unparseable = raw[unparseable_mask][[
    "_source_file", "_block", "_folder_year",
    location_col, date_col, "_raw_date_str"
]].copy()

print(f"ISSUE 1 — Unparseable dates: {len(unparseable)} records")
if len(unparseable):
    print(unparseable["_raw_date_str"].value_counts().head(10).to_string())
    unparseable.to_csv(AUDIT_DIR / "bad_dates_unparseable.csv", index=False)
    print(f"  → saved: bad_dates_unparseable.csv\n")

# Work with parseable dates from here
raw["_parsed_date"] = parsed

# Audit 2: Placeholder dates (1970, future years, etc.)
parseable = raw[~unparseable_mask].copy()
parseable["_date_year"] = parseable["_parsed_date"].dt.year.astype(int)

placeholder_mask = (
    (parseable["_date_year"] < 2000) |   # 1970 default, any old date
    (parseable["_date_year"] > 2025)      # future dates
)

placeholder = parseable[placeholder_mask][[
    "_source_file", "_block", "_folder_year",
    location_col, date_col, "_raw_date_str", "_date_year"
]].copy()

print(f"ISSUE 2 — Placeholder / impossible dates: {len(placeholder)} records")
if len(placeholder):
    print(placeholder.groupby(["_source_file", "_raw_date_str"]).size()
          .sort_values(ascending=False).head(15).to_string())
    placeholder.to_csv(AUDIT_DIR / "bad_dates_placeholder.csv", index=False)
    print(f"  → saved: bad_dates_placeholder.csv\n")

# Audit 3: Date year doesn't match folder year
valid_dates = parseable[~placeholder_mask].copy()
wrong_year_mask = (
    (valid_dates["_date_year"] != valid_dates["_folder_year"]) &
    (valid_dates["_date_year"] != valid_dates["_folder_year"] + 1)
)

wrong_year = valid_dates[wrong_year_mask][[
    "_source_file", "_block", "_folder_year",
    location_col, date_col, "_raw_date_str", "_date_year"
]].copy()
wrong_year["year_difference"] = wrong_year["_date_year"] - wrong_year["_folder_year"]

print(f"ISSUE 3 — Date year ≠ folder year: {len(wrong_year)} records")
if len(wrong_year):
    print("\n  Breakdown by file and year difference:")
    print(wrong_year.groupby(["_source_file", "year_difference"]).size()
          .sort_values(ascending=False).head(20).to_string())

    print("\n  Unique (date_year, folder_year) combos seen:")
    combos = wrong_year.groupby(["_date_year", "_folder_year"]).size().reset_index(name="count")
    print(combos.sort_values("count", ascending=False).to_string(index=False))

    wrong_year.to_csv(AUDIT_DIR / "bad_dates_wrong_year.csv", index=False)
    print(f"\n  → saved: bad_dates_wrong_year.csv\n")

# Audit 4: Non-numeric parameter values
print(f"ISSUE 4 — Non-numeric parameter values:")
nonnumeric_rows = []

for param in PARAM_LIMITS:
    if param not in raw.columns:
        print(f"  MISSING COLUMN: {param}")
        continue

    numeric_vals = pd.to_numeric(raw[param], errors="coerce")
    bad_mask = numeric_vals.isna() & raw[param].notna()   # was present but couldn't convert
    bad_rows  = raw[bad_mask].copy()

    if len(bad_rows):
        bad_rows["_bad_param"]  = param
        bad_rows["_bad_value"]  = bad_rows[param].astype(str)
        nonnumeric_rows.append(bad_rows[[
            "_source_file", "_block", "_folder_year",
            location_col, date_col, "_bad_param", "_bad_value"
        ]])
        print(f"  {param}: {len(bad_rows)} non-numeric values")
        print("    Sample bad values:", raw[bad_mask][param].value_counts().head(5).to_dict())

if nonnumeric_rows:
    nonnumeric_df = pd.concat(nonnumeric_rows, ignore_index=True)
    nonnumeric_df.to_csv(AUDIT_DIR / "bad_parameters_nonnumeric.csv", index=False)
    print(f"  → saved: bad_parameters_nonnumeric.csv\n")
else:
    print("  None found — all parameter values are numeric or null\n")

# Audit 5: Physically impossible / outlier parameter values
print(f"ISSUE 5 — Physically impossible parameter values:")
outlier_rows = []

for param, (lo, hi) in PARAM_LIMITS.items():
    if param not in raw.columns:
        continue

    vals = pd.to_numeric(raw[param], errors="coerce")
    out_mask = vals.notna() & ((vals < lo) | (vals > hi))
    out_rows  = raw[out_mask].copy()

    if len(out_rows):
        out_rows["_bad_param"]  = param
        out_rows["_bad_value"]  = vals[out_mask].values
        out_rows["_limit_low"]  = lo
        out_rows["_limit_high"] = hi
        outlier_rows.append(out_rows[[
            "_source_file", "_block", "_folder_year",
            location_col, date_col, "_bad_param", "_bad_value",
            "_limit_low", "_limit_high"
        ]])
        print(f"  {param}: {len(out_rows)} outliers  (expected {lo}–{hi})")
        print(f"    Min={vals[out_mask].min():.2f}  Max={vals[out_mask].max():.2f}")
        print(f"    Top files: {out_rows['_source_file'].value_counts().head(3).to_dict()}")

if outlier_rows:
    outlier_df = pd.concat(outlier_rows, ignore_index=True)
    outlier_df.to_csv(AUDIT_DIR / "bad_parameters_outliers.csv", index=False)
    print(f"  → saved: bad_parameters_outliers.csv\n")
else:
    print("  None found\n")

# Summary report
summary_lines = [
    "DATA QUALITY AUDIT SUMMARY",
    "=" * 60,
    f"Total records loaded       : {len(raw):,}",
    "",
    "DATE ISSUES",
    f"  Unparseable dates        : {len(unparseable)}",
    f"  Placeholder dates (<2000 or >2025) : {len(placeholder)}",
    f"  Date year ≠ folder year  : {len(wrong_year)}",
    f"  Total date issues        : {len(unparseable) + len(placeholder) + len(wrong_year)}",
    "",
    "PARAMETER ISSUES",
    f"  Non-numeric values       : {sum(len(r) for r in nonnumeric_rows) if nonnumeric_rows else 0}",
    f"  Out-of-range values      : {sum(len(r) for r in outlier_rows) if outlier_rows else 0}",
    "",
    "FILES SAVED TO: diagnostics/audit/",
    "  bad_dates_unparseable.csv",
    "  bad_dates_placeholder.csv",
    "  bad_dates_wrong_year.csv",
    "  bad_parameters_nonnumeric.csv",
    "  bad_parameters_outliers.csv",
]

if len(wrong_year):
    summary_lines += [
        "",
        "WRONG YEAR BREAKDOWN BY FILE:",
    ]
    for fname, count in wrong_year["_source_file"].value_counts().items():
        summary_lines.append(f"  {count:4d} records  →  {fname}")

summary_txt = "\n".join(summary_lines)
print("\n" + summary_txt)

with open(AUDIT_DIR / "summary.txt", "w") as f:
    f.write(summary_txt)

print(f"\nFull summary saved: diagnostics/audit/summary.txt")
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)