import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path("src/data/Water quality data- 5 blocks")
OUTPUT_DIR = Path("src/temporal_results")

PARAMETERS = [
    "pH (NA)",
    "TDS (mg/l)",
    "Total Hardness (As CaCO3) (mg/l)",
    "Chloride (as Cl) (mg/l)",
    "Fluoride (as F) (mg/l)",
    "Total Alkalinity (as Calcium Carbonate) (mg/l)",
    "Sulphate (as SO4) (mg/l)",
    "Nitrate (as NO3) (mg/l)",
]

STANDARDS = {
    "pH (NA)": {"Sn": 8.5, "Videal": 7},
    "TDS (mg/l)": {"Sn": 500, "Videal": 0},
    "Total Hardness (As CaCO3) (mg/l)": {"Sn": 200, "Videal": 0},
    "Chloride (as Cl) (mg/l)": {"Sn": 250, "Videal": 0},
    "Fluoride (as F) (mg/l)": {"Sn": 1.0, "Videal": 0},
    "Total Alkalinity (as Calcium Carbonate) (mg/l)": {"Sn": 200, "Videal": 0},
    "Sulphate (as SO4) (mg/l)": {"Sn": 200, "Videal": 0},
    "Nitrate (as NO3) (mg/l)": {"Sn": 45, "Videal": 0},
}

K = 1 / sum(1 / spec["Sn"] for spec in STANDARDS.values())
WEIGHTS = {param: K / spec["Sn"] for param, spec in STANDARDS.items()}


def should_skip(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(("~", ".", "identifier")) or name.endswith((".tmp", ".bak"))


def extract_year(folder_name: str) -> int | None:
    match = re.search(r"(\d{4})", folder_name)
    return int(match.group(1)) if match else None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"\s*\(\d{4}-\d{4}\)", "", str(col)).strip()
        for col in df.columns
    ]
    return df


def parse_date(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (int, float)) and 30000 <= float(value) <= 60000:
        return pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(value, errors="coerce", dayfirst=True)


def calc_qn(value: float, sn: float, videal: float) -> float:
    if pd.isna(value):
        return np.nan
    qn = ((value - videal) / (sn - videal)) * 100
    return min(max(qn, 0), 300)


def calc_wqi(row: pd.Series) -> float:
    total = 0.0
    for param, spec in STANDARDS.items():
        qn = calc_qn(row[param], spec["Sn"], spec["Videal"])
        if pd.isna(qn):
            return np.nan
        total += qn * WEIGHTS[param]
    return total


def decimal_year(ts: pd.Timestamp) -> float:
    year_start = pd.Timestamp(year=ts.year, month=1, day=1)
    next_year = pd.Timestamp(year=ts.year + 1, month=1, day=1)
    return ts.year + (ts - year_start).days / (next_year - year_start).days


def load_raw_records(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    errors = []

    for block_dir in sorted(data_root.iterdir()):
        if not block_dir.is_dir() or should_skip(block_dir):
            continue
        for year_dir in sorted(block_dir.iterdir()):
            if not year_dir.is_dir() or should_skip(year_dir):
                continue
            folder_year = extract_year(year_dir.name)
            for excel_file in sorted(year_dir.iterdir()):
                if should_skip(excel_file) or excel_file.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
                    continue
                try:
                    xls = pd.ExcelFile(excel_file)
                    if len(xls.sheet_names) < 2:
                        continue
                    df = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
                    df = normalize_columns(df)
                    sno_col = next(
                        (c for c in df.columns if c.lower().replace(" ", "") in {"s.no.", "s.no"}),
                        None,
                    )
                    if sno_col:
                        df = df[pd.to_numeric(df[sno_col], errors="coerce").fillna(0) > 0]
                    if df.empty:
                        continue

                    df["_source_block_folder"] = block_dir.name
                    df["_source_year_folder"] = year_dir.name
                    df["_folder_year"] = folder_year
                    df["_source_file"] = excel_file.name
                    records.append(df)
                except Exception as exc:
                    errors.append({"file": str(excel_file), "error": str(exc)})

    if not records:
        raise RuntimeError(f"No Excel records loaded from {data_root}")

    combined = pd.concat(records, ignore_index=True, sort=False)
    combined.columns = [str(col).strip() for col in combined.columns]
    return combined, pd.DataFrame(errors)


def prepare_observed_monthly(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["Block", "Village", "Sample Collection date", *PARAMETERS]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw[required + ["_folder_year", "_source_file"]].copy()
    df["Sample Collection date"] = df["Sample Collection date"].map(parse_date)
    df = df.dropna(subset=["Sample Collection date", "Block", "Village"])

    parsed_year = df["Sample Collection date"].dt.year
    folder_year = pd.to_numeric(df["_folder_year"], errors="coerce")
    year_mismatch = folder_year.notna() & (parsed_year != folder_year)

    # Preserve parsed month/day but use folder year where source files contain placeholder years.
    df.loc[year_mismatch, "Sample Collection date"] = [
        ts.replace(year=int(fy))
        for ts, fy in zip(df.loc[year_mismatch, "Sample Collection date"], folder_year[year_mismatch])
    ]

    for param in PARAMETERS:
        df[param] = pd.to_numeric(df[param], errors="coerce")

    df["WQI"] = df.apply(calc_wqi, axis=1)
    df = df.dropna(subset=["WQI"])
    df["Month"] = df["Sample Collection date"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["Month"].dt.year
    df["Decimal_Year"] = df["Month"].map(decimal_year)

    value_cols = PARAMETERS + ["WQI"]
    monthly = (
        df.groupby(["Block", "Village", "Month"], as_index=False)
        .agg({**{col: "mean" for col in value_cols}, "Sample Collection date": "count"})
        .rename(columns={"Sample Collection date": "n_samples"})
    )
    monthly["Year"] = monthly["Month"].dt.year
    monthly["Decimal_Year"] = monthly["Month"].map(decimal_year)

    date_quality = pd.DataFrame(
        {
            "metric": ["raw_records", "usable_records", "date_year_folder_mismatches"],
            "value": [len(raw), len(df), int(year_mismatch.sum())],
        }
    )
    return monthly, date_quality


def interpolate_short_monthly_gaps(monthly: pd.DataFrame, max_gap_months: int = 4) -> pd.DataFrame:
    value_cols = PARAMETERS + ["WQI"]
    frames = []

    for (block, village), grp in monthly.groupby(["Block", "Village"], sort=True):
        grp = grp.sort_values("Month").set_index("Month")
        full_index = pd.date_range(grp.index.min(), grp.index.max(), freq="MS")
        out = grp.reindex(full_index)
        out.index.name = "Month"
        out["Block"] = block
        out["Village"] = village
        out["Observed"] = out["n_samples"].notna()
        out["n_samples"] = out["n_samples"].fillna(0).astype(int)
        if len(out) > 1:
            local_limit = min(max_gap_months, len(out) - 1)
            out[value_cols] = out[value_cols].interpolate(
                method="linear",
                limit=local_limit,
                limit_area="inside",
            )
        out = out.dropna(subset=["WQI"])
        out["Interpolated"] = ~out["Observed"]
        frames.append(out.reset_index())

    result = pd.concat(frames, ignore_index=True)
    result["Year"] = result["Month"].dt.year
    result["Decimal_Year"] = result["Month"].map(decimal_year)
    return result[
        ["Block", "Village", "Month", "Year", "Decimal_Year", "Observed", "Interpolated", "n_samples", *value_cols]
    ]


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def mann_kendall_sen(x: np.ndarray, y: np.ndarray) -> dict[str, float | str | int]:
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    n = len(y)
    if n < 3:
        return {
            "n_points": n,
            "mk_s": np.nan,
            "mk_z": np.nan,
            "mk_p": np.nan,
            "sen_slope_per_year": np.nan,
            "trend": "insufficient",
        }

    slopes = []
    s_value = 0
    for i in range(n - 1):
        dy = y[i + 1 :] - y[i]
        dx = x[i + 1 :] - x[i]
        s_value += int(np.nansum(np.sign(dy)))
        valid_dx = dx != 0
        slopes.extend((dy[valid_dx] / dx[valid_dx]).tolist())

    _, tie_counts = np.unique(y, return_counts=True)
    tie_term = sum(count * (count - 1) * (2 * count + 5) for count in tie_counts if count > 1)
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    if variance == 0:
        z_value = 0.0
    elif s_value > 0:
        z_value = (s_value - 1) / math.sqrt(variance)
    elif s_value < 0:
        z_value = (s_value + 1) / math.sqrt(variance)
    else:
        z_value = 0.0

    p_value = 2 * (1 - normal_cdf(abs(z_value)))
    sen_slope = float(np.nanmedian(slopes)) if slopes else np.nan
    if p_value < 0.05 and sen_slope > 0:
        trend = "increasing"
    elif p_value < 0.05 and sen_slope < 0:
        trend = "decreasing"
    else:
        trend = "no significant trend"

    return {
        "n_points": n,
        "mk_s": s_value,
        "mk_z": z_value,
        "mk_p": p_value,
        "sen_slope_per_year": sen_slope,
        "trend": trend,
    }


def build_trend_table(monthly_filled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trend_parameters = PARAMETERS + ["WQI"]
    for (block, village), grp in monthly_filled.groupby(["Block", "Village"], sort=True):
        x = grp["Decimal_Year"].to_numpy(dtype=float)
        for param in trend_parameters:
            result = mann_kendall_sen(x, grp[param].to_numpy(dtype=float))
            rows.append({"Block": block, "Village": village, "Parameter": param, **result})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, errors = load_raw_records(DATA_ROOT)
    monthly_observed, date_quality = prepare_observed_monthly(raw)
    monthly_filled = interpolate_short_monthly_gaps(monthly_observed, max_gap_months=4)
    trend_table = build_trend_table(monthly_filled)

    coverage = (
        monthly_filled.groupby(["Block", "Village"], as_index=False)
        .agg(
            first_month=("Month", "min"),
            last_month=("Month", "max"),
            total_months=("Month", "count"),
            observed_months=("Observed", "sum"),
            interpolated_months=("Interpolated", "sum"),
            mean_wqi=("WQI", "mean"),
        )
    )
    coverage["observed_fraction"] = coverage["observed_months"] / coverage["total_months"]

    block_summary = (
        trend_table[trend_table["Parameter"] == "WQI"]
        .groupby(["Block", "trend"], as_index=False)
        .size()
        .rename(columns={"size": "n_villages"})
    )

    monthly_observed.to_csv(OUTPUT_DIR / "monthly_observed_wqi_parameters.csv", index=False)
    monthly_filled.to_csv(OUTPUT_DIR / "monthly_interpolated_wqi_parameters.csv", index=False)
    trend_table.to_csv(OUTPUT_DIR / "mann_kendall_sen_trends.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "monthly_coverage_summary.csv", index=False)
    block_summary.to_csv(OUTPUT_DIR / "wqi_trend_block_summary.csv", index=False)
    date_quality.to_csv(OUTPUT_DIR / "date_quality_summary.csv", index=False)
    errors.to_csv(OUTPUT_DIR / "load_errors.csv", index=False)

    print("Temporal analysis complete")
    print(f"Raw records loaded: {len(raw)}")
    print(f"Observed monthly rows: {len(monthly_observed)}")
    print(f"Interpolated monthly rows: {int(monthly_filled['Interpolated'].sum())}")
    print(f"Trend tests: {len(trend_table)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
