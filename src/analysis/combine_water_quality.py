"""Combine the water-quality Excel workbooks into one CSV file, and add a WQI column."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import pandas as pd

# Keywords we look for in a header row. A row only needs to match enough
# of these (not all four) to be accepted as the table header, since real
# sheets vary (e.g. some omit "District", some spell "Sl.No" differently).
HEADER_KEYWORDS = {
    "serial": [r"^s\.?\s*no\.?$", r"^sl\.?\s*no\.?$"],
    "state": [r"^state"],
    "district": [r"^district"],
    "block": [r"^block"],
}
MIN_HEADER_MATCHES = 2  # require at least this many keyword groups to match

# ---------------------------------------------------------------------------
# WQI calculation (mirrors the reference calc_wqi.py exactly).
# ---------------------------------------------------------------------------
WQI_STANDARDS = {
    "pH (NA)": {"Sn": 8.5, "Videal": 7},
    "TDS (mg/l)": {"Sn": 500, "Videal": 0},
    "Total Hardness (As CaCO3) (mg/l)": {"Sn": 200, "Videal": 0},
    "Chloride (as Cl) (mg/l)": {"Sn": 250, "Videal": 0},
    "Fluoride (as F) (mg/l)": {"Sn": 1.0, "Videal": 0},
    "Total Alkalinity (as Calcium Carbonate) (mg/l)": {"Sn": 200, "Videal": 0},
    "Sulphate (as SO4) (mg/l)": {"Sn": 200, "Videal": 0},
    "Nitrate (as NO3) (mg/l)": {"Sn": 45, "Videal": 0},
}

# If automatic column matching (below) picks the wrong column for a
# parameter, add an explicit override here: standards key -> exact column
# name as it appears in the combined data. Left empty by default.
COLUMN_OVERRIDES: dict[str, str] = {}


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _matches_any(cell: str, patterns: list[str]) -> bool:
    return any(re.search(p, cell) for p in patterns)


def _normalize_param_name(name: str) -> str:
    """Strip units/parentheticals and punctuation for fuzzy column matching."""
    text = name.lower()
    text = re.sub(r"\(as\s+([^)]*)\)", r"as \1", text)  # "(as Cl)" -> "as cl"
    text = re.sub(r"\([^)]*\)", " ", text)  # drop remaining parentheticals (units)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_table_sheet(workbook_path: Path, scan_rows: int = 30):
    """Find the header row on the SECOND worksheet (index 1) of the workbook.

    Returns (sheet_name, header_row_index, best_row_cells) on success,
    or (None, None, best_row_cells) on failure — best_row_cells is the
    closest-matching row found, for diagnostics.
    """
    workbook = pd.ExcelFile(workbook_path)
    if len(workbook.sheet_names) < 2:
        return None, None, [f"workbook only has {len(workbook.sheet_names)} sheet(s), expected >= 2"]

    sheet_name = workbook.sheet_names[1]
    preview = pd.read_excel(workbook, sheet_name=sheet_name, header=None, nrows=scan_rows)

    best_match_count = -1
    best_row_cells: list[str] = []
    for row_number, row in preview.iterrows():
        cells = [_normalize(v) for v in row.dropna()]
        if not cells:
            continue
        match_count = sum(
            1 for patterns in HEADER_KEYWORDS.values()
            if any(_matches_any(c, patterns) for c in cells)
        )
        if match_count > best_match_count:
            best_match_count = match_count
            best_row_cells = cells
        if match_count >= MIN_HEADER_MATCHES:
            return sheet_name, int(row_number), cells

    return None, None, best_row_cells


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and remove known Excel schema artifacts."""
    frame = frame.copy()
    frame.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in frame.columns]
    unnamed = [c for c in frame.columns if c.lower().startswith("unnamed")]
    if unnamed:
        print(f"  Removing {len(unnamed)} Excel-generated Unnamed column(s)")
        frame = frame.drop(columns=unnamed)

    for artifact in ("Contaminated parameters", "Chemical parameters"):
        if artifact in frame.columns:
            frame = frame.drop(columns=artifact)

    standard_tds = "TDS (mg/l)"
    if "TDS" in frame.columns:
        if standard_tds in frame.columns:
            frame[standard_tds] = frame[standard_tds].combine_first(frame["TDS"])
            frame = frame.drop(columns="TDS")
        else:
            frame = frame.rename(columns={"TDS": standard_tds})

    return frame


def normalize_combined_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Consolidate state/year fields and discard report-footer rows."""
    frame = frame.copy()

    state_columns = [
        column
        for column in frame.columns
        if column == "State" or column.startswith("State (")
    ]
    if state_columns:
        state = pd.Series(pd.NA, index=frame.index, dtype="object")
        for column in state_columns:
            values = frame[column].astype("string").str.strip()
            values = values.mask(values.str.startswith("Report Printed On", na=False))
            values = values.mask(values.eq(""))
            state = state.fillna(values)
        frame["State"] = state
        columns_to_drop = [column for column in state_columns if column != "State"]
        if columns_to_drop:
            frame = frame.drop(columns=columns_to_drop)

    if "Year" in frame.columns:
        frame = frame.drop(columns="Year")

    serial_numbers = pd.to_numeric(frame["S.No."], errors="coerce")
    footer_rows = serial_numbers.isna() | serial_numbers.le(0)
    if footer_rows.any():
        print(f"  Removing {int(footer_rows.sum())} non-data report/footer row(s)")
        frame = frame.loc[~footer_rows].reset_index(drop=True)

    return frame


def build_column_map(available_columns: list[str]) -> dict[str, str]:
    """Match each WQI_STANDARDS parameter to a real column name.

    Uses exact match first, then normalized (unit/punctuation-stripped)
    match, then closest fuzzy match as a last resort. Raises with a
    detailed message (including near-miss candidates) if any parameter
    can't be confidently matched, rather than silently guessing wrong.
    """
    normalized_lookup: dict[str, str] = {}
    for col in available_columns:
        normalized_lookup.setdefault(_normalize_param_name(col), col)

    column_map: dict[str, str] = {}
    problems: list[str] = []

    for param in WQI_STANDARDS:
        if param in COLUMN_OVERRIDES:
            override = COLUMN_OVERRIDES[param]
            if override not in available_columns:
                problems.append(f"  - {param!r}: override {override!r} not found in data columns")
            else:
                column_map[param] = override
            continue

        if param in available_columns:
            column_map[param] = param
            continue

        norm_param = _normalize_param_name(param)
        if norm_param in normalized_lookup:
            column_map[param] = normalized_lookup[norm_param]
            continue

        close = difflib.get_close_matches(norm_param, normalized_lookup.keys(), n=3, cutoff=0.6)
        if len(close) == 1:
            column_map[param] = normalized_lookup[close[0]]
        else:
            candidates = [normalized_lookup[c] for c in close] if close else "(no close matches)"
            problems.append(f"  - {param!r}: no confident match; closest candidates: {candidates}")

    if problems:
        raise ValueError(
            "Could not confidently match every WQI parameter to a data column:\n"
            + "\n".join(problems)
            + "\n\nAvailable columns:\n  " + "\n  ".join(available_columns)
            + "\n\nAdd an explicit entry to COLUMN_OVERRIDES at the top of this script to fix a mismatch."
        )
    return column_map


def calc_qn(value: float, Sn: float, Videal: float) -> float:
    qn = ((value - Videal) / (Sn - Videal)) * 100
    qn = min(qn, 300)
    return max(qn, 0)


def add_wqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a WQI column at the end, computed exactly like the reference script."""
    column_map = build_column_map(list(df.columns))
    print("WQI parameter -> data column mapping:")
    for param, col in column_map.items():
        marker = " (override)" if param in COLUMN_OVERRIDES else ""
        print(f"  {param!r} -> {col!r}{marker}")

    K = 1 / sum(1 / spec["Sn"] for spec in WQI_STANDARDS.values())
    weights = {param: K / spec["Sn"] for param, spec in WQI_STANDARDS.items()}

    numeric = {
        param: pd.to_numeric(df[col], errors="coerce")
        for param, col in column_map.items()
    }

    wqi = pd.Series(0.0, index=df.index)
    any_missing = pd.Series(False, index=df.index)
    for param, spec in WQI_STANDARDS.items():
        values = numeric[param]
        any_missing |= values.isna()
        qn = ((values - spec["Videal"]) / (spec["Sn"] - spec["Videal"])) * 100
        qn = qn.clip(lower=0, upper=300)
        wqi = wqi + qn.fillna(0) * weights[param]

    wqi[any_missing] = pd.NA  # don't report a partial/wrong WQI for incomplete rows

    missing_count = int(any_missing.sum())
    if missing_count:
        print(
            f"Note: {missing_count} of {len(df)} rows are missing/non-numeric in at least one "
            "WQI parameter column; WQI left blank for those rows."
        )

    df = df.copy()
    df["WQI"] = wqi
    return df


def combine_workbooks(input_dir: Path, output_file: Path) -> tuple[int, list[str]]:
    """Read all data workbooks below input_dir and write one combined CSV with WQI."""
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir.resolve()}")

    all_xlsx = sorted(input_dir.rglob("*.xlsx"))
    real_xlsx = [p for p in all_xlsx if not p.name.startswith("~$")]
    print(f"Found {len(real_xlsx)} .xlsx file(s) under {input_dir.resolve()}")
    if not real_xlsx:
        raise RuntimeError(
            f"No .xlsx files found under {input_dir.resolve()}. "
            "Check the --input-dir path and that files end in .xlsx (not .xls)."
        )

    for workbook_path in real_xlsx:
        try:
            sheet_name, header_row, near_match = find_table_sheet(workbook_path)
            if sheet_name is None:
                skipped.append(
                    f"{workbook_path}: table header not found on 2nd sheet "
                    f"(closest row seen: {near_match!r})"
                )
                continue

            frame = pd.read_excel(
                workbook_path,
                sheet_name=sheet_name,
                header=header_row,
            ).dropna(how="all")
            frame = clean_columns(frame)
            if frame.empty:
                skipped.append(f"{workbook_path}: table is empty")
                continue

            relative_path = workbook_path.relative_to(input_dir)
            block_folder = relative_path.parts[0]
            year_folder = relative_path.parts[1]
            # Some blocks (e.g. "Salem") prefix the year folder with the
            # block name ("Salem 2022-2023") — strip that redundant prefix
            # so "Academic year" is consistently just the year range.
            academic_year = re.sub(rf"^{re.escape(block_folder)}\s*", "", year_folder).strip()
            location = workbook_path.stem  # filename without .xlsx, e.g. "Andipatti"

            # The source workbooks already contain their own "State" /
            # "District" / "Block" columns (that's what header detection
            # matches on), so our folder-derived metadata columns use a
            # distinct prefix to avoid colliding with real data columns.
            meta_columns = {
                "Source file": relative_path.as_posix(),
                "Folder: Block": block_folder,
                "Folder: Academic year": academic_year,
                "Folder: Location": location,
            }
            for offset, (col_name, value) in enumerate(meta_columns.items()):
                if col_name in frame.columns:
                    raise ValueError(
                        f"metadata column {col_name!r} collides with an existing data column"
                    )
                frame.insert(offset, col_name, value)
            frames.append(frame)
        except Exception as error:  # Keep one malformed workbook from stopping the merge.
            skipped.append(f"{workbook_path}: {type(error).__name__}: {error}")

    if not frames:
        detail = "\n".join(f"- {m}" for m in skipped) or "(no per-file diagnostics captured)"
        raise RuntimeError(
            f"Found {len(real_xlsx)} .xlsx file(s) but none produced readable data.\n{detail}"
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = normalize_combined_columns(combined)
    combined = add_wqi_column(combined)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)
    return len(frames), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"C:\Mohit\Repositories\WQI_Research_VIT\src\data\Water quality data- 5 blocks"),
        help="Folder containing the block/year Excel files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\Mohit\Repositories\WQI_Research_VIT\src\data\combined_water_quality_data.csv"),
        help="Path for the combined CSV.",
    )
    args = parser.parse_args()

    files_read, skipped = combine_workbooks(args.input_dir, args.output)
    print(f"Combined {files_read} workbooks into {args.output}.")
    if skipped:
        print(f"Skipped {len(skipped)} files:")
        for message in skipped:
            print(f"- {message}")


if __name__ == "__main__":
    main()