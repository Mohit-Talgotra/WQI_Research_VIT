"""Combine the water-quality Excel workbooks into one CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_table_sheet(workbook_path: Path) -> tuple[str, int] | None:
    """Find the worksheet and row containing the tabular data header."""
    workbook = pd.ExcelFile(workbook_path)
    for sheet_name in workbook.sheet_names:
        preview = pd.read_excel(workbook, sheet_name=sheet_name, header=None, nrows=20)
        for row_number, row in preview.iterrows():
            names = {
                str(value).strip().lower()
                for value in row.dropna()
            }
            has_serial_number = "s.no." in names
            has_state = any(name.startswith("state") for name in names)
            if has_serial_number and has_state and {"district", "block"}.issubset(names):
                return sheet_name, int(row_number)
    return None


def combine_workbooks(input_dir: Path, output_file: Path) -> tuple[int, list[str]]:
    """Read all data workbooks below input_dir and write one combined CSV."""
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for workbook_path in sorted(input_dir.rglob("*.xlsx")):
        if workbook_path.name.startswith("~$"):
            continue

        try:
            table_location = find_table_sheet(workbook_path)
            if table_location is None:
                skipped.append(f"{workbook_path}: table header not found")
                continue

            sheet_name, header_row = table_location
            frame = pd.read_excel(
                workbook_path,
                sheet_name=sheet_name,
                header=header_row,
            ).dropna(how="all")
            if frame.empty:
                skipped.append(f"{workbook_path}: table is empty")
                continue

            relative_path = workbook_path.relative_to(input_dir)
            frame.insert(0, "Source file", relative_path.as_posix())
            frame.insert(1, "Block folder", relative_path.parts[0])
            frame.insert(2, "Academic year", relative_path.parts[1])
            frames.append(frame)
        except Exception as error:  # Keep one malformed workbook from stopping the merge.
            skipped.append(f"{workbook_path}: {error}")

    if not frames:
        raise RuntimeError(f"No readable data workbooks found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(output_file, index=False)
    return len(frames), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("src/data/Water quality data- 5 blocks"),
        help="Folder containing the block/year Excel files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/data/combined_water_quality_data.csv"),
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