from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from pymannkendall import original_test
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.structural import UnobservedComponents


PARAMETER_COLS = [
    "pH_mean",
    "TDS_mean",
    "Hardness_mean",
    "Chloride_mean",
    "Fluoride_mean",
    "Alkalinity_mean",
    "Sulphate_mean",
    "Nitrate_mean",
]


def _decimal_year(dates: pd.Series) -> pd.Series:
    return dates.dt.year + (dates.dt.month - 1) / 12.0


def _median_pairwise_slope(years: np.ndarray, values: np.ndarray) -> float:
    slopes: list[float] = []
    for idx in range(len(values) - 1):
        year_deltas = years[idx + 1 :] - years[idx]
        valid = year_deltas != 0
        if np.any(valid):
            slopes.extend(((values[idx + 1 :][valid] - values[idx]) / year_deltas[valid]).tolist())
    return float(np.median(slopes)) if slopes else np.nan


def compute_sens_slopes(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Sen's slope features for each (Block, Location) time series.

    The slope columns are numeric units per year. Callers should pass only the
    training portion of the monthly data to avoid leaking holdout trends.
    """
    rows: list[dict[str, object]] = []
    for (block, location), group in monthly_df.groupby(["Block", "Location"], sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        decimal_year = _decimal_year(group["Date"])
        row: dict[str, object] = {"Block": block, "Location": location}
        for param in PARAMETER_COLS:
            series = pd.DataFrame(
                {"decimal_year": decimal_year, "value": group[param]}
            ).dropna(subset=["value"])
            if len(series) < 4:
                row[f"{param}_sens_slope"] = np.nan
                row[f"{param}_mk_trend"] = "unknown"
                continue
            try:
                row[f"{param}_sens_slope"] = _median_pairwise_slope(
                    series["decimal_year"].to_numpy(dtype=float),
                    series["value"].to_numpy(dtype=float),
                )
                row[f"{param}_mk_trend"] = original_test(series["value"].to_numpy()).trend
            except Exception:
                row[f"{param}_sens_slope"] = np.nan
                row[f"{param}_mk_trend"] = "unknown"
        rows.append(row)
    return pd.DataFrame(rows)


def attach_sens_slopes(feature_df: pd.DataFrame, slopes_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join Sen's slope features onto a feature dataset."""
    return feature_df.merge(slopes_df, on=["Block", "Location"], how="left")


def get_sens_slope_feature_columns() -> list[str]:
    """Return the numeric Sen's slope columns used as model features."""
    return [f"{param}_sens_slope" for param in PARAMETER_COLS]



def kalman_fill_location(group: pd.DataFrame, param: str) -> pd.DataFrame:
    """
    Add a Kalman-filled value column for one parameter at one location.

    Original observed values are passed through unchanged. Only missing monthly
    gap rows receive smoothed values. If the state-space model cannot be fit,
    the function falls back to forward/backward fill for the lag source only.
    """
    group = group.sort_values("Date").copy()
    series = group.set_index("Date")[param].asfreq("MS")
    observed_mask = ~series.isna()
    filled_col = f"{param}_kalman_filled"

    if observed_mask.sum() < 3:
        fallback = series.ffill().bfill()
        group[filled_col] = group["Date"].map(fallback)
        return group

    try:
        model = UnobservedComponents(series, level="local linear trend")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = model.fit(disp=False, maxiter=200)
        smoothed_level = pd.Series(result.smoothed_state[0], index=series.index)
        filled_series = series.copy()
        filled_series.loc[~observed_mask] = smoothed_level.loc[~observed_mask]
        group[filled_col] = group["Date"].map(filled_series)
    except Exception:
        fallback = series.ffill().bfill()
        group[filled_col] = group["Date"].map(fallback)

    return group


def kalman_fill_monthly_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete monthly location spines and Kalman-fill parameter gaps.

    Returned rows include original observations plus synthetic gap months.
    The `is_observed` flag marks real rows; callers must use only those rows
    as prediction targets and use synthetic rows only for lag lookup.
    """
    groups: list[pd.DataFrame] = []
    for (block, location), group in monthly_df.groupby(["Block", "Location"], sort=False):
        group = group.sort_values("Date").reset_index(drop=True).copy()
        group["is_observed"] = True
        full_dates = pd.date_range(group["Date"].min(), group["Date"].max(), freq="MS")
        spine = pd.DataFrame({"Date": full_dates, "Block": block, "Location": location})
        merged = spine.merge(group, on=["Block", "Location", "Date"], how="left")
        merged["is_observed"] = merged["is_observed"].fillna(False).astype(bool)

        for param in PARAMETER_COLS:
            merged = kalman_fill_location(merged, param)

        groups.append(merged)

    if not groups:
        return pd.DataFrame(columns=["Block", "Location", "Date", "is_observed"])

    return (
        pd.concat(groups, ignore_index=True)
        .sort_values(["Block", "Location", "Date"])
        .reset_index(drop=True)
    )
