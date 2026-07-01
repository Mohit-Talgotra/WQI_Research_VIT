from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.forecasting.panel_forecaster import (  # noqa: E402
    HOLDOUT_START,
    TARGET_COL,
    build_training_dataset,
    load_monthly_data,
)
from src.forecasting.parameter_forecaster import select_backtest_dates  # noqa: E402

OUTPUT_DIR = ROOT / "src" / "data" / "forecasting_gam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["Location"]
MODEL_NAME = "GAM"


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_decimal_year(dataset: pd.DataFrame) -> pd.DataFrame:
    if "decimal_year" in dataset.columns:
        return dataset
    dataset = dataset.copy()
    dataset["decimal_year"] = dataset["Date"].dt.year + (dataset["Date"].dt.month - 1) / 12.0
    return dataset


def get_gam_feature_columns() -> list[str]:
    return [
        "decimal_year",
        "WQI_lag_1",
        "Gap_lag_1_months",
        "Month_sin",
        "Month_cos",
        "Location",
    ]


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        train_out[col] = encoder.fit_transform(train_df[col].astype(str))
        class_to_index = {label: idx for idx, label in enumerate(encoder.classes_)}
        test_out[col] = test_df[col].astype(str).map(lambda value: class_to_index.get(value, 0)).astype(int)
        encoders[col] = encoder

    return train_out, test_out, encoders


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_enc, test_enc, _ = encode_categoricals(train_df, test_df, CATEGORICAL_COLS)
    x_train = train_enc[feature_cols].copy()
    x_test = test_enc[feature_cols].copy()

    numeric_cols = [col for col in feature_cols if col not in CATEGORICAL_COLS]
    train_medians = x_train[numeric_cols].median()
    x_train[numeric_cols] = x_train[numeric_cols].fillna(train_medians)
    x_test[numeric_cols] = x_test[numeric_cols].fillna(train_medians)

    train_mins = x_train[numeric_cols].min()
    train_maxs = x_train[numeric_cols].max()
    x_test[numeric_cols] = x_test[numeric_cols].clip(lower=train_mins, upper=train_maxs, axis=1)

    for col in CATEGORICAL_COLS:
        x_train[col] = x_train[col].astype(int)
        x_test[col] = x_test[col].astype(int)

    return x_train.to_numpy(dtype=float), x_test.to_numpy(dtype=float)


def build_gam() -> LinearGAM:
    return LinearGAM(
        s(0, spline_order=3, n_splines=10)
        + s(1, spline_order=3, n_splines=8)
        + s(2, spline_order=3, n_splines=6)
        + s(3, spline_order=3, n_splines=6)
        + s(4, spline_order=3, n_splines=6)
        + f(5)
    )


def fit_gam(x_train: np.ndarray, y_train: np.ndarray) -> LinearGAM:
    gam = build_gam()
    lam_grid = np.logspace(-3, 3, 13)
    return gam.gridsearch(x_train, y_train, lam=lam_grid, progress=False)


def rolling_origin_gam_backtest(
    dataset: pd.DataFrame,
    backtest_dates: list[pd.Timestamp],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    n_folds = len(backtest_dates)
    for fold_idx, cutoff_date in enumerate(backtest_dates, 1):
        print(f"  [GAM] fold {fold_idx}/{n_folds}: {cutoff_date.date()}", flush=True)
        train_df = dataset[dataset["Date"] < cutoff_date].copy()
        test_df = dataset[dataset["Date"] == cutoff_date].copy()
        if train_df.empty or test_df.empty:
            continue

        x_train, x_test = prepare_xy(train_df, test_df, feature_cols)
        model = fit_gam(x_train, train_df[TARGET_COL].to_numpy(dtype=float))
        predictions = model.predict(x_test)

        metric_rows.append(
            {
                "model": MODEL_NAME,
                "cutoff_date": cutoff_date.date().isoformat(),
                "n_test": int(len(test_df)),
                "rmse": rmse(test_df[TARGET_COL], predictions),
                "mae": float(mean_absolute_error(test_df[TARGET_COL], predictions)),
                "r2": float(r2_score(test_df[TARGET_COL], predictions)),
            }
        )
        for idx, (_, row) in enumerate(test_df.iterrows()):
            prediction_rows.append(
                {
                    "model": MODEL_NAME,
                    "Date": cutoff_date.date().isoformat(),
                    "Block": row["Block"],
                    "Location": row["Location"],
                    "Actual_WQI": float(row[TARGET_COL]),
                    "Predicted_WQI": float(predictions[idx]),
                    "Absolute_Error": float(abs(row[TARGET_COL] - predictions[idx])),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_gam_holdout(dataset: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = dataset[dataset["Date"] < HOLDOUT_START].copy()
    holdout_df = dataset[dataset["Date"] >= HOLDOUT_START].copy()
    if holdout_df.empty:
        raise ValueError(f"No holdout rows found on or after {HOLDOUT_START.date()}.")

    x_train, x_holdout = prepare_xy(train_df, holdout_df, feature_cols)
    model = fit_gam(x_train, train_df[TARGET_COL].to_numpy(dtype=float))
    predictions = model.predict(x_holdout)

    holdout_metrics = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "rmse": rmse(holdout_df[TARGET_COL], predictions),
                "mae": float(mean_absolute_error(holdout_df[TARGET_COL], predictions)),
                "r2": float(r2_score(holdout_df[TARGET_COL], predictions)),
                "n_rows": int(len(holdout_df)),
            }
        ]
    )
    holdout_predictions = holdout_df[["Date", "Block", "Location", TARGET_COL]].copy()
    holdout_predictions["model"] = MODEL_NAME
    holdout_predictions["Actual_WQI"] = holdout_predictions[TARGET_COL].astype(float)
    holdout_predictions["Predicted_WQI"] = predictions.astype(float)
    holdout_predictions["Absolute_Error"] = (
        holdout_predictions["Actual_WQI"] - holdout_predictions["Predicted_WQI"]
    ).abs()
    holdout_predictions = holdout_predictions[
        ["model", "Date", "Block", "Location", "Actual_WQI", "Predicted_WQI", "Absolute_Error"]
    ]

    return holdout_metrics, holdout_predictions


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    return (
        metrics_df.groupby("model")
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            folds=("cutoff_date", "count"),
        )
        .reset_index()
        .fillna({"std_rmse": 0.0})
        .sort_values(["mean_rmse", "mean_mae"])
        .reset_index(drop=True)
    )


def main() -> None:
    print("=" * 72)
    print("GAM WQI FORECASTER")
    print("=" * 72)

    monthly_df = load_monthly_data()
    dataset = add_decimal_year(build_training_dataset(monthly_df))
    feature_cols = get_gam_feature_columns()
    validation_dataset = dataset[dataset["Date"] < HOLDOUT_START].copy()
    backtest_dates = select_backtest_dates(validation_dataset)
    if not backtest_dates:
        raise ValueError("No validation dates available before the holdout start.")

    print(f"Monthly observed rows    : {len(monthly_df):,}")
    print(f"Feature rows            : {len(dataset):,}")
    print(f"Validation rows         : {len(validation_dataset):,}")
    print(f"Holdout rows            : {int((dataset['Date'] >= HOLDOUT_START).sum()):,}")
    print(f"Validation dates        : {[d.date().isoformat() for d in backtest_dates]}")
    print(f"Holdout start           : {HOLDOUT_START.date().isoformat()}")

    metrics_df, predictions_df = rolling_origin_gam_backtest(dataset, backtest_dates, feature_cols)
    summary_df = summarize_metrics(metrics_df)
    holdout_df, holdout_predictions_df = evaluate_gam_holdout(dataset, feature_cols)

    print("\nValidation summary:")
    print(summary_df.to_string(index=False))
    print("\nUntouched holdout:")
    print(holdout_df.to_string(index=False))

    dataset.to_csv(OUTPUT_DIR / "gam_feature_dataset.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "gam_validation_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "gam_validation_predictions.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "gam_model_comparison.csv", index=False)
    holdout_df.to_csv(OUTPUT_DIR / "gam_holdout_comparison.csv", index=False)
    holdout_predictions_df.to_csv(OUTPUT_DIR / "gam_holdout_predictions.csv", index=False)

    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
