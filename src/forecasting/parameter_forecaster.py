from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.forecasting.panel_forecaster import (
    HOLDOUT_START,
    build_training_dataset as build_direct_dataset,
    fit_and_predict as fit_direct_predict,
    get_feature_columns as get_direct_feature_columns,
    load_monthly_data as load_direct_monthly_data,
)
from src.forecasting.temporal_features import (
    attach_sens_slopes,
    compute_sens_slopes,
    get_sens_slope_feature_columns,
    kalman_fill_monthly_dataset,
)

OUTPUT_DIR = ROOT / 'src' / 'data' / 'forecasting_parameters'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETER_DATASET_PATH = ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_parameter_dataset.csv'
LAG_STEPS = (1, 2, 3)
CATEGORICAL_COLS = ['Block', 'Location']
TARGET_COL = 'WQI_target'
PARAMETER_STANDARDS = {
    'pH_mean': {'Sn': 8.5, 'Videal': 7.0},
    'TDS_mean': {'Sn': 500.0, 'Videal': 0.0},
    'Hardness_mean': {'Sn': 200.0, 'Videal': 0.0},
    'Chloride_mean': {'Sn': 250.0, 'Videal': 0.0},
    'Fluoride_mean': {'Sn': 1.0, 'Videal': 0.0},
    'Alkalinity_mean': {'Sn': 200.0, 'Videal': 0.0},
    'Sulphate_mean': {'Sn': 200.0, 'Videal': 0.0},
    'Nitrate_mean': {'Sn': 45.0, 'Videal': 0.0},
}
K = 1.0 / sum(1.0 / spec['Sn'] for spec in PARAMETER_STANDARDS.values())
WEIGHTS = {name: K / spec['Sn'] for name, spec in PARAMETER_STANDARDS.items()}
PARAMETER_COLS = list(PARAMETER_STANDARDS)
MIN_TRAIN_ROWS = 250
MIN_TEST_ROWS = 5
MAX_BACKTEST_PERIODS = 10
TUNING_BACKTEST_PERIODS = 4
MIN_HISTORY_FOR_LOCATION_FORECAST = 3
MAX_RECOMMENDED_FORECAST_GAP_MONTHS = 6
DEFAULT_PARAMETER_MODEL_PARAMS = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'depth': 6,
    'learning_rate': 0.03,
    'iterations': 500,
    'l2_leaf_reg': 5.0,
    'random_seed': 42,
    'thread_count': -1,
    'verbose': False,
}
PARAMETER_TUNING_CANDIDATES = [
    {'name': 'baseline', 'params': dict(DEFAULT_PARAMETER_MODEL_PARAMS)},
    {'name': 'shallower_regularized', 'params': {**DEFAULT_PARAMETER_MODEL_PARAMS, 'depth': 4, 'learning_rate': 0.05, 'iterations': 350, 'l2_leaf_reg': 7.0}},
    {'name': 'balanced_medium', 'params': {**DEFAULT_PARAMETER_MODEL_PARAMS, 'depth': 5, 'learning_rate': 0.04, 'iterations': 450, 'l2_leaf_reg': 6.0}},
    {'name': 'deeper_slow', 'params': {**DEFAULT_PARAMETER_MODEL_PARAMS, 'depth': 7, 'learning_rate': 0.02, 'iterations': 650, 'l2_leaf_reg': 6.0}},
]


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calc_wqi_from_row(row: pd.Series) -> float:
    total = 0.0
    for param, spec in PARAMETER_STANDARDS.items():
        value = float(row[param])
        qn = ((value - spec['Videal']) / (spec['Sn'] - spec['Videal'])) * 100.0
        total += float(np.clip(qn, 0.0, 300.0)) * WEIGHTS[param]
    return total


def classify_wqi(value: float) -> str:
    if value <= 25:
        return 'Excellent'
    if value <= 50:
        return 'Good'
    if value <= 75:
        return 'Poor'
    if value <= 100:
        return 'Very Poor'
    return 'Unsuitable'


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def load_parameter_monthly_data() -> pd.DataFrame:
    df = pd.read_csv(PARAMETER_DATASET_PATH, parse_dates=['Date'])
    required = {'Block', 'Location', 'Date', 'WQI_mean', *PARAMETER_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Parameter monthly dataset is missing columns: {sorted(missing)}')
    return df.sort_values(['Block', 'Location', 'Date']).reset_index(drop=True)


def build_parameter_feature_dataset(
    monthly_df: pd.DataFrame,
    filled_monthly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    min_year = int(monthly_df['Date'].dt.year.min())
    lag_source_by_location: dict[tuple[object, object], pd.DataFrame] = {}
    if filled_monthly_df is not None:
        for key, source_group in filled_monthly_df.groupby(['Block', 'Location'], sort=False):
            lag_source_by_location[key] = source_group.sort_values('Date').reset_index(drop=True)

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for (block, location), group in monthly_df.groupby(['Block', 'Location'], sort=False):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) < 2:
            continue
        lag_source = lag_source_by_location.get((block, location), group)
        lag_source = lag_source.sort_values('Date').reset_index(drop=True)
        for idx in range(1, len(group)):
            current = group.iloc[idx]
            current_date = pd.Timestamp(current['Date'])
            observed_history = group.iloc[:idx].sort_values('Date', ascending=False).reset_index(drop=True)
            lag_history = lag_source[lag_source['Date'] < current_date].sort_values('Date', ascending=False).reset_index(drop=True)
            row: dict[str, float | str | pd.Timestamp] = {
                'Block': block,
                'Location': location,
                'Date': current_date,
                'Year': int(current_date.year),
                'Year_index': int(current_date.year - min_year),
                'Month': int(current_date.month),
                'Month_sin': float(np.sin(2 * np.pi * current_date.month / 12)),
                'Month_cos': float(np.cos(2 * np.pi * current_date.month / 12)),
                'History_points': int(len(observed_history)),
                TARGET_COL: float(current['WQI_mean']),
            }
            for lag in LAG_STEPS:
                if lag <= len(lag_history):
                    previous = lag_history.iloc[lag - 1]
                    previous_date = pd.Timestamp(previous['Date'])
                    if filled_monthly_df is not None:
                        previous_params = {
                            param: float(previous.get(f'{param}_kalman_filled', previous[param]))
                            for param in PARAMETER_COLS
                        }
                        row[f'WQI_lag_{lag}'] = float(calc_wqi_from_row(pd.Series(previous_params)))
                    else:
                        previous_params = {param: float(previous[param]) for param in PARAMETER_COLS}
                        row[f'WQI_lag_{lag}'] = float(previous['WQI_mean'])
                    row[f'Gap_lag_{lag}_months'] = float(
                        (current_date.year - previous_date.year) * 12
                        + (current_date.month - previous_date.month)
                    )
                    for param, value in previous_params.items():
                        row[f'{param}_lag_{lag}'] = value
                else:
                    row[f'WQI_lag_{lag}'] = np.nan
                    row[f'Gap_lag_{lag}_months'] = np.nan
                    for param in PARAMETER_COLS:
                        row[f'{param}_lag_{lag}'] = np.nan
            for param in PARAMETER_COLS:
                row[f'{param}_target'] = float(current[param])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['Date', 'Block', 'Location']).reset_index(drop=True)


def get_parameter_feature_columns() -> list[str]:
    feature_cols = ['Block', 'Location', 'Year', 'Year_index', 'Month', 'Month_sin', 'Month_cos', 'History_points']
    for lag in LAG_STEPS:
        feature_cols.append(f'WQI_lag_{lag}')
        feature_cols.append(f'Gap_lag_{lag}_months')
        for param in PARAMETER_COLS:
            feature_cols.append(f'{param}_lag_{lag}')
    return feature_cols


def build_parameter_model(model_params: dict[str, object] | None = None) -> CatBoostRegressor:
    params = {**DEFAULT_PARAMETER_MODEL_PARAMS}
    if model_params:
        params.update(model_params)
    return CatBoostRegressor(**params)


def fit_predict_parameter_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, CatBoostRegressor]]:
    predictions = pd.DataFrame(index=test_df.index)
    models: dict[str, CatBoostRegressor] = {}
    x_train = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        x_train[col] = x_train[col].astype(str)
        x_test[col] = x_test[col].astype(str)
    for param in PARAMETER_COLS:
        model = build_parameter_model(model_params=model_params)
        model.fit(x_train, train_df[f'{param}_target'], cat_features=CATEGORICAL_COLS)
        predictions[param] = model.predict(x_test)
        models[param] = model
    return predictions, models


def select_backtest_dates(dataset: pd.DataFrame) -> list[pd.Timestamp]:
    counts = dataset.groupby('Date').size().sort_index()
    eligible_dates: list[pd.Timestamp] = []
    for current_date, row_count in counts.items():
        if row_count < MIN_TEST_ROWS:
            continue
        if int((dataset['Date'] < current_date).sum()) < MIN_TRAIN_ROWS:
            continue
        eligible_dates.append(pd.Timestamp(current_date))
    return eligible_dates[-MAX_BACKTEST_PERIODS:]


def evaluate_parameter_model(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    backtest_dates: list[pd.Timestamp],
    model_name: str = 'ParameterCatBoost',
    model_params: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for cutoff_date in backtest_dates:
        train_df = dataset[dataset['Date'] < cutoff_date].copy()
        test_df = dataset[dataset['Date'] == cutoff_date].copy()
        pred_params, _ = fit_predict_parameter_models(train_df, test_df, feature_cols, model_params=model_params)
        pred_wqi = pred_params.apply(calc_wqi_from_row, axis=1)
        metric_rows.append(
            {
                'model': model_name,
                'cutoff_date': cutoff_date.date().isoformat(),
                'n_test': int(len(test_df)),
                'rmse': rmse(test_df[TARGET_COL], pred_wqi),
                'mae': float(mean_absolute_error(test_df[TARGET_COL], pred_wqi)),
                'r2': float(r2_score(test_df[TARGET_COL], pred_wqi)),
            }
        )
        for idx, (_, row) in enumerate(test_df.iterrows()):
            prediction_rows.append(
                {
                    'model': model_name,
                    'Date': cutoff_date.date().isoformat(),
                    'Block': row['Block'],
                    'Location': row['Location'],
                    'Actual_WQI': float(row[TARGET_COL]),
                    'Predicted_WQI': float(pred_wqi.iloc[idx]),
                    'Absolute_Error': float(abs(row[TARGET_COL] - pred_wqi.iloc[idx])),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def build_baseline_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-free WQI baselines from observations before each cutoff."""
    global_mean = float(train_df[TARGET_COL].mean())
    location_mean = train_df.groupby(["Block", "Location"])[TARGET_COL].mean()
    block_mean = train_df.groupby("Block")[TARGET_COL].mean()
    month_mean = train_df.groupby("Month")[TARGET_COL].mean()

    predictions = pd.DataFrame(index=test_df.index)
    predictions["Persistence"] = test_df["WQI_lag_1"].fillna(global_mean)
    predictions["LocationMean"] = [
        location_mean.get((row.Block, row.Location), block_mean.get(row.Block, global_mean))
        for row in test_df.itertuples()
    ]
    predictions["SeasonalMean"] = test_df["Month"].map(month_mean).fillna(global_mean)
    return predictions


def evaluate_baselines(
    dataset: pd.DataFrame,
    backtest_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for cutoff_date in backtest_dates:
        train_df = dataset[dataset["Date"] < cutoff_date].copy()
        test_df = dataset[dataset["Date"] == cutoff_date].copy()
        predictions = build_baseline_predictions(train_df, test_df)
        for model_name, values in predictions.items():
            metric_rows.append(
                {
                    "model": model_name,
                    "cutoff_date": cutoff_date.date().isoformat(),
                    "n_test": int(len(test_df)),
                    "rmse": rmse(test_df[TARGET_COL], values),
                    "mae": float(mean_absolute_error(test_df[TARGET_COL], values)),
                    "r2": float(r2_score(test_df[TARGET_COL], values)),
                }
            )
            for idx, (_, row) in enumerate(test_df.iterrows()):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "Date": cutoff_date.date().isoformat(),
                        "Block": row["Block"],
                        "Location": row["Location"],
                        "Actual_WQI": float(row[TARGET_COL]),
                        "Predicted_WQI": float(values.iloc[idx]),
                        "Absolute_Error": float(abs(row[TARGET_COL] - values.iloc[idx])),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def tune_parameter_model(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    backtest_dates: list[pd.Timestamp],
) -> tuple[str, dict[str, object], pd.DataFrame]:
    tuning_dates = backtest_dates[-min(TUNING_BACKTEST_PERIODS, len(backtest_dates)) :]
    result_rows: list[dict[str, object]] = []
    best_name = str(PARAMETER_TUNING_CANDIDATES[0]['name'])
    best_params = dict(PARAMETER_TUNING_CANDIDATES[0]['params'])
    best_score: tuple[float, float] | None = None

    for candidate in PARAMETER_TUNING_CANDIDATES:
        candidate_name = str(candidate['name'])
        candidate_params = dict(candidate['params'])
        metrics_df, _ = evaluate_parameter_model(
            dataset,
            feature_cols,
            tuning_dates,
            model_name=f'Tuning::{candidate_name}',
            model_params=candidate_params,
        )
        mean_rmse = float(metrics_df['rmse'].mean())
        mean_mae = float(metrics_df['mae'].mean())
        current_score = (mean_rmse, mean_mae)
        if best_score is None or current_score < best_score:
            best_name = candidate_name
            best_params = candidate_params
            best_score = current_score
        result_rows.append(
            {
                'candidate_name': candidate_name,
                'mean_rmse': mean_rmse,
                'mean_mae': mean_mae,
                'mean_r2': float(metrics_df['r2'].mean()),
                'folds': int(len(metrics_df)),
                'params_json': json.dumps(candidate_params, sort_keys=True),
            }
        )

    tuning_results = pd.DataFrame(result_rows).sort_values(['mean_rmse', 'mean_mae']).reset_index(drop=True)
    tuning_results['selected'] = tuning_results['candidate_name'] == best_name
    return best_name, best_params, tuning_results

def evaluate_direct_baseline(backtest_dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct_monthly = load_direct_monthly_data()
    direct_dataset = build_direct_dataset(direct_monthly)
    direct_feature_cols = get_direct_feature_columns()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for cutoff_date in backtest_dates:
        train_df = direct_dataset[direct_dataset['Date'] < cutoff_date].copy()
        test_df = direct_dataset[direct_dataset['Date'] == cutoff_date].copy()
        predictions = fit_direct_predict('CatBoost', direct_feature_cols, train_df, test_df)
        metric_rows.append(
            {
                'model': 'DirectCatBoost',
                'cutoff_date': cutoff_date.date().isoformat(),
                'n_test': int(len(test_df)),
                'rmse': rmse(test_df['WQI_target'], predictions),
                'mae': float(mean_absolute_error(test_df['WQI_target'], predictions)),
                'r2': float(r2_score(test_df['WQI_target'], predictions)),
            }
        )
        for idx, (_, row) in enumerate(test_df.iterrows()):
            prediction_rows.append(
                {
                    'model': 'DirectCatBoost',
                    'Date': cutoff_date.date().isoformat(),
                    'Block': row['Block'],
                    'Location': row['Location'],
                    'Actual_WQI': float(row['WQI_target']),
                    'Predicted_WQI': float(predictions[idx]),
                    'Absolute_Error': float(abs(row['WQI_target'] - predictions[idx])),
                }
            )
    return direct_dataset, pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby('model')
        .agg(
            mean_rmse=('rmse', 'mean'),
            std_rmse=('rmse', 'std'),
            mean_mae=('mae', 'mean'),
            mean_r2=('r2', 'mean'),
            folds=('cutoff_date', 'count'),
        )
        .reset_index()
        .sort_values(['mean_rmse', 'mean_mae'])
        .reset_index(drop=True)
    )
    summary['std_rmse'] = summary['std_rmse'].fillna(0.0)
    return summary


def summarize_predictions_by_location(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, block, location), group in predictions_df.groupby(['model', 'Block', 'Location']):
        rows.append(
            {
                'model': model_name,
                'Block': block,
                'Location': location,
                'n_predictions': int(len(group)),
                'rmse': rmse(group['Actual_WQI'], group['Predicted_WQI']),
                'mae': float(mean_absolute_error(group['Actual_WQI'], group['Predicted_WQI'])),
                'mean_actual': float(group['Actual_WQI'].mean()),
                'mean_predicted': float(group['Predicted_WQI'].mean()),
                'mean_bias': float((group['Predicted_WQI'] - group['Actual_WQI']).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(['model', 'rmse', 'mae']).reset_index(drop=True)



def summarize_predictions_by_month(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, date_value), group in predictions_df.groupby(['model', 'Date']):
        rows.append(
            {
                'model': model_name,
                'Date': date_value,
                'n_predictions': int(len(group)),
                'rmse': rmse(group['Actual_WQI'], group['Predicted_WQI']),
                'mae': float(mean_absolute_error(group['Actual_WQI'], group['Predicted_WQI'])),
                'mean_actual': float(group['Actual_WQI'].mean()),
                'mean_predicted': float(group['Predicted_WQI'].mean()),
                'mean_bias': float((group['Predicted_WQI'] - group['Actual_WQI']).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(['Date', 'model']).reset_index(drop=True)



def summarize_head_to_head(predictions_df: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions_df.pivot_table(
        index=['Date', 'Block', 'Location', 'Actual_WQI'],
        columns='model',
        values='Absolute_Error',
        aggfunc='first',
    ).reset_index()
    if 'ParameterCatBoostTuned' not in pivot.columns or 'DirectCatBoost' not in pivot.columns:
        return pd.DataFrame()
    pivot['parameter_beats_direct'] = pivot['ParameterCatBoostTuned'] < pivot['DirectCatBoost']
    pivot['direct_beats_parameter'] = pivot['DirectCatBoost'] < pivot['ParameterCatBoostTuned']
    pivot['tie'] = pivot['DirectCatBoost'] == pivot['ParameterCatBoostTuned']
    summary = (
        pivot.groupby(['Block', 'Location'])
        .agg(
            n_predictions=('Actual_WQI', 'count'),
            parameter_wins=('parameter_beats_direct', 'sum'),
            direct_wins=('direct_beats_parameter', 'sum'),
            ties=('tie', 'sum'),
            parameter_mean_abs_error=('ParameterCatBoostTuned', 'mean'),
            direct_mean_abs_error=('DirectCatBoost', 'mean'),
        )
        .reset_index()
    )
    summary['mean_abs_error_gain'] = summary['direct_mean_abs_error'] - summary['parameter_mean_abs_error']
    return summary.sort_values('mean_abs_error_gain', ascending=False).reset_index(drop=True)



def build_uncertainty_profile(predictions_df: pd.DataFrame, model_name: str) -> dict[str, object]:
    parameter_predictions = predictions_df[predictions_df['model'] == model_name].copy()
    residuals = parameter_predictions['Predicted_WQI'] - parameter_predictions['Actual_WQI']
    absolute_errors = residuals.abs()
    profile: dict[str, object] = {
        'global_mae': float(absolute_errors.mean()),
        'global_rmse': rmse(parameter_predictions['Actual_WQI'], parameter_predictions['Predicted_WQI']),
        'global_residual_std': float(residuals.std(ddof=0)) if len(residuals) > 1 else float(absolute_errors.mean()),
        'abs_error_p50': float(absolute_errors.quantile(0.50)),
        'abs_error_p80': float(absolute_errors.quantile(0.80)),
        'abs_error_p90': float(absolute_errors.quantile(0.90)),
        'interval_80_empirical_coverage': float((absolute_errors <= absolute_errors.quantile(0.80)).mean()),
        'n_validation_predictions': int(len(parameter_predictions)),
        'per_location': {},
    }
    per_location: dict[str, dict[str, float | int]] = {}
    for (block, location), group in parameter_predictions.groupby(['Block', 'Location']):
        key = f'{block}|||{location}'
        per_location[key] = {
            'n_predictions': int(len(group)),
            'mae': float(mean_absolute_error(group['Actual_WQI'], group['Predicted_WQI'])),
            'rmse': rmse(group['Actual_WQI'], group['Predicted_WQI']),
        }
    profile['per_location'] = per_location
    return profile


def evaluate_holdout(
    parameter_dataset: pd.DataFrame,
    feature_cols: list[str],
    direct_dataset: pd.DataFrame,
    default_model_params: dict[str, object],
    tuned_model_params: dict[str, object],
) -> pd.DataFrame:
    train_param = parameter_dataset[parameter_dataset['Date'] < HOLDOUT_START].copy()
    holdout_param = parameter_dataset[parameter_dataset['Date'] >= HOLDOUT_START].copy()
    default_pred_params, _ = fit_predict_parameter_models(
        train_param,
        holdout_param,
        feature_cols,
        model_params=default_model_params,
    )
    default_pred_wqi = default_pred_params.apply(calc_wqi_from_row, axis=1)
    tuned_pred_params, _ = fit_predict_parameter_models(
        train_param,
        holdout_param,
        feature_cols,
        model_params=tuned_model_params,
    )
    tuned_pred_wqi = tuned_pred_params.apply(calc_wqi_from_row, axis=1)

    train_direct = direct_dataset[direct_dataset['Date'] < HOLDOUT_START].copy()
    holdout_direct = direct_dataset[direct_dataset['Date'] >= HOLDOUT_START].copy()
    direct_predictions = fit_direct_predict('CatBoost', get_direct_feature_columns(), train_direct, holdout_direct)

    baseline_predictions = build_baseline_predictions(train_param, holdout_param)
    rows = [
        {
            'model': 'ParameterCatBoostDefault',
            'rmse': rmse(holdout_param[TARGET_COL], default_pred_wqi),
            'mae': float(mean_absolute_error(holdout_param[TARGET_COL], default_pred_wqi)),
            'r2': float(r2_score(holdout_param[TARGET_COL], default_pred_wqi)),
            'n_rows': int(len(holdout_param)),
        },
        {
            'model': 'ParameterCatBoostTuned',
            'rmse': rmse(holdout_param[TARGET_COL], tuned_pred_wqi),
            'mae': float(mean_absolute_error(holdout_param[TARGET_COL], tuned_pred_wqi)),
            'r2': float(r2_score(holdout_param[TARGET_COL], tuned_pred_wqi)),
            'n_rows': int(len(holdout_param)),
        },
        {
            'model': 'DirectCatBoost',
            'rmse': rmse(holdout_direct['WQI_target'], direct_predictions),
            'mae': float(mean_absolute_error(holdout_direct['WQI_target'], direct_predictions)),
            'r2': float(r2_score(holdout_direct['WQI_target'], direct_predictions)),
            'n_rows': int(len(holdout_direct)),
        },
    ]
    for model_name, values in baseline_predictions.items():
        rows.append(
            {
                "model": model_name,
                "rmse": rmse(holdout_param[TARGET_COL], values),
                "mae": float(mean_absolute_error(holdout_param[TARGET_COL], values)),
                "r2": float(r2_score(holdout_param[TARGET_COL], values)),
                "n_rows": int(len(holdout_param)),
            }
        )
    return pd.DataFrame(rows)


def fit_final_parameter_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict[str, object] | None = None,
) -> dict[str, CatBoostRegressor]:
    _, models = fit_predict_parameter_models(dataset, dataset, feature_cols, model_params=model_params)
    return models


def predict_future(
    artifact: dict[str, object],
    location: str,
    target_date: str,
    block: str | None = None,
) -> dict[str, object]:
    monthly_df: pd.DataFrame = artifact['monthly_history']
    target_timestamp = pd.Timestamp(target_date).replace(day=1)
    history = monthly_df[monthly_df['Location'] == location].copy()
    if block is not None:
        history = history[history['Block'] == block].copy()
    if history.empty:
        raise ValueError(f"Location '{location}' not found in parameter history.")
    if block is None:
        block = str(history['Block'].iloc[0])
    history = history[history['Date'] < target_timestamp].sort_values('Date').reset_index(drop=True)
    if history.empty:
        raise ValueError(f"Location '{location}' has no history before {target_timestamp.date()}.")

    min_history = int(artifact.get('minimum_history_for_location_forecast', MIN_HISTORY_FOR_LOCATION_FORECAST))
    if len(history) < min_history:
        return {
            'location': location,
            'block': block,
            'target_date': target_timestamp.date().isoformat(),
            'forecast_status': 'insufficient_history',
            'forecast_recommended': False,
            'reason': f'Only {len(history)} historical observed months are available; at least {min_history} are required.',
        }

    min_year = int(artifact['min_year'])
    row: dict[str, float | str | pd.Timestamp] = {
        'Block': block,
        'Location': location,
        'Date': target_timestamp,
        'Year': int(target_timestamp.year),
        'Year_index': int(target_timestamp.year - min_year),
        'Month': int(target_timestamp.month),
        'Month_sin': float(np.sin(2 * np.pi * target_timestamp.month / 12)),
        'Month_cos': float(np.cos(2 * np.pi * target_timestamp.month / 12)),
        'History_points': int(len(history)),
    }
    filled_history = artifact.get('kalman_filled_history')
    lag_history = history
    if isinstance(filled_history, pd.DataFrame) and not filled_history.empty:
        filled_lag_history = filled_history[
            (filled_history['Block'] == block)
            & (filled_history['Location'] == location)
            & (filled_history['Date'] < target_timestamp)
        ].copy()
        if not filled_lag_history.empty:
            lag_history = filled_lag_history

    recent = lag_history.sort_values('Date', ascending=False).reset_index(drop=True)
    for lag in LAG_STEPS:
        if lag <= len(recent):
            prev = recent.iloc[lag - 1]
            prev_date = pd.Timestamp(prev['Date'])
            if 'kalman_filled_history' in artifact and isinstance(filled_history, pd.DataFrame) and not filled_history.empty:
                previous_params = {
                    param: float(prev.get(f'{param}_kalman_filled', prev[param]))
                    for param in PARAMETER_COLS
                }
                row[f'WQI_lag_{lag}'] = float(calc_wqi_from_row(pd.Series(previous_params)))
            else:
                previous_params = {param: float(prev[param]) for param in PARAMETER_COLS}
                row[f'WQI_lag_{lag}'] = float(prev['WQI_mean'])
            row[f'Gap_lag_{lag}_months'] = float((target_timestamp.year - prev_date.year) * 12 + (target_timestamp.month - prev_date.month))
            for param, value in previous_params.items():
                row[f'{param}_lag_{lag}'] = value
        else:
            row[f'WQI_lag_{lag}'] = np.nan
            row[f'Gap_lag_{lag}_months'] = np.nan
            for param in PARAMETER_COLS:
                row[f'{param}_lag_{lag}'] = np.nan

    feature_df = pd.DataFrame([row])
    slopes_df = artifact.get('sens_slopes')
    if isinstance(slopes_df, pd.DataFrame) and not slopes_df.empty:
        feature_df = attach_sens_slopes(feature_df, slopes_df)
    else:
        for col in get_sens_slope_feature_columns():
            feature_df[col] = np.nan
    feature_df = feature_df[artifact['feature_columns']]
    for col in CATEGORICAL_COLS:
        feature_df[col] = feature_df[col].astype(str)

    predicted_params = {
        param: float(model.predict(feature_df)[0])
        for param, model in artifact['models'].items()
    }
    predicted_wqi = float(calc_wqi_from_row(pd.Series(predicted_params)))

    uncertainty_profile: dict[str, object] = artifact.get('uncertainty_profile', {})
    global_mae = float(uncertainty_profile.get('global_mae', 12.0))
    global_rmse = float(uncertainty_profile.get('global_rmse', max(global_mae, 14.0)))
    global_std = float(uncertainty_profile.get('global_residual_std', max(global_mae / 1.25, 10.0)))
    abs_error_p80 = float(uncertainty_profile.get('abs_error_p80', max(global_mae * 1.25, 15.0)))
    per_location = uncertainty_profile.get('per_location', {})
    location_key = f'{block}|||{location}'
    location_profile = per_location.get(location_key, {}) if isinstance(per_location, dict) else {}

    uncertainty_scale = 1.0
    history_points = int(len(history))
    if history_points < 6:
        uncertainty_scale *= 1.15
    if history_points < 3:
        uncertainty_scale *= 1.10
    gap_lag_1 = row.get('Gap_lag_1_months')
    max_gap = int(artifact.get('max_recommended_forecast_gap_months', MAX_RECOMMENDED_FORECAST_GAP_MONTHS))
    stale_history = pd.notna(gap_lag_1) and float(gap_lag_1) > max_gap
    if pd.notna(gap_lag_1) and float(gap_lag_1) > 3:
        uncertainty_scale *= 1.15
    if stale_history:
        uncertainty_scale *= 1.25

    expected_abs_error = global_mae
    residual_sigma = max(global_std, global_mae / 1.25, 1.0)
    if location_profile and int(location_profile.get('n_predictions', 0)) >= 2:
        expected_abs_error = (global_mae + float(location_profile['mae'])) / 2.0
        residual_sigma = max((global_rmse + float(location_profile['rmse'])) / 2.0, 1.0)

    expected_abs_error *= uncertainty_scale
    residual_sigma *= uncertainty_scale
    interval_half_width = max(abs_error_p80 * uncertainty_scale, expected_abs_error)
    lower_bound = predicted_wqi - interval_half_width
    upper_bound = predicted_wqi + interval_half_width
    confidence_score = max(0.05, min(0.95, 1.0 - (expected_abs_error / 25.0)))
    if confidence_score >= 0.70:
        confidence_label = 'high'
    elif confidence_score >= 0.45:
        confidence_label = 'medium'
    else:
        confidence_label = 'low'

    exceedance_probabilities: dict[str, float] = {}
    exceedance_flags: dict[str, bool] = {}
    for threshold in (50.0, 75.0, 100.0):
        probability = 1.0 - normal_cdf((threshold - predicted_wqi) / residual_sigma)
        probability = max(0.0, min(1.0, probability))
        key = f'wqi_gt_{int(threshold)}'
        exceedance_probabilities[key] = round(probability, 3)
        exceedance_flags[key] = probability >= 0.5

    return {
        'location': location,
        'block': block,
        'target_date': target_timestamp.date().isoformat(),
        'forecast_status': 'stale_history' if stale_history else 'ok',
        'forecast_recommended': not stale_history,
        'last_observed_gap_months': round(float(gap_lag_1), 2) if pd.notna(gap_lag_1) else None,
        'predicted_wqi': round(predicted_wqi, 2),
        'predicted_category': classify_wqi(predicted_wqi),
        'exceedance_flags': exceedance_flags,
        'exceedance_probabilities': exceedance_probabilities,
        'uncertainty': {
            'expected_abs_error': round(expected_abs_error, 2),
            'prediction_interval_80': [round(lower_bound, 2), round(upper_bound, 2)],
            'confidence_score': round(confidence_score, 3),
            'confidence_label': confidence_label,
        },
        'predicted_params': {key: round(value, 3) for key, value in predicted_params.items()},
    }


def main() -> None:
    print('=' * 72)
    print('IRREGULAR-TIME PARAMETER WQI PREDICTOR')
    print('=' * 72)

    parameter_monthly = load_parameter_monthly_data()
    slopes_df = compute_sens_slopes(parameter_monthly[parameter_monthly['Date'] < HOLDOUT_START])
    filled_monthly = kalman_fill_monthly_dataset(parameter_monthly)
    parameter_dataset = build_parameter_feature_dataset(parameter_monthly, filled_monthly)
    parameter_dataset = attach_sens_slopes(parameter_dataset, slopes_df)
    feature_cols = get_parameter_feature_columns() + get_sens_slope_feature_columns()

    validation_dataset = parameter_dataset[parameter_dataset['Date'] < HOLDOUT_START].copy()
    holdout_dataset = parameter_dataset[parameter_dataset['Date'] >= HOLDOUT_START].copy()
    if holdout_dataset.empty:
        raise ValueError(f'No holdout rows found on or after {HOLDOUT_START.date()}.')
    validation_dates = select_backtest_dates(validation_dataset)
    if not validation_dates:
        raise ValueError(
            'No validation dates available before the holdout start. '
            'Move FORECAST_HOLDOUT_START later or lower the minimum train/test row settings.'
        )

    tuned_candidate_name, tuned_model_params, tuning_results_df = tune_parameter_model(
        validation_dataset,
        feature_cols,
        validation_dates,
    )
    tuning_dates = validation_dates[-min(TUNING_BACKTEST_PERIODS, len(validation_dates)) :]

    direct_dataset, direct_metrics, direct_predictions = evaluate_direct_baseline(validation_dates)
    baseline_metrics, baseline_predictions = evaluate_baselines(parameter_dataset, validation_dates)
    default_parameter_metrics, default_parameter_predictions = evaluate_parameter_model(
        parameter_dataset,
        feature_cols,
        validation_dates,
        model_name='ParameterCatBoostDefault',
        model_params=DEFAULT_PARAMETER_MODEL_PARAMS,
    )
    tuned_parameter_metrics, tuned_parameter_predictions = evaluate_parameter_model(
        parameter_dataset,
        feature_cols,
        validation_dates,
        model_name='ParameterCatBoostTuned',
        model_params=tuned_model_params,
    )

    combined_metrics = pd.concat(
        [default_parameter_metrics, tuned_parameter_metrics, direct_metrics, baseline_metrics],
        ignore_index=True,
    )
    combined_predictions = pd.concat(
        [default_parameter_predictions, tuned_parameter_predictions, direct_predictions, baseline_predictions],
        ignore_index=True,
    )
    summary_df = summarize_metrics(combined_metrics)
    per_location_df = summarize_predictions_by_location(combined_predictions)
    per_month_df = summarize_predictions_by_month(combined_predictions)
    head_to_head_df = summarize_head_to_head(
        combined_predictions[combined_predictions['model'].isin(['ParameterCatBoostTuned', 'DirectCatBoost'])]
    )
    uncertainty_profile = build_uncertainty_profile(combined_predictions, model_name='ParameterCatBoostTuned')
    holdout_df = evaluate_holdout(
        parameter_dataset,
        feature_cols,
        direct_dataset,
        default_model_params=DEFAULT_PARAMETER_MODEL_PARAMS,
        tuned_model_params=tuned_model_params,
    ).sort_values(['rmse', 'mae']).reset_index(drop=True)

    print(f'Monthly observed rows    : {len(parameter_monthly):,}')
    print(f"Sen's slope rows        : {len(slopes_df):,}")
    print(f'Kalman lag-source rows  : {len(filled_monthly):,}')
    print(f'Feature rows            : {len(parameter_dataset):,}')
    print(f'Validation rows         : {len(validation_dataset):,}')
    print(f'Holdout rows            : {len(holdout_dataset):,}')
    print(f'Validation dates        : {[d.date().isoformat() for d in validation_dates]}')
    print(f'Tuning dates            : {[d.date().isoformat() for d in tuning_dates]}')
    print(f'Holdout start           : {HOLDOUT_START.date().isoformat()}')
    print('\nTuning results:')
    print(tuning_results_df[['candidate_name', 'mean_rmse', 'mean_mae', 'mean_r2', 'selected']].to_string(index=False))
    print('\nValidation comparison:')
    print(summary_df.to_string(index=False))
    print('\nUntouched holdout comparison:')
    print(holdout_df.to_string(index=False))

    parameter_dataset.to_csv(OUTPUT_DIR / 'parameter_feature_dataset.csv', index=False)
    filled_monthly.to_csv(OUTPUT_DIR / 'kalman_filled_monthly_dataset.csv', index=False)
    combined_metrics.to_csv(OUTPUT_DIR / 'rolling_origin_metrics.csv', index=False)
    combined_metrics.to_csv(OUTPUT_DIR / 'validation_metrics.csv', index=False)
    combined_predictions.to_csv(OUTPUT_DIR / 'rolling_origin_predictions.csv', index=False)
    combined_predictions.to_csv(OUTPUT_DIR / 'validation_predictions.csv', index=False)
    tuning_results_df.to_csv(OUTPUT_DIR / 'tuning_results.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    per_location_df.to_csv(OUTPUT_DIR / 'per_location_comparison.csv', index=False)
    per_month_df.to_csv(OUTPUT_DIR / 'per_month_comparison.csv', index=False)
    head_to_head_df.to_csv(OUTPUT_DIR / 'head_to_head_comparison.csv', index=False)
    holdout_df.to_csv(OUTPUT_DIR / 'holdout_comparison.csv', index=False)

    final_models = fit_final_parameter_models(parameter_dataset, feature_cols, model_params=tuned_model_params)
    artifact = {
        'models': final_models,
        'feature_columns': feature_cols,
        'monthly_history': parameter_monthly,
        'kalman_filled_history': filled_monthly,
        'sens_slopes': slopes_df,
        'min_year': int(parameter_monthly['Date'].dt.year.min()),
        'uncertainty_profile': uncertainty_profile,
        'selected_candidate_name': tuned_candidate_name,
        'selected_model_params': tuned_model_params,
        'forecast_task': 'irregular observed-month WQI prediction',
        'minimum_history_for_location_forecast': MIN_HISTORY_FOR_LOCATION_FORECAST,
        'max_recommended_forecast_gap_months': MAX_RECOMMENDED_FORECAST_GAP_MONTHS,
    }
    joblib.dump(artifact, OUTPUT_DIR / 'parameter_forecaster.joblib')

    metadata = {
        'forecast_task': 'irregular observed-month WQI prediction',
        'data_warning': (
            'The source data are sparse and irregular. Metrics are for observed sampling months only; '
            'this artifact should not be presented as a regular monthly per-location forecaster.'
        ),
        'parameter_columns': PARAMETER_COLS,
        'sens_slope_feature_columns': get_sens_slope_feature_columns(),
        'kalman_lag_source_rows': int(len(filled_monthly)),
        'validation_dates': [date.date().isoformat() for date in validation_dates],
        'tuning_dates': [date.date().isoformat() for date in tuning_dates],
        'holdout_start': HOLDOUT_START.date().isoformat(),
        'holdout_rows': int(len(holdout_dataset)),
        'holdout_comparison': holdout_df.to_dict(orient='records'),
        'selected_candidate_name': tuned_candidate_name,
        'selected_model_params': tuned_model_params,
        'baselines': ['Persistence', 'LocationMean', 'SeasonalMean'],
        'final_model_fit_rows': int(len(parameter_dataset)),
        'validation_summary': summary_df.to_dict(orient='records'),
        'head_to_head_summary': {
            'parameter_wins_total': int(head_to_head_df['parameter_wins'].sum()) if not head_to_head_df.empty else 0,
            'direct_wins_total': int(head_to_head_df['direct_wins'].sum()) if not head_to_head_df.empty else 0,
            'ties_total': int(head_to_head_df['ties'].sum()) if not head_to_head_df.empty else 0,
        },
        'uncertainty_profile': {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in uncertainty_profile.items()
            if key != 'per_location'
        },
    }
    (OUTPUT_DIR / 'model_meta.json').write_text(json.dumps(metadata, indent=2))

    print(f'\nSaved outputs to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
