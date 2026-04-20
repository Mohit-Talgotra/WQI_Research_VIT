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


def build_parameter_feature_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    min_year = int(monthly_df['Date'].dt.year.min())
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for (block, location), group in monthly_df.groupby(['Block', 'Location'], sort=False):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) < 2:
            continue
        for idx in range(1, len(group)):
            current = group.iloc[idx]
            history = group.iloc[:idx].sort_values('Date', ascending=False).reset_index(drop=True)
            row: dict[str, float | str | pd.Timestamp] = {
                'Block': block,
                'Location': location,
                'Date': pd.Timestamp(current['Date']),
                'Year': int(current['Date'].year),
                'Year_index': int(current['Date'].year - min_year),
                'Month': int(current['Date'].month),
                'Month_sin': float(np.sin(2 * np.pi * current['Date'].month / 12)),
                'Month_cos': float(np.cos(2 * np.pi * current['Date'].month / 12)),
                'History_points': int(len(history)),
                TARGET_COL: float(current['WQI_mean']),
            }
            for lag in LAG_STEPS:
                if lag <= len(history):
                    previous = history.iloc[lag - 1]
                    row[f'WQI_lag_{lag}'] = float(previous['WQI_mean'])
                    row[f'Gap_lag_{lag}_months'] = float(
                        (current['Date'].year - previous['Date'].year) * 12
                        + (current['Date'].month - previous['Date'].month)
                    )
                    for param in PARAMETER_COLS:
                        row[f'{param}_lag_{lag}'] = float(previous[param])
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
    recent = history.sort_values('Date', ascending=False).reset_index(drop=True)
    for lag in LAG_STEPS:
        if lag <= len(recent):
            prev = recent.iloc[lag - 1]
            row[f'WQI_lag_{lag}'] = float(prev['WQI_mean'])
            row[f'Gap_lag_{lag}_months'] = float((target_timestamp.year - prev['Date'].year) * 12 + (target_timestamp.month - prev['Date'].month))
            for param in PARAMETER_COLS:
                row[f'{param}_lag_{lag}'] = float(prev[param])
        else:
            row[f'WQI_lag_{lag}'] = np.nan
            row[f'Gap_lag_{lag}_months'] = np.nan
            for param in PARAMETER_COLS:
                row[f'{param}_lag_{lag}'] = np.nan

    feature_df = pd.DataFrame([row])[artifact['feature_columns']]
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
    if pd.notna(gap_lag_1) and float(gap_lag_1) > 3:
        uncertainty_scale *= 1.15

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
    print('PARAMETER-LEVEL WQI FORECASTER')
    print('=' * 72)

    parameter_monthly = load_parameter_monthly_data()
    parameter_dataset = build_parameter_feature_dataset(parameter_monthly)
    feature_cols = get_parameter_feature_columns()
    backtest_dates = select_backtest_dates(parameter_dataset)
    tuned_candidate_name, tuned_model_params, tuning_results_df = tune_parameter_model(
        parameter_dataset,
        feature_cols,
        backtest_dates,
    )
    direct_dataset, direct_metrics, direct_predictions = evaluate_direct_baseline(backtest_dates)
    default_parameter_metrics, default_parameter_predictions = evaluate_parameter_model(
        parameter_dataset,
        feature_cols,
        backtest_dates,
        model_name='ParameterCatBoostDefault',
        model_params=DEFAULT_PARAMETER_MODEL_PARAMS,
    )
    tuned_parameter_metrics, tuned_parameter_predictions = evaluate_parameter_model(
        parameter_dataset,
        feature_cols,
        backtest_dates,
        model_name='ParameterCatBoostTuned',
        model_params=tuned_model_params,
    )

    combined_metrics = pd.concat(
        [default_parameter_metrics, tuned_parameter_metrics, direct_metrics],
        ignore_index=True,
    )
    combined_predictions = pd.concat(
        [default_parameter_predictions, tuned_parameter_predictions, direct_predictions],
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
    )

    tuning_dates = backtest_dates[-min(TUNING_BACKTEST_PERIODS, len(backtest_dates)) :]
    print(f'Monthly rows            : {len(parameter_monthly):,}')
    print(f'Feature rows            : {len(parameter_dataset):,}')
    print(f'Backtest dates          : {[d.date().isoformat() for d in backtest_dates]}')
    print(f'Tuning window           : {[d.date().isoformat() for d in tuning_dates]}')
    print('\nTuning results:')
    print(tuning_results_df[['candidate_name', 'mean_rmse', 'mean_mae', 'mean_r2', 'selected']].to_string(index=False))
    print('\nModel comparison:')
    print(summary_df.to_string(index=False))
    print('\nHoldout comparison:')
    print(holdout_df.to_string(index=False))

    parameter_dataset.to_csv(OUTPUT_DIR / 'parameter_feature_dataset.csv', index=False)
    combined_metrics.to_csv(OUTPUT_DIR / 'rolling_origin_metrics.csv', index=False)
    combined_predictions.to_csv(OUTPUT_DIR / 'rolling_origin_predictions.csv', index=False)
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
        'min_year': int(parameter_monthly['Date'].dt.year.min()),
        'uncertainty_profile': uncertainty_profile,
        'selected_candidate_name': tuned_candidate_name,
        'selected_model_params': tuned_model_params,
    }
    joblib.dump(artifact, OUTPUT_DIR / 'parameter_forecaster.joblib')

    metadata = {
        'parameter_columns': PARAMETER_COLS,
        'backtest_dates': [date.date().isoformat() for date in backtest_dates],
        'tuning_dates': [date.date().isoformat() for date in tuning_dates],
        'selected_candidate_name': tuned_candidate_name,
        'selected_model_params': tuned_model_params,
        'holdout_start': HOLDOUT_START.date().isoformat(),
        'holdout_comparison': holdout_df.to_dict(orient='records'),
        'head_to_head_summary': {
            'parameter_wins_total': int(head_to_head_df['parameter_wins'].sum()) if not head_to_head_df.empty else 0,
            'direct_wins_total': int(head_to_head_df['direct_wins'].sum()) if not head_to_head_df.empty else 0,
            'ties_total': int(head_to_head_df['ties'].sum()) if not head_to_head_df.empty else 0,
        },
        'uncertainty_profile': {
            'global_mae': round(float(uncertainty_profile['global_mae']), 4),
            'global_rmse': round(float(uncertainty_profile['global_rmse']), 4),
            'global_residual_std': round(float(uncertainty_profile['global_residual_std']), 4),
            'abs_error_p50': round(float(uncertainty_profile['abs_error_p50']), 4),
            'abs_error_p80': round(float(uncertainty_profile['abs_error_p80']), 4),
            'abs_error_p90': round(float(uncertainty_profile['abs_error_p90']), 4),
        },
    }
    with open(OUTPUT_DIR / 'model_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    print('\nSample prediction:')
    sample = predict_future(artifact, location='Valaiyakaranur', block='Ayodhiyapattanam', target_date='2026-01-01')
    print(sample)

    print('\nSaved outputs:')
    print(f'  - {OUTPUT_DIR / "parameter_feature_dataset.csv"}')
    print(f'  - {OUTPUT_DIR / "rolling_origin_metrics.csv"}')
    print(f'  - {OUTPUT_DIR / "rolling_origin_predictions.csv"}')
    print(f'  - {OUTPUT_DIR / "tuning_results.csv"}')
    print(f'  - {OUTPUT_DIR / "model_comparison.csv"}')
    print(f'  - {OUTPUT_DIR / "per_location_comparison.csv"}')
    print(f'  - {OUTPUT_DIR / "per_month_comparison.csv"}')
    print(f'  - {OUTPUT_DIR / "head_to_head_comparison.csv"}')
    print(f'  - {OUTPUT_DIR / "holdout_comparison.csv"}')
    print(f'  - {OUTPUT_DIR / "parameter_forecaster.joblib"}')
    print(f'  - {OUTPUT_DIR / "model_meta.json"}')


if __name__ == '__main__':
    main()
