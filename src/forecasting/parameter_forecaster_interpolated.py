from __future__ import annotations

import json
import math
import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Save outputs to a dedicated interpolated results directory
if os.environ.get('ALLOW_INTERPOLATED_EXPERIMENTS') != '1':
    raise RuntimeError(
        'parameter_forecaster_interpolated.py is an exploratory experiment only. '
        'It uses interpolated targets and must not be used for model-performance claims. '
        'Set ALLOW_INTERPOLATED_EXPERIMENTS=1 to run it deliberately.'
    )

OUTPUT_DIR = ROOT / 'src' / 'data' / 'forecasting_interpolated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the linear-interpolated parameter dataset we created
PARAMETER_DATASET_PATH = ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_parameter_interpolated_linear.csv'
HOLDOUT_START = pd.Timestamp('2025-01-01')

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
PARAMETER_COLS = list(PARAMETER_STANDARDS.keys())

MIN_TRAIN_ROWS = 600  # Increased since we now have full 2,590 monthly rows
MIN_TEST_ROWS = 15
MAX_BACKTEST_PERIODS = 6
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

def build_parameter_feature_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    # Sort history to be absolutely sure
    monthly_df = monthly_df.sort_values(['Block', 'Location', 'Date']).reset_index(drop=True)
    
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
                'Month': int(current['Date'].month),
                'Decimal_Year': float(current['Decimal_Year']),
                'Month_sin': float(np.sin(2 * np.pi * current['Date'].month / 12)),
                'Month_cos': float(np.cos(2 * np.pi * current['Date'].month / 12)),
                'History_points': int(len(history)),
                TARGET_COL: float(current['WQI_mean']),
            }
            # Lags (regular monthly steps since dataset is fully interpolated)
            for lag in LAG_STEPS:
                if lag <= len(history):
                    previous = history.iloc[lag - 1]
                    row[f'WQI_lag_{lag}'] = float(previous['WQI_mean'])
                    row[f'Gap_lag_{lag}_months'] = float(lag)  # Fixed monthly gaps now!
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
    feature_cols = ['Block', 'Location', 'Decimal_Year', 'Month_sin', 'Month_cos', 'History_points']
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
    n = len(backtest_dates)
    for fold_idx, cutoff_date in enumerate(backtest_dates, 1):
        print(f'  [{model_name}] fold {fold_idx}/{n}: {cutoff_date.date()}', flush=True)
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
    tuning_dates = backtest_dates[-min(TUNING_BACKTEST_PERIODS, len(backtest_dates)):]
    result_rows: list[dict[str, object]] = []
    best_name = str(PARAMETER_TUNING_CANDIDATES[0]['name'])
    best_params = dict(PARAMETER_TUNING_CANDIDATES[0]['params'])
    best_score: tuple[float, float] | None = None

    n_candidates = len(PARAMETER_TUNING_CANDIDATES)
    for cand_idx, candidate in enumerate(PARAMETER_TUNING_CANDIDATES, 1):
        candidate_name = str(candidate['name'])
        candidate_params = dict(candidate['params'])
        print(f'  Tuning candidate {cand_idx}/{n_candidates}: {candidate_name}', flush=True)
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

    print(f'  Best candidate: {best_name}', flush=True)
    tuning_results = pd.DataFrame(result_rows).sort_values(['mean_rmse', 'mean_mae']).reset_index(drop=True)
    tuning_results['selected'] = tuning_results['candidate_name'] == best_name
    return best_name, best_params, tuning_results

def evaluate_direct_baseline(
    dataset: pd.DataFrame, 
    backtest_dates: list[pd.Timestamp]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Evaluate a direct WQI forecasting model (where target is directly WQI_target instead of parameters)
    feature_cols = get_parameter_feature_columns()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    n = len(backtest_dates)
    for fold_idx, cutoff_date in enumerate(backtest_dates, 1):
        print(f'  [DirectCatBoost] fold {fold_idx}/{n}: {cutoff_date.date()}', flush=True)
        train_df = dataset[dataset['Date'] < cutoff_date].copy()
        test_df = dataset[dataset['Date'] == cutoff_date].copy()
        
        x_train = train_df[feature_cols].copy()
        x_test = test_df[feature_cols].copy()
        for col in CATEGORICAL_COLS:
            x_train[col] = x_train[col].astype(str)
            x_test[col] = x_test[col].astype(str)
            
        model = CatBoostRegressor(**DEFAULT_PARAMETER_MODEL_PARAMS)
        model.fit(x_train, train_df[TARGET_COL], cat_features=CATEGORICAL_COLS)
        predictions = model.predict(x_test)
        
        metric_rows.append(
            {
                'model': 'DirectCatBoost',
                'cutoff_date': cutoff_date.date().isoformat(),
                'n_test': int(len(test_df)),
                'rmse': rmse(test_df[TARGET_COL], predictions),
                'mae': float(mean_absolute_error(test_df[TARGET_COL], predictions)),
                'r2': float(r2_score(test_df[TARGET_COL], predictions)),
            }
        )
        for idx, (_, row) in enumerate(test_df.iterrows()):
            prediction_rows.append(
                {
                    'model': 'DirectCatBoost',
                    'Date': cutoff_date.date().isoformat(),
                    'Block': row['Block'],
                    'Location': row['Location'],
                    'Actual_WQI': float(row[TARGET_COL]),
                    'Predicted_WQI': float(predictions[idx]),
                    'Absolute_Error': float(abs(row[TARGET_COL] - predictions[idx])),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)

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

def evaluate_holdout(
    parameter_dataset: pd.DataFrame,
    feature_cols: list[str],
    default_model_params: dict[str, object],
    tuned_model_params: dict[str, object],
) -> pd.DataFrame:
    train_param = parameter_dataset[parameter_dataset['Date'] < HOLDOUT_START].copy()
    holdout_param = parameter_dataset[parameter_dataset['Date'] >= HOLDOUT_START].copy()
    
    # 1. Default Parameter CatBoost
    default_pred_params, _ = fit_predict_parameter_models(
        train_param, holdout_param, feature_cols, model_params=default_model_params,
    )
    default_pred_wqi = default_pred_params.apply(calc_wqi_from_row, axis=1)
    
    # 2. Tuned Parameter CatBoost
    tuned_pred_params, _ = fit_predict_parameter_models(
        train_param, holdout_param, feature_cols, model_params=tuned_model_params,
    )
    tuned_pred_wqi = tuned_pred_params.apply(calc_wqi_from_row, axis=1)

    # 3. Direct WQI CatBoost
    x_train = train_param[feature_cols].copy()
    x_holdout = holdout_param[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        x_train[col] = x_train[col].astype(str)
        x_holdout[col] = x_holdout[col].astype(str)
        
    direct_model = CatBoostRegressor(**DEFAULT_PARAMETER_MODEL_PARAMS)
    direct_model.fit(x_train, train_param[TARGET_COL], cat_features=CATEGORICAL_COLS)
    direct_predictions = direct_model.predict(x_holdout)

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
            'rmse': rmse(holdout_param[TARGET_COL], direct_predictions),
            'mae': float(mean_absolute_error(holdout_param[TARGET_COL], direct_predictions)),
            'r2': float(r2_score(holdout_param[TARGET_COL], direct_predictions)),
            'n_rows': int(len(holdout_param)),
        },
    ]
    return pd.DataFrame(rows)

def main() -> None:
    print('=' * 72)
    print('INTERPOLATED PARAMETER-LEVEL WQI FORECASTER')
    print('=' * 72)

    parameter_monthly = pd.read_csv(PARAMETER_DATASET_PATH, parse_dates=['Date'])
    
    print(f'\nBuilding feature dataset ({len(parameter_monthly):,} rows)...')
    parameter_dataset = build_parameter_feature_dataset(parameter_monthly)
    feature_cols = get_parameter_feature_columns()
    backtest_dates = select_backtest_dates(parameter_dataset)
    
    print(f'\nTuning model ({len(PARAMETER_TUNING_CANDIDATES)} candidates × '
          f'{len(backtest_dates)} folds × {len(PARAMETER_COLS)} params)...')
    tuned_candidate_name, tuned_model_params, tuning_results_df = tune_parameter_model(
        parameter_dataset, feature_cols, backtest_dates,
    )

    print(f'\nEvaluating direct baseline ({len(backtest_dates)} folds)...')
    direct_metrics, direct_predictions = evaluate_direct_baseline(parameter_dataset, backtest_dates)

    print(f'\nEvaluating default parameter model ({len(backtest_dates)} folds × {len(PARAMETER_COLS)} params)...')
    default_parameter_metrics, default_parameter_predictions = evaluate_parameter_model(
        parameter_dataset, feature_cols, backtest_dates,
        model_name='ParameterCatBoostDefault',
        model_params=DEFAULT_PARAMETER_MODEL_PARAMS,
    )

    print(f'\nEvaluating tuned parameter model ({len(backtest_dates)} folds × {len(PARAMETER_COLS)} params)...')
    tuned_parameter_metrics, tuned_parameter_predictions = evaluate_parameter_model(
        parameter_dataset, feature_cols, backtest_dates,
        model_name='ParameterCatBoostTuned',
        model_params=tuned_model_params,
    )

    combined_metrics = pd.concat(
        [default_parameter_metrics, tuned_parameter_metrics, direct_metrics], ignore_index=True,
    )
    summary_df = summarize_metrics(combined_metrics)

    print('\nRunning holdout evaluation...')
    holdout_df = evaluate_holdout(
        parameter_dataset, feature_cols,
        default_model_params=DEFAULT_PARAMETER_MODEL_PARAMS,
        tuned_model_params=tuned_model_params,
    )

    print(f'\nFeature rows            : {len(parameter_dataset):,}')
    print(f'Backtest dates          : {[d.date().isoformat() for d in backtest_dates]}')
    print('\nTuning results:')
    print(tuning_results_df[['candidate_name', 'mean_rmse', 'mean_mae', 'mean_r2', 'selected']].to_string(index=False))
    print('\nModel comparison (Interpolated Data):')
    print(summary_df.to_string(index=False))
    print('\nHoldout comparison (Interpolated Data):')
    print(holdout_df.to_string(index=False))

    # Save outputs
    parameter_dataset.to_csv(OUTPUT_DIR / 'interpolated_feature_dataset.csv', index=False)
    combined_metrics.to_csv(OUTPUT_DIR / 'interpolated_metrics.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'interpolated_model_comparison.csv', index=False)
    holdout_df.to_csv(OUTPUT_DIR / 'interpolated_holdout_comparison.csv', index=False)
    
    print('\nSaved outputs to:', OUTPUT_DIR)

if __name__ == '__main__':
    main()
