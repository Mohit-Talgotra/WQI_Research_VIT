from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.forecasting.panel_forecaster import (  # noqa: E402
    build_feature_row,
    build_training_dataset,
    classify_wqi,
    fit_and_predict,
    get_feature_columns,
    load_monthly_data,
)

load_dotenv()

OUTPUT_DIR = ROOT / 'src' / 'data' / 'forecasting_hss'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FEATURE_COLS = get_feature_columns()
HSS_FEATURE_COLS = [col for col in BASELINE_FEATURE_COLS if col != 'Location']
HSS_CATEGORICAL_COLS = ['Block']
HSS_NUMERIC_COLS = [col for col in HSS_FEATURE_COLS if col not in HSS_CATEGORICAL_COLS]
TARGET_COL = 'WQI_target'
HOLDOUT_START = pd.Timestamp('2025-01-01')
MIN_TRAIN_ROWS = 250
MIN_TEST_ROWS = 5
MAX_BACKTEST_PERIODS = 10


@dataclass(frozen=True)
class HSSConfig:
    ridge_alpha: float
    process_var: float
    observation_var: float
    initial_var: float
    block_shrink: float

    @property
    def label(self) -> str:
        return (
            f'alpha={self.ridge_alpha:g},q={self.process_var:g},'
            f'r={self.observation_var:g},p0={self.initial_var:g},'
            f'shrink={self.block_shrink:g}'
        )


PARAM_GRID = [
    HSSConfig(1.0, 1.0, 49.0, 49.0, 0.50),
    HSSConfig(1.0, 4.0, 49.0, 49.0, 0.50),
    HSSConfig(1.0, 9.0, 81.0, 81.0, 0.50),
    HSSConfig(5.0, 4.0, 81.0, 81.0, 0.50),
    HSSConfig(5.0, 9.0, 81.0, 81.0, 0.50),
    HSSConfig(5.0, 16.0, 121.0, 121.0, 0.50),
    HSSConfig(20.0, 4.0, 81.0, 81.0, 0.75),
    HSSConfig(20.0, 9.0, 121.0, 121.0, 0.75),
]


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


class HierarchicalStateSpaceForecaster:
    def __init__(self, config: HSSConfig) -> None:
        self.config = config
        self.fixed_model: Pipeline | None = None
        self.global_residual_mean = 0.0
        self.block_residual_means: dict[str, float] = {}
        self.location_states: dict[str, dict[str, Any]] = {}
        self.min_year = 0

    def _build_fixed_model(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    'prep',
                    ColumnTransformer(
                        transformers=[
                            ('categorical', OneHotEncoder(handle_unknown='ignore'), HSS_CATEGORICAL_COLS),
                            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), HSS_NUMERIC_COLS),
                        ]
                    ),
                ),
                ('ridge', Ridge(alpha=self.config.ridge_alpha)),
            ]
        )

    def _initial_state_mean(self, block: str) -> float:
        block_mean = float(self.block_residual_means.get(block, self.global_residual_mean))
        return (
            self.config.block_shrink * block_mean
            + (1.0 - self.config.block_shrink) * self.global_residual_mean
        )

    def fit(self, train_df: pd.DataFrame) -> 'HierarchicalStateSpaceForecaster':
        self.min_year = int(train_df['Year'].min())
        self.fixed_model = self._build_fixed_model()
        self.fixed_model.fit(train_df[HSS_FEATURE_COLS], train_df[TARGET_COL])

        working = train_df.copy().sort_values(['Location', 'Date']).reset_index(drop=True)
        working['fixed_prediction'] = self.fixed_model.predict(working[HSS_FEATURE_COLS])
        working['residual'] = working[TARGET_COL] - working['fixed_prediction']

        self.global_residual_mean = float(working['residual'].mean())
        self.block_residual_means = (
            working.groupby('Block')['residual'].mean().astype(float).to_dict()
        )
        self.location_states = self._fit_location_states(working)
        return self

    def _fit_location_states(self, train_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for location, group in train_df.groupby('Location', sort=False):
            group = group.sort_values('Date').reset_index(drop=True)
            block = str(group['Block'].iloc[0])
            state_mean = self._initial_state_mean(block)
            state_var = self.config.initial_var
            last_date: pd.Timestamp | None = None

            for row in group.itertuples(index=False):
                current_date = pd.Timestamp(row.Date)
                gap = 1 if last_date is None else max(
                    1,
                    (current_date.year - last_date.year) * 12
                    + (current_date.month - last_date.month),
                )
                state_var = state_var + gap * self.config.process_var
                innovation = float(row.residual) - state_mean
                gain = state_var / (state_var + self.config.observation_var)
                state_mean = state_mean + gain * innovation
                state_var = (1.0 - gain) * state_var
                last_date = current_date

            states[location] = {
                'mean': float(state_mean),
                'variance': float(state_var),
                'last_date': last_date,
                'block': block,
            }
        return states

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        if self.fixed_model is None:
            raise RuntimeError('Model must be fitted before prediction.')

        fixed_pred = self.fixed_model.predict(feature_df[HSS_FEATURE_COLS])
        residual_pred: list[float] = []
        for row in feature_df.itertuples(index=False):
            location_state = self.location_states.get(str(row.Location))
            if location_state is None:
                residual_pred.append(self._initial_state_mean(str(row.Block)))
            else:
                residual_pred.append(float(location_state['mean']))
        return fixed_pred + np.array(residual_pred)

    def to_artifact(self) -> dict[str, Any]:
        if self.fixed_model is None:
            raise RuntimeError('Model must be fitted before serialization.')
        return {
            'config': asdict(self.config),
            'fixed_model': self.fixed_model,
            'global_residual_mean': self.global_residual_mean,
            'block_residual_means': self.block_residual_means,
            'location_states': self.location_states,
            'min_year': self.min_year,
            'feature_columns': BASELINE_FEATURE_COLS,
            'hss_feature_columns': HSS_FEATURE_COLS,
        }


def predict_with_artifact(artifact: dict[str, Any], feature_df: pd.DataFrame) -> np.ndarray:
    fixed_model: Pipeline = artifact['fixed_model']
    fixed_pred = fixed_model.predict(feature_df[artifact['hss_feature_columns']])
    residual_pred: list[float] = []
    block_means = artifact['block_residual_means']
    global_mean = float(artifact['global_residual_mean'])
    shrink = float(artifact['config']['block_shrink'])
    states = artifact['location_states']

    for row in feature_df.itertuples(index=False):
        state = states.get(str(row.Location))
        if state is None:
            block_mean = float(block_means.get(str(row.Block), global_mean))
            residual_pred.append(shrink * block_mean + (1.0 - shrink) * global_mean)
        else:
            residual_pred.append(float(state['mean']))

    return fixed_pred + np.array(residual_pred)


def evaluate_model(
    dataset: pd.DataFrame,
    config: HSSConfig,
    backtest_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for cutoff_date in backtest_dates:
        train_df = dataset[dataset['Date'] < cutoff_date].copy()
        test_df = dataset[dataset['Date'] == cutoff_date].copy()
        model = HierarchicalStateSpaceForecaster(config).fit(train_df)
        predictions = model.predict(test_df)

        fold_rows.append(
            {
                'model': 'HierarchicalStateSpace',
                'config': config.label,
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
                    'model': 'HierarchicalStateSpace',
                    'config': config.label,
                    'Date': cutoff_date.date().isoformat(),
                    'Block': row['Block'],
                    'Location': row['Location'],
                    'Actual_WQI': float(row[TARGET_COL]),
                    'Predicted_WQI': float(predictions[idx]),
                    'Absolute_Error': float(abs(row[TARGET_COL] - predictions[idx])),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def evaluate_baseline_models(
    dataset: pd.DataFrame,
    backtest_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_names = ['NaiveLastValue', 'CatBoost']

    for model_name in baseline_names:
        for cutoff_date in backtest_dates:
            train_df = dataset[dataset['Date'] < cutoff_date].copy()
            test_df = dataset[dataset['Date'] == cutoff_date].copy()
            predictions = fit_and_predict(model_name, BASELINE_FEATURE_COLS, train_df, test_df)

            fold_rows.append(
                {
                    'model': model_name,
                    'config': 'baseline',
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
                        'model': model_name,
                        'config': 'baseline',
                        'Date': cutoff_date.date().isoformat(),
                        'Block': row['Block'],
                        'Location': row['Location'],
                        'Actual_WQI': float(row[TARGET_COL]),
                        'Predicted_WQI': float(predictions[idx]),
                        'Absolute_Error': float(abs(row[TARGET_COL] - predictions[idx])),
                    }
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


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
    dataset: pd.DataFrame,
    config: HSSConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_df = dataset[dataset['Date'] < HOLDOUT_START].copy()
    test_df = dataset[dataset['Date'] >= HOLDOUT_START].copy()
    model = HierarchicalStateSpaceForecaster(config).fit(train_df)
    predictions = model.predict(test_df)
    result_df = test_df[['Date', 'Block', 'Location', TARGET_COL]].copy()
    result_df['Predicted_WQI'] = predictions
    result_df['Absolute_Error'] = (result_df[TARGET_COL] - result_df['Predicted_WQI']).abs()
    metrics = {
        'rmse': rmse(result_df[TARGET_COL], result_df['Predicted_WQI']),
        'mae': float(mean_absolute_error(result_df[TARGET_COL], result_df['Predicted_WQI'])),
        'r2': float(r2_score(result_df[TARGET_COL], result_df['Predicted_WQI'])),
        'n_rows': int(len(result_df)),
    }
    return result_df, metrics


def evaluate_baseline_holdout(dataset: pd.DataFrame) -> pd.DataFrame:
    train_df = dataset[dataset['Date'] < HOLDOUT_START].copy()
    test_df = dataset[dataset['Date'] >= HOLDOUT_START].copy()
    rows: list[dict[str, Any]] = []
    for model_name in ['NaiveLastValue', 'CatBoost']:
        predictions = fit_and_predict(model_name, BASELINE_FEATURE_COLS, train_df, test_df)
        rows.append(
            {
                'model': model_name,
                'rmse': rmse(test_df[TARGET_COL], predictions),
                'mae': float(mean_absolute_error(test_df[TARGET_COL], predictions)),
                'r2': float(r2_score(test_df[TARGET_COL], predictions)),
                'n_rows': int(len(test_df)),
            }
        )
    return pd.DataFrame(rows)


def save_plot(predictions_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for model_name, color in [('HierarchicalStateSpace', '#1f77b4'), ('CatBoost', '#ff7f0e')]:
        subset = predictions_df[predictions_df['model'] == model_name].copy()
        if subset.empty:
            continue
        axes[0].scatter(
            subset['Actual_WQI'],
            subset['Predicted_WQI'],
            alpha=0.55,
            label=model_name,
            color=color,
            edgecolors='black',
            linewidth=0.2,
        )
    combined = predictions_df[['Actual_WQI', 'Predicted_WQI']]
    low = float(min(combined['Actual_WQI'].min(), combined['Predicted_WQI'].min()))
    high = float(max(combined['Actual_WQI'].max(), combined['Predicted_WQI'].max()))
    axes[0].plot([low, high], [low, high], linestyle='--', color='red', linewidth=1.5)
    axes[0].set_title('Backtest: Actual vs Predicted')
    axes[0].set_xlabel('Actual WQI')
    axes[0].set_ylabel('Predicted WQI')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    summary = (
        predictions_df.groupby(['Date', 'model'])['Absolute_Error']
        .mean()
        .reset_index()
    )
    summary['Date'] = pd.to_datetime(summary['Date'])
    for model_name, color in [('HierarchicalStateSpace', '#1f77b4'), ('CatBoost', '#ff7f0e'), ('NaiveLastValue', '#2ca02c')]:
        subset = summary[summary['model'] == model_name].copy()
        if subset.empty:
            continue
        axes[1].plot(subset['Date'], subset['Absolute_Error'], marker='o', label=model_name, color=color)
    axes[1].set_title('Mean Absolute Error by Backtest Month')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'backtest_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


def predict_future(artifact: dict[str, Any], monthly_df: pd.DataFrame, location: str, target_date: str, block: str | None = None) -> dict[str, Any]:
    target_timestamp = pd.Timestamp(target_date).replace(day=1)
    location_history = monthly_df[monthly_df['Location'] == location].copy()
    if block is not None:
        location_history = location_history[location_history['Block'] == block].copy()
    if location_history.empty:
        raise ValueError(f"Location '{location}' not found in monthly history.")
    if block is None:
        block = str(location_history['Block'].iloc[0])

    history = location_history[location_history['Date'] < target_timestamp].copy()
    if history.empty:
        raise ValueError(f"Location '{location}' has no history before {target_timestamp.date()}.")

    feature_row = build_feature_row(
        history=history,
        target_date=target_timestamp,
        block=block,
        location=location,
        min_year=int(artifact['min_year']),
    )
    feature_df = pd.DataFrame([feature_row])[BASELINE_FEATURE_COLS]
    predicted_wqi = float(predict_with_artifact(artifact, feature_df)[0])
    return {
        'location': location,
        'block': block,
        'target_date': target_timestamp.date().isoformat(),
        'predicted_wqi': round(predicted_wqi, 2),
        'category': classify_wqi(predicted_wqi),
        'lag_1_wqi': None if pd.isna(feature_row['WQI_lag_1']) else round(float(feature_row['WQI_lag_1']), 2),
        'lag_2_wqi': None if pd.isna(feature_row['WQI_lag_2']) else round(float(feature_row['WQI_lag_2']), 2),
        'lag_3_wqi': None if pd.isna(feature_row['WQI_lag_3']) else round(float(feature_row['WQI_lag_3']), 2),
    }


def main() -> None:
    print('=' * 72)
    print('HIERARCHICAL STATE-SPACE FORECASTER')
    print('=' * 72)

    monthly_df = load_monthly_data()
    dataset = build_training_dataset(monthly_df)
    backtest_dates = select_backtest_dates(dataset)

    print(f'Monthly rows            : {len(monthly_df):,}')
    print(f'Feature rows            : {len(dataset):,}')
    print(f'Backtest dates          : {[d.date().isoformat() for d in backtest_dates]}')

    tuning_rows: list[dict[str, Any]] = []
    best_config: HSSConfig | None = None
    best_metrics_df = pd.DataFrame()
    best_predictions_df = pd.DataFrame()
    best_score = float('inf')

    for config in PARAM_GRID:
        metrics_df, predictions_df = evaluate_model(dataset, config, backtest_dates)
        score = float(metrics_df['rmse'].mean())
        tuning_rows.append(
            {
                'config': config.label,
                'mean_rmse': float(metrics_df['rmse'].mean()),
                'mean_mae': float(metrics_df['mae'].mean()),
                'mean_r2': float(metrics_df['r2'].mean()),
            }
        )
        print(f'  Tried {config.label} -> mean RMSE {score:.3f}, mean MAE {metrics_df["mae"].mean():.3f}')
        if score < best_score:
            best_score = score
            best_config = config
            best_metrics_df = metrics_df
            best_predictions_df = predictions_df

    if best_config is None:
        raise RuntimeError('Failed to choose a hierarchical state-space configuration.')

    baseline_metrics_df, baseline_predictions_df = evaluate_baseline_models(dataset, backtest_dates)
    combined_metrics_df = pd.concat([best_metrics_df, baseline_metrics_df], ignore_index=True)
    combined_predictions_df = pd.concat([best_predictions_df, baseline_predictions_df], ignore_index=True)
    summary_df = summarize_metrics(combined_metrics_df)

    print('\nBest HSS config:')
    print(f'  {best_config.label}')
    print('\nBacktest summary:')
    print(summary_df.to_string(index=False))

    tuning_df = pd.DataFrame(tuning_rows).sort_values('mean_rmse').reset_index(drop=True)
    tuning_df.to_csv(OUTPUT_DIR / 'hss_tuning_results.csv', index=False)
    combined_metrics_df.to_csv(OUTPUT_DIR / 'rolling_origin_metrics.csv', index=False)
    combined_predictions_df.to_csv(OUTPUT_DIR / 'rolling_origin_predictions.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    save_plot(combined_predictions_df)

    final_model = HierarchicalStateSpaceForecaster(best_config).fit(dataset)
    artifact = final_model.to_artifact()
    artifact['monthly_history'] = monthly_df
    joblib.dump(artifact, OUTPUT_DIR / 'hierarchical_state_space.joblib')

    holdout_df, holdout_metrics = evaluate_holdout(dataset, best_config)
    holdout_df.to_csv(OUTPUT_DIR / 'final_holdout_predictions.csv', index=False)
    baseline_holdout_df = evaluate_baseline_holdout(dataset)
    hss_holdout_df = pd.DataFrame([
        {
            'model': 'HierarchicalStateSpace',
            'rmse': holdout_metrics['rmse'],
            'mae': holdout_metrics['mae'],
            'r2': holdout_metrics['r2'],
            'n_rows': holdout_metrics['n_rows'],
        }
    ])
    holdout_comparison_df = pd.concat([hss_holdout_df, baseline_holdout_df], ignore_index=True)
    holdout_comparison_df.to_csv(OUTPUT_DIR / 'holdout_comparison.csv', index=False)

    metadata = {
        'best_config': asdict(best_config),
        'backtest_dates': [date.date().isoformat() for date in backtest_dates],
        'holdout_start': HOLDOUT_START.date().isoformat(),
        'holdout_metrics': holdout_metrics,
    }
    with open(OUTPUT_DIR / 'model_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    print('\nHoldout comparison:')
    print(holdout_comparison_df.to_string(index=False))

    print('\nSample future predictions:')
    for location, block in [('Valaiyakaranur', 'Ayodhiyapattanam'), ('Ammapalayam', 'Panamarathupatti')]:
        forecast = predict_future(artifact, monthly_df, location=location, block=block, target_date='2026-01-01')
        print(f"  - {forecast['location']} ({forecast['block']}): {forecast['predicted_wqi']} [{forecast['category']}]")

    print('\nSaved outputs:')
    print(f'  - {OUTPUT_DIR / "hss_tuning_results.csv"}')
    print(f'  - {OUTPUT_DIR / "rolling_origin_metrics.csv"}')
    print(f'  - {OUTPUT_DIR / "rolling_origin_predictions.csv"}')
    print(f'  - {OUTPUT_DIR / "model_comparison.csv"}')
    print(f'  - {OUTPUT_DIR / "holdout_comparison.csv"}')
    print(f'  - {OUTPUT_DIR / "hierarchical_state_space.joblib"}')
    print(f'  - {OUTPUT_DIR / "model_meta.json"}')


if __name__ == '__main__':
    main()
