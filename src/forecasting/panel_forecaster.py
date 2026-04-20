from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'src' / 'data' / 'forecasting_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETER_ALIASES = {
    'pH_mean': 'ph',
    'TDS_mean': 'tds',
    'Total_mean': 'hardness',
    'Chloride_mean': 'chloride',
    'Fluoride_mean': 'fluoride',
    'Sulphate_mean': 'sulphate',
    'Nitrate_mean': 'nitrate',
}

LAG_STEPS = (1, 2, 3)
CATEGORICAL_COLS = ['Block', 'Location']
TARGET_COL = 'WQI_target'
MIN_TRAIN_ROWS = int(os.environ.get('FORECAST_MIN_TRAIN_ROWS', '250'))
MIN_TEST_ROWS = int(os.environ.get('FORECAST_MIN_TEST_ROWS', '5'))
MAX_BACKTEST_PERIODS = int(os.environ.get('FORECAST_MAX_BACKTEST_PERIODS', '10'))
HOLDOUT_START = pd.Timestamp(os.environ.get('FORECAST_HOLDOUT_START', '2025-01-01'))


def monthly_data_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get('MONTHLY_WQI_PATH')
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_dataset.csv',
            ROOT / 'src' / 'data' / 'monthly_wqi_dataset.csv',
            ROOT / 'data' / 'monthly_wqi_dataset.csv',
        ]
    )
    return candidates


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def resolve_existing_path(candidates: list[Path]) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        'Could not find a monthly WQI dataset. Checked:\n- ' + '\n- '.join(checked)
    )


def month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


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


def load_monthly_data() -> pd.DataFrame:
    monthly_path = resolve_existing_path(monthly_data_candidates())
    df = pd.read_csv(monthly_path, parse_dates=['Date'])
    required = {'Block', 'Location', 'Date', 'WQI_mean'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Monthly dataset is missing required columns: {sorted(missing)}')

    if 'n_records' not in df.columns:
        df['n_records'] = np.nan

    df = df.sort_values(['Block', 'Location', 'Date']).reset_index(drop=True)
    return df


def build_feature_row(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    block: str,
    location: str,
    min_year: int,
) -> dict[str, float | str | pd.Timestamp]:
    ordered_history = history.sort_values('Date').reset_index(drop=True)
    recent_history = ordered_history.sort_values('Date', ascending=False).reset_index(drop=True)

    row: dict[str, float | str | pd.Timestamp] = {
        'Block': block,
        'Location': location,
        'Date': target_date,
        'Year': int(target_date.year),
        'Year_index': int(target_date.year - min_year),
        'Month': int(target_date.month),
        'Quarter': int(((target_date.month - 1) // 3) + 1),
        'Month_sin': float(np.sin(2 * np.pi * target_date.month / 12)),
        'Month_cos': float(np.cos(2 * np.pi * target_date.month / 12)),
        'History_points': int(len(ordered_history)),
    }

    lag_values: list[float] = []
    lag_gaps: list[float] = []

    for lag in LAG_STEPS:
        if lag <= len(recent_history):
            previous = recent_history.iloc[lag - 1]
            previous_date = pd.Timestamp(previous['Date'])
            previous_wqi = float(previous['WQI_mean'])
            gap = float(month_diff(target_date, previous_date))

            row[f'WQI_lag_{lag}'] = previous_wqi
            row[f'Gap_lag_{lag}_months'] = gap
            row[f'n_records_lag_{lag}'] = float(previous.get('n_records', np.nan))

            lag_values.append(previous_wqi)
            lag_gaps.append(gap)

            for source_col, alias in PARAMETER_ALIASES.items():
                if source_col in previous and pd.notna(previous[source_col]):
                    row[f'{alias}_lag_{lag}'] = float(previous[source_col])
                else:
                    row[f'{alias}_lag_{lag}'] = np.nan
        else:
            row[f'WQI_lag_{lag}'] = np.nan
            row[f'Gap_lag_{lag}_months'] = np.nan
            row[f'n_records_lag_{lag}'] = np.nan
            for alias in PARAMETER_ALIASES.values():
                row[f'{alias}_lag_{lag}'] = np.nan

    row['Lag_WQI_mean'] = float(np.mean(lag_values)) if lag_values else np.nan
    row['Lag_WQI_std'] = float(np.std(lag_values)) if len(lag_values) >= 2 else np.nan
    row['Lag_gap_mean_months'] = float(np.mean(lag_gaps)) if lag_gaps else np.nan
    row['Lag_gap_max_months'] = float(np.max(lag_gaps)) if lag_gaps else np.nan

    if not pd.isna(row['WQI_lag_1']) and not pd.isna(row['WQI_lag_2']):
        row['WQI_trend_1_2'] = float(row['WQI_lag_1'] - row['WQI_lag_2'])
    else:
        row['WQI_trend_1_2'] = np.nan

    if not pd.isna(row['WQI_lag_2']) and not pd.isna(row['WQI_lag_3']):
        row['WQI_trend_2_3'] = float(row['WQI_lag_2'] - row['WQI_lag_3'])
    else:
        row['WQI_trend_2_3'] = np.nan

    return row


def build_training_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    min_year = int(monthly_df['Date'].dt.year.min())
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for (block, location), group in monthly_df.groupby(['Block', 'Location'], sort=False):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) < 2:
            continue

        for idx in range(1, len(group)):
            target_row = group.iloc[idx]
            history = group.iloc[:idx].copy()
            feature_row = build_feature_row(
                history=history,
                target_date=pd.Timestamp(target_row['Date']),
                block=block,
                location=location,
                min_year=min_year,
            )
            feature_row[TARGET_COL] = float(target_row['WQI_mean'])
            rows.append(feature_row)

    dataset = pd.DataFrame(rows).sort_values(['Date', 'Block', 'Location']).reset_index(drop=True)
    return dataset


def get_feature_columns() -> list[str]:
    feature_cols = [
        'Block',
        'Location',
        'Year',
        'Year_index',
        'Month',
        'Quarter',
        'Month_sin',
        'Month_cos',
        'History_points',
        'Lag_WQI_mean',
        'Lag_WQI_std',
        'Lag_gap_mean_months',
        'Lag_gap_max_months',
        'WQI_trend_1_2',
        'WQI_trend_2_3',
    ]

    for lag in LAG_STEPS:
        feature_cols.extend(
            [
                f'WQI_lag_{lag}',
                f'Gap_lag_{lag}_months',
                f'n_records_lag_{lag}',
            ]
        )
        for alias in PARAMETER_ALIASES.values():
            feature_cols.append(f'{alias}_lag_{lag}')

    return feature_cols


def build_random_forest(feature_cols: list[str]) -> Pipeline:
    numeric_cols = [col for col in feature_cols if col not in CATEGORICAL_COLS]
    return Pipeline(
        steps=[
            (
                'prep',
                ColumnTransformer(
                    transformers=[
                        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
                        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols),
                    ]
                ),
            ),
            (
                'model',
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_catboost() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='RMSE',
        depth=6,
        learning_rate=0.03,
        iterations=800,
        l2_leaf_reg=5.0,
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )


def build_xgboost(feature_cols: list[str]) -> Pipeline:
    numeric_cols = [col for col in feature_cols if col not in CATEGORICAL_COLS]
    return Pipeline(
        steps=[
            (
                'prep',
                ColumnTransformer(
                    transformers=[
                        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
                        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols),
                    ]
                ),
            ),
            (
                'model',
                XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.04,
                    min_child_weight=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=3.0,
                    gamma=0.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist',
                ),
            ),
        ]
    )


def fit_and_predict(
    model_name: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    if model_name == 'NaiveLastValue':
        return test_df['WQI_lag_1'].to_numpy(dtype=float)

    if model_name == 'RandomForest':
        model = build_random_forest(feature_cols)
        model.fit(train_df[feature_cols], train_df[TARGET_COL])
        return model.predict(test_df[feature_cols])

    if model_name == 'XGBoost':
        model = build_xgboost(feature_cols)
        model.fit(train_df[feature_cols], train_df[TARGET_COL])
        return model.predict(test_df[feature_cols])

    if model_name == 'CatBoost':
        model = build_catboost()
        x_train = train_df[feature_cols].copy()
        x_test = test_df[feature_cols].copy()
        for col in CATEGORICAL_COLS:
            x_train[col] = x_train[col].astype(str)
            x_test[col] = x_test[col].astype(str)
        model.fit(x_train, train_df[TARGET_COL], cat_features=CATEGORICAL_COLS)
        return model.predict(x_test)

    raise ValueError(f'Unsupported model: {model_name}')


def select_backtest_dates(dataset: pd.DataFrame) -> list[pd.Timestamp]:
    counts = dataset.groupby('Date').size().sort_index()
    eligible_dates: list[pd.Timestamp] = []

    for current_date, row_count in counts.items():
        if row_count < MIN_TEST_ROWS:
            continue
        train_rows = int((dataset['Date'] < current_date).sum())
        if train_rows < MIN_TRAIN_ROWS:
            continue
        eligible_dates.append(pd.Timestamp(current_date))

    return eligible_dates[-MAX_BACKTEST_PERIODS:]


def rolling_origin_backtest(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    backtest_dates = select_backtest_dates(dataset)
    if not backtest_dates:
        raise RuntimeError('No eligible backtest dates were found. Lower MIN_TRAIN_ROWS or MIN_TEST_ROWS.')

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for model_name in model_names:
        for cutoff_date in backtest_dates:
            train_df = dataset[dataset['Date'] < cutoff_date].copy()
            test_df = dataset[dataset['Date'] == cutoff_date].copy()
            if len(test_df) < MIN_TEST_ROWS or len(train_df) < MIN_TRAIN_ROWS:
                continue

            predictions = fit_and_predict(model_name, feature_cols, train_df, test_df)
            fold_rmse = rmse(test_df[TARGET_COL], predictions)
            fold_mae = float(mean_absolute_error(test_df[TARGET_COL], predictions))
            fold_r2 = float(r2_score(test_df[TARGET_COL], predictions)) if len(test_df) >= 2 else np.nan

            metric_rows.append(
                {
                    'model': model_name,
                    'cutoff_date': cutoff_date.date().isoformat(),
                    'n_test': int(len(test_df)),
                    'rmse': fold_rmse,
                    'mae': fold_mae,
                    'r2': fold_r2,
                }
            )

            for index, (_, row) in enumerate(test_df.iterrows()):
                prediction_rows.append(
                    {
                        'model': model_name,
                        'Date': cutoff_date.date().isoformat(),
                        'Block': row['Block'],
                        'Location': row['Location'],
                        'Actual_WQI': float(row[TARGET_COL]),
                        'Predicted_WQI': float(predictions[index]),
                        'Absolute_Error': float(abs(row[TARGET_COL] - predictions[index])),
                    }
                )

    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df = pd.DataFrame(metric_rows)
    return predictions_df, metrics_df, backtest_dates


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for model_name, group in metrics_df.groupby('model'):
        summary_rows.append(
            {
                'model': model_name,
                'mean_rmse': float(group['rmse'].mean()),
                'std_rmse': float(group['rmse'].std(ddof=0)),
                'mean_mae': float(group['mae'].mean()),
                'mean_r2': float(group['r2'].mean()),
                'folds': int(len(group)),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(['mean_rmse', 'mean_mae']).reset_index(drop=True)


def evaluate_final_holdout(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    train_df = dataset[dataset['Date'] < HOLDOUT_START].copy()
    test_df = dataset[dataset['Date'] >= HOLDOUT_START].copy()
    if train_df.empty or test_df.empty:
        return pd.DataFrame(), None

    predictions = fit_and_predict(model_name, feature_cols, train_df, test_df)
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


def fit_final_model(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> object:
    if model_name == 'RandomForest':
        model = build_random_forest(feature_cols)
        model.fit(dataset[feature_cols], dataset[TARGET_COL])
        return model

    if model_name == 'XGBoost':
        model = build_xgboost(feature_cols)
        model.fit(dataset[feature_cols], dataset[TARGET_COL])
        return model

    if model_name == 'CatBoost':
        model = build_catboost()
        x_train = dataset[feature_cols].copy()
        for col in CATEGORICAL_COLS:
            x_train[col] = x_train[col].astype(str)
        model.fit(x_train, dataset[TARGET_COL], cat_features=CATEGORICAL_COLS)
        return model

    raise ValueError(f'Unsupported final model: {model_name}')


def predict_from_history(
    model: object,
    monthly_df: pd.DataFrame,
    feature_cols: list[str],
    min_year: int,
    location: str,
    target_date: str,
    block: str | None = None,
) -> dict[str, object]:
    target_timestamp = pd.Timestamp(target_date).replace(day=1)

    location_history = monthly_df[monthly_df['Location'] == location].copy()
    if block is not None:
        location_history = location_history[location_history['Block'] == block].copy()

    if location_history.empty:
        raise ValueError(f"Location '{location}' was not found in the monthly history.")

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
        min_year=min_year,
    )
    feature_frame = pd.DataFrame([feature_row])[feature_cols]

    if isinstance(model, CatBoostRegressor):
        for col in CATEGORICAL_COLS:
            feature_frame[col] = feature_frame[col].astype(str)

    predicted_wqi = float(model.predict(feature_frame)[0])
    return {
        'location': location,
        'block': block,
        'target_date': target_timestamp.date().isoformat(),
        'predicted_wqi': round(predicted_wqi, 2),
        'category': classify_wqi(predicted_wqi),
        'lag_1_wqi': None if pd.isna(feature_row['WQI_lag_1']) else round(float(feature_row['WQI_lag_1']), 2),
        'lag_2_wqi': None if pd.isna(feature_row['WQI_lag_2']) else round(float(feature_row['WQI_lag_2']), 2),
        'lag_3_wqi': None if pd.isna(feature_row['WQI_lag_3']) else round(float(feature_row['WQI_lag_3']), 2),
        'months_since_last_observation': None if pd.isna(feature_row['Gap_lag_1_months']) else int(feature_row['Gap_lag_1_months']),
    }


def save_feature_importance(model: object, feature_cols: list[str]) -> None:
    if not isinstance(model, CatBoostRegressor):
        return

    importance_df = pd.DataFrame(
        {
            'feature': feature_cols,
            'importance': model.get_feature_importance(),
        }
    ).sort_values('importance', ascending=False)
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    top_features = importance_df.head(15).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features['feature'], top_features['importance'], color='#1f77b4')
    ax.set_title('Top CatBoost Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=200, bbox_inches='tight')
    plt.close()


def save_backtest_plot(predictions_df: pd.DataFrame, best_model_name: str) -> None:
    best_predictions = predictions_df[predictions_df['model'] == best_model_name].copy()
    if best_predictions.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(
        best_predictions['Actual_WQI'],
        best_predictions['Predicted_WQI'],
        alpha=0.7,
        edgecolors='black',
        linewidth=0.3,
    )
    low = min(best_predictions['Actual_WQI'].min(), best_predictions['Predicted_WQI'].min())
    high = max(best_predictions['Actual_WQI'].max(), best_predictions['Predicted_WQI'].max())
    axes[0].plot([low, high], [low, high], linestyle='--', color='red', linewidth=1.5)
    axes[0].set_title(f'{best_model_name} Backtest: Actual vs Predicted')
    axes[0].set_xlabel('Actual WQI')
    axes[0].set_ylabel('Predicted WQI')
    axes[0].grid(alpha=0.3)

    by_date = (
        best_predictions.groupby('Date')[['Actual_WQI', 'Predicted_WQI']]
        .mean()
        .reset_index()
    )
    by_date['Date'] = pd.to_datetime(by_date['Date'])
    axes[1].plot(by_date['Date'], by_date['Actual_WQI'], marker='o', label='Actual')
    axes[1].plot(by_date['Date'], by_date['Predicted_WQI'], marker='o', label='Predicted')
    axes[1].set_title('Mean WQI by Backtest Month')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Mean WQI')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'backtest_summary.png', dpi=200, bbox_inches='tight')
    plt.close()


def main() -> None:
    print('=' * 72)
    print('WQI PANEL FORECASTER')
    print('=' * 72)

    monthly_df = load_monthly_data()
    feature_dataset = build_training_dataset(monthly_df)
    feature_cols = get_feature_columns()
    min_year = int(monthly_df['Date'].dt.year.min())

    print(f'Monthly rows            : {len(monthly_df):,}')
    print(f'Locations               : {monthly_df["Location"].nunique()}')
    print(f'Blocks                  : {monthly_df["Block"].nunique()}')
    print(f'Feature rows            : {len(feature_dataset):,}')
    print(f'Date range              : {feature_dataset["Date"].min().date()} to {feature_dataset["Date"].max().date()}')
    print(f'Holdout starts          : {HOLDOUT_START.date()}')

    feature_dataset.to_csv(OUTPUT_DIR / 'panel_feature_dataset.csv', index=False)

    model_names = ['NaiveLastValue', 'RandomForest', 'XGBoost', 'CatBoost']
    predictions_df, metrics_df, backtest_dates = rolling_origin_backtest(
        dataset=feature_dataset,
        feature_cols=feature_cols,
        model_names=model_names,
    )
    summary_df = summarize_metrics(metrics_df)

    predictions_df.to_csv(OUTPUT_DIR / 'rolling_origin_predictions.csv', index=False)
    metrics_df.to_csv(OUTPUT_DIR / 'rolling_origin_metrics.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)

    print('\nBacktest months:')
    for date_value in backtest_dates:
        print(f'  - {date_value.date().isoformat()}')

    print('\nModel comparison:')
    print(summary_df.to_string(index=False))

    best_model_name = str(summary_df.iloc[0]['model'])
    best_model = fit_final_model(feature_dataset, feature_cols, best_model_name)
    save_feature_importance(best_model, feature_cols)
    save_backtest_plot(predictions_df, best_model_name)

    holdout_df, holdout_metrics = evaluate_final_holdout(feature_dataset, feature_cols, best_model_name)
    if holdout_metrics is not None:
        holdout_df.to_csv(OUTPUT_DIR / 'final_holdout_predictions.csv', index=False)
        print('\nFinal holdout:')
        print(json.dumps(holdout_metrics, indent=2))

    artifacts = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_columns': feature_cols,
        'categorical_columns': CATEGORICAL_COLS,
        'monthly_history': monthly_df,
        'min_year': min_year,
    }
    joblib.dump(artifacts, OUTPUT_DIR / 'wqi_panel_forecaster.joblib')

    metadata = {
        'model_name': best_model_name,
        'monthly_rows': int(len(monthly_df)),
        'feature_rows': int(len(feature_dataset)),
        'n_locations': int(monthly_df['Location'].nunique()),
        'n_blocks': int(monthly_df['Block'].nunique()),
        'feature_columns': feature_cols,
        'backtest_dates': [date_value.date().isoformat() for date_value in backtest_dates],
        'holdout_start': HOLDOUT_START.date().isoformat(),
        'holdout_metrics': holdout_metrics,
    }
    with open(OUTPUT_DIR / 'model_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    print('\nSample predictions:')
    demo_locations = (
        monthly_df.groupby(['Block', 'Location']).size().reset_index(name='n')
        .sort_values('n', ascending=False)
        .head(3)
    )
    for _, row in demo_locations.iterrows():
        forecast = predict_from_history(
            model=best_model,
            monthly_df=monthly_df,
            feature_cols=feature_cols,
            min_year=min_year,
            location=str(row['Location']),
            block=str(row['Block']),
            target_date='2026-01-01',
        )
        print(f"  - {forecast['location']} ({forecast['block']}): {forecast['predicted_wqi']} [{forecast['category']}]")

    print('\nSaved outputs:')
    print(f"  - {OUTPUT_DIR / 'panel_feature_dataset.csv'}")
    print(f"  - {OUTPUT_DIR / 'rolling_origin_predictions.csv'}")
    print(f"  - {OUTPUT_DIR / 'rolling_origin_metrics.csv'}")
    print(f"  - {OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"  - {OUTPUT_DIR / 'wqi_panel_forecaster.joblib'}")
    print(f"  - {OUTPUT_DIR / 'model_meta.json'}")


if __name__ == '__main__':
    main()
