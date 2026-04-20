# WQI Forecasting

The strongest forecasting path in this repo right now is:

1. `uv sync`
2. `uv run python src/forecasting/parameter_forecaster.py`

This model forecasts the main chemistry parameters first, then recomputes WQI from those predicted parameters.

Current result on the same rolling future-month benchmark:

- `ParameterCatBoost`: best current model
- `DirectCatBoost`: strong baseline for direct WQI prediction
- `HierarchicalStateSpace`: experimental alternative, currently weaker

Outputs from the parameter-first model are written to `src/data/forecasting_parameters/`.

Other forecasting paths:

- `uv run python src/forecasting/panel_forecaster.py`
  Direct WQI forecasting baseline with CatBoost, XGBoost, RandomForest, and naive last-value comparison.
- `uv run python src/forecasting/hierarchical_state_space.py`
  Experimental location-level state-space model with block-level pooling.

Why the parameter-first path is preferred:

- It matches the domain better than predicting a single composite score directly.
- It now beats the direct WQI CatBoost model on both rolling backtests and the 2025 holdout.
- It keeps the WQI formula explicit instead of forcing the model to learn the index indirectly.
