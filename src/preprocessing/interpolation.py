import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

PARAMETER_DATASET_PATH = ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_parameter_dataset.csv'
OUTPUT_DIR = ROOT / 'src' / 'data' / 'constructed_data'

PARAMETER_STANDARDS = {
    'pH_mean': {'Sn': 8.5, 'Videal': 7.0, 'min': 0.0, 'max': 14.0},
    'TDS_mean': {'Sn': 500.0, 'Videal': 0.0, 'min': 0.0, 'max': 10000.0},
    'Hardness_mean': {'Sn': 200.0, 'Videal': 0.0, 'min': 0.0, 'max': 5000.0},
    'Chloride_mean': {'Sn': 250.0, 'Videal': 0.0, 'min': 0.0, 'max': 5000.0},
    'Fluoride_mean': {'Sn': 1.0, 'Videal': 0.0, 'min': 0.0, 'max': 20.0},
    'Alkalinity_mean': {'Sn': 200.0, 'Videal': 0.0, 'min': 0.0, 'max': 5000.0},
    'Sulphate_mean': {'Sn': 200.0, 'Videal': 0.0, 'min': 0.0, 'max': 5000.0},
    'Nitrate_mean': {'Sn': 45.0, 'Videal': 0.0, 'min': 0.0, 'max': 1000.0},
}
PARAMETER_COLS = list(PARAMETER_STANDARDS.keys())

K = 1.0 / sum(1.0 / spec['Sn'] for spec in PARAMETER_STANDARDS.values())
WEIGHTS = {name: K / spec['Sn'] for name, spec in PARAMETER_STANDARDS.items()}

def calc_wqi_from_row(row: pd.Series) -> float:
    total = 0.0
    for param, spec in PARAMETER_STANDARDS.items():
        value = float(row[param])
        qn = ((value - spec['Videal']) / (spec['Sn'] - spec['Videal'])) * 100.0
        total += float(np.clip(qn, 0.0, 300.0)) * WEIGHTS[param]
    return total

def calculate_decimal_year(dates: pd.Series) -> pd.Series:
    return dates.dt.year + (dates.dt.month - 1) / 12.0

def interpolate_location_series(group: pd.DataFrame, method: str) -> pd.DataFrame:
    group = group.sort_values('Date').reset_index(drop=True)
    if len(group) < 2:
        # Cannot interpolate with less than 2 points, just return group as is
        group['interpolated'] = False
        return group

    min_date = group['Date'].min()
    max_date = group['Date'].max()
    full_dates = pd.date_range(start=min_date, end=max_date, freq='MS')
    
    # Reindex to full dates
    block = group['Block'].iloc[0]
    location = group['Location'].iloc[0]
    
    df_reindexed = pd.DataFrame({'Date': full_dates})
    df_reindexed['Block'] = block
    df_reindexed['Location'] = location
    
    # Merge with observed data
    df_merged = pd.merge(df_reindexed, group, on=['Block', 'Location', 'Date'], how='left')
    
    # Track which rows are interpolated
    df_merged['interpolated'] = df_merged['WQI_mean'].isna()
    
    # We will interpolate each parameter
    x = (df_merged['Date'] - min_date).dt.days.values
    observed_mask = ~df_merged['interpolated'].values
    x_obs = x[observed_mask]
    
    for param in PARAMETER_COLS:
        y_obs = df_merged.loc[observed_mask, param].values
        
        if method == 'linear' or len(x_obs) < 3:
            f = interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')
        elif method == 'spline':
            # Use quadratic spline (order=2) if we have enough points, otherwise linear
            kind = 'quadratic' if len(x_obs) > 2 else 'linear'
            f = interp1d(x_obs, y_obs, kind=kind, fill_value='extrapolate')
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
            
        y_interp = f(x)
        # Apply physical boundary limits (clipping)
        spec = PARAMETER_STANDARDS[param]
        y_interp = np.clip(y_interp, spec['min'], spec['max'])
        df_merged[param] = y_interp

    # Recalculate WQI for all rows
    df_merged['WQI_mean'] = df_merged.apply(calc_wqi_from_row, axis=1)
    
    # Recalculate Year, Month
    df_merged['Year'] = df_merged['Date'].dt.year
    df_merged['Month'] = df_merged['Date'].dt.month
    
    return df_merged

def main() -> None:
    print("=" * 72)
    print("RUNNING TEMPORAL INTERPOLATION AND DECIMAL YEAR CALCULATION")
    print("=" * 72)

    if not PARAMETER_DATASET_PATH.exists():
        print(f"Error: Parameter dataset not found at {PARAMETER_DATASET_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(PARAMETER_DATASET_PATH, parse_dates=['Date'])
    
    # Sort
    df = df.sort_values(['Block', 'Location', 'Date']).reset_index(drop=True)
    
    # Add decimal year for observed data as well
    df['Decimal_Year'] = calculate_decimal_year(df['Date'])
    
    # Group by location and interpolate
    linear_dfs = []
    spline_dfs = []
    
    for _, group in df.groupby(['Block', 'Location']):
        linear_dfs.append(interpolate_location_series(group, 'linear'))
        spline_dfs.append(interpolate_location_series(group, 'spline'))
        
    df_linear = pd.concat(linear_dfs, ignore_index=True)
    df_spline = pd.concat(spline_dfs, ignore_index=True)
    
    # Add Decimal Year to interpolated dataframes
    df_linear['Decimal_Year'] = calculate_decimal_year(df_linear['Date'])
    df_spline['Decimal_Year'] = calculate_decimal_year(df_spline['Date'])
    
    # Save files
    linear_out = OUTPUT_DIR / 'monthly_wqi_parameter_interpolated_linear.csv'
    spline_out = OUTPUT_DIR / 'monthly_wqi_parameter_interpolated_spline.csv'
    
    df_linear.to_csv(linear_out, index=False)
    df_spline.to_csv(spline_out, index=False)
    
    print(f"\nInterpolation complete:")
    print(f"  Linear interpolated dataset saved to: {linear_out}")
    print(f"  Spline interpolated dataset saved to: {spline_out}")
    print(f"  Original dataset rows: {len(df)}")
    print(f"  Interpolated dataset rows: {len(df_linear)}")
    print(f"  New interpolated rows: {df_linear['interpolated'].sum()} out of {len(df_linear)}")

if __name__ == '__main__':
    main()
