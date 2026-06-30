import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pymannkendall as mk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

LINEAR_INTERP_PATH = ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_parameter_interpolated_linear.csv'
SPLINE_INTERP_PATH = ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_parameter_interpolated_spline.csv'
ORIGINAL_PATH = ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_parameter_dataset.csv'

OUTPUT_DIR = ROOT / 'src' / 'data' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETER_COLS = [
    'pH_mean', 'TDS_mean', 'Hardness_mean', 'Chloride_mean', 
    'Fluoride_mean', 'Alkalinity_mean', 'Sulphate_mean', 'Nitrate_mean',
    'WQI_mean'
]

def run_mk_analysis(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Per Location Trend Analysis
    location_rows = []
    for (block, location), group in df.groupby(['Block', 'Location']):
        group = group.sort_values('Date').reset_index(drop=True)
        # We need at least 4 observations for Mann-Kendall test
        if len(group) < 4:
            continue
            
        for param in PARAMETER_COLS:
            series = group[param].dropna()
            if len(series) < 4:
                continue
            
            try:
                res = mk.original_test(series)
                location_rows.append({
                    'Dataset': dataset_name,
                    'Block': block,
                    'Location': location,
                    'Parameter': param,
                    'Trend': res.trend,
                    'h': bool(res.h),
                    'p_value': float(res.p),
                    'z_stat': float(res.z),
                    'sens_slope': float(res.slope),
                    'intercept': float(res.intercept),
                    'n_obs': len(series)
                })
            except Exception as e:
                # Handle cases with constant values or other exceptions
                pass
                
    df_location = pd.DataFrame(location_rows)
    
    # 2. Global Trend Analysis (averaged across Salem Block)
    global_rows = []
    global_monthly = df.groupby('Date')[PARAMETER_COLS].mean().reset_index().sort_values('Date')
    
    for param in PARAMETER_COLS:
        series = global_monthly[param].dropna()
        if len(series) < 4:
            continue
            
        try:
            res = mk.original_test(series)
            global_rows.append({
                'Dataset': dataset_name,
                'Parameter': param,
                'Trend': res.trend,
                'h': bool(res.h),
                'p_value': float(res.p),
                'z_stat': float(res.z),
                'sens_slope': float(res.slope),
                'intercept': float(res.intercept),
                'n_obs': len(series)
            })
        except Exception as e:
            pass
            
    df_global = pd.DataFrame(global_rows)
    
    return df_location, df_global

def main() -> None:
    print("=" * 72)
    print("RUNNING MANN-KENDALL TREND TEST AND SEN'S SLOPE ESTIMATION")
    print("=" * 72)
    
    # Run analysis for each dataset type
    datasets = {
        'Original': ORIGINAL_PATH,
        'Linear_Interpolated': LINEAR_INTERP_PATH,
        'Spline_Interpolated': SPLINE_INTERP_PATH
    }
    
    all_loc_dfs = []
    all_glob_dfs = []
    
    for name, path in datasets.items():
        if not path.exists():
            print(f"Dataset {name} not found at {path}, skipping.")
            continue
            
        print(f"Analyzing {name} dataset...")
        df = pd.read_csv(path, parse_dates=['Date'])
        df_loc, df_glob = run_mk_analysis(df, name)
        
        all_loc_dfs.append(df_loc)
        all_glob_dfs.append(df_glob)
        
    if not all_loc_dfs:
        print("No datasets found to analyze.")
        return
        
    df_loc_total = pd.concat(all_loc_dfs, ignore_index=True)
    df_glob_total = pd.concat(all_glob_dfs, ignore_index=True)
    
    # Save outputs
    loc_out = OUTPUT_DIR / 'trend_summary_per_location.csv'
    glob_out = OUTPUT_DIR / 'trend_summary_global.csv'
    
    df_loc_total.to_csv(loc_out, index=False)
    df_glob_total.to_csv(glob_out, index=False)
    
    print(f"\nTrend analysis complete:")
    print(f"  Per-location trend summary saved to: {loc_out}")
    print(f"  Global trend summary saved to: {glob_out}")
    
    # Print a nice summary of significant global trends for the interpolated datasets
    print("\n--- SIGNIFICANT GLOBAL TRENDS (Linear Interpolated) ---")
    sig_linear = df_glob_total[(df_glob_total['Dataset'] == 'Linear_Interpolated') & (df_glob_total['h'] == True)]
    if sig_linear.empty:
        print("No significant global trends found.")
    else:
        print(sig_linear[['Parameter', 'Trend', 'p_value', 'sens_slope']].to_string(index=False))
        
    print("\n--- SIGNIFICANT GLOBAL TRENDS (Spline Interpolated) ---")
    sig_spline = df_glob_total[(df_glob_total['Dataset'] == 'Spline_Interpolated') & (df_glob_total['h'] == True)]
    if sig_spline.empty:
        print("No significant global trends found.")
    else:
        print(sig_spline[['Parameter', 'Trend', 'p_value', 'sens_slope']].to_string(index=False))

    # Print summary of location-level trends for linear interpolated WQI
    print("\n--- LOCATION-LEVEL WQI TREND SUMMARY (Linear Interpolated) ---")
    wqi_loc_linear = df_loc_total[(df_loc_total['Dataset'] == 'Linear_Interpolated') & (df_loc_total['Parameter'] == 'WQI_mean')]
    if not wqi_loc_linear.empty:
        counts = wqi_loc_linear.groupby('Trend').size()
        print("WQI Trend distribution across locations:")
        for trend_type, count in counts.items():
            print(f"  {trend_type}: {count} locations")
    else:
        print("No location-level WQI trends analyzed.")

if __name__ == '__main__':
    main()
