"""
Sensitivity analysis for the 300% sub-index cap (Concern 3.2).
Evaluates WQI distributions, outlier locations (Sanyasigundu, Sarkar kollapatti),
and weight contributions under 200%, 300%, and Uncapped (1,000,000%) configurations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARAMETER_DATASET_PATH = ROOT / 'src' / 'data' / 'monthly_wqi_parameter_dataset.csv'
OUT_DIR = ROOT / 'reproducibility_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def calc_wqi_custom_cap(row: pd.Series, cap: float) -> tuple[float, dict[str, float]]:
    total = 0.0
    contribs = {}
    for param, spec in PARAMETER_STANDARDS.items():
        value = float(row[param])
        qn = ((value - spec['Videal']) / (spec['Sn'] - spec['Videal'])) * 100.0
        qn_clipped = float(np.clip(qn, 0.0, cap))
        w = WEIGHTS[param]
        c = qn_clipped * w
        total += c
        contribs[param] = c
    return total, contribs


def run_sensitivity():
    if not PARAMETER_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {PARAMETER_DATASET_PATH}")

    df = pd.read_csv(PARAMETER_DATASET_PATH, parse_dates=['Date'])
    caps = [200.0, 300.0, 1e6]
    labels = ['Cap_200pct', 'Cap_300pct_Default', 'Uncapped']

    summary_rows = []
    outlier_locs = ['Sanyasigundu', 'Sarkar Gollapatti', 'Sarkar kollapatti']

    for cap, label in zip(caps, labels):
        wqis = []
        f_contribs = []
        for _, row in df.iterrows():
            wqi_val, c_dict = calc_wqi_custom_cap(row, cap)
            wqis.append(wqi_val)
            f_contribs.append(c_dict['Fluoride_mean'])

        df_temp = df.copy()
        df_temp['WQI_sim'] = wqis
        df_temp['F_contrib'] = f_contribs
        df_temp['F_share_pct'] = (df_temp['F_contrib'] / df_temp['WQI_sim']) * 100.0

        # Outlier location check
        outlier_rows = df_temp[df_temp['Location'].str.strip().isin(outlier_locs)]
        outlier_mean_wqi = float(outlier_rows['WQI_sim'].mean()) if len(outlier_rows) > 0 else np.nan

        summary_rows.append({
            'Configuration': label,
            'Cap_Value': cap if cap < 1e5 else 'None (Inf)',
            'Mean_WQI': round(float(np.mean(wqis)), 2),
            'Median_WQI': round(float(np.median(wqis)), 2),
            'Max_WQI': round(float(np.max(wqis)), 2),
            'Std_WQI': round(float(np.std(wqis)), 2),
            'Mean_Fluoride_Contribution_Pct': round(float(df_temp['F_share_pct'].mean()), 2),
            'Outlier_Locations_Mean_WQI': round(outlier_mean_wqi, 2),
            'N_Samples': len(df_temp),
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = OUT_DIR / 'cap_sensitivity_summary.csv'
    summary_df.to_csv(out_csv, index=False)
    print("=" * 60)
    print("SUB-INDEX CAP SENSITIVITY ANALYSIS (Concern 3.2)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {out_csv}")


if __name__ == '__main__':
    run_sensitivity()
