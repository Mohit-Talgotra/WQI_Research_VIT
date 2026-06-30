import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

LINEAR_INTERP_PATH = ROOT / 'src' / 'data' / 'constructed_data' / 'monthly_wqi_parameter_interpolated_linear.csv'
OUTPUT_DIR = ROOT / 'src' / 'data' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['WQI_mean', 'Fluoride_mean', 'TDS_mean', 'Hardness_mean', 'Nitrate_mean']

def fit_lme_for_target(df: pd.DataFrame, target: str) -> dict[str, float | str]:
    print(f"\nFitting Linear Mixed-Effects Model for: {target}")
    print("-" * 50)
    
    # Model: Target ~ Decimal_Year with random intercept & slope for each Location
    # We use MixedLM from statsmodels
    try:
        model = smf.mixedlm(
            formula=f"{target} ~ Decimal_Year",
            data=df,
            groups=df["Location"],
            re_formula="~Decimal_Year"
        )
        result = model.fit()
        print(result.summary())
        
        # Save results in a dictionary
        fixed_effects = result.fe_params
        p_values = result.pvalues
        cov_re = result.cov_re
        
        return {
            'Parameter': target,
            'Intercept_coef': float(fixed_effects['Intercept']),
            'Intercept_p_val': float(p_values['Intercept']),
            'Decimal_Year_slope': float(fixed_effects['Decimal_Year']),
            'Decimal_Year_p_val': float(p_values['Decimal_Year']),
            'Random_Intercept_var': float(cov_re.iloc[0, 0]),
            'Random_Slope_var': float(cov_re.iloc[1, 1]),
            'Random_Covariance': float(cov_re.iloc[0, 1]),
            'Converged': bool(result.converged)
        }
    except Exception as e:
        print(f"Error fitting model for {target}: {e}")
        # Try a simpler random intercept model if the random slope model fails to converge
        try:
            print("Attempting simpler random-intercept only model...")
            model = smf.mixedlm(
                formula=f"{target} ~ Decimal_Year",
                data=df,
                groups=df["Location"]
            )
            result = model.fit()
            print(result.summary())
            
            fixed_effects = result.fe_params
            p_values = result.pvalues
            cov_re = result.cov_re
            
            return {
                'Parameter': target,
                'Intercept_coef': float(fixed_effects['Intercept']),
                'Intercept_p_val': float(p_values['Intercept']),
                'Decimal_Year_slope': float(fixed_effects['Decimal_Year']),
                'Decimal_Year_p_val': float(p_values['Decimal_Year']),
                'Random_Intercept_var': float(cov_re.iloc[0, 0]),
                'Random_Slope_var': 0.0,
                'Random_Covariance': 0.0,
                'Converged': bool(result.converged)
            }
        except Exception as ex:
            print(f"Failed simpler model: {ex}")
            return {
                'Parameter': target,
                'Intercept_coef': np.nan,
                'Intercept_p_val': np.nan,
                'Decimal_Year_slope': np.nan,
                'Decimal_Year_p_val': np.nan,
                'Random_Intercept_var': np.nan,
                'Random_Slope_var': np.nan,
                'Random_Covariance': np.nan,
                'Converged': False
            }

def main() -> None:
    print("=" * 72)
    print("RUNNING LINEAR MIXED-EFFECTS MODELS FOR TEMPORAL TREND STUDY")
    print("=" * 72)

    if not LINEAR_INTERP_PATH.exists():
        print(f"Error: Interpolated dataset not found at {LINEAR_INTERP_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(LINEAR_INTERP_PATH, parse_dates=['Date'])
    
    # We clean the location names to prevent formatting issues
    df['Location'] = df['Location'].astype(str).str.strip()
    
    results = []
    for target in TARGETS:
        res = fit_lme_for_target(df, target)
        results.append(res)
        
    df_results = pd.DataFrame(results)
    
    out_path = OUTPUT_DIR / 'mixed_effects_temporal_summary.csv'
    df_results.to_csv(out_path, index=False)
    
    print("\n" + "=" * 50)
    print(f"Linear Mixed-Effects Analysis Complete. Summary saved to: {out_path}")
    print("=" * 50)
    print(df_results[['Parameter', 'Decimal_Year_slope', 'Decimal_Year_p_val', 'Converged']].to_string(index=False))

if __name__ == '__main__':
    main()
