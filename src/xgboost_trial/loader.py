import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    
    df['Sample Collection date'] = pd.to_datetime(df['Sample Collection date'], dayfirst=True, errors='coerce')
    df['Sample tested date'] = pd.to_datetime(df['Sample tested date'], dayfirst=True, errors='coerce')
    
    df = df.sort_values(['Village', 'Sample Collection date']).reset_index(drop=True)
    
    numeric_cols = ['Turbidity (NTU)', 'pH (NA)', 'TDS (mg/l)', 
                    'Total Alkalinity (as Calcium Carbonate) (mg/l)',
                    'Chloride (as Cl) (mg/l)', 'Fluoride (as F) (mg/l)',
                    'Nitrate (as NO3) (mg/l)', 'Sulphate (as SO4) (mg/l)',
                    'Total Hardness (As CaCO3) (mg/l)', 'Iron (As Fe) (mg/l)',
                    'Free residual Chlorine (mg/l)', 'WQI']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill missing with median
            df[col].fillna(df[col].median(), inplace=True)
    
    df = df.dropna(subset=['Sample Collection date', 'WQI'])
    
    print(f"Loaded {len(df)} samples")
    print(f"Date range: {df['Sample Collection date'].min()} to {df['Sample Collection date'].max()}")
    print(f"Number of unique locations: {df['Village'].nunique()}")
    
    return df