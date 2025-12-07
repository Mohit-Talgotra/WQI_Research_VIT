from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_basic_features(df):
    df['year'] = df['Sample Collection date'].dt.year
    df['month'] = df['Sample Collection date'].dt.month
    df['quarter'] = df['Sample Collection date'].dt.quarter
    df['day_of_year'] = df['Sample Collection date'].dt.dayofyear

    le_state = LabelEncoder()
    le_district = LabelEncoder()
    le_block = LabelEncoder()
    le_village = LabelEncoder()
    
    df['state_encoded'] = le_state.fit_transform(df['State'].fillna('Unknown'))
    df['district_encoded'] = le_district.fit_transform(df['District'].fillna('Unknown'))
    df['block_encoded'] = le_block.fit_transform(df['Block'].fillna('Unknown'))
    df['village_encoded'] = le_village.fit_transform(df['Village'].fillna('Unknown'))
    
    if 'Sample source' in df.columns:
        le_source = LabelEncoder()
        df['source_encoded'] = le_source.fit_transform(df['Sample source'].fillna('Unknown'))
    
    print(f"Features created. New shape: {df.shape}")
    
    return df

def prepare_features_target(df):
    feature_cols = [
        'Turbidity (NTU)', 'pH (NA)', 'TDS (mg/l)',
        'Total Alkalinity (as Calcium Carbonate) (mg/l)',
        'Chloride (as Cl) (mg/l)', 'Fluoride (as F) (mg/l)',
        'Nitrate (as NO3) (mg/l)', 'Sulphate (as SO4) (mg/l)',
        'Total Hardness (As CaCO3) (mg/l)', 'Iron (As Fe) (mg/l)',
        'Free residual Chlorine (mg/l)',
        'year', 'month', 'quarter', 'day_of_year',
        'state_encoded', 'district_encoded', 'block_encoded', 'village_encoded'
    ]
    
    if 'source_encoded' in df.columns:
        feature_cols.append('source_encoded')
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df['WQI'].copy()
    
    dates = df['Sample Collection date'].copy()
    villages = df['Village'].copy()
    
    return X, y, dates, villages, feature_cols