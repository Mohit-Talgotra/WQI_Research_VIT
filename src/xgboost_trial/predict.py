import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def predict_future(model, df, feature_cols, target_date, target_village):    
    village_data = df[df['Village'] == target_village].sort_values('Sample Collection date').iloc[-1]
    
    pred_features = {}
    
    chem_cols = ['Turbidity (NTU)', 'pH (NA)', 'TDS (mg/l)',
                 'Total Alkalinity (as Calcium Carbonate) (mg/l)',
                 'Chloride (as Cl) (mg/l)', 'Fluoride (as F) (mg/l)',
                 'Nitrate (as NO3) (mg/l)', 'Sulphate (as SO4) (mg/l)',
                 'Total Hardness (As CaCO3) (mg/l)', 'Iron (As Fe) (mg/l)',
                 'Free residual Chlorine (mg/l)']
    
    for col in chem_cols:
        if col in feature_cols:
            pred_features[col] = village_data[col]
    
    pred_features['year'] = target_date.year
    pred_features['month'] = target_date.month
    pred_features['quarter'] = (target_date.month - 1) // 3 + 1
    pred_features['day_of_year'] = target_date.timetuple().tm_yday
    
    location_cols = ['state_encoded', 'district_encoded', 'block_encoded', 'village_encoded']
    if 'source_encoded' in feature_cols:
        location_cols.append('source_encoded')
    
    for col in location_cols:
        if col in feature_cols:
            pred_features[col] = village_data[col]
    
    X_pred = pd.DataFrame([pred_features])[feature_cols]
    
    prediction = model.predict(X_pred)[0]
    
    print(f"\nPrediction for {target_village} on {target_date.date()}:")
    print(f"Predicted WQI: {prediction:.2f}")
    
    return prediction