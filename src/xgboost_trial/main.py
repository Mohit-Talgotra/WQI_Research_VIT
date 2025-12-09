import pandas as pd
import numpy as np
import joblib
import pickle
from loader import load_and_prepare_data
from features import create_basic_features, prepare_features_target
from train import time_based_split, train_xgboost
from evals import evaluate_model, plot_results
from predict import predict_future
import warnings
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

def main(filepath):
    print("="*80)
    print("WATER QUALITY FORECASTING WITH XGBOOST")
    print("="*80)
    
    df = load_and_prepare_data(filepath)
    
    df = create_basic_features(df)
    
    X, y, dates, villages, feature_cols = prepare_features_target(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y, dates)
    
    print("NaN in y_train:", np.isnan(y_train).sum())
    print("Inf in y_train:", np.isinf(y_train).sum())
    print("NaN in y_val:", np.isnan(y_val).sum())
    print("Inf in y_val:", np.isinf(y_val).sum())
    
    model = train_xgboost(X_train, y_train, X_val, y_val)

    train_metrics, val_metrics, test_metrics = evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    plot_results(y_test, test_metrics['predictions'], feature_cols, model)
    
    target_date = pd.Timestamp('2025-06-15')
    target_village = df['Village'].iloc[0]
    predict_future(model, df, feature_cols, target_date, target_village)
    
    print("\nSaving model...")
    joblib.dump(model, 'water_quality_model.joblib')
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("✓ Model saved as 'water_quality_model.joblib'")
    print("✓ Feature columns saved as 'feature_columns.pkl'")
    
    
    print("\n" + "="*80)
    print("FORECASTING COMPLETE!")
    print("="*80)
    
    return model, df, feature_cols

if __name__ == "__main__":
    filepath = os.environ["WQI_CALCULATED_DATA_FILE_PATH"]
    
    model, df, feature_cols = main(filepath)
    
    # model.predict(new_data)