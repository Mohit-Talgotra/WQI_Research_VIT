import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def time_based_split(X, y, dates, test_size=0.2, val_size=0.15):
    df_split = pd.DataFrame({
        'date': dates,
        'y': y
    })
    df_split = pd.concat([df_split, X.reset_index(drop=True)], axis=1)
    df_split = df_split.sort_values('date')
    
    n = len(df_split)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_df = df_split.iloc[:train_end]
    val_df = df_split.iloc[train_end:val_end]
    test_df = df_split.iloc[val_end:]
    
    X_train = train_df.drop(['date', 'y'], axis=1)
    y_train = train_df['y']
    
    X_val = val_df.drop(['date', 'y'], axis=1)
    y_val = val_df['y']
    
    X_test = test_df.drop(['date', 'y'], axis=1)
    y_test = test_df['y']
    
    print(f"\nTime-based split:")
    print(f"Train: {len(X_train)} samples ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Val:   {len(X_val)} samples ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test:  {len(X_test)} samples ({test_df['date'].min()} to {test_df['date'].max()})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_xgboost(X_train, y_train, X_val, y_val):    
    print("\nTraining XGBoost model...")
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print("Model training complete!")
    
    return model