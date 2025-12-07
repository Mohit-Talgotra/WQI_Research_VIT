import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    def calculate_metrics(y_true, y_pred, set_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"\n{set_name} Set Metrics:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'predictions': y_pred}
    
    train_metrics = calculate_metrics(y_train, train_pred, "Train")
    val_metrics = calculate_metrics(y_val, val_pred, "Validation")
    test_metrics = calculate_metrics(y_test, test_pred, "Test")
    
    return train_metrics, val_metrics, test_metrics

def plot_results(y_test, test_pred, feature_cols, model):    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].scatter(y_test, test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual WQI', fontsize=12)
    axes[0, 0].set_ylabel('Predicted WQI', fontsize=12)
    axes[0, 0].set_title('Actual vs Predicted WQI', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    residuals = y_test - test_pred
    axes[0, 1].scatter(test_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted WQI', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]
    
    axes[1, 0].barh(range(len(indices)), importance[indices], color='steelblue')
    axes[1, 0].set_yticks(range(len(indices)))
    axes[1, 0].set_yticklabels([feature_cols[i] for i in indices], fontsize=10)
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Prediction Error', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('water_quality_forecast_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'water_quality_forecast_results.png'")
    plt.show()