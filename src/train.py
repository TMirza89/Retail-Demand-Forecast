import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ── 1. LOAD FEATURES ──────────────────────────────────────────────────────────
print("Loading features...")
df = pd.read_csv('../data/features.csv', parse_dates=['date'])
print(f"Shape: {df.shape}")

# ── 2. DEFINE FEATURES AND TARGET ─────────────────────────────────────────────
# Drop columns that would leak future info or are non-numeric identifiers
DROP_COLS = [
    'date', 'sales',           # target + index
    'event_name_1',            # text — encoded below
    'event_name_2',            # text
    'wday',                    # replaced by cyclical encoding
    'month',                   # replaced by cyclical encoding
    'is_closed',               # not available at prediction time
]

TARGET = 'sales'

# Encode event names as integers
le = LabelEncoder()
df['event_name_1_enc'] = le.fit_transform(df['event_name_1'].fillna('none'))
df['event_name_2_enc'] = le.fit_transform(df['event_name_2'].fillna('none'))

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
print(f"\nFeatures used: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

X = df[FEATURE_COLS]
y = df[TARGET]

# ── 3. TIME-BASED TRAIN / VALIDATION SPLIT ────────────────────────────────────
# WHY time-based and not random?
# Random split leaks future data into training — the model would "see" future sales
# In production you always train on past, predict on future
# We use last 6 months as validation
SPLIT_DATE = '2016-01-01'
train_mask = df['date'] < SPLIT_DATE
val_mask   = df['date'] >= SPLIT_DATE

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

print(f"\nTrain: {X_train.shape} | {df[train_mask]['date'].min().date()} → {df[train_mask]['date'].max().date()}")
print(f"Val  : {X_val.shape}   | {df[val_mask]['date'].min().date()} → {df[val_mask]['date'].max().date()}")

# ── 4. METRICS HELPER ─────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{label}")
    print(f"  MAE  : {mae:.0f} units")
    print(f"  RMSE : {rmse:.0f} units")
    print(f"  MAPE : {mape:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

# ── 5. MLFLOW SETUP ───────────────────────────────────────────────────────────
# MLflow tracks every experiment — params, metrics, models
# This is what separates DS notebooks from MLE workflows
mlflow.set_tracking_uri('../mlruns')
mlflow.set_experiment('retail_demand_forecast')
print("\nMLflow tracking started...")

# ── 6. MODEL 1 — XGBOOST ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("Training XGBoost...")
print("="*50)

xgb_params = {
    'n_estimators'    : 500,
    'max_depth'       : 6,
    'learning_rate'   : 0.05,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'random_state'    : 42,
    'n_jobs'          : -1,
}

with mlflow.start_run(run_name='xgboost_v1'):
    mlflow.log_params(xgb_params)

    xgb = XGBRegressor(**xgb_params)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    xgb_preds = xgb.predict(X_val)
    xgb_metrics = evaluate(y_val, xgb_preds, "XGBoost Validation")

    mlflow.log_metrics(xgb_metrics)
    mlflow.sklearn.log_model(xgb, 'xgboost_model')
    print("XGBoost logged to MLflow ✅")

# ── 7. MODEL 2 — LIGHTGBM ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("Training LightGBM...")
print("="*50)

lgbm_params = {
    'n_estimators'  : 500,
    'max_depth'     : 6,
    'learning_rate' : 0.05,
    'subsample'     : 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'random_state'  : 42,
    'n_jobs'        : -1,
    'verbose'       : -1,
}

with mlflow.start_run(run_name='lightgbm_v1'):
    mlflow.log_params(lgbm_params)

    lgbm = LGBMRegressor(**lgbm_params)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    lgbm_preds = lgbm.predict(X_val)
    lgbm_metrics = evaluate(y_val, lgbm_preds, "LightGBM Validation")

    mlflow.log_metrics(lgbm_metrics)
    mlflow.sklearn.log_model(lgbm, 'lightgbm_model')
    print("LightGBM logged to MLflow ✅")

# ── 8. COMPARE MODELS ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"{'Metric':<10} {'XGBoost':>12} {'LightGBM':>12}")
print("-" * 36)
for metric in ['mae', 'rmse', 'mape']:
    print(f"{metric.upper():<10} {xgb_metrics[metric]:>12.2f} {lgbm_metrics[metric]:>12.2f}")

# Pick best model
best_model  = xgb  if xgb_metrics['mae'] < lgbm_metrics['mae'] else lgbm
best_name   = 'XGBoost' if xgb_metrics['mae'] < lgbm_metrics['mae'] else 'LightGBM'
best_preds  = xgb_preds if best_name == 'XGBoost' else lgbm_preds
print(f"\nBest model: {best_name}")

# ── 9. SAVE BEST MODEL ────────────────────────────────────────────────────────
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/best_model.pkl')
joblib.dump(FEATURE_COLS, '../models/feature_cols.pkl')
print(f"\nModel saved → ../models/best_model.pkl")
print(f"Features saved → ../models/feature_cols.pkl")

# ── 10. FEATURE IMPORTANCE ────────────────────────────────────────────────────
print("\nPlotting feature importance...")
importance = pd.DataFrame({
    'feature'   : FEATURE_COLS,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(importance['feature'][::-1], importance['importance'][::-1], color='steelblue')
plt.title(f'Top 15 Feature Importances — {best_name}')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../data/feature_importance.png', dpi=150)
plt.show()

# ── 11. FORECAST vs ACTUAL PLOT ───────────────────────────────────────────────
print("Plotting forecast vs actual...")
val_dates = df[val_mask]['date']

plt.figure(figsize=(16, 5))
plt.plot(val_dates.values, y_val.values,   label='Actual',   linewidth=1.5, color='steelblue')
plt.plot(val_dates.values, best_preds,     label='Forecast', linewidth=1.5, color='coral', alpha=0.8)
plt.title(f'Forecast vs Actual — {best_name} | Validation Period')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.savefig('../data/forecast_vs_actual.png', dpi=150)
plt.show()

# ── 12. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n========== TRAINING SUMMARY ==========")
print(f"Best model        : {best_name}")
print(f"MAE               : {min(xgb_metrics['mae'], lgbm_metrics['mae']):.0f} units/day")
print(f"MAPE              : {min(xgb_metrics['mape'], lgbm_metrics['mape']):.2f}%")
print(f"Top feature       : {importance.iloc[0]['feature']}")
print(f"Model saved to    : ../models/best_model.pkl")
print(f"MLflow UI         : run 'mlflow ui' in terminal to view experiments")
print("======================================")

# saving xgboost for comparison
joblib.dump(xgb, '../models/xgb_model.pkl')
print("XGBoost also saved → ../models/xgb_model.pkl")