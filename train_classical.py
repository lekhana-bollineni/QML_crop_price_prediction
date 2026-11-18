"""
Train and tune Classical Machine Learning Models:
1. Random Forest Regressor
2. XGBoost Regressor
3. LightGBM Regressor

Tuned on validation dataset (2022-2023). Zero test data used for tuning.
Saves trained models to backend/saved_models/.
"""
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def safe_mape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-5)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def main():
    print("=" * 80)
    print(">> STEP 1: LOADING DATASETS (LEAKAGE-FREE CHRONOLOGICAL SPLITS)")
    print("=" * 80)
    
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")
    
    with open("backend/saved_models/feature_config.json", "r") as f:
        config = json.load(f)
        
    feature_cols = config["feature_cols"]
    target_col = config["target_col"]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    sample_weight = train_df["sample_weight"].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()
    
    print(f"Training split (2014-2021): {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation split (2022-2023): {X_val.shape[0]} samples")
    
    os.makedirs("backend/saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    validation_tuning_results = []
    
    # ==========================================================================
    # 1. RANDOM FOREST REGRESSOR
    # ==========================================================================
    print("\n" + "=" * 80)
    print(">> 1. TUNING & TRAINING RANDOM FOREST REGRESSOR")
    print("=" * 80)
    rf_candidates = [
        {"n_estimators": 80, "max_depth": 12, "min_samples_split": 5},
        {"n_estimators": 120, "max_depth": 16, "min_samples_split": 4},
        {"n_estimators": 150, "max_depth": 20, "min_samples_split": 3}
    ]
    
    best_rf_params = None
    best_rf_val_mae = float("inf")
    
    for params in rf_candidates:
        print(f"Evaluating RF config: {params}...")
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        preds = m.predict(X_val)
        val_mae = mean_absolute_error(y_val, preds)
        val_rmse = root_mean_squared_error(y_val, preds)
        print(f"  Validation MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f}")
        validation_tuning_results.append({
            "model": "Random Forest",
            "params": str(params),
            "val_mae": val_mae,
            "val_rmse": val_rmse
        })
        if val_mae < best_rf_val_mae:
            best_rf_val_mae = val_mae
            best_rf_params = params
            
    print(f"Best RF Parameters: {best_rf_params} (Validation MAE: {best_rf_val_mae:.2f})")
    print("Fitting final Random Forest on complete training partition...")
    t0 = time.perf_counter()
    best_rf = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train, sample_weight=sample_weight)
    rf_train_time = time.perf_counter() - t0
    
    joblib.dump(best_rf, "backend/saved_models/random_forest.joblib")
    print(f"Saved random_forest.joblib (Training time: {rf_train_time:.2f}s)")
    
    # ==========================================================================
    # 2. XGBOOST REGRESSOR
    # ==========================================================================
    print("\n" + "=" * 80)
    print(">> 2. TUNING & TRAINING XGBOOST REGRESSOR")
    print("=" * 80)
    xgb_candidates = [
        {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.8},
        {"n_estimators": 180, "max_depth": 7, "learning_rate": 0.06, "subsample": 0.85},
        {"n_estimators": 220, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.9}
    ]
    
    best_xgb_params = None
    best_xgb_val_mae = float("inf")
    
    for params in xgb_candidates:
        print(f"Evaluating XGB config: {params}...")
        m = XGBRegressor(**params, objective="reg:squarederror", random_state=42, n_jobs=-1)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        preds = m.predict(X_val)
        val_mae = mean_absolute_error(y_val, preds)
        val_rmse = root_mean_squared_error(y_val, preds)
        print(f"  Validation MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f}")
        validation_tuning_results.append({
            "model": "XGBoost",
            "params": str(params),
            "val_mae": val_mae,
            "val_rmse": val_rmse
        })
        if val_mae < best_xgb_val_mae:
            best_xgb_val_mae = val_mae
            best_xgb_params = params
            
    print(f"Best XGB Parameters: {best_xgb_params} (Validation MAE: {best_xgb_val_mae:.2f})")
    print("Fitting final XGBoost on complete training partition...")
    t0 = time.perf_counter()
    best_xgb = XGBRegressor(**best_xgb_params, objective="reg:squarederror", random_state=42, n_jobs=-1)
    best_xgb.fit(X_train, y_train, sample_weight=sample_weight)
    xgb_train_time = time.perf_counter() - t0
    
    joblib.dump(best_xgb, "backend/saved_models/xgboost.joblib")
    print(f"Saved xgboost.joblib (Training time: {xgb_train_time:.2f}s)")
    
    # ==========================================================================
    # 3. LIGHTGBM REGRESSOR
    # ==========================================================================
    print("\n" + "=" * 80)
    print(">> 3. TUNING & TRAINING LIGHTGBM REGRESSOR")
    print("=" * 80)
    lgb_candidates = [
        {"n_estimators": 120, "num_leaves": 31, "learning_rate": 0.08, "subsample": 0.8},
        {"n_estimators": 180, "num_leaves": 45, "learning_rate": 0.06, "subsample": 0.85},
        {"n_estimators": 220, "num_leaves": 63, "learning_rate": 0.05, "subsample": 0.9}
    ]
    
    best_lgb_params = None
    best_lgb_val_mae = float("inf")
    
    for params in lgb_candidates:
        print(f"Evaluating LightGBM config: {params}...")
        m = LGBMRegressor(**params, objective="regression", random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        preds = m.predict(X_val)
        val_mae = mean_absolute_error(y_val, preds)
        val_rmse = root_mean_squared_error(y_val, preds)
        print(f"  Validation MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f}")
        validation_tuning_results.append({
            "model": "LightGBM",
            "params": str(params),
            "val_mae": val_mae,
            "val_rmse": val_rmse
        })
        if val_mae < best_lgb_val_mae:
            best_lgb_val_mae = val_mae
            best_lgb_params = params
            
    print(f"Best LightGBM Parameters: {best_lgb_params} (Validation MAE: {best_lgb_val_mae:.2f})")
    print("Fitting final LightGBM on complete training partition...")
    t0 = time.perf_counter()
    best_lgb = LGBMRegressor(**best_lgb_params, objective="regression", random_state=42, n_jobs=-1, verbose=-1)
    best_lgb.fit(X_train, y_train, sample_weight=sample_weight)
    lgb_train_time = time.perf_counter() - t0
    
    joblib.dump(best_lgb, "backend/saved_models/lightgbm.joblib")
    print(f"Saved lightgbm.joblib (Training time: {lgb_train_time:.2f}s)")
    
    # Save training times metadata
    times_meta = {
        "Random Forest": {
            "training_time_seconds": rf_train_time,
            "training_samples": len(X_train),
            "params": best_rf_params
        },
        "XGBoost": {
            "training_time_seconds": xgb_train_time,
            "training_samples": len(X_train),
            "params": best_xgb_params
        },
        "LightGBM": {
            "training_time_seconds": lgb_train_time,
            "training_samples": len(X_train),
            "params": best_lgb_params
        }
    }
    
    meta_path = "backend/saved_models/classical_meta.json"
    with open(meta_path, "w") as f:
        json.dump(times_meta, f, indent=2)
        
    pd.DataFrame(validation_tuning_results).to_csv("results/validation_tuning_classical.csv", index=False)
    print("\nClassical models successfully tuned and trained!")

if __name__ == "__main__":
    main()
