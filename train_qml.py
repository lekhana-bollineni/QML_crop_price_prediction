"""
Train and evaluate Hybrid Quantum Machine Learning Models:
1. Variational Quantum Regressor (VQR)
2. Quantum Neural Network Regressor (QNN)
3. Quantum Kernel Support Vector Regression (QSVR)

Features: Top-8 quantum features mapped to [0, pi] angle encoding.
Supports controlled QML experiments with sample size logging.
Saves models to backend/saved_models/.
"""
import os
import time
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from qml_models import VariationalQuantumRegressor, QuantumNeuralNetworkRegressor, QuantumKernelSVR

def safe_mape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-5)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Quantum Machine Learning Models")
    parser.add_argument("--sample-size", type=int, default=5000, help="Training sample size for QML simulation (default: 5000)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for VQR and QNN (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for VQR and QNN (default: 64)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 80)
    print(">> STEP 2: HYBRID QUANTUM MACHINE LEARNING (QML) TRAINING PIPELINE")
    print(f">> Configuration: sample_size={args.sample_size}, epochs={args.epochs}, batch_size={args.batch_size}")
    print("=" * 80)

    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")

    top_8_quantum_features = [
        'price_current_month',
        'price_min_current_month',
        'tmin_current_month',
        'price_max_current_month',
        'price_change_ratio_1m',
        'rainfall_total_current_month',
        'price_median_current_month',
        'sin_month'
    ]
    top_8_angle_cols = [f"{c}_angle" for c in top_8_quantum_features]
    target_col = "target_price_next_month"

    # Verify columns exist
    for c in top_8_angle_cols:
        assert c in train_df.columns, f"Missing angle column: {c}"

    X_train = train_df[top_8_angle_cols].values
    y_train = train_df[target_col].values

    # Validation subset for rapid QML checkpoint evaluation
    val_sub_n = min(1500, len(val_df))
    np.random.seed(args.random_state)
    val_idx = np.random.choice(len(val_df), size=val_sub_n, replace=False)
    X_val = val_df[top_8_angle_cols].values[val_idx]
    y_val = val_df[target_col].values[val_idx]

    os.makedirs("backend/saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    qml_results = []
    meta = {}

    # ==========================================================================
    # 1. VARIATIONAL QUANTUM REGRESSOR (VQR)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(">> 1. TRAINING VARIATIONAL QUANTUM REGRESSOR (VQR)")
    print(f"   Architecture: 8 Qubits, 2 Entangling Layers, Angle Embedding (Ry), PauliZ Readout")
    print("=" * 80)
    vqr = VariationalQuantumRegressor(
        n_qubits=8,
        n_layers=2,
        lr=0.02,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        random_state=args.random_state
    )
    vqr.fit(X_train, y_train)
    vqr.save("backend/saved_models/vqr_model.joblib")
    
    val_preds_vqr = vqr.predict(X_val)
    vqr_val_mae = mean_absolute_error(y_val, val_preds_vqr)
    vqr_val_rmse = root_mean_squared_error(y_val, val_preds_vqr)
    vqr_val_r2 = r2_score(y_val, val_preds_vqr)
    vqr_val_mape = safe_mape(y_val, val_preds_vqr)
    
    print(f"VQR Trained in {vqr.training_time_:.2f}s on {vqr.training_samples_} samples")
    print(f"Validation MAE: {vqr_val_mae:.2f} | RMSE: {vqr_val_rmse:.2f} | R2: {vqr_val_r2:.4f} | MAPE: {vqr_val_mape:.2f}%")
    
    meta["VQR"] = {
        "training_time_seconds": vqr.training_time_,
        "training_samples": vqr.training_samples_,
        "n_qubits": 8,
        "n_layers": 2,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    qml_results.append({
        "model": "VQR",
        "training_samples": vqr.training_samples_,
        "training_time_seconds": vqr.training_time_,
        "val_mae": vqr_val_mae,
        "val_rmse": vqr_val_rmse,
        "val_r2": vqr_val_r2,
        "val_mape": vqr_val_mape
    })

    # ==========================================================================
    # 2. QUANTUM NEURAL NETWORK REGRESSOR (QNN)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(">> 2. TRAINING QUANTUM NEURAL NETWORK REGRESSOR (QNN)")
    print(f"   Architecture: 8 Qubits, 8-Expectation Vector -> Classical MLP (8->16->ReLU->1)")
    print("=" * 80)
    qnn = QuantumNeuralNetworkRegressor(
        n_qubits=8,
        n_layers=1,
        hidden_dim=16,
        lr=0.01,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        random_state=args.random_state
    )
    qnn.fit(X_train, y_train)
    qnn.save("backend/saved_models/qnn_model.joblib")
    
    val_preds_qnn = qnn.predict(X_val)
    qnn_val_mae = mean_absolute_error(y_val, val_preds_qnn)
    qnn_val_rmse = root_mean_squared_error(y_val, val_preds_qnn)
    qnn_val_r2 = r2_score(y_val, val_preds_qnn)
    qnn_val_mape = safe_mape(y_val, val_preds_qnn)
    
    print(f"QNN Trained in {qnn.training_time_:.2f}s on {qnn.training_samples_} samples")
    print(f"Validation MAE: {qnn_val_mae:.2f} | RMSE: {qnn_val_rmse:.2f} | R2: {qnn_val_r2:.4f} | MAPE: {qnn_val_mape:.2f}%")
    
    meta["QNN"] = {
        "training_time_seconds": qnn.training_time_,
        "training_samples": qnn.training_samples_,
        "n_qubits": 8,
        "n_layers": 1,
        "hidden_dim": 16,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    qml_results.append({
        "model": "QNN",
        "training_samples": qnn.training_samples_,
        "training_time_seconds": qnn.training_time_,
        "val_mae": qnn_val_mae,
        "val_rmse": qnn_val_rmse,
        "val_r2": qnn_val_r2,
        "val_mape": qnn_val_mape
    })

    # ==========================================================================
    # 3. QUANTUM KERNEL SUPPORT VECTOR REGRESSION (QSVR)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(">> 3. TRAINING QUANTUM KERNEL SUPPORT VECTOR REGRESSION (QSVR)")
    print(f"   Architecture: 8 Qubits, Analytical State-Vector Fidelity Gram Matrix K(xi, xj)")
    print("=" * 80)
    qsvr = QuantumKernelSVR(
        n_qubits=8,
        C=10.0,
        epsilon=0.1,
        sample_size=args.sample_size,
        random_state=args.random_state
    )
    qsvr.fit(X_train, y_train)
    qsvr.save("backend/saved_models/qsvr_model.joblib")
    
    val_preds_qsvr = qsvr.predict(X_val)
    qsvr_val_mae = mean_absolute_error(y_val, val_preds_qsvr)
    qsvr_val_rmse = root_mean_squared_error(y_val, val_preds_qsvr)
    qsvr_val_r2 = r2_score(y_val, val_preds_qsvr)
    qsvr_val_mape = safe_mape(y_val, val_preds_qsvr)
    
    print(f"QSVR Trained in {qsvr.training_time_:.2f}s on {qsvr.training_samples_} samples")
    print(f"Validation MAE: {qsvr_val_mae:.2f} | RMSE: {qsvr_val_rmse:.2f} | R2: {qsvr_val_r2:.4f} | MAPE: {qsvr_val_mape:.2f}%")
    
    meta["QSVR"] = {
        "training_time_seconds": qsvr.training_time_,
        "training_samples": qsvr.training_samples_,
        "n_qubits": 8,
        "C": 10.0,
        "epsilon": 0.1
    }
    qml_results.append({
        "model": "QSVR",
        "training_samples": qsvr.training_samples_,
        "training_time_seconds": qsvr.training_time_,
        "val_mae": qsvr_val_mae,
        "val_rmse": qsvr_val_rmse,
        "val_r2": qsvr_val_r2,
        "val_mape": qsvr_val_mape
    })

    with open("backend/saved_models/qml_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame(qml_results).to_csv("results/validation_tuning_qml.csv", index=False)
    print("\nHybrid Quantum ML models successfully trained, verified, and saved!")

if __name__ == "__main__":
    main()
