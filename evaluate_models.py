"""
Evaluation, Benchmarking, Prediction Generation, and Visualization Suite.
Evaluates all 6 models on the 2024 out-of-time test partition (5,109 records):
1. Random Forest (Classical)
2. XGBoost (Classical)
3. LightGBM (Classical)
4. Variational Quantum Regressor - VQR (Hybrid QML)
5. Quantum Neural Network Regressor - QNN (Hybrid QML)
6. Quantum Kernel SVR - QSVR (Hybrid QML)

Outputs:
- results/model_metrics.csv
- results/test_predictions.csv
- results/final_report.csv
- results/final_predictions.csv
- results/computational_comparison.csv
- results/visualizations/ (12 publication-quality PNG charts)
"""
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy import stats
from qml_models import VariationalQuantumRegressor, QuantumNeuralNetworkRegressor, QuantumKernelSVR

# Matplotlib publication configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

def safe_mape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-5)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def main():
    print("=" * 80)
    print(">> STEP 3: RIGOROUS EVALUATION & BENCHMARKING ON 2024 TEST DATASET")
    print("=" * 80)
    
    test_df = pd.read_csv("test.csv")
    with open("backend/saved_models/feature_config.json", "r") as f:
        config = json.load(f)
        
    feature_cols = config["feature_cols"]
    top_8_quantum_features = config["top_8_quantum_features"]
    top_8_angle_cols = [f"{c}_angle" for c in top_8_quantum_features]
    target_col = config["target_col"]
    
    X_test_classical = test_df[feature_cols].copy()
    X_test_quantum = test_df[top_8_angle_cols].values
    y_test = test_df[target_col].values
    
    test_samples = len(y_test)
    print(f"Total test observations (Year 2024): {test_samples:,}")
    
    # Load metadata
    with open("backend/saved_models/classical_meta.json", "r") as f:
        classical_meta = json.load(f)
    with open("backend/saved_models/qml_meta.json", "r") as f:
        qml_meta = json.load(f)
        
    # Load Models
    print("\nLoading trained models...")
    models = {
        "Random Forest": {
            "type": "Classical",
            "model": joblib.load("backend/saved_models/random_forest.joblib"),
            "data": X_test_classical,
            "train_time": classical_meta["Random Forest"]["training_time_seconds"],
            "train_samples": classical_meta["Random Forest"]["training_samples"]
        },
        "XGBoost": {
            "type": "Classical",
            "model": joblib.load("backend/saved_models/xgboost.joblib"),
            "data": X_test_classical,
            "train_time": classical_meta["XGBoost"]["training_time_seconds"],
            "train_samples": classical_meta["XGBoost"]["training_samples"]
        },
        "LightGBM": {
            "type": "Classical",
            "model": joblib.load("backend/saved_models/lightgbm.joblib"),
            "data": X_test_classical,
            "train_time": classical_meta["LightGBM"]["training_time_seconds"],
            "train_samples": classical_meta["LightGBM"]["training_samples"]
        },
        "VQR": {
            "type": "Quantum",
            "model": VariationalQuantumRegressor.load("backend/saved_models/vqr_model.joblib"),
            "data": X_test_quantum,
            "train_time": qml_meta["VQR"]["training_time_seconds"],
            "train_samples": qml_meta["VQR"]["training_samples"]
        },
        "QNN": {
            "type": "Quantum",
            "model": QuantumNeuralNetworkRegressor.load("backend/saved_models/qnn_model.joblib"),
            "data": X_test_quantum,
            "train_time": qml_meta["QNN"]["training_time_seconds"],
            "train_samples": qml_meta["QNN"]["training_samples"]
        },
        "QSVR": {
            "type": "Quantum",
            "model": QuantumKernelSVR.load("backend/saved_models/qsvr_model.joblib"),
            "data": X_test_quantum,
            "train_time": qml_meta["QSVR"]["training_time_seconds"],
            "train_samples": qml_meta["QSVR"]["training_samples"]
        }
    }
    
    metrics_records = []
    all_predictions_records = []
    model_predictions_dict = {}
    
    print("\nRunning inference & calculating metrics on 2024 test data...")
    for name, item in models.items():
        m = item["model"]
        m_type = item["type"]
        X_in = item["data"]
        train_time = item["train_time"]
        train_samples = item["train_samples"]
        
        # Measure inference time accurately
        t0 = time.perf_counter()
        preds = m.predict(X_in)
        pred_time = time.perf_counter() - t0
        
        preds = np.asarray(preds, dtype=np.float64).flatten()
        model_predictions_dict[name] = preds
        
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mape = float(safe_mape(y_test, preds))
        
        print(f"[{m_type:9s}] {name:15s} | MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | R2: {r2:6.4f} | MAPE: {mape:5.2f}% | Infer: {pred_time:6.2f}s")
        
        metrics_records.append({
            "model": name,
            "type": m_type,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2),
            "training_time_seconds": round(train_time, 2),
            "prediction_time_seconds": round(pred_time, 2),
            "training_samples": train_samples,
            "test_samples": test_samples
        })
        
        # Test predictions rows
        for idx, row in test_df.iterrows():
            actual = float(y_test[idx])
            pred_val = float(preds[idx])
            err = pred_val - actual
            abs_err = abs(err)
            pct_err = (abs_err / max(actual, 1e-5)) * 100.0
            
            all_predictions_records.append({
                "date": f"{int(row['year'])}-{int(row['month']):02d}",
                "target_date": f"{int(row['target_year'])}-{int(row['target_month']):02d}",
                "state": row["state"],
                "district": row["district"],
                "market": row["market"],
                "actual_price": round(actual, 2),
                "predicted_price": round(pred_val, 2),
                "model": name,
                "error": round(err, 2),
                "absolute_error": round(abs_err, 2),
                "percentage_error": round(pct_err, 2)
            })

    # DataFrames
    metrics_df = pd.DataFrame(metrics_records)
    preds_df = pd.DataFrame(all_predictions_records)
    
    # Save CSVs
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    metrics_path = "results/model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")
    
    test_preds_path = "results/test_predictions.csv"
    preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved {test_preds_path} ({len(preds_df):,} rows)")
    
    final_report_path = "results/final_report.csv"
    metrics_df.to_csv(final_report_path, index=False)
    print(f"Saved {final_report_path}")
    
    final_preds_path = "results/final_predictions.csv"
    preds_df.to_csv(final_preds_path, index=False)
    print(f"Saved {final_preds_path}")
    
    comp_df = metrics_df[["model", "type", "training_time_seconds", "prediction_time_seconds", "training_samples", "test_samples"]].copy()
    comp_path = "results/computational_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved {comp_path}")
    
    # Dynamic best calculation (Zero hardcoding)
    best_by_mae = metrics_df.loc[metrics_df["MAE"].idxmin()]["model"]
    best_by_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin()]["model"]
    best_by_r2 = metrics_df.loc[metrics_df["R2"].idxmax()]["model"]
    best_by_mape = metrics_df.loc[metrics_df["MAPE"].idxmin()]["model"]
    fastest_train = metrics_df.loc[metrics_df["training_time_seconds"].idxmin()]["model"]
    fastest_pred = metrics_df.loc[metrics_df["prediction_time_seconds"].idxmin()]["model"]
    
    summary_findings = {
        "best_by_MAE": best_by_mae,
        "best_by_RMSE": best_by_rmse,
        "best_by_R2": best_by_r2,
        "best_by_MAPE": best_by_mape,
        "fastest_training_model": fastest_train,
        "fastest_prediction_model": fastest_pred
    }
    with open("results/summary_findings.json", "w") as f:
        json.dump(summary_findings, f, indent=2)
        
    print("\n" + "=" * 80)
    print(">> BENCHMARK SUMMARY & OBJECTIVE FINDINGS (DYNAMICALLY EVALUATED)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("\nObjective Leaders:")
    for k, v in summary_findings.items():
        print(f"  {k:26s}: {v}")
        
    # ==========================================================================
    # GENERATE 12 PUBLICATION-QUALITY VISUALIZATIONS
    # ==========================================================================
    print("\n" + "=" * 80)
    print(">> GENERATING 12 PUBLICATION-QUALITY CHARTS (300 DPI)")
    print("=" * 80)
    
    model_colors = {
        "Random Forest": "#2E7D32",   # Dark green
        "XGBoost": "#1565C0",         # Blue
        "LightGBM": "#00838F",        # Teal
        "VQR": "#E65100",             # Deep orange
        "QNN": "#6A1B9A",             # Purple
        "QSVR": "#AD1457"             # Magenta/Pink
    }
    palette = [model_colors[m] for m in metrics_df["model"]]
    
    # 1. Actual vs Predicted price (Scatter plot with parity line)
    plt.figure(figsize=(9, 7), dpi=300)
    # Take stratified sample of 1,000 points for readable scatter
    sample_indices = np.random.RandomState(42).choice(len(y_test), size=min(1200, len(y_test)), replace=False)
    for name in ["Random Forest", "XGBoost", "LightGBM", "QSVR", "QNN", "VQR"]:
        plt.scatter(y_test[sample_indices], model_predictions_dict[name][sample_indices], 
                    label=name, color=model_colors[name], alpha=0.45, s=22)
    max_val = max(np.max(y_test), np.max([model_predictions_dict[m] for m in models]))
    plt.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label='Perfect Parity ($y = \\hat{y}$)')
    plt.xlabel("Actual Onion Modal Price (Rs/quintal)", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted Onion Modal Price (Rs/quintal)", fontsize=12, fontweight='bold')
    plt.title("1. Actual vs. Predicted Wholesale Onion Price (2024 Test Set)", fontsize=14, fontweight='bold', pad=12)
    plt.grid(True)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    plt.tight_layout()
    plt.savefig("results/visualizations/01_actual_vs_predicted.png", dpi=300)
    plt.close()
    print("Saved 01_actual_vs_predicted.png")
    
    # 2. MAE Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(metrics_df["model"], metrics_df["MAE"], color=palette, width=0.55, edgecolor='#222', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 10, f"₹{h:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel("Mean Absolute Error (Rs/qtl)", fontsize=12, fontweight='bold')
    plt.title("2. Test Set Mean Absolute Error (MAE) Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0, metrics_df["MAE"].max() * 1.15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/visualizations/02_mae_comparison.png", dpi=300)
    plt.close()
    print("Saved 02_mae_comparison.png")
    
    # 3. RMSE Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(metrics_df["model"], metrics_df["RMSE"], color=palette, width=0.55, edgecolor='#222', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 15, f"₹{h:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel("Root Mean Squared Error (Rs/qtl)", fontsize=12, fontweight='bold')
    plt.title("3. Test Set Root Mean Squared Error (RMSE) Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0, metrics_df["RMSE"].max() * 1.15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/visualizations/03_rmse_comparison.png", dpi=300)
    plt.close()
    print("Saved 03_rmse_comparison.png")
    
    # 4. R2 Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(metrics_df["model"], metrics_df["R2"], color=palette, width=0.55, edgecolor='#222', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 0.02 if h >= 0 else h - 0.05, f"{h:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel("Coefficient of Determination ($R^2$)", fontsize=12, fontweight='bold')
    plt.title("4. Test Set $R^2$ Score Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(min(0, metrics_df["R2"].min() - 0.1), max(1.0, metrics_df["R2"].max() * 1.15))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/visualizations/04_r2_comparison.png", dpi=300)
    plt.close()
    print("Saved 04_r2_comparison.png")
    
    # 5. MAPE Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(metrics_df["model"], metrics_df["MAPE"], color=palette, width=0.55, edgecolor='#222', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 0.8, f"{h:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel("Mean Absolute Percentage Error (%)", fontsize=12, fontweight='bold')
    plt.title("5. Test Set MAPE Comparison (Safe-Epsilon Handling)", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0, metrics_df["MAPE"].max() * 1.15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/visualizations/05_mape_comparison.png", dpi=300)
    plt.close()
    print("Saved 05_mape_comparison.png")
    
    # 6. Training Time Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(metrics_df["model"], metrics_df["training_time_seconds"], color=palette, width=0.55, edgecolor='#222', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 1, f"{h:.2f}s", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel("Wall-Clock Training Time (seconds)", fontsize=12, fontweight='bold')
    plt.title("6. Model Training Time Benchmark", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0, metrics_df["training_time_seconds"].max() * 1.15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/visualizations/06_training_time_comparison.png", dpi=300)
    plt.close()
    print("Saved 06_training_time_comparison.png")
    
    # 7. Prediction / Inference Time Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(metrics_df["model"], metrics_df["prediction_time_seconds"], color=palette, width=0.55, edgecolor='#222', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h + 0.2, f"{h:.2f}s", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel(f"Prediction Time for {test_samples:,} Records (seconds)", fontsize=12, fontweight='bold')
    plt.title("7. Out-of-Time Test Set Inference Latency Benchmark", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0, metrics_df["prediction_time_seconds"].max() * 1.18)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/visualizations/07_prediction_time_comparison.png", dpi=300)
    plt.close()
    print("Saved 07_prediction_time_comparison.png")
    
    # 8. Historical onion price trend (Aggregated across major mandis)
    plt.figure(figsize=(12, 5.5), dpi=300)
    full_df = pd.read_csv("cleaned_onion_monthly.csv")
    monthly_trend = full_df.groupby(["year", "month"])["price_current_month"].agg(['mean', 'median', lambda x: np.percentile(x, 75), lambda x: np.percentile(x, 25)]).reset_index()
    monthly_trend.columns = ["year", "month", "mean_price", "median_price", "p75", "p25"]
    monthly_trend["date_str"] = monthly_trend.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
    
    x_axis = range(len(monthly_trend))
    plt.plot(x_axis, monthly_trend["median_price"], color='#004D40', lw=2.2, label='National Median Modal Price')
    plt.plot(x_axis, monthly_trend["mean_price"], color='#E65100', lw=1.8, linestyle='--', label='National Mean Modal Price')
    plt.fill_between(x_axis, monthly_trend["p25"], monthly_trend["p75"], color='#80CBC4', alpha=0.35, label='Interquartile Range (IQR: 25th-75th %ile)')
    
    # Annotate partitions
    # 2014-2021 = Train, 2022-2023 = Val, 2024 = Test
    train_end_idx = monthly_trend[(monthly_trend['year'] == 2021) & (monthly_trend['month'] == 12)].index[0]
    val_end_idx = monthly_trend[(monthly_trend['year'] == 2023) & (monthly_trend['month'] == 12)].index[0]
    
    plt.axvline(train_end_idx, color='#D32F2F', linestyle=':', lw=2, label='Train / Val Boundary (End of 2021)')
    plt.axvline(val_end_idx, color='#7B1FA2', linestyle=':', lw=2, label='Val / Test Boundary (End of 2023)')
    
    # Set x-ticks once per year
    year_ticks = []
    year_labels = []
    for yr in range(2014, 2025):
        match = monthly_trend[monthly_trend['year'] == yr]
        if not match.empty:
            year_ticks.append(match.index[0])
            year_labels.append(str(yr))
    plt.xticks(year_ticks, year_labels, fontsize=10)
    plt.ylabel("Wholesale Price (Rs/quintal)", fontsize=12, fontweight='bold')
    plt.title("8. Historical Wholesale Onion Price Trajectory & Crisis Spikes (2014–2024)", fontsize=14, fontweight='bold', pad=12)
    plt.legend(frameon=True, loc='upper left', fontsize=9.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/visualizations/08_historical_price_trend.png", dpi=300)
    plt.close()
    print("Saved 08_historical_price_trend.png")
    
    # 9. Model prediction comparison across test timeline (Monthly 2024 aggregate predictions)
    plt.figure(figsize=(11, 5.5), dpi=300)
    test_monthly_agg = preds_df.groupby(["target_date", "model"])["predicted_price"].mean().unstack()
    actual_monthly = preds_df.groupby("target_date")["actual_price"].mean()
    
    plt.plot(actual_monthly.index, actual_monthly.values, color='#000000', lw=3.0, marker='o', label='Actual Market Average')
    for m_name in ["Random Forest", "XGBoost", "LightGBM", "QSVR", "QNN", "VQR"]:
        if m_name in test_monthly_agg.columns:
            plt.plot(test_monthly_agg.index, test_monthly_agg[m_name], color=model_colors[m_name], 
                     lw=1.8, marker='s', markersize=5, linestyle='-', label=f"{m_name} Forecast")
                     
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel("Forecast Target Month (2024)", fontsize=12, fontweight='bold')
    plt.ylabel("Average Predicted Price (Rs/quintal)", fontsize=12, fontweight='bold')
    plt.title("9. Monthly Forecast Trajectory Comparison Across Models (2024 Out-of-Time Test)", fontsize=14, fontweight='bold', pad=12)
    plt.legend(frameon=True, loc='upper left', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/visualizations/09_model_prediction_comparison.png", dpi=300)
    plt.close()
    print("Saved 09_model_prediction_comparison.png")
    
    # 10. Model error distribution (KDE / Histogram)
    plt.figure(figsize=(10, 6), dpi=300)
    for m_name in ["Random Forest", "XGBoost", "LightGBM", "QSVR", "QNN", "VQR"]:
        m_errs = preds_df[preds_df["model"] == m_name]["error"]
        sns.kdeplot(m_errs, label=m_name, color=model_colors[m_name], lw=2.0)
    plt.axvline(0, color='black', linestyle='--', lw=1.2)
    plt.xlim(-1500, 1500)
    plt.xlabel("Prediction Residual Error ($e = \\hat{y} - y$, Rs/qtl)", fontsize=12, fontweight='bold')
    plt.ylabel("Density", fontsize=12, fontweight='bold')
    plt.title("10. Residual Error Distribution Density Across Models", fontsize=14, fontweight='bold', pad=12)
    plt.legend(frameon=True, loc='upper right', fontsize=9.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/visualizations/10_model_error_distribution.png", dpi=300)
    plt.close()
    print("Saved 10_model_error_distribution.png")
    
    # 11. Classical vs Quantum Computational Time (Grouped Bar Chart)
    plt.figure(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(metrics_df))
    width = 0.35
    
    plt.bar(x - width/2, metrics_df["training_time_seconds"], width, label='Training Time (s)', color='#1976D2', edgecolor='#222')
    plt.bar(x + width/2, metrics_df["prediction_time_seconds"], width, label='Inference Time (s)', color='#F57C00', edgecolor='#222')
    
    for i, row in metrics_df.iterrows():
        plt.text(i - width/2, row["training_time_seconds"] + 1, f"{row['training_time_seconds']:.1f}s", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        plt.text(i + width/2, row["prediction_time_seconds"] + 0.3, f"{row['prediction_time_seconds']:.1f}s", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    plt.xticks(x, [f"{m}\n({t})" for m, t in zip(metrics_df["model"], metrics_df["type"])], fontsize=10)
    plt.ylabel("Wall-Clock Seconds", fontsize=12, fontweight='bold')
    plt.title("11. Computational Complexity: Classical vs. Quantum Runtimes", fontsize=14, fontweight='bold', pad=12)
    plt.legend(frameon=True, loc='upper right', fontsize=10)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/visualizations/11_classical_vs_quantum_computational_time.png", dpi=300)
    plt.close()
    print("Saved 11_classical_vs_quantum_computational_time.png")
    
    # 12. Classical vs Quantum Metric Comparison (Multi-Panel Radar / Grouped Subplots)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Classical group vs Quantum group summary averages
    class_df = metrics_df[metrics_df["type"] == "Classical"]
    quant_df = metrics_df[metrics_df["type"] == "Quantum"]
    
    # Subplot A: Accuracy Metrics (MAE & RMSE)
    labels = ["MAE (Rs/qtl)", "RMSE (Rs/qtl)"]
    class_means = [class_df["MAE"].mean(), class_df["RMSE"].mean()]
    quant_means = [quant_df["MAE"].mean(), quant_df["RMSE"].mean()]
    
    x_idx = np.arange(len(labels))
    w = 0.3
    axes[0].bar(x_idx - w/2, class_means, w, label='Classical Avg (RF, XGB, LGB)', color='#2E7D32', edgecolor='#222')
    axes[0].bar(x_idx + w/2, quant_means, w, label='Quantum Avg (VQR, QNN, QSVR)', color='#6A1B9A', edgecolor='#222')
    for i in range(len(labels)):
        axes[0].text(i - w/2, class_means[i] + 12, f"₹{class_means[i]:.1f}", ha='center', fontweight='bold')
        axes[0].text(i + w/2, quant_means[i] + 12, f"₹{quant_means[i]:.1f}", ha='center', fontweight='bold')
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(labels, fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Error Scale (Rs/qtl - Lower is Better)", fontsize=11, fontweight='bold')
    axes[0].set_title("A: Error Scale Benchmarks", fontsize=12, fontweight='bold')
    axes[0].legend(frameon=True, fontsize=9.5)
    axes[0].grid(axis='y')
    
    # Subplot B: Fit Quality & Percentage Error (R2 & MAPE)
    labels_b = ["R2 Score", "MAPE (%)"]
    class_means_b = [class_df["R2"].mean(), class_df["MAPE"].mean()]
    quant_means_b = [quant_df["R2"].mean(), quant_df["MAPE"].mean()]
    
    # Normalized for side-by-side or individual bar
    x_models = np.arange(len(metrics_df))
    bars = axes[1].bar(x_models, metrics_df["R2"], color=palette, width=0.55, edgecolor='#222')
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2., h + 0.02 if h >= 0 else h - 0.06, f"{h:.3f}", ha='center', fontsize=9, fontweight='bold')
    axes[1].set_xticks(x_models)
    axes[1].set_xticklabels(metrics_df["model"], fontsize=10, rotation=25)
    axes[1].set_ylabel("R2 Score (Higher is Better)", fontsize=11, fontweight='bold')
    axes[1].set_title("B: Model Explained Variance ($R^2$)", fontsize=12, fontweight='bold')
    axes[1].grid(axis='y')
    
    plt.suptitle("12. Objective Classical vs. Hybrid Quantum Performance Matrix", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("results/visualizations/12_classical_vs_quantum_metric_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 12_classical_vs_quantum_metric_comparison.png")
    
    print("\nAll 12 visualizations successfully generated and saved to results/visualizations/!")

if __name__ == "__main__":
    main()
