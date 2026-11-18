"""
Compute statistical comparison tests (paired t-test / p-test, F-test / ANOVA on residuals),
standard deviations of error, and document optimizers and optimized hyperparameters.
Updates results/model_metrics.csv, results/statistical_comparison.csv, and results/final_report.csv.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

def compute_statistical_tests():
    base_dir = r"C:\Users\bolli\OneDrive\Documents\capstone_qml_project"
    preds_file = os.path.join(base_dir, "results", "test_predictions.csv")
    metrics_file = os.path.join(base_dir, "results", "model_metrics.csv")
    
    print(f"Reading test predictions from: {preds_file}")
    df_preds = pd.read_csv(preds_file)
    df_metrics = pd.read_csv(metrics_file)
    
    # Verify models
    models = ["Random Forest", "XGBoost", "LightGBM", "VQR", "QNN", "QSVR"]
    
    # Extract residuals and absolute errors per model
    model_errors = {}
    model_residuals = {}
    for m in models:
        sub = df_preds[df_preds["model"] == m].sort_values(by=["date", "market"])
        model_errors[m] = sub["absolute_error"].values
        model_residuals[m] = sub["error"].values
        print(f"Model: {m:15s} | Count: {len(sub)} | Mean Abs Err: {np.mean(sub['absolute_error'].values):.2f}")
        
    # Baseline model for paired t-test is the best performing model (Random Forest, lowest MAE)
    ref_model = "Random Forest"
    ref_errors = model_errors[ref_model]
    ref_residuals = model_residuals[ref_model]
    ref_var = np.var(ref_residuals, ddof=1)
    
    # Mapping of optimizers and optimization status
    optimizers = {
        "Random Forest": "Randomized/Grid CV (n_estimators=150, max_depth=20, split=3)",
        "XGBoost": "Newton-Raphson / 2nd-order Hist-GBDT (lr=0.05, depth=8, subsample=0.9)",
        "LightGBM": "Hist-based GBDT / Gradient Descent (num_leaves=63, lr=0.05, subsample=0.9)",
        "VQR": "Adam Optimizer (lr=0.02, batch_size=64, epochs=5, MSE loss)",
        "QNN": "Adam Optimizer (lr=0.01, batch_size=64, epochs=5, MSE loss)",
        "QSVR": "SMO / Dual Quadratic Programming Optimizer (C=10.0, epsilon=0.1)"
    }
    
    stat_rows = []
    updated_metrics_rows = []
    
    for idx, row in df_metrics.iterrows():
        m = row["model"]
        m_type = row["type"]
        mae = float(row["MAE"])
        rmse = float(row["RMSE"])
        r2 = float(row["R2"])
        mape = float(row["MAPE"])
        train_time = float(row["training_time_seconds"])
        pred_time = float(row["prediction_time_seconds"])
        train_samples = int(row["training_samples"])
        test_samples = int(row["test_samples"])
        
        errs = model_errors[m]
        resids = model_residuals[m]
        
        # Standard deviations
        err_std = float(np.std(errs, ddof=1)) # std dev of absolute error
        resid_std = float(np.std(resids, ddof=1)) # std dev of residual error
        resid_var = float(np.var(resids, ddof=1))
        
        # Paired t-test (p-test) against reference model
        if m == ref_model:
            t_stat = 0.0
            p_val_t = 1.0
            t_sig = "Reference Benchmark"
        else:
            t_res = stats.ttest_rel(errs, ref_errors)
            t_stat = float(t_res.statistic)
            p_val_t = float(t_res.pvalue)
            t_sig = "Statistically Significant (p < 0.05)" if p_val_t < 0.05 else "Not Significant (p >= 0.05)"
            
        # Two-tailed F-test (variance ratio of residuals)
        if m == ref_model:
            f_stat = 1.0
            p_val_f = 1.0
            f_sig = "Reference Benchmark"
        else:
            f_stat = float(resid_var / ref_var)
            dfn = len(resids) - 1
            dfd = len(ref_residuals) - 1
            # Two-tailed p-value from F-distribution
            cdf_val = stats.f.cdf(f_stat, dfn, dfd)
            p_val_f = float(2.0 * min(cdf_val, 1.0 - cdf_val))
            f_sig = "Significant Variance Difference (p < 0.05)" if p_val_f < 0.05 else "Equal Variance (p >= 0.05)"
            
        opt = optimizers.get(m, "Optimized")
        
        updated_metrics_rows.append({
            "model": m,
            "type": m_type,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2),
            "training_time_seconds": round(train_time, 2),
            "prediction_time_seconds": round(pred_time, 2),
            "training_samples": train_samples,
            "test_samples": test_samples,
            "error_std_dev": round(err_std, 2),
            "residual_std_dev": round(resid_std, 2),
            "t_statistic": round(t_stat, 4),
            "p_value_paired_ttest": f"{p_val_t:.4e}" if p_val_t < 0.0001 else f"{p_val_t:.4f}",
            "f_statistic": round(f_stat, 4),
            "f_test_p_value": f"{p_val_f:.4e}" if p_val_f < 0.0001 else f"{p_val_f:.4f}",
            "optimizer": opt,
            "optimization_status": "Optimized via Validation Tuning"
        })
        
        stat_rows.append({
            "model": m,
            "type": m_type,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "error_std_dev": round(err_std, 2),
            "residual_std_dev": round(resid_std, 2),
            "paired_t_statistic": round(t_stat, 4),
            "paired_t_p_value": f"{p_val_t:.4e}" if p_val_t < 0.0001 else f"{p_val_t:.4f}",
            "t_test_conclusion": t_sig,
            "f_statistic": round(f_stat, 4),
            "f_test_p_value": f"{p_val_f:.4e}" if p_val_f < 0.0001 else f"{p_val_f:.4f}",
            "f_test_conclusion": f_sig,
            "optimizer": opt
        })
        
    df_metrics_updated = pd.DataFrame(updated_metrics_rows)
    df_stat = pd.DataFrame(stat_rows)
    
    # Save back to CSVs
    df_metrics_updated.to_csv(metrics_file, index=False)
    print(f"Updated {metrics_file}")
    
    stat_file = os.path.join(base_dir, "results", "statistical_comparison.csv")
    df_stat.to_csv(stat_file, index=False)
    print(f"Created {stat_file}")
    
    final_report_file = os.path.join(base_dir, "results", "final_report.csv")
    df_metrics_updated.to_csv(final_report_file, index=False)
    print(f"Updated {final_report_file}")
    
    print("\nStatistical Comparison Table:")
    print(df_stat.to_string(index=False))

if __name__ == "__main__":
    compute_statistical_tests()
