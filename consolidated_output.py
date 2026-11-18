"""
Consolidated Output Viewer for Model Benchmarks & Evaluation
Prints a clean, formatted terminal summary of all models from results/model_metrics.csv.
"""
import os
import json
import pandas as pd

def main():
    metrics_path = "results/model_metrics.csv"
    findings_path = "results/summary_findings.json"
    comp_path = "results/computational_comparison.csv"
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Please run 'python evaluate_models.py' first.")
        return

    df_metrics = pd.read_csv(metrics_path)
    
    print("\n" + "=" * 95)
    print("      QUANTUM-ENHANCED CROP PRICE PREDICTION - CONSOLIDATED MODEL PERFORMANCE")
    print("=" * 95)
    
    header = f"{'Model':<15} {'Type':<12} {'MAE (Rs/qtl)':<14} {'RMSE (Rs/qtl)':<15} {'R2 Score':<12} {'MAPE (%)':<12}"
    print(header)
    print("-" * 95)
    
    for _, row in df_metrics.iterrows():
        print(f"{row['model']:<15} {row['type']:<12} Rs {row['MAE']:<11.2f} Rs {row['RMSE']:<12.2f} {row['R2']:<12.4f} {row['MAPE']:<11.2f}%")
        
    print("=" * 95)
    
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        print("\n" + "=" * 95)
        print("                         COMPUTATIONAL RUNTIME BENCHMARKS")
        print("=" * 95)
        comp_header = f"{'Model':<15} {'Type':<12} {'Train Samples':<15} {'Test Samples':<15} {'Train Time':<15} {'Inference Time':<15}"
        print(comp_header)
        print("-" * 95)
        for _, row in df_comp.iterrows():
            print(f"{row['model']:<15} {row['type']:<12} {int(row['training_samples']):<15,d} {int(row['test_samples']):<15,d} {row['training_time_seconds']:<12.2f}s   {row['prediction_time_seconds']:<12.2f}s")
        print("=" * 95)

    if os.path.exists(findings_path):
        with open(findings_path, "r") as f:
            findings = json.load(f)
        print("\n" + "=" * 95)
        print("                           OBJECTIVE BENCHMARK LEADERS")
        print("=" * 95)
        for k, v in findings.items():
            clean_k = k.replace("_", " ").title()
            print(f"  * {clean_k:<28}: {v}")
        print("=" * 95)

    print("\nGenerated Artifact Files:")
    print("  * Full Metrics Table : results/model_metrics.csv")
    print("  * Executive Report   : results/final_report.csv")
    print("  * Test Predictions   : results/test_predictions.csv (30,654 rows)")
    print("  * Computational Stats: results/computational_comparison.csv")
    print("  * 12 Visual Charts   : results/visualizations/\n")

if __name__ == "__main__":
    main()
