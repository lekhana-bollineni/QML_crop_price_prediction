
import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add root folder to sys.path so qml_models can be unpickled/imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from qml_models import VariationalQuantumRegressor, QuantumNeuralNetworkRegressor, QuantumKernelSVR

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "frontend"), static_url_path="")
CORS(app)

# Global variables for models, lookups, and configuration
MODELS = {}
MARKET_LOOKUP = {}
FEATURE_CONFIG = {}
CATEGORICAL_ENCODERS = {}
FEATURE_SCALER = None
ANGLE_SCALER = None
METRICS_CACHE = []
FINDINGS_CACHE = {}
STATISTICAL_CACHE = []

def load_resources():
    global MODELS, MARKET_LOOKUP, FEATURE_CONFIG, CATEGORICAL_ENCODERS
    global FEATURE_SCALER, ANGLE_SCALER, METRICS_CACHE, FINDINGS_CACHE, STATISTICAL_CACHE
    
    print(">> Initializing Backend Resources & Loading Models Once at Startup...")
    saved_models_dir = os.path.join(BASE_DIR, "backend", "saved_models")
    
    # 1. Feature Config & Encoders
    with open(os.path.join(saved_models_dir, "feature_config.json"), "r") as f:
        FEATURE_CONFIG = json.load(f)
        
    CATEGORICAL_ENCODERS = joblib.load(os.path.join(saved_models_dir, "categorical_encoder.joblib"))
    FEATURE_SCALER = joblib.load(os.path.join(saved_models_dir, "feature_scaler.joblib"))
    ANGLE_SCALER = joblib.load(os.path.join(saved_models_dir, "angle_scaler.joblib"))
    
    # 2. Market Lookup
    lookup_file = os.path.join(BASE_DIR, "backend", "market_lookup.json")
    if os.path.exists(lookup_file):
        with open(lookup_file, "r", encoding="utf-8") as f:
            MARKET_LOOKUP = json.load(f)
    else:
        print("Warning: market_lookup.json not found! Run build_market_lookup.py first.")
        
    # 3. Load Classical Models
    print("Loading Classical Models...")
    MODELS["Random Forest"] = joblib.load(os.path.join(saved_models_dir, "random_forest.joblib"))
    MODELS["XGBoost"] = joblib.load(os.path.join(saved_models_dir, "xgboost.joblib"))
    MODELS["LightGBM"] = joblib.load(os.path.join(saved_models_dir, "lightgbm.joblib"))
    
    # 4. Load Quantum Models
    print("Loading Hybrid Quantum Models...")
    MODELS["VQR"] = VariationalQuantumRegressor.load(os.path.join(saved_models_dir, "vqr_model.joblib"))
    MODELS["QNN"] = QuantumNeuralNetworkRegressor.load(os.path.join(saved_models_dir, "qnn_model.joblib"))
    MODELS["QSVR"] = QuantumKernelSVR.load(os.path.join(saved_models_dir, "qsvr_model.joblib"))
    
    # 5. Load Metrics and Findings
    metrics_file = os.path.join(BASE_DIR, "results", "model_metrics.csv")
    if os.path.exists(metrics_file):
        df_m = pd.read_csv(metrics_file)
        METRICS_CACHE = df_m.to_dict(orient="records")
        
    findings_file = os.path.join(BASE_DIR, "results", "summary_findings.json")
    if os.path.exists(findings_file):
        with open(findings_file, "r") as f:
            FINDINGS_CACHE = json.load(f)

    stat_file = os.path.join(BASE_DIR, "results", "statistical_comparison.csv")
    if os.path.exists(stat_file):
        df_s = pd.read_csv(stat_file)
        STATISTICAL_CACHE = df_s.to_dict(orient="records")
            
    print(">> All Models and Lookup Indexes Successfully Loaded!")

# Load resources at server start
load_resources()

# ==============================================================================
# STATIC & FRONTEND ROUTES
# ==============================================================================
@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "frontend"), "index.html")

@app.route("/visualizations/<path:filename>")
def get_visualization(filename):
    # Security: validate filename and prevent directory traversal
    safe_name = os.path.basename(filename)
    return send_from_directory(os.path.join(BASE_DIR, "results", "visualizations"), safe_name)

# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================
@app.route("/api/states", methods=["GET"])
def get_states():
    states = MARKET_LOOKUP.get("states", [])
    return jsonify({"status": "success", "states": states})

@app.route("/api/districts", methods=["GET"])
def get_districts():
    state = request.args.get("state", "").strip()
    if not state:
        return jsonify({"error": "Missing 'state' query parameter"}), 400
    districts = MARKET_LOOKUP.get("districts_by_state", {}).get(state, [])
    if not districts:
        return jsonify({"error": f"No districts found for state '{state}'"}), 404
    return jsonify({"status": "success", "state": state, "districts": districts})

@app.route("/api/markets", methods=["GET"])
def get_markets():
    state = request.args.get("state", "").strip()
    district = request.args.get("district", "").strip()
    if not state or not district:
        return jsonify({"error": "Missing 'state' or 'district' query parameter"}), 400
    markets = MARKET_LOOKUP.get("markets_by_district", {}).get(state, {}).get(district, [])
    if not markets:
        return jsonify({"error": f"No markets found for district '{district}', state '{state}'"}), 404
    return jsonify({"status": "success", "state": state, "district": district, "markets": markets})

@app.route("/api/historical-prices", methods=["GET"])
def get_historical_prices():
    state = request.args.get("state", "").strip()
    district = request.args.get("district", "").strip()
    market = request.args.get("market", "").strip()
    
    if not state or not district or not market:
        return jsonify({"error": "Please provide 'state', 'district', and 'market'"}), 400
        
    key = f"{state}|{district}|{market}"
    series = MARKET_LOOKUP.get("historical_series", {}).get(key, [])
    if not series:
        return jsonify({"error": f"No historical price records found for market '{market}'"}), 404
        
    return jsonify({
        "status": "success",
        "market": market,
        "district": district,
        "state": state,
        "series": series
    })

@app.route("/api/model-metrics", methods=["GET"])
def get_model_metrics():
    return jsonify({
        "status": "success",
        "metrics": METRICS_CACHE,
        "findings": FINDINGS_CACHE
    })

@app.route("/api/comparison", methods=["GET"])
def get_comparison():
    comp_file = os.path.join(BASE_DIR, "results", "computational_comparison.csv")
    if os.path.exists(comp_file):
        df_comp = pd.read_csv(comp_file)
        comp_records = df_comp.to_dict(orient="records")
    else:
        comp_records = []
        
    return jsonify({
        "status": "success",
        "metrics": METRICS_CACHE,
        "computational": comp_records,
        "leaders": FINDINGS_CACHE,
        "statistical_tests": STATISTICAL_CACHE
    })

@app.route("/api/statistical-tests", methods=["GET"])
def get_statistical_tests():
    return jsonify({
        "status": "success",
        "reference_model": "Random Forest",
        "significance_threshold_alpha": 0.05,
        "statistical_tests": STATISTICAL_CACHE
    })

@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    # Filterable predictions endpoint
    model_filter = request.args.get("model", "").strip()
    limit = int(request.args.get("limit", 100))
    limit = min(max(1, limit), 1000)
    
    preds_file = os.path.join(BASE_DIR, "results", "test_predictions.csv")
    if not os.path.exists(preds_file):
        return jsonify({"error": "test_predictions.csv not found"}), 404
        
    df_p = pd.read_csv(preds_file)
    if model_filter:
        df_p = df_p[df_p["model"].str.lower() == model_filter.lower()]
        
    sample = df_p.head(limit).to_dict(orient="records")
    return jsonify({
        "status": "success",
        "count": len(sample),
        "total_available": len(df_p),
        "predictions": sample
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict next-month wholesale onion market price using historical market data and selected model.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # 1. Input Validation
    state = str(data.get("state", "")).strip()
    district = str(data.get("district", "")).strip()
    market = str(data.get("market", "")).strip()
    model_name = str(data.get("model", "")).strip()
    
    try:
        month = int(data.get("month", 0))
        year = int(data.get("year", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Month and Year must be valid integers"}), 400

    quantity = data.get("quantity")
    if quantity is not None:
        try:
            quantity = float(quantity)
            if quantity < 0:
                return jsonify({"error": "Quantity must be a non-negative number"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Quantity must be a valid numeric value"}), 400

    # Validate state, district, market
    if not state or not district or not market:
        return jsonify({"error": "state, district, and market are mandatory fields"}), 400

    # Validate month and year
    if not (1 <= month <= 12):
        return jsonify({"error": "Prediction month must be between 1 and 12"}), 400
    if not (2014 <= year <= 2035):
        return jsonify({"error": "Prediction year must be between 2014 and 2035"}), 400

    # Validate model
    if model_name not in MODELS:
        available_models = list(MODELS.keys())
        return jsonify({"error": f"Invalid model '{model_name}'. Available: {available_models}"}), 400

    # 2. Historical Feature Generation from Database
    key = f"{state}|{district}|{market}"
    latest_rec = MARKET_LOOKUP.get("latest_records", {}).get(key)
    
    if not latest_rec:
        return jsonify({
            "error": f"No historical trade data found for market '{market}' ({district}, {state}). Insufficient history to generate features."
        }), 404

    # Build the 37 tabular features
    feature_cols = FEATURE_CONFIG["feature_cols"]
    row_dict = {}
    for c in feature_cols:
        row_dict[c] = float(latest_rec.get(c, 0.0))

    # Update dynamic target temporal features
    row_dict["quarter"] = float((month - 1) // 3 + 1)
    row_dict["sin_month"] = float(np.sin(2 * np.pi * month / 12.0))
    row_dict["cos_month"] = float(np.cos(2 * np.pi * month / 12.0))

    # Encoded categorical features
    row_dict["market_encoded"] = float(CATEGORICAL_ENCODERS.get("market", {}).get(market, 0.0))
    row_dict["district_encoded"] = float(CATEGORICAL_ENCODERS.get("district", {}).get(district, 0.0))
    row_dict["state_encoded"] = float(CATEGORICAL_ENCODERS.get("state", {}).get(state, 0.0))
    row_dict["variety_encoded"] = float(CATEGORICAL_ENCODERS.get("variety", {}).get(latest_rec.get("variety", "Other"), 0.0))
    row_dict["crop_season_onion_encoded"] = float(CATEGORICAL_ENCODERS.get("crop_season_onion", {}).get(latest_rec.get("crop_season_onion", "Rabi"), 0.0))

    input_df = pd.DataFrame([row_dict])[feature_cols]

    # 3. Model Inference Execution
    t0 = time.perf_counter()
    selected_model = MODELS[model_name]
    
    if model_name in ["Random Forest", "XGBoost", "LightGBM"]:
        # Classical model takes the 37 raw/encoded tabular features
        pred_val = float(selected_model.predict(input_df)[0])
    else:
        # Quantum models (VQR, QNN, QSVR) take the 8 angle-encoded features
        top_8 = FEATURE_CONFIG["top_8_quantum_features"]
        # Transform via fitted angle_scaler into [0, pi]
        angle_features = ANGLE_SCALER.transform(input_df)
        angle_df = pd.DataFrame(angle_features, columns=feature_cols)
        X_qml = angle_df[top_8].values
        pred_val = float(selected_model.predict(X_qml)[0])
        
    inference_latency = time.perf_counter() - t0
    
    # Clip negative predictions to safe realistic lower boundary (Rs 100/qtl)
    predicted_price = max(100.0, round(pred_val, 2))

    # 4. Revenue Calculation
    estimated_revenue = None
    if quantity is not None:
        estimated_revenue = round(predicted_price * quantity, 2)

    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_name = month_names[month - 1]

    response = {
        "status": "success",
        "crop": "Onion (Wholesale)",
        "state": state,
        "district": district,
        "market": market,
        "month": month,
        "month_name": month_name,
        "year": year,
        "model": model_name,
        "predicted_price": predicted_price,
        "unit": "Rs/quintal",
        "inference_latency_seconds": round(inference_latency, 4),
        "latest_historical_mandi_price": round(float(latest_rec.get("latest_price", 0.0)), 2),
        "latest_data_period": f"{latest_rec.get('year')}-{latest_rec.get('month'):02d}"
    }
    
    if quantity is not None:
        response["quantity_quintals"] = quantity
        response["estimated_revenue_rs"] = estimated_revenue

    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f">> Starting Flask Application on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
