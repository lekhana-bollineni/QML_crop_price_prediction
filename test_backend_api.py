"""
End-to-end verification of Flask API endpoints and all 6 ML/QML models.
Uses Flask's test_client to verify responses without port conflicts.
"""
import os
import sys
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

base_dir = r"C:\Users\bolli\OneDrive\Documents\capstone_qml_project"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "backend"))

from app import app

def test_api():
    print("=" * 80)
    print(">> TESTING CAPSTONE BACKEND FLASK API ENDPOINTS")
    print("=" * 80)
    
    client = app.test_client()
    
    # 1. GET /
    res = client.get("/")
    assert res.status_code == 200, f"GET / failed with {res.status_code}"
    print(f" [PASS] GET / -> Status {res.status_code} (HTML length: {len(res.data)} bytes)")
    
    # 2. GET /api/states
    res = client.get("/api/states")
    assert res.status_code == 200, f"GET /api/states failed with {res.status_code}"
    data = res.get_json()
    states = data.get("states", [])
    print(f" [PASS] GET /api/states -> {len(states)} states found. Top 3: {states[:3]}")
    
    # 3. GET /api/districts
    res = client.get("/api/districts?state=Maharashtra")
    assert res.status_code == 200, f"GET /api/districts failed with {res.status_code}"
    data = res.get_json()
    districts = data.get("districts", [])
    print(f" [PASS] GET /api/districts?state=Maharashtra -> {len(districts)} districts found. Sample: {districts[:3]}")
    
    # 4. GET /api/markets
    res = client.get("/api/markets?state=Maharashtra&district=Nashik")
    assert res.status_code == 200, f"GET /api/markets failed with {res.status_code}"
    data = res.get_json()
    markets = data.get("markets", [])
    print(f" [PASS] GET /api/markets?state=Maharashtra&district=Nashik -> {len(markets)} mandis found: {markets}")
    
    # 5. GET /api/model-metrics
    res = client.get("/api/model-metrics")
    assert res.status_code == 200, f"GET /api/model-metrics failed with {res.status_code}"
    data = res.get_json()
    metrics = data.get("metrics", [])
    print(f" [PASS] GET /api/model-metrics -> {len(metrics)} models loaded.")
    
    # 6. GET /api/comparison
    res = client.get("/api/comparison")
    assert res.status_code == 200, f"GET /api/comparison failed with {res.status_code}"
    data = res.get_json()
    print(f" [PASS] GET /api/comparison -> Status {res.status_code}. Leaders: {data.get('leaders')}")
    
    # 7. GET /api/statistical-tests
    res = client.get("/api/statistical-tests")
    assert res.status_code == 200, f"GET /api/statistical-tests failed with {res.status_code}"
    data = res.get_json()
    print(f" [PASS] GET /api/statistical-tests -> {len(data.get('statistical_tests', []))} statistical test rows returned.")
    
    # 8. POST /api/predict for all 6 models
    models = ["Random Forest", "XGBoost", "LightGBM", "QSVR", "QNN", "VQR"]
    print("\n" + "=" * 80)
    print(">> TESTING LIVE INFERENCE ACROSS ALL 6 MODELS")
    print("=" * 80)
    
    for m in models:
        payload = {
            "state": "Maharashtra",
            "district": "Nashik",
            "market": "Nasik",
            "month": 1,
            "year": 2025,
            "model": m,
            "quantity": 25.0
        }
        res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        assert res.status_code == 200, f"POST /api/predict for {m} failed: {res.data}"
        pdata = res.get_json()
        print(f" [PASS] Model: {m:15s} | Price: Rs. {pdata['predicted_price']:.2f} / qtl | Latency: {pdata['inference_latency_seconds']*1000:.2f}ms | Revenue (25 qtl): Rs. {pdata['estimated_revenue_rs']:,.2f}")
        
    print("\n" + "=" * 80)
    print(">> ALL BACKEND APIS AND 6 PREDICTION MODELS PASSED 100% SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_api()
