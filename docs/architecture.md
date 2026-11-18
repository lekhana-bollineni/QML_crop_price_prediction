# System Architecture & Technical Specifications

## 1. System Overview
The **Quantum-Enhanced Crop Market Price Prediction System** is structured as an end-to-end, reproducible research and deployment framework. It couples state-of-the-art Classical Machine Learning with parameterized Quantum Machine Learning (QML) simulation to predict wholesale agricultural commodity prices (specifically Indian wholesale onions).

```
+-----------------------------------------------------------------------------------+
|                                PRESENTATION TIER                                  |
|   HTML5 / CSS3 / Vanilla JavaScript / Chart.js 4.4 Dashboard (Responsive UI)      |
+------------------------------------------+----------------------------------------+
                                           | HTTP REST / JSON
                                           v
+-----------------------------------------------------------------------------------+
|                               APPLICATION TIER                                    |
|   Flask 3.1 WSGI Application (backend/app.py)                                     |
|   - Dynamic Mandi Hierarchy Lookup & Fast Feature Vector Synthesis                 |
|   - Single-Instance Model Registry (Zero Runtime Retraining)                     |
|   - Real-time Inference Dispatcher & Dynamic Revenue Calculator                   |
+---------------------+---------------------------------------+---------------------+
                      |                                       |
                      v                                       v
+-----------------------------------+   +-------------------------------------------+
|       CLASSICAL ML ENGINE         |   |             HYBRID QML ENGINE             |
|   - Random Forest Regressor       |   |   - Variational Quantum Regressor (VQR)   |
|   - XGBoost Regressor             |   |   - Quantum Neural Network (QNN)          |
|   - LightGBM Regressor            |   |   - Quantum Kernel SVR (QSVR)             |
|   (37 Tabular Feature Pipeline)   |   |   (8-Qubit Register | Angle Embedding)    |
+-----------------+-----------------+   +---------------------+---------------------+
                  |                                           |
                  +---------------------+---------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                                  DATA TIER                                        |
|   - Cleaned Monthly Observations (54,002 rows, 812 mandis, 24 states)             |
|   - Chronological Leakage-Free Partitions:                                        |
|     * train.csv (2014-2021 | 38,017 obs)                                          |
|     * validation.csv (2022-2023 | 10,876 obs)                                     |
|     * test.csv (2024 | 5,109 obs)                                                 |
|   - Saved Artifacts: Scalers, Encoders, Precomputed State Vectors, Lookup Indexes |
+-----------------------------------------------------------------------------------+
```

---

## 2. Directory Layout & Module Structure
```
capstone_project/
├── backend/
│   ├── app.py                      # Flask REST API server and static frontend host
│   ├── market_lookup.json          # Preprocessed lightweight mandi hierarchy & records
│   └── saved_models/
│       ├── random_forest.joblib    # Tuned Random Forest model
│       ├── xgboost.joblib          # Tuned XGBoost model
│       ├── lightgbm.joblib         # Tuned LightGBM model
│       ├── vqr_model.joblib        # Trained Variational Quantum Regressor state & weights
│       ├── qnn_model.joblib        # Trained Quantum Neural Network state & weights
│       ├── qsvr_model.joblib       # Trained Quantum Kernel SVR model & state vectors
│       ├── feature_scaler.joblib   # RobustScaler fitted strictly on training data
│       ├── angle_scaler.joblib     # MinMaxScaler [0, pi] fitted strictly on training data
│       ├── categorical_encoder.joblib # Frequency maps for market, district, state
│       ├── feature_config.json     # Feature name registry and definitions
│       ├── classical_meta.json     # Classical training times and parameter metadata
│       └── qml_meta.json           # QML training times, sample sizes, and qubit metadata
├── docs/
│   ├── architecture.md             # This document
│   ├── methodology.md              # Empirical data science and temporal integrity
│   ├── qml_methodology.md          # Quantum circuit mathematics, ansatz, and kernels
│   └── evaluation.md               # Benchmarking results and comparative metrics
├── frontend/
│   ├── index.html                  # Responsive modern web dashboard
│   ├── css/
│   │   └── style.css               # Agricultural emerald + Quantum deep indigo styling
│   └── js/
│       └── app.js                  # Dynamic client logic, API calls, Chart.js graphs
├── results/
│   ├── model_metrics.csv           # Final standardized test metrics (all 6 models)
│   ├── test_predictions.csv        # 30,654 test prediction rows (actual vs predicted)
│   ├── final_report.csv            # Executive summary report per model
│   ├── final_predictions.csv       # Complete out-of-time prediction records
│   ├── computational_comparison.csv# Wall-clock training and inference latency
│   ├── summary_findings.json       # Dynamic objective leaders (zero hardcoding)
│   ├── validation_tuning_classical.csv # Validation hyperparameter logs
│   ├── validation_tuning_qml.csv   # QML validation checkpoint logs
│   └── visualizations/             # 12 Publication-quality 300 DPI PNG figures
├── build_market_lookup.py          # Script to generate market_lookup.json & scalers
├── qml_models.py                   # Clean PennyLane + PyTorch QML class implementations
├── train_classical.py              # Classical ML training and validation tuning pipeline
├── train_qml.py                    # Hybrid QML training with sample size accounting
├── evaluate_models.py              # Test set evaluation and publication figure generator
├── requirements.txt                # Production dependency specifications
├── cleaned_onion_monthly.csv       # Master monthly dataset (54,002 rows)
├── train.csv                       # Training partition (2014-2021)
├── validation.csv                  # Validation partition (2022-2023)
├── test.csv                        # Out-of-time test partition (2024)
└── README.md                       # Comprehensive project guide and running instructions
```

---

## 3. Component Details

### A. Data Preprocessing & Leakage Elimination
- **Temporal Boundaries**: Training (2014–2021), Validation (2022–2023), and Test (2024).
- **Target Formulation**: Supervised forecasting of $P_{t+1}$ using information strictly available at or before month $t$.
- **Zero Forward Leakage**: All rolling means, lags ($t-1, t-3, t-6, t-12$), and frequency encodings are derived without future data contamination.

### B. Classical Model Pipeline
- **37 Numerical & Categorical Features**: Historical price percentiles, arrival momentum, monsoon rainfall, temperature indicators, and frequency encodings.
- **Hyperparameter Tuning**: Hyperparameters tuned via Grid Evaluation on `validation.csv` (2022–2023). Zero test data used for tuning.

### C. Quantum Machine Learning Pipeline
- **Simulator Backend**: PennyLane with `lightning.qubit` state-vector simulator and C++ acceleration.
- **Differentiable Quantum Layers**: Adjoint differentiation for fast backpropagation through quantum rotation gates.
- **Angle Encoding**: Top-8 quantum features mapped to $[0, \pi]$ rotation angles.
- **State-Vector Kernel**: Exact fidelity inner products $K(x_i, x_j) = |\langle\psi(x_i)|\psi(x_j)\rangle|^2$ for QSVR.

### D. Flask Backend API & Frontend Integration
- **Models Loaded Once at Startup**: Zero training on user request. Inference executes in milliseconds (10 ms for classical, ~100 ms for quantum single-sample).
- **Dynamic Lookup**: Translates user-selected `(state, district, market)` into historical time-series vector on-the-fly.
