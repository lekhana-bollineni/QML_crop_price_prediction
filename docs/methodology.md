# Empirical Research Methodology & Temporal Validation

## 1. Problem Definition & Formulation
Wholesale vegetable markets in India (mandis) exhibit extreme price volatility driven by seasonal harvest cycles, climatic shocks (monsoon anomalies, heatwaves), perishable storage limits, and localized transportation disruptions.

### Target Formulation ($P_{t+1}$)
The objective is to forecast the **next-month wholesale onion modal price** ($P_{t+1}$ in Rs/quintal) for any given market node $m$:

$$\hat{P}_{m, t+1} = f(\mathbf{x}_{m, t})$$

where $\mathbf{x}_{m, t}$ represents the multidimensional information state observed at or prior to the final trading day of month $t$.

---

## 2. End-to-End Data Pipeline Flow

```
+-------------------------------------------------------------------------------+
| Raw Daily Agmarknet & Weather Records: 965,314 rows across 56 columns (2014-2024)|
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| Data Cleaning & Anomaly Correction: 964,570 clean daily rows                  |
| - Stripped zero-price entries and reporting scale typos (> Rs 25,000/qtl)     |
| - Resolved mandi geographic aliases and coordinate coordinates                |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| Monthly Aggregation & Alignment: 54,002 continuous monthly observations       |
| - Calculated intra-month modal price statistics (mean, median, min, max, std) |
| - Summed monthly cumulative mandi arrivals & cumulative rainfall              |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| Feature Engineering (37 Tabular / 8 Quantum Features):                        |
| - Autoregressive Price Memory: Lag 1, Lag 2, Lag 3, Lag 6, Lag 12             |
| - Rolling Smoothed Price Regimes: 3-month, 6-month, 12-month rolling means   |
| - Supply Arrival Dynamics: Current total, daily average, lag-1, 3m rolling    |
| - Meteorological Stressors: Rainfall sum, temp avg, thermal max/min           |
| - Cyclical Annual Calendars: sin(2πm/12), cos(2πm/12), calendar quarter       |
| - Geographical Frequency Encoding: Mandi, district, and state frequency share |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| Strict Chronological Partitioning (Zero Lookahead Bias):                     |
| 1. Training Set:   2014 - 2021 (8 Years | 38,017 Observations)                |
| 2. Validation Set: 2022 - 2023 (2 Years | 10,876 Observations)                |
| 3. Test Set:       2024        (1 Year  |  5,109 Observations)                |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| Model Training & Hyperparameter Tuning on Validation Partition                |
| - Random Forest, XGBoost, LightGBM tuned strictly on 2022-2023 validation     |
| - QML models trained on controlled budget (5,000 samples) via adjoint sim     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| Out-of-Time 2024 Test Benchmark & Dynamic Dashboard Deployment                |
| - Standardized evaluation on 5,109 untouched test observations                 |
| - Measured MAE, RMSE, R², safe-epsilon MAPE, and inference latency            |
+-------------------------------------------------------------------------------+
```

---

## 3. Strict Absence of Data Leakage
1. **No Target Leakage**: Current-month statistics (`price_current_month`, `price_max_current_month`, etc.) are computed exclusively from days $[1, D_t]$ of month $t$. The forecast target is month $t+1$.
2. **Scaler Integrity**: All numerical transformers (`RobustScaler`) and quantum angle normalizers (`MinMaxScaler(0, pi)`) were fitted **exclusively on the training split** (2014–2021). The validation and test splits were transformed using frozen training parameters.
3. **Categorical Encoders**: Frequency maps for high-cardinality mandi, district, and state identifiers were derived strictly from the training partition. Unseen entities default to zero.
4. **Zero Test Data Tuning**: The 2024 test partition was kept completely untouched during feature selection and hyperparameter optimization.
