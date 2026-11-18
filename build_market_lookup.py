"""
Script to build backend/market_lookup.json and fit/save feature scalers and categorical encoders.
"""
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, MinMaxScaler

def build_lookup_and_scalers():
    print("Loading cleaned dataset and train split...")
    data_path = "cleaned_onion_monthly.csv"
    train_path = "train.csv"
    
    df_clean = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    
    # 1. Categorical Frequency Encoders based strictly on training split
    cat_cols = ['market', 'district', 'state', 'variety', 'crop_season_onion']
    categorical_encoders = {}
    for col in cat_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        categorical_encoders[col] = freq_map
        
    os.makedirs("backend/saved_models", exist_ok=True)
    joblib.dump(categorical_encoders, "backend/saved_models/categorical_encoder.joblib")
    print("Saved categorical_encoder.joblib")
    
    # 2. Fit Feature Scalers strictly on training split
    feature_cols = [
        'price_current_month', 'price_median_current_month', 'price_min_current_month',
        'price_max_current_month', 'price_std_current_month', 'days_reported',
        'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_6', 'price_lag_12',
        'price_rolling_3m_mean', 'price_rolling_6m_mean', 'price_rolling_12m_mean',
        'price_rolling_3m_std', 'price_change_ratio_1m',
        'arrivals_total_current_month', 'arrivals_daily_avg_current_month',
        'arrivals_lag_1', 'arrivals_rolling_3m_mean', 'arrivals_change_ratio_1m',
        'rainfall_total_current_month', 'rainfall_lag_1', 'rainfall_rolling_3m_sum',
        'temp_avg_current_month', 'temp_avg_lag_1', 'temp_rolling_3m_mean',
        'tmax_current_month', 'tmin_current_month',
        'quarter', 'sin_month', 'cos_month',
        'market_encoded', 'district_encoded', 'state_encoded', 'variety_encoded', 'crop_season_onion_encoded'
    ]
    
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
    
    robust_scaler = RobustScaler()
    robust_scaler.fit(train_df[feature_cols])
    joblib.dump(robust_scaler, "backend/saved_models/feature_scaler.joblib")
    print("Saved feature_scaler.joblib")
    
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    angle_scaler.fit(train_df[feature_cols])
    joblib.dump(angle_scaler, "backend/saved_models/angle_scaler.joblib")
    print("Saved angle_scaler.joblib")
    
    feature_config = {
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'top_8_quantum_features': top_8_quantum_features,
        'target_col': 'target_price_next_month'
    }
    with open("backend/saved_models/feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)
    print("Saved feature_config.json")
    
    # 3. Build Market Hierarchy & Historical Trends
    print("Building market lookup hierarchy...")
    states = sorted(df_clean['state'].dropna().unique().tolist())
    
    districts_by_state = {}
    markets_by_district = {}
    
    for state in states:
        state_df = df_clean[df_clean['state'] == state]
        districts = sorted(state_df['district'].dropna().unique().tolist())
        districts_by_state[state] = districts
        markets_by_district[state] = {}
        for d in districts:
            m_list = sorted(state_df[state_df['district'] == d]['market'].dropna().unique().tolist())
            markets_by_district[state][d] = m_list
            
    # Latest observation per market
    # Sort chronologically by year and month
    df_sorted = df_clean.sort_values(by=['year', 'month']).copy()
    
    # Latest features map: key = f"{state}|{district}|{market}"
    latest_market_records = {}
    historical_price_series = {}
    
    grouped = df_sorted.groupby(['state', 'district', 'market'])
    
    for (st, dst, mkt), group in grouped:
        key = f"{st}|{dst}|{mkt}"
        last_row = group.iloc[-1]
        
        # Save features needed to construct next-month vector
        rec = {}
        for c in feature_cols:
            if c in last_row:
                rec[c] = float(last_row[c]) if pd.notnull(last_row[c]) else 0.0
        rec['year'] = int(last_row['year'])
        rec['month'] = int(last_row['month'])
        rec['variety'] = str(last_row.get('variety', 'Other'))
        rec['crop_season_onion'] = str(last_row.get('crop_season_onion', 'Rabi'))
        rec['lat'] = float(last_row.get('lat', 20.0))
        rec['lon'] = float(last_row.get('lon', 78.0))
        rec['latest_price'] = float(last_row['price_current_month'])
        
        latest_market_records[key] = rec
        
        # Historical series (recent up to 36 months for lightweight payload)
        recent_pts = group.tail(36)
        series = []
        for _, r in recent_pts.iterrows():
            series.append({
                "date": f"{int(r['year'])}-{int(r['month']):02d}",
                "price": round(float(r['price_current_month']), 2),
                "arrivals": round(float(r.get('arrivals_total_current_month', 0.0)), 1)
            })
        historical_price_series[key] = series
        
    lookup_data = {
        "states": states,
        "districts_by_state": districts_by_state,
        "markets_by_district": markets_by_district,
        "latest_records": latest_market_records,
        "historical_series": historical_price_series
    }
    
    out_file = "backend/market_lookup.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lookup_data, f)
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"Generated {out_file} successfully! File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    build_lookup_and_scalers()
