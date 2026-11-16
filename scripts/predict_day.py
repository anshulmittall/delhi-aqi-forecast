"""
predict_day.py

Forecasts one day's AQI with the saved XGBoost model and explains the
forecast with SHAP. This is the "live" demo: one command, one day, one answer.

Run from the project root, after 05_train_eval.py:
    python scripts/predict_day.py 2019-11-03

The date is the day being PREDICTED (target_date). The model only sees the
features of the day before it, exactly as in training and testing.

Input:
  models/xgb_aqi.joblib, models/feature_names.json
  data/processed/delhi_features.csv
Output:
  printed only (forecast, its CPCB category, the persistence forecast, the
  actual AQI if known, and the features that pushed the forecast most)
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

from common import bucket_names

# ---- CONFIG ----
TOP_PUSHES = 5   # how many SHAP contributions to print
# ----------------

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

if len(sys.argv) != 2:
    sys.exit("Usage: python scripts/predict_day.py YYYY-MM-DD   (the day to forecast)")
target = pd.Timestamp(sys.argv[1])

# ---- 1. Load the saved model and its input columns, in training order ----
model = joblib.load(os.path.join(MODEL_DIR, "xgb_aqi.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    feature_cols = json.load(f)

# ---- 2. Find the row whose target is the requested day ----
table = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_features.csv"),
                    parse_dates=["date", "target_date"])
row = table[table["target_date"] == target]
if row.empty:
    first, last = table["target_date"].min().date(), table["target_date"].max().date()
    sys.exit(f"No row predicts {target.date()}. Choose a day from {first} to {last}.")
X = row[feature_cols]

# ---- 3. Forecast, and the "tomorrow = today" baseline for comparison ----
forecast = float(model.predict(X)[0])
today = row["aqi"].fillna(row["aqi_lag1"]).iloc[0]   # same rule as step 5
actual = row["aqi_tomorrow"].iloc[0]

print(f"Forecast for {target.date()} (using data up to {row['date'].iloc[0].date()})")
print(f"  XGBoost forecast : {forecast:.0f} ({bucket_names([forecast])[0]})")
print(f"  Persistence      : {today:.0f} (today's AQI)")
if pd.notna(actual):
    print(f"  Actual AQI       : {actual:.0f} ({bucket_names([actual])[0]}), "
          f"error {forecast - actual:+.0f}")

# ---- 4. Why: SHAP splits the forecast into one push per feature ----
explanation = shap.TreeExplainer(model)(X)
pushes = pd.Series(explanation.values[0], index=feature_cols)
base = float(explanation.base_values[0])
# Same check as step 6: average forecast + all pushes = this forecast.
assert np.isclose(base + pushes.sum(), forecast, atol=0.1)

print(f"\nAverage forecast {base:.1f}; biggest pushes (AQI points):")
for name in pushes.abs().sort_values(ascending=False).index[:TOP_PUSHES]:
    print(f"  {name:18s} = {X[name].iloc[0]:8.4g}   push {pushes[name]:+6.1f}")
