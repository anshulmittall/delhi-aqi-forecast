"""
09_existing_projects_comparison.py

Most existing AQI projects on this Kaggle dataset predict TODAY's AQI from
TODAY's pollutants, with a random train/test split. This script runs that
approach on our Delhi data, so it can be compared fairly with our next-day
model:

  A. their method as published: same-day AQI from same-day PM2.5, PM10, NO2
     and CO, random 80/20 split (pre-COVID days)
  B. the same kind of model used as a real forecast: trained on our training
     days, then given today's pollutants to predict tomorrow, scored on our
     265 test days
  C. our persistence baseline and our XGBoost, on the same 265 test days

The XGBoost settings are the ones chosen for our own model in step 5, so the
only difference is what the model is asked to predict.

Run from the project root, after 05_train_eval.py:
    python scripts/09_existing_projects_comparison.py

Input:
  data/processed/delhi_daily.csv, data/processed/delhi_features.csv
  outputs/metrics/model_comparison.csv
Output:
  outputs/metrics/existing_projects_comparison.csv
"""

import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from common import TEST_END, TRAIN_END

# ---- CONFIG ----
# The pollutants used by the typical existing project
SAME_DAY_INPUTS = ["pm25", "pm10", "no2", "co"]
# Our tuned XGBoost settings (step 5), reused so the comparison is fair
XGB_SETTINGS = dict(max_depth=3, learning_rate=0.03, n_estimators=300,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=42)
# ----------------

PROCESSED_DIR = "data/processed"
METRICS_DIR = "outputs/metrics"

daily = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_daily.csv"), parse_dates=["date"])
table = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_features.csv"),
                    parse_dates=["date", "target_date"])
rows = []

# ---- A. Their method as published: today's pollutants -> today's AQI ----
# Only pre-COVID days, and only days that have an AQI to predict.
pre_covid = daily[daily["date"] <= TEST_END].dropna(subset=["aqi"])
X_train, X_test, y_train, y_test = train_test_split(
    pre_covid[SAME_DAY_INPUTS], pre_covid["aqi"], test_size=0.2, random_state=42)
same_day = XGBRegressor(**XGB_SETTINGS).fit(X_train, y_train).predict(X_test)
rows.append({
    "approach": "Existing method: same-day AQI from same-day pollutants, random 80/20 split",
    "test_days": len(y_test),
    "MAE": mean_absolute_error(y_test, same_day),
    "R2": r2_score(y_test, same_day),
})

# ---- B. The same kind of model used to forecast tomorrow ----
# Trained on our training days only, then given today's pollutants and asked
# for tomorrow's AQI, which is what a forecast actually has to do.
train_days = pre_covid[pre_covid["date"] <= TRAIN_END]
model = XGBRegressor(**XGB_SETTINGS).fit(train_days[SAME_DAY_INPUTS], train_days["aqi"])
test = table[(table["target_date"] > TRAIN_END) & (table["target_date"] <= TEST_END)]
as_forecast = model.predict(test[SAME_DAY_INPUTS])
rows.append({
    "approach": "Existing method used as a next-day forecast",
    "test_days": len(test),
    "MAE": mean_absolute_error(test["aqi_tomorrow"], as_forecast),
    "R2": r2_score(test["aqi_tomorrow"], as_forecast),
})

# ---- C. Our baseline and our model on the same test days (from step 5) ----
ours = pd.read_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index_col=0)
for name in ["Persistence (tomorrow = today)", "XGBoost"]:
    rows.append({"approach": f"This project: {name}", "test_days": len(test),
                 "MAE": ours.loc[name, "MAE"], "R2": ours.loc[name, "R2"]})

result = pd.DataFrame(rows).round(2)
# How much lower our error is than the existing method used as a forecast
gain = (result.loc[1, "MAE"] - result.loc[3, "MAE"]) / result.loc[1, "MAE"] * 100
result["note"] = ["", "", "", f"MAE {gain:.1f}% lower than the existing method as a forecast"]
result.to_csv(os.path.join(METRICS_DIR, "existing_projects_comparison.csv"), index=False)
print(result.to_string(index=False))
