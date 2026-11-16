"""
08_report_figures.py

Makes the two extra figures used in the report and works out every number
the report quotes that is not already in outputs/metrics/. It trains nothing:
it only reads files written by steps 02, 04 and 05.

Run from the project root, after 05_train_eval.py:
    python scripts/08_report_figures.py

Input:
  data/processed/delhi_daily.csv, data/processed/delhi_features.csv
  models/xgb_aqi.joblib, models/feature_names.json
Output:
  outputs/figures/report_01_aqi_by_year.png   average AQI per calendar year
  outputs/figures/report_02_diwali.png        AQI in the two weeks around Diwali
  reports/report_numbers.csv                  one row per number quoted
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import TimeSeriesSplit

from common import (
    AXIS, BLUE, BLUE_DARK, BLUE_WASH, INK_SOFT, ORANGE, TEST_END, TRAIN_END,
    aqi_to_bucket, label_aqi_bands, use_project_style,
)

# ---- CONFIG ----
# Diwali dates, 2015-2019 (sources in the report's reference list)
DIWALI = ["2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27"]
WINDOW = 7          # days shown before and after Diwali
RAIN_MM = 1.0       # a "rainy" day has more than this much rain
CV_FOLDS = 5        # must match 05_train_eval.py
# ----------------

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
FIG_DIR = "outputs/figures"
NUMBERS_DIR = "reports"
os.makedirs(NUMBERS_DIR, exist_ok=True)

use_project_style()
numbers = []   # (topic, name, value) rows for report_numbers.csv


def note(topic, name, value):
    numbers.append({"topic": topic, "name": name, "value": value})
    print(f"  {topic:14s} {name:40s} {value}")


daily = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_daily.csv"),
                    parse_dates=["date"]).set_index("date")
aqi = daily["aqi"]

# ---- 1. Average AQI per year (Figure 8 in the report) ----
yearly = aqi.groupby(aqi.index.year).mean()
for year, value in yearly.items():
    note("yearly_mean", str(year), round(value, 1))

fig, ax = plt.subplots(figsize=(7, 4))
labels = [str(y) if y < 2020 else "2020\n(Jan-Jul 1)" for y in yearly.index]
colours = [BLUE if y < 2020 else AXIS for y in yearly.index]  # 2020 is not comparable
bars = ax.bar(labels, yearly.values, width=0.6, color=colours)
ax.bar_label(bars, fmt="%.0f", fontsize=9, color=INK_SOFT)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Mean daily AQI")
ax.set_title("Average AQI per year (2020 is half a year, incl. lockdown)")
fig.savefig(os.path.join(FIG_DIR, "report_01_aqi_by_year.png"))
plt.close(fig)

# ---- 2. Diwali: AQI from 7 days before to 7 days after (Figure 11) ----
fig, ax = plt.subplots(figsize=(8.5, 4.4))
shades = [BLUE_WASH, "#9cc5f3", "#5c9ce6", BLUE, BLUE_DARK]   # older years lighter
for day, colour in zip(DIWALI, shades):
    diwali = pd.Timestamp(day)
    window = aqi.loc[diwali - pd.Timedelta(days=WINDOW): diwali + pd.Timedelta(days=WINDOW)]
    offsets = (window.index - diwali).days
    ax.plot(offsets, window.values, marker="o", markersize=3, linewidth=2,
            color=colour, label=f"{diwali.year} ({diwali:%d %b})")
    # "before" = mean of the 7 days before Diwali; "after" = the day after
    before = aqi.loc[diwali - pd.Timedelta(days=WINDOW): diwali - pd.Timedelta(days=1)].mean()
    after = aqi.loc[diwali + pd.Timedelta(days=1)]
    note("diwali", f"{diwali.year} mean of 7 days before", round(before, 0))
    note("diwali", f"{diwali.year} day after", round(after, 0))
ax.axvline(0, color=ORANGE, linewidth=1.5)
ax.text(0.2, 760, "Diwali", color=ORANGE, fontsize=10)
ax.set_xlim(-WINDOW, WINDOW)
ax.set_ylim(0, 800)
label_aqi_bands(ax)
ax.set_xlabel("Days from Diwali")
ax.set_ylabel("AQI")
ax.set_title("AQI in the two weeks around Diwali, each year")
ax.legend(loc="upper left", fontsize=9, frameon=False)
fig.savefig(os.path.join(FIG_DIR, "report_02_diwali.png"))
plt.close(fig)

# ---- 3. Training-period facts: categories and autocorrelation ----
train_aqi = aqi[aqi.index <= TRAIN_END].dropna()
note("training", "days with an AQI", len(train_aqi))
for name, count in aqi_to_bucket(train_aqi).value_counts().items():
    note("category", f"{name} days (share %)", f"{count} ({count / len(train_aqi):.1%})")
note("training", "lag-1 autocorrelation of AQI", round(aqi[aqi.index <= TRAIN_END].autocorr(1), 2))

# ---- 4. The five validation folds used for tuning (Table 6) ----
table = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_features.csv"),
                    parse_dates=["date", "target_date"])
train = table[table["target_date"] <= TRAIN_END].reset_index(drop=True)
for fold, (fit, val) in enumerate(TimeSeriesSplit(n_splits=CV_FOLDS).split(train), start=1):
    note("cv_fold", f"fold {fold} fitted on",
         f"{len(fit)} rows, up to {train['target_date'][fit[-1]].date()}")
    note("cv_fold", f"fold {fold} validated on",
         f"{train['target_date'][val[0]].date()} to {train['target_date'][val[-1]].date()} "
         f"({len(val)} rows)")

# ---- 5. Rain's SHAP push on rainy vs dry test days ----
model = joblib.load(os.path.join(MODEL_DIR, "xgb_aqi.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    feature_cols = json.load(f)
test = table[(table["target_date"] > TRAIN_END) & (table["target_date"] <= TEST_END)]
rain_push = shap.TreeExplainer(model)(test[feature_cols]).values[:, feature_cols.index("precip_sum")]
rainy = (test["precip_sum"] > RAIN_MM).to_numpy()
dry = (test["precip_sum"] == 0).to_numpy()
note("rain_shap", f"rainy days (> {RAIN_MM} mm)", int(rainy.sum()))
note("rain_shap", "mean push on rainy days", round(float(rain_push[rainy].mean()), 1))
note("rain_shap", "dry days (0 mm)", int(dry.sum()))
note("rain_shap", "mean push on dry days", round(float(rain_push[dry].mean()), 1))

pd.DataFrame(numbers).to_csv(os.path.join(NUMBERS_DIR, "report_numbers.csv"), index=False)
print(f"\nsaved {FIG_DIR}/report_01_aqi_by_year.png, {FIG_DIR}/report_02_diwali.png, "
      f"{NUMBERS_DIR}/report_numbers.csv")
