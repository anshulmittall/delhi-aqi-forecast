"""
07_covid_case_study.py

Scores the frozen XGBoost model on the COVID lockdown period. The model never
saw these days, and it was trained on a Delhi with normal traffic and industry,
so this is a test of how it copes when the world behind the data changes
(a "regime shift" or "distribution shift").

Nothing is retrained or tuned here: the model from step 5 is used as-is.

Run from the project root, after 05_train_eval.py:
    python scripts/07_covid_case_study.py

Input:
  models/xgb_aqi.joblib, models/feature_names.json
  data/processed/delhi_features.csv
  data/processed/delhi_daily.csv
Output:
  outputs/metrics/covid_regime.csv          test vs lockdown scores
  outputs/metrics/lockdown_window_by_year.csv
  outputs/figures/covid_01_forecast.png
  outputs/figures/covid_02_window_by_year.png
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from common import (
    BLUE, INK_SOFT, LOCKDOWN_START, ORANGE, TEST_END, TRAIN_END,
    label_aqi_bands, score, use_project_style,
)

# ---- CONFIG ----
PLOT_FROM = "2020-01-01"   # figure 1 starts here, to show the run-up to lockdown
# ----------------

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
METRICS_DIR = "outputs/metrics"
FIG_DIR = "outputs/figures"

use_project_style()

# ---- 1. Load the frozen model and both held-out periods ----
model = joblib.load(os.path.join(MODEL_DIR, "xgb_aqi.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    feature_cols = json.load(f)

table = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_features.csv"),
                    parse_dates=["date", "target_date"])
periods = {
    "Test (pre-COVID)": table[(table["target_date"] > TRAIN_END) & (table["target_date"] <= TEST_END)],
    "COVID lockdown": table[table["target_date"] >= LOCKDOWN_START],
}

# ---- 2. Score XGBoost and persistence on each period ----
rows = []
forecasts = {}
for period, part in periods.items():
    today = part["aqi"].fillna(part["aqi_lag1"])
    forecast = pd.Series(model.predict(part[feature_cols]), index=part.index)
    forecasts[period] = forecast
    for name, pred in [("XGBoost", forecast), ("Persistence (tomorrow = today)", today)]:
        rows.append({
            "period": period,
            "model": name,
            "days": len(part),
            "mean_actual_aqi": part["aqi_tomorrow"].mean(),
            **score(part["aqi_tomorrow"], pred, today),
        })
results = pd.DataFrame(rows).round(2)
results.to_csv(os.path.join(METRICS_DIR, "covid_regime.csv"), index=False)

print("Frozen model, test period vs COVID lockdown:")
print(results[["period", "model", "days", "mean_actual_aqi", "MAE", "MAPE_pct",
               "bias", "R2", "direction_accuracy_pct"]].to_string(index=False))

xgb = results[results["model"] == "XGBoost"].set_index("period")
pers = results[results["model"] != "XGBoost"].set_index("period")
for period in periods:
    gain = (pers.loc[period, "MAE"] - xgb.loc[period, "MAE"]) / pers.loc[period, "MAE"]
    print(f"  {period}: XGBoost beats persistence on MAE by {gain:.1%}")

# ---- 3. How different was the lockdown? Same calendar window, every year ----
# Compare 22 March - 1 July of each year (the dates the lockdown rows predict),
# so the comparison isn't distorted by the seasons.
daily = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_daily.csv"),
                    parse_dates=["date"]).set_index("date")
covid_targets = periods["COVID lockdown"]["target_date"]
first_md = covid_targets.min().strftime("%m-%d")
last_md = covid_targets.max().strftime("%m-%d")
window_label = f"{covid_targets.min():%-d %b} to {covid_targets.max():%-d %b}"  # e.g. "22 Mar to 1 Jul"
month_day = daily.index.strftime("%m-%d")   # "MM-DD" strings sort like dates
window = daily[(month_day >= first_md) & (month_day <= last_md)]
by_year = window.groupby(window.index.year)["aqi"].agg(["mean", "median"]).round(1)
by_year.index.name = "year"
by_year.to_csv(os.path.join(METRICS_DIR, "lockdown_window_by_year.csv"))

earlier = by_year.loc[by_year.index < 2020, "mean"]
drop = 1 - by_year.loc[2020, "mean"] / earlier.mean()
print(f"\nMean AQI, {window_label}, by year:")
print(by_year.to_string())
print(f"2020 is {drop:.0%} below the 2015-2019 average for the same dates "
      f"(lowest earlier year: {earlier.min():.0f})")

# ---- 4. Figures ----
print("\nMaking figures...")

# Figure 1: forecast vs actual through the switch into lockdown, with errors
recent = table[table["target_date"] >= PLOT_FROM]
recent_forecast = pd.Series(model.predict(recent[feature_cols]), index=recent.index)
dates = recent["target_date"]
lockdown = pd.Timestamp(LOCKDOWN_START)

fig, (ax_top, ax_err) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
for ax in (ax_top, ax_err):
    ax.axvspan(lockdown, dates.max(), color=ORANGE, alpha=0.12, linewidth=0)
ax_top.plot(dates, recent["aqi_tomorrow"], color=BLUE, linewidth=1.5, label="Actual AQI")
ax_top.plot(dates, recent_forecast, color=ORANGE, linewidth=1.5, label="XGBoost forecast")
ax_top.text(lockdown + (dates.max() - lockdown) / 2, 0.97, "COVID lockdown",
            transform=ax_top.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=INK_SOFT)
ax_top.set_ylim(0, 550)
ax_top.set_ylabel("AQI")
ax_top.legend(loc="upper left", bbox_to_anchor=(0.18, 1))
ax_top.set_title(
    f"Frozen model through the lockdown: MAE {xgb.loc['Test (pre-COVID)', 'MAE']:.1f} before, "
    f"{xgb.loc['COVID lockdown', 'MAE']:.1f} during")
label_aqi_bands(ax_top)

error = recent_forecast - recent["aqi_tomorrow"]
ax_err.vlines(dates, 0, error, color=BLUE, linewidth=1.2)
ax_err.axhline(0, color=INK_SOFT, linewidth=1)
ax_err.set_ylabel("Forecast − actual")
ax_err.set_xlim(dates.min(), dates.max())
fig.savefig(os.path.join(FIG_DIR, "covid_01_forecast.png"))
plt.close(fig)
print(f"  saved {FIG_DIR}/covid_01_forecast.png")

# Figure 2: the same calendar window across years
fig, ax = plt.subplots(figsize=(8, 4.2))
years = by_year.index.tolist()
colors = [ORANGE if year == 2020 else BLUE for year in years]
ax.bar(range(len(years)), by_year["mean"], width=0.6, color=colors)
for x, value in enumerate(by_year["mean"]):
    ax.text(x, value + 4, f"{value:.0f}", ha="center", fontsize=9, color=INK_SOFT)
ax.set_xticks(range(len(years)), [f"{y}\n(lockdown)" if y == 2020 else str(y) for y in years])
ax.grid(axis="x", visible=False)
ax.set_ylim(0, by_year["mean"].max() * 1.15)
ax.set_ylabel("Mean AQI")
ax.set_title(f"Mean AQI, {window_label}: the same dates each year")
fig.savefig(os.path.join(FIG_DIR, "covid_02_window_by_year.png"))
plt.close(fig)
print(f"  saved {FIG_DIR}/covid_02_window_by_year.png")
