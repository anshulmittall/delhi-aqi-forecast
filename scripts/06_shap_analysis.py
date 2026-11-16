"""
06_shap_analysis.py

Explains the saved XGBoost model with SHAP: which features drive tomorrow's
forecast, in which direction, and why the model said what it said on one
particular day.

SHAP in one sentence: for every single prediction, it splits
"this forecast minus the model's average forecast" into one contribution per
feature, and those contributions add up exactly to the forecast. A SHAP value
is in AQI points: +40 means "this feature pushed tomorrow's forecast up by 40".

Run from the project root, after 05_train_eval.py:
    python scripts/06_shap_analysis.py

Input:
  models/xgb_aqi.joblib, models/feature_names.json
  data/processed/delhi_features.csv
Output:
  outputs/figures/shap_01_beeswarm.png        every test day, every top feature
  outputs/figures/shap_02_importance.png      average size of each feature's push
  outputs/figures/shap_03_dependence.png      how the push changes with the value
  outputs/figures/shap_04_waterfall_severe.png  one severe November day, explained
  outputs/figures/shap_05_waterfall_clean.png   one clean monsoon day, explained
  outputs/metrics/shap_importance.csv
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import (
    AXIS, BLUE, INK_SOFT, TEST_END, TRAIN_END, plot_waterfall, use_project_style,
)
from features import WEATHER_COLUMNS

# ---- CONFIG ----
TOP_N = 15   # features shown in the summary plots
# ----------------

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
METRICS_DIR = "outputs/metrics"
FIG_DIR = "outputs/figures"

use_project_style()


def save_current(filename):
    """Save whatever SHAP just drew (SHAP plots draw on the current figure)."""
    path = os.path.join(FIG_DIR, filename)
    plt.gcf().savefig(path)
    plt.close("all")
    print(f"  saved {path}")


# ---- 1. Load the model and the test days ----
model = joblib.load(os.path.join(MODEL_DIR, "xgb_aqi.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    feature_cols = json.load(f)

table = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_features.csv"),
                    parse_dates=["date", "target_date"])
test = table[(table["target_date"] > TRAIN_END) & (table["target_date"] <= TEST_END)]
test = test.reset_index(drop=True)
X_test = test[feature_cols]
print(f"Explaining {len(test)} test-day forecasts with {len(feature_cols)} features")

# ---- 2. SHAP values ----
# TreeExplainer computes exact SHAP values for tree models by walking the
# trees, rather than approximating them by sampling.
explainer = shap.TreeExplainer(model)
explanation = explainer(X_test)   # one row of SHAP values per test day

# The defining property: average forecast + all contributions = the forecast.
reconstructed = explanation.base_values + explanation.values.sum(axis=1)
assert np.allclose(reconstructed, model.predict(X_test), atol=0.1), \
    "SHAP values don't add up to the model's predictions"
print(f"Check: base value {explanation.base_values[0]:.1f} + SHAP values "
      f"= the model's forecast, on every test day")

# ---- 3. Rankings ----
# (XGBoost works in 32-bit floats; convert so printed numbers are clean)
mean_abs = pd.Series(np.abs(explanation.values).mean(axis=0).astype(float), index=feature_cols)

# XGBoost's own "gain" importance: how much a feature's splits reduced the
# training error. It's measured on training data and per split, whereas SHAP
# measures how much each feature actually moved the forecasts on test days.
gain = pd.Series(model.get_booster().get_score(importance_type="gain"))
gain = gain.reindex(feature_cols).fillna(0)   # features never split on get 0

ranking = pd.DataFrame({
    "mean_abs_shap": mean_abs.round(2),
    "shap_rank": mean_abs.rank(ascending=False).astype(int),
    "gain_rank": gain.rank(ascending=False, method="min").astype(int),
}).sort_values("shap_rank")
ranking.index.name = "feature"
ranking.to_csv(os.path.join(METRICS_DIR, "shap_importance.csv"))

print(f"\nTop {TOP_N} features by mean |SHAP| (AQI points), with XGBoost's gain rank:")
print(ranking.head(TOP_N).to_string())
unused = (gain == 0).sum()
print(f"\n{unused} of {len(feature_cols)} features were never used in any split.")

# ---- 4. Figures ----
print("\nMaking figures...")

# Figure 1: beeswarm. One dot per test day per feature. Horizontal position
# = SHAP value (push on the forecast); colour = the feature's own value that
# day (red high, blue low).
shap.plots.beeswarm(explanation, max_display=TOP_N, show=False)
plt.title(f"What drives the forecast: top {TOP_N} features (test period)", loc="left")
save_current("shap_01_beeswarm.png")

# Figure 2: average size of each feature's push, drawn directly so the bars
# match the rest of the report.
top = ranking.head(TOP_N)
fig, ax = plt.subplots(figsize=(8, 6))
positions = np.arange(len(top))
ax.barh(positions, top["mean_abs_shap"], height=0.6, color=BLUE)
for y, v in zip(positions, top["mean_abs_shap"]):
    ax.text(v + top["mean_abs_shap"].max() * 0.015, y, f"{v:.1f}",
            va="center", fontsize=9, color=INK_SOFT)
ax.set_yticks(positions, top.index)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(0, top["mean_abs_shap"].max() * 1.15)
ax.set_xlabel("Mean |SHAP value| (AQI points)")
ax.set_title("Average size of each feature's push on the forecast")
fig.savefig(os.path.join(FIG_DIR, "shap_02_importance.png"))
plt.close(fig)
print(f"  saved {FIG_DIR}/shap_02_importance.png")

# Figure 3: dependence. For the two strongest features and the strongest
# weather feature: feature value (x) against its SHAP value (y).
weather_like = [c for c in ranking.index if c in WEATHER_COLUMNS or c.startswith(("wind", "humidity"))]
panels = list(ranking.index[:2]) + [weather_like[0]]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, name in zip(axes, panels):
    column = feature_cols.index(name)
    ax.scatter(X_test[name], explanation.values[:, column], s=12, color=BLUE,
               alpha=0.6, linewidth=0)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(f"{name} (feature value)")
    ax.set_title(name, fontsize=11)
axes[0].set_ylabel("SHAP value (AQI points)")
fig.suptitle("How each feature's push depends on its value (test period)", x=0.01, y=1.04, ha="left")
fig.savefig(os.path.join(FIG_DIR, "shap_03_dependence.png"))
plt.close(fig)
print(f"  saved {FIG_DIR}/shap_03_dependence.png")

# Figures 4 and 5: waterfalls for two contrasting days. A waterfall starts at
# the average forecast and adds one feature's push at a time until it reaches
# that day's forecast. (Drawn by plot_waterfall in common.py rather than
# shap.plots.waterfall, whose value labels overlap the feature names.)
november = test[test["target_date"].dt.month == 11]
monsoon = test[test["target_date"].dt.month.isin([7, 8, 9])]
chosen = [
    (november["aqi_tomorrow"].idxmax(), "severe", "worst November day"),
    (monsoon["aqi_tomorrow"].idxmin(), "clean", "cleanest monsoon day"),
]
for row, tag, description in chosen:
    day = test.loc[row]
    labels = [f"{name} = {value:.4g}" for name, value in zip(feature_cols, X_test.loc[row])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    forecast = plot_waterfall(ax, explanation.base_values[row], explanation.values[row], labels)
    ax.set_title(f"Forecast for {day['target_date'].date()} ({description}): "
                 f"predicted {forecast:.0f}, actual {day['aqi_tomorrow']:.0f}")
    filename = f"shap_0{4 if tag == 'severe' else 5}_waterfall_{tag}.png"
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  saved {FIG_DIR}/{filename}  ({description}: {day['target_date'].date()}, "
          f"predicted {forecast:.0f}, actual {day['aqi_tomorrow']:.0f})")
