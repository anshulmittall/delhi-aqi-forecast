"""
05_train_eval.py

Trains five ways of predicting tomorrow's AQI, from a naive baseline up to
XGBoost, and scores them all on the same unseen test period.

Run from the project root, after 04_build_features.py:
    python scripts/05_train_eval.py

Input:
  data/processed/delhi_features.csv
  data/processed/delhi_daily.csv          (only for the leakage demo)
Output:
  models/xgb_aqi.joblib                   the tuned XGBoost model
  models/feature_names.json               its input columns, in order
  outputs/metrics/model_comparison.csv    every model's test scores
  outputs/metrics/test_predictions.csv    every model's prediction, per test day
  outputs/metrics/leakage_demo.csv        honest vs deliberately leaky model
  outputs/figures/model_01_... to model_05_...

The split is by target_date, the day being predicted:
  train   up to TRAIN_END
  test    the day after TRAIN_END up to TEST_END
  COVID   LOCKDOWN_START onwards; not touched here (see 07_covid_case_study.py)

Hyperparameters are chosen using the training rows only, with
TimeSeriesSplit: every validation fold comes strictly after the rows the model
was fitted on, the way a forecast is used in real life. The test set is used
once, at the very end, to score models that are already fixed.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import (
    AQI_BUCKET_LABELS, AXIS, BLUE, INK_SOFT, ORANGE, SURFACE, TEST_END,
    TRAIN_END, bucket_names, label_aqi_bands, score, use_project_style,
)

# ---- CONFIG ----
RANDOM_SEED = 42      # fixed so every run gives the same models and numbers
CV_FOLDS = 5

# Small grids on purpose: with ~1,600 training rows, a big search would mostly
# find settings that happen to suit these particular folds.
RIDGE_GRID = {"model__alpha": [0.1, 1, 10, 100, 1000]}
FOREST_GRID = {
    "model__min_samples_leaf": [1, 5],
    "model__max_features": [0.33, 1.0],   # share of features tried at each split
}
XGB_GRID = {
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [300, 600],
    "min_child_weight": [1, 5],
}

# Train one extra model that cheats (sees tomorrow's PM2.5), to show what
# leakage does to the scores. It is reported and then thrown away.
RUN_LEAKAGE_DEMO = True
# ----------------

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
METRICS_DIR = "outputs/metrics"
FIG_DIR = "outputs/figures"
for folder in (MODEL_DIR, METRICS_DIR, FIG_DIR):
    os.makedirs(folder, exist_ok=True)

use_project_style()


def tune(name, model, grid, X, y):
    """Choose hyperparameters by time-ordered cross-validation on training rows."""
    search = GridSearchCV(
        model, grid,
        cv=TimeSeriesSplit(n_splits=CV_FOLDS),   # never shuffled, never KFold
        scoring="neg_mean_absolute_error",
    )
    search.fit(X, y)  # afterwards, refits the best settings on ALL training rows
    cv_mae = -search.best_score_
    # MAE on each validation fold for the winning settings, oldest fold first
    best = search.best_index_
    per_fold = [-search.cv_results_[f"split{i}_test_score"][best] for i in range(CV_FOLDS)]
    print(f"  {name:14s} CV MAE {cv_mae:5.1f}   per fold {np.round(per_fold, 1)}")
    print(f"  {'':14s} best {search.best_params_}")
    return search.best_estimator_, cv_mae


def save(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---- 1. Load and split by the date being predicted ----
table = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_features.csv"),
                    parse_dates=["date", "target_date"])
feature_cols = [c for c in table.columns if c not in ("date", "target_date", "aqi_tomorrow")]

train = table[table["target_date"] <= TRAIN_END]
test = table[(table["target_date"] > TRAIN_END) & (table["target_date"] <= TEST_END)]

# No day may be both learned from and scored on.
assert train["target_date"].max() < test["target_date"].min(), "train and test overlap"

X_train, y_train = train[feature_cols], train["aqi_tomorrow"]
X_test, y_test = test[feature_cols], test["aqi_tomorrow"]
print(f"Train: {len(train)} rows, predicting {train['target_date'].min().date()} "
      f"to {train['target_date'].max().date()}")
print(f"Test:  {len(test)} rows, predicting {test['target_date'].min().date()} "
      f"to {test['target_date'].max().date()}")
print(f"Features: {len(feature_cols)}")

# Today's AQI, the reference point for the persistence baseline and for
# "did it go up or down". If today's reading is missing, use yesterday's.
today_aqi = test["aqi"].fillna(test["aqi_lag1"])

# ---- 2. Baselines: no learning at all ----
predictions = {
    "Persistence (tomorrow = today)": today_aqi,
    "7-day average": test["aqi_roll7_mean"].fillna(today_aqi),
}

# ---- 3. Learned models, each tuned the same way ----
print(f"\nTuning on training rows ({CV_FOLDS} time-ordered folds):")
cv_scores = {}

# Ridge: linear regression with a penalty that keeps the coefficients small.
# It can't handle missing values and needs features on a common scale, so
# those two steps go INSIDE the pipeline. That way the medians and scaling
# are learned from the training folds only, never from validation or test.
ridge = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", Ridge()),
])
ridge, cv_scores["Ridge regression"] = tune("Ridge", ridge, RIDGE_GRID, X_train, y_train)

# Random forest: many decision trees on random subsets, averaged. Trees don't
# care about scale, but scikit-learn's still needs the gaps filled.
forest = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)),
])
forest, cv_scores["Random forest"] = tune("Random forest", forest, FOREST_GRID, X_train, y_train)

# XGBoost: trees built one after another, each correcting the errors of the
# ones before. It handles missing values itself: at every split it learns
# which side missing values should go to.
xgb = XGBRegressor(
    subsample=0.8,           # each tree sees a random 80% of the rows...
    colsample_bytree=0.8,    # ...and 80% of the features, which curbs overfitting
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
xgb, cv_scores["XGBoost"] = tune("XGBoost", xgb, XGB_GRID, X_train, y_train)

for name, model in [("Ridge regression", ridge), ("Random forest", forest), ("XGBoost", xgb)]:
    predictions[name] = pd.Series(model.predict(X_test), index=test.index)

# ---- 4. Score everything on the test period ----
results = pd.DataFrame(
    {name: score(y_test, pred, today_aqi) for name, pred in predictions.items()}
).T
results.insert(0, "cv_MAE", pd.Series(cv_scores))
results.index.name = "model"
results = results.round(2)

print("\nTest-period scores (lower MAE/RMSE/MAPE is better; higher R2/accuracy is better):")
print(results.to_string())
results.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"))

persistence_mae = results.loc["Persistence (tomorrow = today)", "MAE"]
xgb_mae = results.loc["XGBoost", "MAE"]
print(f"\nXGBoost vs persistence: MAE {xgb_mae:.1f} vs {persistence_mae:.1f} "
      f"({(persistence_mae - xgb_mae) / persistence_mae:+.1%} improvement)")

pred_table = test[["date", "target_date"]].assign(actual=y_test, today=today_aqi)
for name, pred in predictions.items():
    pred_table[name] = pred.round(1)
pred_table.to_csv(os.path.join(METRICS_DIR, "test_predictions.csv"), index=False)

# ---- 5. Save the headline model ----
# XGBoost is the headline model because it is the model this project set out
# to explain with SHAP, decided before any test score existed. It is NOT
# picked for its test score: choosing whichever model scores best on the test
# set would make that score optimistic.
#
# Note that cross-validation actually favours Ridge. The early folds train on
# a few hundred rows and are then validated on smog worse than anything in
# their training data. Trees can't predict outside the range of values they
# were trained on, while a linear model can extend its line. With more data
# (the later folds, and the full training set) the trees catch up.
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_aqi.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

# ---- 6. Leakage demo ----
if RUN_LEAKAGE_DEMO:
    # DELIBERATE CHEAT, for demonstration only: give the model tomorrow's
    # PM2.5. PM2.5 is the main ingredient of tomorrow's AQI, so this is like
    # handing over the answer key. Same hyperparameters, same split.
    daily_pm25 = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_daily.csv"),
                             parse_dates=["date"]).set_index("date")["pm25"]
    X_train_leaky = X_train.assign(pm25_tomorrow=daily_pm25.reindex(train["target_date"]).to_numpy())
    X_test_leaky = X_test.assign(pm25_tomorrow=daily_pm25.reindex(test["target_date"]).to_numpy())

    cheat = clone(xgb).fit(X_train_leaky, y_train)   # clone = same settings, untrained
    demo = pd.DataFrame({
        "Honest XGBoost": score(y_test, predictions["XGBoost"], today_aqi),
        "Leaky XGBoost (sees tomorrow's PM2.5)": score(y_test, cheat.predict(X_test_leaky), today_aqi),
    }).T.round(2)
    print("\nLeakage demo (the leaky model is discarded, never saved):")
    print(demo[["MAE", "R2", "bucket_accuracy_pct"]].to_string())
    demo.to_csv(os.path.join(METRICS_DIR, "leakage_demo.csv"), index_label="model")

# ---- 7. Figures ----
print("\nMaking figures...")
dates = test["target_date"]

# Figure 1: the test period, actual vs XGBoost
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(dates, y_test, color=BLUE, linewidth=1.5, label="Actual AQI")
ax.plot(dates, predictions["XGBoost"], color=ORANGE, linewidth=1.5, label="XGBoost forecast")
ax.set_ylim(0, 700)
ax.set_xlim(dates.min(), dates.max())
ax.set_ylabel("AQI")
ax.set_title(f"Next-day forecast vs actual, test period (MAE {xgb_mae:.1f})")
ax.legend(loc="upper left")
label_aqi_bands(ax)
save(fig, "model_01_test_forecast.png")

# Figure 2: predicted vs actual, persistence beside XGBoost
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
for ax, name in zip(axes, ["Persistence (tomorrow = today)", "XGBoost"]):
    ax.scatter(y_test, predictions[name], s=14, color=BLUE, alpha=0.5, linewidth=0)
    ax.plot([0, 700], [0, 700], color=AXIS, linewidth=1)  # perfect forecasts lie on this line
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual AQI")
    ax.set_title(f"{name}\nMAE {results.loc[name, 'MAE']:.1f}, R² {results.loc[name, 'R2']:.2f}",
                 fontsize=11)
axes[0].set_ylabel("Predicted AQI")
save(fig, "model_02_predicted_vs_actual.png")

# Figure 3: XGBoost's errors
error = predictions["XGBoost"] - y_test   # positive = forecast too high
fig, (ax_hist, ax_time) = plt.subplots(1, 2, figsize=(12, 4.2),
                                       gridspec_kw={"width_ratios": [1, 2]})
ax_hist.hist(error, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
ax_hist.axvline(0, color=INK_SOFT, linewidth=1)
ax_hist.set_xlabel("Forecast − actual (AQI)")
ax_hist.set_ylabel("Number of days")
ax_hist.set_title(f"Error distribution (mean {error.mean():+.1f})")
ax_time.vlines(dates, 0, error, color=BLUE, linewidth=1.2)
ax_time.axhline(0, color=INK_SOFT, linewidth=1)
ax_time.set_xlim(dates.min(), dates.max())
ax_time.set_ylabel("Forecast − actual (AQI)")
ax_time.set_title("Error by day: largest when AQI jumps suddenly")
save(fig, "model_03_errors.png")

# Figure 4: CPCB category confusion matrix
matrix = confusion_matrix(bucket_names(y_test), bucket_names(predictions["XGBoost"]),
                          labels=AQI_BUCKET_LABELS)
blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", BLUE, "#0d366b"])
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(matrix, annot=True, fmt="d", mask=(matrix == 0), cmap=blues,
            xticklabels=AQI_BUCKET_LABELS, yticklabels=AQI_BUCKET_LABELS,
            linewidths=1, linecolor=SURFACE, cbar=False, ax=ax)
ax.grid(False)
ax.tick_params(axis="y", rotation=0)
ax.set_xlabel("Predicted category")
ax.set_ylabel("Actual category")
ax.set_title(f"XGBoost: CPCB category, test period "
             f"({results.loc['XGBoost', 'bucket_accuracy_pct']:.0f}% exactly right)")
save(fig, "model_04_category_confusion.png")

# Figure 5: every model side by side
fig, (ax_mae, ax_dir) = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
positions = np.arange(len(results))
for ax, col, label in [(ax_mae, "MAE", "MAE on test (AQI points, lower is better)"),
                       (ax_dir, "direction_accuracy_pct", "Up/down called right (%, higher is better)")]:
    values = results[col]
    ax.barh(positions, values, height=0.6, color=BLUE)
    for y, v in zip(positions, values):
        ax.text(v + values.max() * 0.015, y, f"{v:.1f}", va="center", fontsize=9, color=INK_SOFT)
    ax.set_xlim(0, values.max() * 1.18)
    ax.set_xlabel(label)
    ax.grid(axis="y", visible=False)
ax_mae.set_yticks(positions, results.index)
ax_mae.invert_yaxis()
fig.suptitle("Model comparison, test period", x=0.01, ha="left")
save(fig, "model_05_comparison.png")
