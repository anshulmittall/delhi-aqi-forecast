"""
03_eda.py

Exploratory figures for the report.

Every figure except the first uses the TRAINING period only. The test period
stays unseen until the models are scored, so nothing noticed here can quietly
shape decisions that are later judged on that same data. Figure 1 shows the
whole timeline because its job is to show where the split falls.

Run from the project root, after 02_build_dataset.py:
    python scripts/03_eda.py

Input:
  data/processed/delhi_daily.csv
  outputs/metrics/missingness_delhi.csv

Output (outputs/figures/):
  eda_01_aqi_timeline.png        AQI over time, with train/test/lockdown periods
  eda_02_aqi_distribution.png    histogram + counts per CPCB category
  eda_03_aqi_by_month.png        seasonality: box plot per month
  eda_04_aqi_by_weekday.png      weekly pattern: box plot per weekday
  eda_05_correlation.png         Spearman correlation heatmap
  eda_06_autocorrelation.png     how strongly AQI is tied to its own past
  eda_07_aqi_vs_weather.png      AQI against wind, humidity, boundary layer
  eda_08_missing_data.png        missing values per column
"""

import calendar
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from common import (
    AQI_BUCKET_LABELS, AXIS, BLUE, BLUE_DARK, BLUE_WASH, GRID, INK_SOFT,
    LOCKDOWN_START, NEUTRAL, ORANGE, RED, SURFACE, TEST_END, TRAIN_END,
    aqi_to_bucket, label_aqi_bands, use_project_style,
)

# ---- CONFIG ----
ROLLING_DAYS = 30   # smoothing window for the trend line in figure 1
MAX_LAG = 30        # how many days back the autocorrelation plot looks

# Columns for the correlation heatmap, with readable names. NO, NOx and
# Toluene are left out because they move almost in lockstep with NO2 and
# Benzene, and would only make the grid bigger without adding information.
HEATMAP_COLUMNS = {
    "aqi": "AQI",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "co": "CO",
    "so2": "SO2",
    "o3": "O3",
    "nh3": "NH3",
    "benzene": "Benzene",
    "temp_mean": "Temperature",
    "humidity_mean": "Humidity",
    "wind_speed_mean": "Wind speed",
    "precip_sum": "Rain",
    "pressure_mean": "Pressure",
    "blh_mean": "Boundary layer (mean)",
    "blh_min": "Boundary layer (min)",
}
# ----------------

PROCESSED_DIR = "data/processed"
METRICS_DIR = "outputs/metrics"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

use_project_style()


def save(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---- Load and split ----
df = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_daily.csv"),
                 parse_dates=["date"]).set_index("date")
train = df.loc[:TRAIN_END]

print(f"All data:  {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")
print(f"Training:  {len(train)} days, up to {TRAIN_END} (used for figures 2-7)")
print("\nMaking figures...")

# ---- Figure 1: the whole timeline ----
test_start = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(df.index, df["aqi"], color=BLUE, alpha=0.35, linewidth=0.8, label="Daily AQI")
# Trailing window (each point averages the previous 30 days), the same
# past-only convention the features will use.
trend = df["aqi"].rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS // 2).mean()
ax.plot(df.index, trend, color=BLUE_DARK, linewidth=2, label=f"{ROLLING_DAYS}-day average")

ax.axvspan(test_start, pd.Timestamp(TEST_END), color=GRID, alpha=0.7, linewidth=0)
ax.axvspan(pd.Timestamp(LOCKDOWN_START), df.index.max(), color=ORANGE, alpha=0.12, linewidth=0)

# Period names along the top. x is a date, y is a fraction of the axes height.
spans = [
    (df.index.min(), pd.Timestamp(TRAIN_END), "Train"),
    (test_start, pd.Timestamp(TEST_END), "Test"),
    (pd.Timestamp(LOCKDOWN_START), df.index.max(), "COVID\nlockdown"),
]
for start, end, name in spans:
    ax.text(start + (end - start) / 2, 0.97, name, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=INK_SOFT)

ax.set_ylim(0, 800)
ax.set_xlim(df.index.min(), df.index.max())
ax.set_ylabel("AQI")
ax.set_title("Delhi daily AQI, 2015 to mid-2020")
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
label_aqi_bands(ax)
save(fig, "eda_01_aqi_timeline.png")

# ---- Figure 2: distribution ----
fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.2),
                                      gridspec_kw={"width_ratios": [1.3, 1]})

sns.histplot(train["aqi"].dropna(), bins=40, kde=True, ax=ax_hist,
             color=BLUE, alpha=0.6, edgecolor=SURFACE, linewidth=1,
             line_kws={"linewidth": 2})
ax_hist.lines[0].set_color(BLUE_DARK)  # the KDE curve; histplot colours it like the bars
ax_hist.set_xlabel("AQI")
ax_hist.set_ylabel("Number of days")
ax_hist.set_title("AQI distribution (training period)")

counts = aqi_to_bucket(train["aqi"].dropna()).value_counts().reindex(AQI_BUCKET_LABELS)
positions = np.arange(len(counts))
ax_bar.barh(positions, counts.values, height=0.6, color=BLUE)
ax_bar.set_yticks(positions, counts.index)
ax_bar.invert_yaxis()  # Good at the top, Severe at the bottom
ax_bar.grid(axis="y", visible=False)
for y, n in zip(positions, counts.values):
    ax_bar.text(n + counts.max() * 0.02, y, f"{n}  ({n / counts.sum():.0%})",
                va="center", fontsize=9, color=INK_SOFT)
ax_bar.set_xlim(0, counts.max() * 1.3)  # room for the labels
ax_bar.set_xlabel("Number of days")
ax_bar.set_title("Days per CPCB category")
save(fig, "eda_02_aqi_distribution.png")

# ---- Figures 3 and 4: seasonality and weekly pattern ----
box_style = dict(color=BLUE_WASH, linecolor=BLUE_DARK, linewidth=1, fliersize=2, width=0.6)

fig, ax = plt.subplots(figsize=(10, 4.2))
sns.boxplot(x=train.index.month, y=train["aqi"], order=range(1, 13), ax=ax, **box_style)
ax.set_xticks(range(12), calendar.month_abbr[1:])
ax.set_xlabel("")
ax.set_ylabel("AQI")
ax.set_ylim(0, 800)
ax.set_title("AQI by month (training period): winter is the smog season")
label_aqi_bands(ax)
save(fig, "eda_03_aqi_by_month.png")

fig, ax = plt.subplots(figsize=(8, 4.2))
sns.boxplot(x=train.index.dayofweek, y=train["aqi"], order=range(7), ax=ax, **box_style)
ax.set_xticks(range(7), calendar.day_abbr)
ax.set_xlabel("")
ax.set_ylabel("AQI")
ax.set_ylim(0, 800)
ax.set_title("AQI by day of week (training period)")
label_aqi_bands(ax)
save(fig, "eda_04_aqi_by_weekday.png")

# ---- Figure 5: correlation heatmap ----
# Spearman = Pearson correlation computed on ranks. It captures any steady
# up-or-down relationship, not just straight lines, and isn't dragged around
# by a handful of extreme smog days the way Pearson is.
corr = train[list(HEATMAP_COLUMNS)].rename(columns=HEATMAP_COLUMNS).corr(method="spearman")
# The matrix is symmetric and its diagonal is all 1s, so keep only the
# lower triangle. Its first row and last column would be empty, so drop them.
corr_lower = corr.iloc[1:, :-1]
upper = np.triu(np.ones_like(corr_lower, dtype=bool), k=1)
diverging = LinearSegmentedColormap.from_list("blue_red", [BLUE, NEUTRAL, RED])

fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(corr_lower, mask=upper, cmap=diverging, vmin=-1, vmax=1, center=0,
            # Labels rounded to 1 decimal; adding 0.0 turns "-0.0" into "0.0".
            annot=corr_lower.round(1) + 0.0, fmt=".1f", annot_kws={"fontsize": 7.5},
            linewidths=1, linecolor=SURFACE, square=True, ax=ax,
            cbar_kws={"shrink": 0.6, "label": "Spearman correlation"})
ax.grid(False)
ax.set_title("Correlation between pollutants, weather and AQI (training period, same day)")
save(fig, "eda_05_correlation.png")

# ---- Figure 6: autocorrelation ----
# autocorr(lag) = correlation between AQI today and AQI `lag` days earlier.
# pandas skips any day where either value is missing.
lags = np.arange(1, MAX_LAG + 1)
acf = np.array([train["aqi"].autocorr(lag) for lag in lags])
# Correlations inside +/- 1.96/sqrt(n) could be pure chance (95% band).
noise_band = 1.96 / np.sqrt(train["aqi"].notna().sum())

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.axhspan(-noise_band, noise_band, color=GRID, alpha=0.8, linewidth=0)
ax.vlines(lags, 0, acf, color=BLUE, linewidth=2)
ax.scatter(lags, acf, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
for lag in (1, 7, MAX_LAG):
    ax.annotate(f"{acf[lag - 1]:.2f}", (lag, acf[lag - 1]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9, color=INK_SOFT)
ax.axhline(0, color=AXIS, linewidth=0.8)
ax.set_xlim(0, MAX_LAG + 1)
ax.set_ylim(-0.1, 1.05)
ax.set_xlabel("Lag (days back)")
ax.set_ylabel("Correlation with today's AQI")
ax.set_title("Autocorrelation of AQI (training period): yesterday is a strong guide to today")
save(fig, "eda_06_autocorrelation.png")

# ---- Figure 7: AQI against weather ----
# Each dot is one day. The dark line joins the median AQI within each tenth
# of the x-axis values (deciles), which shows the trend through the cloud.
weather_panels = [
    ("wind_speed_mean", "Mean wind speed (km/h)"),
    ("humidity_mean", "Mean relative humidity (%)"),
    ("blh_min", "Night-time boundary layer height (m)"),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (col, xlabel) in zip(axes, weather_panels):
    pair = train[[col, "aqi"]].dropna()
    ax.scatter(pair[col], pair["aqi"], s=8, color=BLUE, alpha=0.25, linewidth=0, label="One day")
    deciles = pd.qcut(pair[col], 10, duplicates="drop")
    trend = pair.groupby(deciles, observed=True).median()
    ax.plot(trend[col], trend["aqi"], color=BLUE_DARK, linewidth=2, marker="o",
            markersize=5, markeredgecolor=SURFACE, label="Median per decile")
    ax.set_xlabel(xlabel)
axes[0].set_ylabel("AQI (same day)")
axes[0].legend(loc="upper right")
axes[-1].set_xscale("log")  # boundary-layer height spans 10 m to 1,000+ m
axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig.suptitle("AQI against weather (training period)", x=0.01, ha="left")
save(fig, "eda_07_aqi_vs_weather.png")

# ---- Figure 8: missing data (whole dataset, before filling) ----
missing = pd.read_csv(os.path.join(METRICS_DIR, "missingness_delhi.csv"), index_col="column")
missing = missing.drop(index="aqi_bucket")  # always missing on the same days as aqi
missing = missing.sort_values("missing_frac")
colors = [ORANGE if dropped else BLUE for dropped in missing["dropped"]]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(missing.index, missing["missing_frac"], height=0.6, color=colors)
for y, frac in enumerate(missing["missing_frac"]):
    ax.text(frac + 0.006, y, f"{frac:.1%}", va="center", fontsize=9, color=INK_SOFT)
ax.grid(axis="y", visible=False)
ax.set_xlim(0, missing["missing_frac"].max() * 1.2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Share of days missing")
ax.set_title("Missing values per column (all data, before filling)")
# Legend built by hand because the colours come from one bar call.
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                   plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["Kept", "Dropped in step 2"], loc="lower right")
save(fig, "eda_08_missing_data.png")

# ---- Numbers worth quoting in the report ----
print("\nMedian AQI by month (training period):")
print(train["aqi"].groupby(train.index.month).median().rename(
    index=lambda m: calendar.month_abbr[m]).to_string())
print("\nMedian AQI by weekday (training period):")
print(train["aqi"].groupby(train.index.dayofweek).median().rename(
    index=lambda d: calendar.day_abbr[d]).to_string())
print(f"\nAutocorrelation: lag 1 = {acf[0]:.2f}, lag 7 = {acf[6]:.2f}, "
      f"lag {MAX_LAG} = {acf[-1]:.2f}")
print("\nSpearman correlation with same-day AQI (training period):")
print(corr["AQI"].drop("AQI").sort_values().round(2).to_string())
