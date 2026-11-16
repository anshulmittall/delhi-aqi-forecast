"""
common.py

Settings and helpers shared by several scripts, kept in one place so they
cannot drift apart. For example, the EDA plots, the model evaluation and the
Streamlit app must all agree on exactly where the test period starts.

Scripts use it with `from common import ...`. That works because Python puts
the running script's own folder (scripts/) on the import path.
"""

import matplotlib

matplotlib.use("Agg")  # draw straight to image files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, r2_score,
    root_mean_squared_error,
)

# ---- Train / test split (all dates inclusive) ----
TRAIN_END = "2019-06-30"   # last day the models may learn from
TEST_END = "2020-03-21"    # test = the day after TRAIN_END up to here

# Janata Curfew was 22 March 2020; Delhi's lockdown followed on the 23rd and
# the national one on the 25th. Traffic and industry fell from the 22nd, so
# the COVID period is scored separately from here on (step 8).
LOCKDOWN_START = "2020-03-22"

# ---- CPCB AQI categories ----
# A category covers (lower edge, upper edge], e.g. Moderate is 101-200.
# Checked in step 2: these reproduce all 1,999 AQI_Bucket labels for Delhi.
AQI_BUCKET_EDGES = [0, 50, 100, 200, 300, 400, float("inf")]
AQI_BUCKET_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def aqi_to_bucket(aqi):
    """Map AQI numbers (a Series or array) to their CPCB category names."""
    return pd.cut(aqi, bins=AQI_BUCKET_EDGES, labels=AQI_BUCKET_LABELS,
                  include_lowest=True)


# ---- Scoring (used by step 5 and the COVID case study) ----
def bucket_names(values):
    """CPCB category name for each AQI value, as a plain array."""
    return np.asarray(aqi_to_bucket(np.asarray(values, dtype=float)))


def score(actual, predicted, today):
    """Every test metric for one model's predictions."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    today = np.asarray(today, dtype=float)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "MAPE_pct": 100 * mean_absolute_percentage_error(actual, predicted),
        # Average of (forecast - actual): positive means forecasts run high
        "bias": np.mean(predicted - actual),
        # Did it land in the right CPCB category (Good ... Severe)?
        "bucket_accuracy_pct": 100 * np.mean(bucket_names(predicted) == bucket_names(actual)),
        # Did it call the move right: up, down, or (rarely) no change?
        # Persistence always predicts "no change", so it scores ~0 here.
        "direction_accuracy_pct": 100 * np.mean(np.sign(predicted - today) == np.sign(actual - today)),
    }


# ---- Plot style ----
# A colour-blind-checked palette: blue carries the data, orange is used only
# to pick out one thing per chart, and everything else is quiet grey.
SURFACE = "#fcfcfb"     # background
INK = "#0b0b0b"         # titles
INK_SOFT = "#52514e"    # axis labels, annotations
INK_MUTED = "#898781"   # secondary annotations
GRID = "#e1e0d9"        # gridlines, neutral shading
AXIS = "#c3c2b7"        # axis lines
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
BLUE_WASH = "#cde2fb"
ORANGE = "#eb6834"
RED = "#e34948"         # positive end of the correlation colour scale
NEUTRAL = "#f0efec"     # midpoint of the correlation colour scale


def use_project_style():
    """Apply one consistent look to every figure in the project."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,     # gridlines behind the data, not over it
        "axes.prop_cycle": cycler(color=[BLUE, ORANGE]),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_SOFT,
        "ytick.labelcolor": INK_SOFT,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })


def label_aqi_bands(ax):
    """Write the CPCB category names just outside the right edge of an AQI axis."""
    top = ax.get_ylim()[1]
    for low, high, name in zip(AQI_BUCKET_EDGES[:-1], AQI_BUCKET_EDGES[1:], AQI_BUCKET_LABELS):
        if low >= top:
            break
        middle = (low + min(high, top)) / 2
        # x is a fraction of the axes width, y is in AQI units
        ax.text(1.01, middle, name, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=INK_MUTED)


def plot_waterfall(ax, base_value, contributions, labels, max_rows=10):
    """Draw one forecast as a SHAP waterfall, and return the forecast.

    Starts at the model's average forecast (base_value) and adds one feature's
    push per row, largest first, until it reaches this day's forecast. Red
    pushes the forecast up (worse air), blue pushes it down.

    contributions: SHAP values in AQI points, one per feature
    labels:        a matching row label per feature, e.g. "pm25 = 320"
    """
    contributions = np.asarray(contributions, dtype=float)
    order = np.argsort(-np.abs(contributions))
    shown, rest = order[: max_rows - 1], order[max_rows - 1:]
    values = list(contributions[shown]) + [contributions[rest].sum()]
    names = [labels[i] for i in shown] + [f"{len(rest)} other features"]

    running = base_value
    for row, value in enumerate(values):
        ax.barh(row, value, left=running, height=0.6, color=RED if value > 0 else BLUE)
        # Numbers sit in their own column right of the plot, so they can
        # never collide with a bar or a label.
        ax.text(1.02, row, f"{value:+.1f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9, color=INK_SOFT)
        running += value
    forecast = running

    ax.axvline(base_value, color=AXIS, linewidth=1)
    ax.axvline(forecast, color=INK_SOFT, linewidth=1)
    ax.text(base_value, -0.75, f"average\n{base_value:.0f}", ha="center", va="bottom",
            fontsize=8, color=INK_MUTED)
    ax.text(forecast, -0.75, f"forecast\n{forecast:.0f}", ha="center", va="bottom",
            fontsize=8, color=INK_SOFT)
    ax.text(1.02, -0.75, "push", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=8, color=INK_MUTED)

    # Pad the x-range so bars and the average/forecast labels clear the edges
    path = base_value + np.concatenate([[0], np.cumsum(values)])
    pad = 0.08 * (path.max() - path.min())
    ax.set_xlim(path.min() - pad, path.max() + pad)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.5, -1.6)   # first row at the top, room for the labels
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Tomorrow's AQI forecast")
    return forecast
