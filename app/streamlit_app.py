"""
streamlit_app.py

Interactive demo of the next-day AQI model. Nothing is trained here: the app
loads the model saved by step 5 and the tables built by steps 2 and 4.

Run from the project root, after the pipeline (scripts 01 to 07):
    streamlit run app/streamlit_app.py

Tabs:
  1. Explore a day   pick a day, compare the forecast with what happened, see why
  2. What-if         change today's conditions and watch tomorrow's forecast move
  3. Model card      scores, how leakage was ruled out, limitations
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# The shared code lives in scripts/. Add that folder to the import path so
# the app uses exactly the same feature and plotting code as the pipeline.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402  (must come after the sys.path line above)
    BLUE, LOCKDOWN_START, ORANGE, TRAIN_END, aqi_to_bucket, plot_waterfall,
    use_project_style,
)
from features import build_features  # noqa: E402

# CPCB's official colour code. Every badge also shows the category's name,
# so the colour is never the only clue. Dark text on all but the darkest.
BUCKET_STYLE = {
    "Good": ("#00b050", "#0b0b0b"),
    "Satisfactory": ("#92d050", "#0b0b0b"),
    "Moderate": ("#ffff00", "#0b0b0b"),
    "Poor": ("#ff9900", "#0b0b0b"),
    "Very Poor": ("#ff0000", "#0b0b0b"),
    "Severe": ("#c00000", "#ffffff"),
}

# Readable column headers for the score tables on the Model card
SCORE_LABELS = {
    "cv_MAE": "CV MAE", "MAPE_pct": "MAPE %", "bias": "Bias",
    "bucket_accuracy_pct": "Category right %", "direction_accuracy_pct": "Up/down right %",
    "mean_actual_aqi": "Mean actual AQI",
}

# Today's conditions the What-if tab lets you change: column -> (label, min, max)
WHAT_IF_INPUTS = {
    "aqi": ("Today's AQI", 0, 800),
    "pm25": ("Today's PM2.5 (µg/m³)", 0, 700),
    "pm10": ("Today's PM10 (µg/m³)", 0, 900),
    "wind_speed_mean": ("Mean wind speed (km/h)", 0, 40),
    "humidity_mean": ("Mean humidity (%)", 0, 100),
    "precip_sum": ("Rain (mm)", 0, 60),
    "blh_min": ("Night-time boundary layer (m)", 10, 1000),
}

st.set_page_config(page_title="Delhi AQI forecast", layout="wide")
use_project_style()


# ---- Loading (cached, so it happens once, not on every click) ----
@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "xgb_aqi.joblib")
    feature_cols = json.loads((ROOT / "models" / "feature_names.json").read_text())
    return model, feature_cols, shap.TreeExplainer(model)


@st.cache_data
def load_tables():
    daily = pd.read_csv(ROOT / "data" / "processed" / "delhi_daily.csv",
                        parse_dates=["date"]).set_index("date")
    table = pd.read_csv(ROOT / "data" / "processed" / "delhi_features.csv",
                        parse_dates=["date", "target_date"])
    # Only days the model never trained on: the test period and the lockdown
    held_out = table[table["target_date"] > TRAIN_END].set_index("target_date")
    model, feature_cols, _ = load_model()
    held_out["forecast"] = model.predict(held_out[feature_cols])
    return daily, held_out


def read_csv_if_exists(name):
    path = ROOT / "outputs" / "metrics" / name
    return pd.read_csv(path) if path.exists() else None


def show_scores(frame):
    """A score table with readable headers, 2 decimals and '—' where empty."""
    st.dataframe(frame.rename(columns=SCORE_LABELS).style.format(precision=2, na_rep="—"),
                 width="stretch")


# ---- Small display helpers ----
def bucket_of(aqi_value):
    return str(aqi_to_bucket(pd.Series([aqi_value])).iloc[0])


def badge(aqi_value):
    name = bucket_of(aqi_value)
    background, text = BUCKET_STYLE[name]
    return (f'<span style="background:{background};color:{text};padding:3px 10px;'
            f'border-radius:4px;font-weight:600">{name}</span>')


def waterfall_figure(features_row, title):
    """SHAP waterfall for one row of features, drawn like the report figures."""
    _, feature_cols, explainer = load_model()
    explanation = explainer(features_row[feature_cols].to_frame().T.astype(float))
    labels = [f"{name} = {value:.4g}" for name, value in features_row[feature_cols].items()]
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_waterfall(ax, explanation.base_values[0], explanation.values[0], labels)
    ax.set_title(title)
    return fig


model, feature_cols, explainer = load_model()
daily, held_out = load_tables()
first_day, last_day = held_out.index.min().date(), held_out.index.max().date()
lockdown = pd.Timestamp(LOCKDOWN_START)

st.title("Delhi next-day AQI forecast")
st.caption(
    "XGBoost model that predicts tomorrow's AQI from today's air quality, the past "
    f"two weeks and today's weather. Trained on days up to {TRAIN_END}; everything "
    "shown here is from days it never saw."
)

tab_explore, tab_what_if, tab_card = st.tabs(["Explore a day", "What-if", "Model card"])

# ---- Tab 1: Explore a day ----
with tab_explore:
    chosen = st.date_input("Forecast for", value=pd.Timestamp("2019-11-03").date(),
                           min_value=first_day, max_value=last_day)
    target = pd.Timestamp(chosen)

    if target not in held_out.index:
        st.warning("No forecast for this day (the data has no AQI for it). Pick another.")
    else:
        row = held_out.loc[target]
        forecast, actual = row["forecast"], row["aqi_tomorrow"]
        if target >= lockdown:
            st.info("This day is in the COVID lockdown, a world unlike anything "
                    "in the model's training data.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Forecast (made the day before)", f"{forecast:.0f}")
        col1.markdown(badge(forecast), unsafe_allow_html=True)
        col2.metric("Actual AQI", f"{actual:.0f}")
        col2.markdown(badge(actual), unsafe_allow_html=True)
        col3.metric("Error (forecast − actual)", f"{forecast - actual:+.0f}")

        left, right = st.columns([3, 2])
        with left:
            st.pyplot(waterfall_figure(row, f"Why the model forecast {forecast:.0f}"))
            st.caption("Starts at the model's average forecast. Each bar is one "
                       "feature's push (SHAP value): red raises tomorrow's forecast, "
                       "blue lowers it.")
        with right:
            nearby = held_out.loc[target - pd.Timedelta(days=30): target + pd.Timedelta(days=30),
                                  ["aqi_tomorrow", "forecast"]]
            st.markdown("**Two months around this day**")
            st.line_chart(nearby.rename(columns={"aqi_tomorrow": "Actual", "forecast": "Forecast"}),
                          color=[BLUE, ORANGE], y_label="AQI")

        with st.expander(f"All {len(feature_cols)} inputs the model saw (all from {row['date']:%Y-%m-%d})"):
            st.dataframe(row[feature_cols].rename("value").to_frame(), width="stretch")

# ---- Tab 2: What-if ----
with tab_what_if:
    st.markdown(
        "Start from a real day, change **today's** conditions, and see how **tomorrow's** "
        "forecast responds. The averages and day-to-day changes are rebuilt from your "
        "values with the same code used in training; everything else stays as recorded."
    )
    start = st.date_input("Starting from the forecast for", value=pd.Timestamp("2019-11-03").date(),
                          min_value=first_day, max_value=last_day, key="what_if_day")
    target = pd.Timestamp(start)

    if target not in held_out.index:
        st.warning("No data for this day. Pick another.")
    else:
        today = held_out.loc[target, "date"]   # the day the inputs describe
        edited = daily.loc[:today].copy()

        sliders, results = st.columns([2, 3])
        with sliders:
            for col, (label, low, high) in WHAT_IF_INPUTS.items():
                recorded = edited.loc[today, col]
                if pd.isna(recorded):
                    st.caption(f"{label}: not recorded on this day")
                    continue
                # Keyed by date, so the sliders reset when you pick another day
                edited.loc[today, col] = st.slider(
                    label, float(low), float(high), float(min(max(recorded, low), high)),
                    key=f"{col}-{today:%Y-%m-%d}",
                )

        original = held_out.loc[target, "forecast"]
        new_row = build_features(edited).loc[today]
        new_forecast = float(model.predict(new_row[feature_cols].to_frame().T.astype(float))[0])

        with results:
            col1, col2 = st.columns(2)
            col1.metric("Original forecast", f"{original:.0f}")
            col1.markdown(badge(original), unsafe_allow_html=True)
            change = round(new_forecast - original)
            col2.metric("What-if forecast", f"{new_forecast:.0f}",
                        f"{change:+d}" if change else None,   # no arrow until something changes
                        delta_color="inverse")   # a rise in AQI is bad, so show it red
            col2.markdown(badge(new_forecast), unsafe_allow_html=True)
            st.pyplot(waterfall_figure(new_row, f"Why the what-if forecast is {new_forecast:.0f}"))

# ---- Tab 3: Model card ----
with tab_card:
    st.subheader("Test-period scores")
    st.caption(f"Test period: forecasts for the day after {TRAIN_END} up to 21 March 2020. "
               "MAE is the average miss in AQI points.")
    comparison = read_csv_if_exists("model_comparison.csv")
    if comparison is not None:
        show_scores(comparison.set_index("model"))

    figure = ROOT / "outputs" / "figures" / "model_01_test_forecast.png"
    if figure.exists():
        st.image(str(figure))

    st.subheader("How data leakage was ruled out")
    st.markdown(
        "- The target is tomorrow's AQI; every input is from today or earlier.\n"
        "- Train and test are split **by date** (the day being predicted), never randomly.\n"
        "- Feature-building is checked by deleting the future: rebuilding features with "
        "every later day removed gives identical values, for all 2,009 days.\n"
        "- Missing values are only carried forward, never filled from later days.\n"
        "- Tuning uses time-ordered cross-validation (TimeSeriesSplit); scaling and "
        "imputation are fitted on training rows only.\n"
        "- A deliberately leaky model (given tomorrow's PM2.5) was trained to show what "
        "leakage looks like, then discarded:"
    )
    leakage = read_csv_if_exists("leakage_demo.csv")
    if leakage is not None:
        show_scores(leakage.set_index("model")[["MAE", "R2", "bucket_accuracy_pct"]])

    st.subheader("COVID lockdown: a regime shift")
    covid = read_csv_if_exists("covid_regime.csv")
    if covid is not None:
        show_scores(covid.set_index(["period", "model"])[["days", "mean_actual_aqi", "MAE", "MAPE_pct", "bias"]])
        st.caption("AQI roughly halved. The model's absolute error fell with it, but its "
                   "relative error rose, it over-forecast on average, and it no longer "
                   "beat the persistence baseline.")

    st.subheader("Limitations")
    st.markdown(
        "- Uses today's weather, not tomorrow's forecast; a real system would add one.\n"
        "- One city-wide value per day, averaged over stations; the station mix changed over the years.\n"
        "- Sudden jumps (smoke plumes, festivals) are caught a day late: nothing in the "
        "inputs sees them coming.\n"
        "- Trained on 2015–2019 Delhi; it does not adapt on its own when conditions change, "
        "as the lockdown showed."
    )
