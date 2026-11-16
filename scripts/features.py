"""
features.py

How the model's inputs are built from the clean daily table. Kept in its own
module so that step 4 (which creates the training data) and the Streamlit app
(which builds inputs for its what-if sliders) run exactly the same code. If
they used two copies, the app could quietly feed the model inputs computed
differently from the ones it was trained on.

Every feature for day t uses only day t and earlier. Step 4 checks this by
deleting the future and confirming that no feature changes.
"""

import numpy as np
import pandas as pd

# Today's measurements, used as-is.
AIR_COLUMNS = ["aqi", "pm25", "pm10", "no", "no2", "nox", "nh3", "co",
               "so2", "o3", "benzene", "toluene"]
WEATHER_COLUMNS = [
    "temp_mean", "temp_max", "temp_min", "humidity_mean", "dew_point_mean",
    "pressure_mean", "wind_speed_mean", "wind_speed_max", "wind_gust_max",
    "precip_sum", "blh_mean", "blh_min",
    # Direction as sin/cos only. Raw degrees would tell the model that 359°
    # and 1° are far apart, when both mean "wind from the north".
    "wind_dir_sin", "wind_dir_cos",
]

AQI_LAGS = [1, 2, 3, 7]   # AQI 1, 2, 3 and 7 days ago (7 = same weekday last week)
PM25_LAGS = [1, 2, 3]
PM10_LAGS = [1]
AQI_ROLL_WINDOWS = [3, 7, 14]


def rolling_mean(series, window):
    """Average of today and the (window - 1) days before it.

    Requires at least half of those days to have data, so a single missing
    day doesn't blank out every window that contains it.
    """
    return series.rolling(window, min_periods=(window + 1) // 2).mean()


def build_features(daily):
    """Every model input for day t, computed from day t and earlier only."""
    aqi = daily["aqi"]
    pm25 = daily["pm25"]
    pm10 = daily["pm10"]
    features = {}

    # -- Today's air quality and weather, as measured --
    for col in AIR_COLUMNS + WEATHER_COLUMNS:
        features[col] = daily[col]

    # -- Step 2's flags: 1 where a value was missing and copied forward --
    for col in daily.columns:
        if col.endswith("_was_missing"):
            features[col] = daily[col]

    # -- Lags: the value k days ago --
    # shift(k) with k > 0 moves every value k rows DOWN, so each row only
    # ever receives a value from an earlier day.
    for k in AQI_LAGS:
        features[f"aqi_lag{k}"] = aqi.shift(k)
    for k in PM25_LAGS:
        features[f"pm25_lag{k}"] = pm25.shift(k)
    for k in PM10_LAGS:
        features[f"pm10_lag{k}"] = pm10.shift(k)

    # -- Change since yesterday: is a pollution episode building or clearing? --
    features["aqi_change1"] = aqi - aqi.shift(1)
    features["pm25_change1"] = pm25 - pm25.shift(1)

    # -- Trailing windows: they end today, which is already known --
    for w in AQI_ROLL_WINDOWS:
        features[f"aqi_roll{w}_mean"] = rolling_mean(aqi, w)
    # How jumpy AQI has been over the last week
    features["aqi_roll7_std"] = aqi.rolling(7, min_periods=4).std()
    features["pm25_roll7_mean"] = rolling_mean(pm25, 7)
    # A few still, humid days in a row is what lets smog build up
    features["wind_speed_roll3_mean"] = rolling_mean(daily["wind_speed_mean"], 3)
    features["humidity_roll3_mean"] = rolling_mean(daily["humidity_mean"], 3)

    # -- Calendar --
    # Day of year as a point on a circle, so 31 December and 1 January end
    # up next to each other instead of 364 apart.
    day_of_year = daily.index.dayofyear
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["day_of_week"] = daily.index.dayofweek          # 0 = Monday
    features["is_weekend"] = (daily.index.dayofweek >= 5).astype(int)

    # -- Domain knowledge --
    month, day = daily.index.month, daily.index.day
    # Crop-residue burning in Punjab and Haryana, roughly 15 Oct - 30 Nov
    features["is_stubble_season"] = (((month == 10) & (day >= 15)) | (month == 11)).astype(int)
    # Dec-Feb: cold nights and temperature inversions trap pollution near the ground
    features["is_winter"] = month.isin([12, 1, 2]).astype(int)

    return pd.DataFrame(features, index=daily.index)
