"""
01_download_data.py

Downloads the two raw data sources for the project and caches them locally,
so every later script runs offline.

  1. Kaggle "Air Quality Data in India" (city_day.csv):
     daily pollutant averages + AQI for 26 Indian cities, 2015-2020.
  2. Open-Meteo historical weather for Delhi, fetched hourly and then
     aggregated to one row per day.

Run from the project root (delhi-aqi-forecast/):
    python scripts/01_download_data.py

Output:
  data/raw/city_day.csv                   (all cities, as downloaded)
  data/raw/weather_hourly_<year>.csv      (raw hourly weather, one file per year)
  data/processed/weather_daily.csv        (one row per day, ready to join in step 2)

Re-running is cheap: any raw file that already exists is reused, not fetched
again. Delete a file from data/raw/ to force a fresh download.
"""

import os
import shutil
import sys
import time

import kagglehub
import numpy as np
import pandas as pd
import requests

# ---- CONFIG ----
KAGGLE_DATASET = "rohanrao/air-quality-data-in-india"
KAGGLE_FILE = "city_day.csv"
CITY = "Delhi"

# Central Delhi. Open-Meteo answers with the weather-model grid cell that
# contains this point, so this is city-scale weather, not street-level.
LATITUDE = 28.6139
LONGITUDE = 77.2090

# Group hours into days by Indian local time, the same way the AQI daily
# averages in city_day.csv are defined. With UTC, each "day" would be shifted
# by 5.5 hours.
TIMEZONE = "Asia/Kolkata"

WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"

# We ask for hourly data rather than Open-Meteo's ready-made daily summary
# because boundary_layer_height only exists hourly, and because we want to
# choose our own daily aggregations (e.g. the night-time minimum of the
# boundary layer, and a correct average of wind direction).
HOURLY_VARS = [
    "temperature_2m",          # °C, 2 m above ground
    "relative_humidity_2m",    # %
    "dew_point_2m",            # °C
    "surface_pressure",        # hPa
    "wind_speed_10m",          # km/h, 10 m above ground
    "wind_gusts_10m",          # km/h
    "wind_direction_10m",      # degrees, direction the wind blows FROM
    "precipitation",           # mm in that hour
    "boundary_layer_height",   # m: height of the air layer pollution mixes into
]
# ----------------

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_city_day():
    """Fetch city_day.csv from Kaggle into data/raw/, unless it's already there."""
    out_path = os.path.join(RAW_DIR, KAGGLE_FILE)
    if os.path.exists(out_path):
        print(f"[kaggle] {out_path} already exists, skipping download.")
        return out_path

    print(f"[kaggle] Downloading {KAGGLE_FILE} from {KAGGLE_DATASET} ...")
    try:
        # path= downloads just this one file. The full dataset also contains
        # much larger hourly and per-station files this project never uses.
        cached_path = kagglehub.dataset_download(KAGGLE_DATASET, path=KAGGLE_FILE)
    except Exception as error:
        print(f"[kaggle] Download failed: {error}")
        print("Manual fallback:")
        print(f"  1. Open https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"  2. Download {KAGGLE_FILE}")
        print(f"  3. Save it as {out_path} and re-run this script.")
        sys.exit(1)

    # kagglehub keeps files in its own cache (~/.cache/kagglehub). Copy it into
    # the project so the data sits next to the code that uses it.
    shutil.copy(cached_path, out_path)
    print(f"[kaggle] Saved {out_path}")
    return out_path


def city_date_range(city_day_path):
    """First and last date that the chosen city has air-quality data for."""
    df = pd.read_csv(city_day_path, usecols=["City", "Date"], parse_dates=["Date"])
    dates = df.loc[df["City"] == CITY, "Date"]
    if dates.empty:
        raise ValueError(f"No rows for City == '{CITY}' in {city_day_path}.")
    return dates.min(), dates.max()


def fetch_weather_year(year, start, end):
    """Hourly weather for one calendar year (clipped to start..end), cached to CSV."""
    out_path = os.path.join(RAW_DIR, f"weather_hourly_{year}.csv")
    if os.path.exists(out_path):
        print(f"[weather] {year}: using cached {out_path}")
        return pd.read_csv(out_path, parse_dates=["time"])

    print(f"[weather] {year}: requesting {start.date()} to {end.date()} ...")
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    response = requests.get(WEATHER_API, params=params, timeout=60)
    if not response.ok:
        # On a bad request Open-Meteo explains why in the body, e.g. which
        # variable name it didn't recognise. Show that instead of a bare 400.
        raise RuntimeError(
            f"Open-Meteo request for {year} failed ({response.status_code}): "
            f"{response.text[:300]}"
        )

    # The JSON has one list per variable under "hourly", all the same length,
    # which maps straight onto DataFrame columns.
    hourly = pd.DataFrame(response.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly.to_csv(out_path, index=False)

    time.sleep(1)  # free public API: space out requests politely
    return hourly


def hourly_to_daily(hourly):
    """Collapse hourly weather into one row per calendar day."""
    hourly = hourly.copy()
    hourly["date"] = hourly["time"].dt.normalize()  # drop the time, keep the day

    # Wind direction is an angle, so it can't be averaged like a number: the
    # mean of 350° and 10° is 180° (a south wind), yet both are northerly.
    # Instead, average the east-west (sin) and north-south (cos) components
    # and turn the averaged pair back into an angle afterwards.
    radians = np.deg2rad(hourly["wind_direction_10m"])
    hourly["wind_dir_sin"] = np.sin(radians)
    hourly["wind_dir_cos"] = np.cos(radians)

    daily = hourly.groupby("date").agg(
        temp_mean=("temperature_2m", "mean"),
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        humidity_mean=("relative_humidity_2m", "mean"),
        dew_point_mean=("dew_point_2m", "mean"),
        pressure_mean=("surface_pressure", "mean"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        wind_speed_max=("wind_speed_10m", "max"),
        wind_gust_max=("wind_gusts_10m", "max"),
        precip_sum=("precipitation", "sum"),
        blh_mean=("boundary_layer_height", "mean"),
        # The night-time minimum is when pollution is squeezed into the
        # thinnest layer of air, so it gets its own column.
        blh_min=("boundary_layer_height", "min"),
        wind_dir_sin=("wind_dir_sin", "mean"),
        wind_dir_cos=("wind_dir_cos", "mean"),
        hours=("time", "count"),
    ).reset_index()

    # Human-readable average direction in degrees (0 = from north, 90 = from
    # east), for plots. The sin/cos columns are what the model will use.
    daily["wind_dir_mean"] = (
        np.rad2deg(np.arctan2(daily["wind_dir_sin"], daily["wind_dir_cos"])) % 360
    )
    return daily


# ---- 1. Air quality (Kaggle) ----
city_day_path = download_city_day()
first_day, last_day = city_date_range(city_day_path)
expected_days = (last_day - first_day).days + 1
print(f"[kaggle] {CITY} has data from {first_day.date()} to {last_day.date()} "
      f"({expected_days} calendar days)")

# ---- 2. Weather (Open-Meteo), one request per year over the same range ----
yearly_frames = []
for year in range(first_day.year, last_day.year + 1):
    start = max(first_day, pd.Timestamp(year=year, month=1, day=1))
    end = min(last_day, pd.Timestamp(year=year, month=12, day=31))
    yearly_frames.append(fetch_weather_year(year, start, end))
hourly = pd.concat(yearly_frames, ignore_index=True)

daily = hourly_to_daily(hourly)

# ---- 3. Sanity checks, printed rather than silently trusted ----
print("\n[check] Days of weather per year:")
print(daily["date"].dt.year.value_counts().sort_index().to_string())

incomplete = daily[daily["hours"] != 24]
print(f"[check] Days without exactly 24 hourly readings: {len(incomplete)}")

if len(daily) != expected_days:
    raise ValueError(
        f"Weather has {len(daily)} days but {CITY}'s air-quality data spans "
        f"{expected_days}. Delete data/raw/weather_hourly_*.csv and re-run."
    )
print(f"[check] Weather covers all {expected_days} days of the air-quality range.")

print("[check] Missing values per weather column:")
print(daily.drop(columns=["date", "hours"]).isna().sum().to_string())

# ---- 4. Save ----
daily = daily.drop(columns=["hours"])  # only needed for the check above
daily_path = os.path.join(PROCESSED_DIR, "weather_daily.csv")
daily.to_csv(daily_path, index=False)

print("\nSaved:")
print(f"  - {city_day_path} (air quality, all cities)")
print(f"  - {RAW_DIR}/weather_hourly_<year>.csv (raw hourly cache)")
print(f"  - {daily_path} ({len(daily)} days x {daily.shape[1] - 1} weather columns)")
