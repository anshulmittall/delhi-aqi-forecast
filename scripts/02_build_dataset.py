"""
02_build_dataset.py

Builds one clean daily table for Delhi: air quality from city_day.csv joined
with the daily weather from step 1. Every later script starts from this file.

Run from the project root, after 01_download_data.py:
    python scripts/02_build_dataset.py

Input:
  data/raw/city_day.csv
  data/processed/weather_daily.csv

Output:
  data/processed/delhi_daily.csv          (one row per calendar day)
  outputs/metrics/missingness_delhi.csv   (missing-data report, plotted in step 3)
"""

import os

import pandas as pd

# ---- CONFIG ----
CITY = "Delhi"

# Drop a pollutant column when more than this fraction of its days is missing.
# For Delhi this removes only Xylene (39% missing, including one unbroken
# 606-day gap). The next-worst column, SO2, is 5.5% missing, so any threshold
# between about 0.06 and 0.38 gives exactly the same result.
MAX_MISSING_FRAC = 0.2

# Forward-fill at most this many consecutive missing days. Carrying yesterday's
# value forward is reasonable for a short sensor outage; for longer gaps we
# would be inventing data, so those stay NaN and are handled inside the model
# pipeline in step 5 (XGBoost handles NaN natively).
FFILL_LIMIT_DAYS = 3
# ----------------

# Short lowercase names that are easy to use in code ("PM2.5" has a dot and
# capitals). Units: CO is mg/m³, every other pollutant is µg/m³.
COLUMN_NAMES = {
    "Date": "date",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO": "no",
    "NO2": "no2",
    "NOx": "nox",
    "NH3": "nh3",
    "CO": "co",
    "SO2": "so2",
    "O3": "o3",
    "Benzene": "benzene",
    "Toluene": "toluene",
    "Xylene": "xylene",
    "AQI": "aqi",
    "AQI_Bucket": "aqi_bucket",
}

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
METRICS_DIR = "outputs/metrics"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def longest_gap(series):
    """Length, in days, of the longest run of consecutive missing values."""
    is_missing = series.isna()
    # The counter goes up at every non-missing value, so all the NaNs inside
    # one gap share the same counter value. Summing NaNs per counter value
    # gives the length of each gap.
    gap_id = (~is_missing).cumsum()
    return int(is_missing.groupby(gap_id).sum().max())


# ---- 1. Load Delhi's rows ----
print("Loading air quality data...")
air = pd.read_csv(os.path.join(RAW_DIR, "city_day.csv"), parse_dates=["Date"])
air = air[air["City"] == CITY].drop(columns=["City"])
air = air.rename(columns=COLUMN_NAMES)
air = air.sort_values("date").set_index("date")
print(f"{CITY}: {len(air)} rows, {air.index.min().date()} to {air.index.max().date()}")

assert not air.index.duplicated().any(), "Found more than one row for the same date"

# ---- 2. Give every calendar day its own row ----
# Lag features (step 4) assume the previous row is exactly yesterday. If a date
# were simply absent from the file, "yesterday" would silently become "the last
# day we have data for", which could be a week earlier. Reindexing turns every
# absent date into an explicit row of NaNs instead.
all_days = pd.date_range(air.index.min(), air.index.max(), freq="D", name="date")
inserted = len(all_days) - len(air)
air = air.reindex(all_days)
print(f"Calendar reindex: {inserted} absent dates added as empty rows")

# ---- 3. Missing-data report ----
pollutants = [c for c in air.columns if c not in ("aqi", "aqi_bucket")]

report = pd.DataFrame({
    "missing_frac": air.isna().mean().round(3),
    "missing_days": air.isna().sum(),
    "longest_gap_days": air.apply(longest_gap),
})
report.index.name = "column"
report["dropped"] = report.index.isin(pollutants) & (report["missing_frac"] > MAX_MISSING_FRAC)

print("\nMissing data (before any filling):")
print(report.to_string())
report.to_csv(os.path.join(METRICS_DIR, "missingness_delhi.csv"))

# ---- 4. Drop pollutants that are missing too often ----
to_drop = report.index[report["dropped"]].tolist()
for col in to_drop:
    print(f"\nDropping '{col}': {report.loc[col, 'missing_frac']:.1%} missing, "
          f"longest gap {report.loc[col, 'longest_gap_days']} days")
air = air.drop(columns=to_drop)
pollutants = [c for c in pollutants if c not in to_drop]

# ---- 5. Fill short gaps in pollutant columns ----
# Forward-fill copies the last known value forward in time, so a value filled
# in on day t only ever comes from day t-1, t-2 or t-3: information that
# already existed on day t. That is why it is safe to do this before the
# train/test split. interpolate() and bfill() would fill day t using days
# AFTER t, leaking the future into the past, so this project never uses them.
#
# AQI is deliberately NOT filled: it is the value we will predict, and a
# made-up target would only teach the model to reproduce our own guesses.
indicator_cols = []
for col in pollutants:
    if air[col].isna().any():
        # Record which days were originally missing BEFORE filling, so the
        # model can tell a real measurement from a copied one.
        indicator = f"{col}_was_missing"
        air[indicator] = air[col].isna().astype(int)
        indicator_cols.append(indicator)
        air[col] = air[col].ffill(limit=FFILL_LIMIT_DAYS)

# ---- 6. Join the weather ----
weather = pd.read_csv(
    os.path.join(PROCESSED_DIR, "weather_daily.csv"), parse_dates=["date"]
).set_index("date")

# Left join keeps every air-quality day. Step 1 guaranteed the weather file has
# no NaNs, so any NaN here can only mean a date failed to match.
daily = air.join(weather, how="left")
unmatched = daily[weather.columns].isna().any(axis=1).sum()
assert unmatched == 0, f"{unmatched} days have no matching weather row"

# Readable column order: target first, then pollutants, flags, weather.
daily = daily[["aqi", "aqi_bucket"] + pollutants + indicator_cols + list(weather.columns)]

# ---- 7. Final checks ----
day_steps = daily.index.to_series().diff().dropna()
assert (day_steps.dt.days == 1).all(), "Dates are not exactly one day apart"
assert daily["aqi"].isna().sum() == report.loc["aqi", "missing_days"], "AQI was modified"

still_missing = daily.isna().sum()
still_missing = still_missing[still_missing > 0]
print(f"\nStill missing after forward-fill (limit {FFILL_LIMIT_DAYS} days):")
print(still_missing.to_string())

# ---- 8. Save ----
out_path = os.path.join(PROCESSED_DIR, "delhi_daily.csv")
daily.reset_index().to_csv(out_path, index=False)

print(f"\nFinal table: {daily.shape[0]} days x {daily.shape[1]} columns")
print(f"  {len(pollutants)} pollutants: {', '.join(pollutants)}")
print(f"  {len(indicator_cols)} missing-value flags, {weather.shape[1]} weather columns")
print("\nSaved:")
print(f"  - {out_path}")
print(f"  - {METRICS_DIR}/missingness_delhi.csv")
