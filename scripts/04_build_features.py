"""
04_build_features.py

Turns the clean daily table into a model-ready table. Each row is one day t:
the features use only day t and earlier, and the target is the AQI of day t+1.

Run from the project root, after 02_build_dataset.py:
    python scripts/04_build_features.py

Input:
  data/processed/delhi_daily.csv
Output:
  data/processed/delhi_features.csv
    date          day t, the day the features describe
    target_date   day t+1, the day being predicted (step 5 splits on this)
    ...features...
    aqi_tomorrow  the target: AQI on target_date

Three leakage checks run before anything is saved, and stop the script if
one fails:
  1. aqi_tomorrow really is the AQI of the next calendar day.
  2. Deleting the future doesn't change the past: features rebuilt from data
     that ends on some cutoff date must match the full-data features up to
     that date exactly. A feature that peeked even one day ahead would differ.
  3. The cleaning script (step 2) never fills a gap using later days.
"""

import os
import textwrap

import numpy as np
import pandas as pd

from common import LOCKDOWN_START, TEST_END, TRAIN_END
from features import build_features  # the feature definitions live there

# ---- CONFIG ----
# The longest look-back is 14 days, so the first 14 days of data don't have
# a full history behind them. They are dropped rather than half-filled.
WARMUP_DAYS = 14
# ----------------

PROCESSED_DIR = "data/processed"


# ---- 1. Load ----
daily = pd.read_csv(os.path.join(PROCESSED_DIR, "delhi_daily.csv"),
                    parse_dates=["date"]).set_index("date")
print(f"Loaded {len(daily)} days, {daily.index.min().date()} to {daily.index.max().date()}")

# ---- 2. Features and target ----
features = build_features(daily)

# A NEGATIVE shift moves every value one row UP, pulling tomorrow's AQI onto
# today's row. This is the only place in the project where data moves
# backwards in time, and it produces the label, never an input.
target = daily["aqi"].shift(-1)

# ---- 3. Leakage checks ----
print("\nLeakage checks:")

# Check 1. Look tomorrow up by calendar date (t + 1 day) rather than by row
# position, so this would also catch rows out of order or a skipped date.
next_day_aqi = daily["aqi"].reindex(daily.index + pd.Timedelta(days=1))
assert np.array_equal(target.to_numpy(), next_day_aqi.to_numpy(), equal_nan=True), \
    "aqi_tomorrow is not the AQI of the next calendar day"
print("  1. aqi_tomorrow equals the next calendar day's AQI on every row")

# Check 2. Rebuild the features pretending each day in turn is the last day
# we have (every later row deleted). If any feature used even one day after
# its own row, the two versions would disagree. Every day is a cutoff, not
# just a sample: some leaks, like back-filling a gap, only show up when the
# cutoff falls inside that gap. Takes a few seconds.
for cutoff in daily.index:
    past_only = build_features(daily.loc[:cutoff])
    pd.testing.assert_frame_equal(past_only, features.loc[:cutoff])
print(f"  2. Features unchanged when the future is deleted (all {len(daily)} days as cutoff)")

# Check 3. Step 2's forward-fill only copies values forward. Make sure no
# method that pulls values from later days ever crept into it.
with open("scripts/02_build_dataset.py") as f:
    cleaning_code = f.read()
for call in [".interpolate(", ".bfill(", ".backfill("]:
    assert call not in cleaning_code, f"02_build_dataset.py uses {call}"
print("  3. Step 2 never fills gaps from later days")

# ---- 4. Assemble the model table ----
table = features.copy()
table.insert(0, "target_date", table.index + pd.Timedelta(days=1))
table["aqi_tomorrow"] = target

rows_before = len(table)
first_full_day = daily.index.min() + pd.Timedelta(days=WARMUP_DAYS)
table = table.loc[first_full_day:]
rows_after_warmup = len(table)

# A row whose tomorrow has no AQI has nothing to learn from or score against.
# This also drops the very last day, whose tomorrow isn't in the data.
table = table[table["aqi_tomorrow"].notna()]

print(f"\nRows: {rows_before} days"
      f" - {rows_before - rows_after_warmup} warm-up days"
      f" - {rows_after_warmup - len(table)} days with no AQI tomorrow"
      f" = {len(table)} rows")

# ---- 5. Report ----
feature_cols = [c for c in table.columns if c not in ("target_date", "aqi_tomorrow")]
print(f"\n{len(feature_cols)} features:")
print(textwrap.fill(", ".join(feature_cols), width=88,
                    initial_indent="  ", subsequent_indent="  "))

nan_counts = table[feature_cols].isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
print("\nFeatures with missing values (handled inside the models in step 5):")
print(nan_counts.to_string())

# Preview of the split step 5 will make, by the date being PREDICTED.
target_dates = table["target_date"]
n_train = (target_dates <= TRAIN_END).sum()
n_test = ((target_dates > TRAIN_END) & (target_dates <= TEST_END)).sum()
n_covid = (target_dates >= LOCKDOWN_START).sum()
print(f"\nSplit by target_date: train {n_train} rows (up to {TRAIN_END}), "
      f"test {n_test} rows (to {TEST_END}), COVID {n_covid} rows (from {LOCKDOWN_START})")

print("\nSpot check (each row's aqi_tomorrow should be the next row's aqi):")
print(table.loc["2019-11-01":"2019-11-05", ["target_date", "aqi", "aqi_lag1", "aqi_tomorrow"]]
      .to_string())

# ---- 6. Save ----
out_path = os.path.join(PROCESSED_DIR, "delhi_features.csv")
table.reset_index().to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({len(table)} rows x {len(feature_cols)} features + target)")
