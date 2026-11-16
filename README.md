# Next-day AQI forecasting for Delhi

Predicts **tomorrow's** air quality index (AQI) for Delhi from **today's and
past** air-quality and weather data, then explains each forecast with SHAP.

- **Data:** Kaggle "Air Quality Data in India" (`city_day.csv`, Delhi,
  2015-01-01 to 2020-07-01) + Open-Meteo historical weather for Delhi.
- **Model:** XGBoost, compared against two baselines, Ridge regression and a
  random forest.
- **Headline:** test-period MAE **26.0** AQI points against **36.4** for
  "tomorrow = today" (28.6% lower), R² 0.91, with the CPCB category exactly
  right on 72% of days.
- **No data leakage, checked by code:** every feature is rebuilt with the
  future deleted, for every day, and must not change.

## How to run

Everything runs from this folder (`delhi-aqi-forecast/`), not from `scripts/`.

```bash
# one-time setup (Python 3.13)
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# the pipeline, in order
python scripts/01_download_data.py      # Kaggle + Open-Meteo -> data/raw/  (needs a Kaggle login)
python scripts/02_build_dataset.py      # clean daily Delhi table
python scripts/03_eda.py                # 8 EDA figures
python scripts/04_build_features.py     # features + target, with leakage checks
python scripts/05_train_eval.py         # train, tune, score, save model (~1 min)
python scripts/06_shap_analysis.py      # SHAP explanations
python scripts/07_covid_case_study.py   # frozen model vs the lockdown
python scripts/08_report_figures.py     # report-only figures + reports/report_numbers.csv
python scripts/09_existing_projects_comparison.py  # the usual GitHub approach vs this project

# one-day forecast with its SHAP explanation (viva demo)
python scripts/predict_day.py 2019-11-03

# the demo app
streamlit run app/streamlit_app.py
```

Step 1 downloads `city_day.csv` with `kagglehub`. If Kaggle authentication
fails, it prints how to download the file by hand. Re-running any step is safe:
downloads are cached and results are deterministic (fixed random seed).

## Project layout

```
scripts/
  common.py               split dates, AQI categories, scoring, plot style
  features.py             how every model input is built (shared with the app)
  01_download_data.py ... 09_existing_projects_comparison.py
  predict_day.py          one-day forecast + SHAP explanation (demo)
app/streamlit_app.py      3 tabs: explore a day, what-if, model card
data/raw/                 downloads (not committed; step 1 recreates them)
data/processed/           delhi_daily.csv, delhi_features.csv, weather_daily.csv
models/                   xgb_aqi.joblib, feature_names.json
outputs/figures/          22 figures (eda_*, model_*, shap_*, covid_*, report_*)
outputs/metrics/          every number quoted below, as CSV
reports/                  report_numbers.csv (extra numbers quoted in the report)
docs/
  final_report/           Delhi_AQI_Forecast_Final_Report (.docx, .pdf)
  milestone_2/            Milestone 2 report (.docx, .pdf), summary, drafts/
  template/               CSET312 mini-project report template
  viva/                   viva_prep.md, demo_script.md, project_analysis.md
  figures/                pipeline, use-case, module and ER diagrams
```

## Method

### Target and split

Each row is one day *t*. The features describe day *t* and earlier; the
target is the AQI of day *t + 1*.

| Period | Days being predicted | Rows | Used for |
| --- | --- | --- | --- |
| Train | 2015-01-16 to 2019-06-30 | 1,617 | fitting and tuning |
| Test | 2019-07-01 to 2020-03-21 | 265 | final scores, used once |
| COVID lockdown | 2020-03-22 to 2020-07-01 | 102 | regime-shift case study |

The split uses the date being **predicted** (`target_date`). Splitting on the
feature date instead would make the last training row's target the AQI of
2019-07-01, a test-period value.

The test period stops on 21 March 2020 so that the lockdown, which changed
Delhi's air overnight, is studied on its own rather than mixed into the test
scores.

### Features (56)

| Group | Features |
| --- | --- |
| Today's air | AQI and 11 pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, benzene, toluene) |
| Missing-value flags | 7 columns marking values carried forward from an earlier day |
| History | AQI 1, 2, 3 and 7 days ago; PM2.5 1-3 days ago; PM10 1 day ago |
| Change | AQI and PM2.5 change since yesterday |
| Rolling | AQI 3/7/14-day mean and 7-day std; PM2.5 7-day mean; 3-day wind and humidity means |
| Today's weather | temperature (mean/max/min), humidity, dew point, pressure, wind speed/gusts, rain, wind direction (sin/cos), boundary-layer height (mean, night-time min) |
| Calendar | day of year (sin/cos), day of week, weekend |
| Domain | stubble-burning season (15 Oct - 30 Nov), winter (Dec - Feb) |

Weather is fetched hourly in Indian local time and aggregated to days, since
that's how the AQI daily averages are defined. Boundary-layer height (the depth
of air pollution mixes into) only exists hourly. Wind direction is averaged as
a vector, because the plain mean of 350° and 10° would be 180°.

### Data cleaning

- Xylene is dropped (39% missing, including a single 606-day gap).
- Other pollutant gaps of up to 3 days are **forward-filled only**, with a
  flag column. Longer gaps stay empty: XGBoost handles them natively, and
  the scikit-learn models impute medians inside their pipelines.
- AQI itself is never filled in, since it is the target.

### Leakage safeguards

1. **Features use only the past, checked by code.** Step 4 rebuilds every
   feature with all later days deleted, once for each of the 2,009 days, and
   requires identical values. Four deliberately injected leaks were all
   caught: a future value, a centred rolling window, back-filling, and
   scaling by the whole dataset's mean.
2. **The target is checked by date.** `aqi_tomorrow` is matched against
   the next calendar day's AQI by date, not by row position.
3. **Nothing is filled from later days.** No `interpolate()` or `bfill()`
   is used anywhere, and step 4 checks that step 2 doesn't use them.
4. **Train and test don't overlap.** The split is by date and the script
   asserts there is no overlap.
5. **Tuning respects time order.** Hyperparameters are tuned with
   `TimeSeriesSplit` (5 folds, oldest to newest). Imputation and scaling sit
   inside the sklearn `Pipeline`, so they are fitted on training folds only.
6. **Leakage demo.** A copy of XGBoost given tomorrow's PM2.5 scores R² 0.98
   against 0.91 for the honest model. It shows what leakage looks like, and
   is discarded.

## Results

### Test period (265 days, 2019-07-01 to 2020-03-21)

| Model | CV MAE | MAE | RMSE | R² | MAPE % | Category right % | Up/down right % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence (tomorrow = today) | – | 36.43 | 51.12 | 0.83 | 17.08 | 64.53 | 1.51 |
| 7-day average | – | 55.57 | 76.89 | 0.63 | 27.07 | 47.92 | 52.08 |
| Ridge regression | 37.59 | 27.84 | 38.95 | 0.90 | 14.57 | 69.06 | 74.72 |
| Random forest | 43.79 | 27.02 | 37.12 | 0.91 | 14.17 | 73.96 | 72.08 |
| **XGBoost** | 43.79 | **26.02** | **36.75** | **0.91** | **13.15** | 72.45 | 72.83 |

Source: `outputs/metrics/model_comparison.csv`. XGBoost settings chosen by
cross-validation: `max_depth=3, learning_rate=0.03, n_estimators=300,
min_child_weight=5`, plus `subsample=0.8, colsample_bytree=0.8`.

What the table shows:

- **Persistence is a strong baseline.** AQI's day-to-day autocorrelation is
  0.89, so "tomorrow = today" already gets R² 0.83. It can never call a rise
  or a fall, though: 1.5% on up/down.
- **Every learned model clearly beats it,** cutting MAE by 24-29% and getting
  the direction right on about 3 days in 4.
- **Cross-validation favours Ridge** (37.6 against 43.8). The early folds
  train on a few hundred rows, then are validated on smog worse than anything
  they've seen. Trees can't predict beyond their training range; a linear
  model can extend its line. Once training passes about 1,000 rows (folds 4
  and 5), the trees match or beat Ridge. XGBoost was fixed as the headline
  model in advance, because the project explains it with SHAP. It was not
  picked for its test score.
- **The biggest misses come on sudden jumps.** For example, 2019-11-03 was
  forecast at 439 against an actual 659. Nothing in today's data sees an
  incoming smoke plume, so the forecast catches up a day late.
- **Category mistakes are almost always one step off** (see
  `model_04_category_confusion.png`).

### What drives the forecast (SHAP, test period)

| Rank | Feature | Mean \|SHAP\| (AQI points) |
| --- | --- | --- |
| 1 | `aqi` (today) | 36.84 |
| 2 | `pm25` (today) | 35.45 |
| 3 | `pm10` (today) | 9.35 |
| 4 | `aqi_roll3_mean` | 7.96 |
| 5 | `aqi_roll7_mean` | 7.67 |
| 6 | `precip_sum` | 7.26 |
| 7 | `pm25_change1` | 5.78 |
| 8 | `aqi_roll14_mean` | 3.45 |
| 9 | `wind_dir_sin` | 3.08 |
| 10 | `pm10_lag1` | 2.77 |

Source: `outputs/metrics/shap_importance.csv`. Points that are useful in a viva:

- **Today's level dominates,** as the autocorrelation suggests.
- **Rain washes the air.** Rainy days (>1 mm) push the forecast down by 11 on
  average; dry days push it up by 5.
- **Wind direction matters more than wind speed.** Westerly and
  north-westerly days push the forecast up (+2.1), and easterly days push it
  down (−3.6). That fits smoke arriving from crop burning in Punjab and
  Haryana, to Delhi's north-west.
- **Day of week barely matters,** matching the flat weekday box plot.
- **SHAP and XGBoost's built-in "gain" importance disagree below the top
  few.** Gain measures how much a feature's splits reduced training error.
  SHAP measures how much it actually moved the test forecasts.

### COVID lockdown: a regime shift

The model from step 5, unchanged, scored on the lockdown period:

| Period | Model | Mean actual AQI | MAE | MAPE % | Bias | R² |
| --- | --- | --- | --- | --- | --- | --- |
| Test (pre-COVID) | XGBoost | 228.69 | 26.02 | 13.15 | 3.49 | 0.91 |
| Test (pre-COVID) | Persistence | 228.69 | 36.43 | 17.08 | 0.11 | 0.83 |
| COVID lockdown | XGBoost | 125.99 | 20.69 | 18.51 | 11.50 | 0.46 |
| COVID lockdown | Persistence | 125.99 | 19.27 | 15.75 | 1.02 | 0.48 |

Source: `outputs/metrics/covid_regime.csv`. Mean AQI for 22 Mar to 1 Jul was
126 in 2020, 51% below the 2015-2019 average for the same dates (303, 264,
265, 233, 214).

- **Absolute error went down only because AQI halved.** Relative error went
  up, from 13% to 19% MAPE.
- **The model over-forecast,** with a bias of +11.5. It had learned a Delhi
  with normal traffic and industry.
- **Its 29% advantage over persistence disappeared.** Persistence was 7%
  better during lockdown. The patterns it learned stopped applying, which is
  what a distribution shift looks like.

## Compared with existing projects

Most AQI projects on GitHub that use this Kaggle dataset predict **today's**
AQI from **today's** pollutants with a random train/test split, and none of the
six checked compares against a baseline. CPCB computes AQI from those
pollutants, so that mostly re-learns the formula rather than forecasting.
Step 09 runs that approach on the same Delhi data with the same XGBoost
settings:

| Approach | Test days | MAE | R² |
| --- | --- | --- | --- |
| Existing method as published (same-day, random split) | 380 | 26.08 | 0.90 |
| The same method used as a next-day forecast | 265 | 30.59 | 0.90 |
| This project: persistence | 265 | 36.43 | 0.83 |
| **This project: next-day XGBoost** | 265 | **26.02** | **0.91** |

Source: `outputs/metrics/existing_projects_comparison.csv`. As a real
forecast, this project's error is 14.9% lower. The repositories checked are
listed in the final report (Section 3.7.8, references [31]–[36]).

## Limitations and future work

- **Today's weather only.** A real system would add tomorrow's weather
  *forecast*. That isn't leakage, since forecasts exist in advance, but it
  was outside this project's "today and past data only" brief.
- **One averaged value for the whole city.** The number and mix of
  monitoring stations changed over the years, and AQI for the same dates
  fell every year from 2015 to 2019. Part of that fall may come from the
  stations rather than the air.
- **No upwind information.** Satellite fire counts for Punjab and Haryana, or
  a festival calendar (Diwali), could help with the sudden jumps it currently
  misses.
- **Retraining.** The model doesn't adapt by itself when conditions change.
  Regular retraining, or monitoring for drift, would be needed in practice.

## Figures

| File | Shows |
| --- | --- |
| `eda_01` to `eda_08` | timeline with split, distribution, month and weekday patterns, correlations, autocorrelation, AQI against weather, missing data (training period only, except the timeline and missing data) |
| `model_01` to `model_05` | test forecast against actual, predicted vs actual, errors, category confusion matrix, model comparison |
| `shap_01` to `shap_05` | beeswarm, importance, dependence, waterfalls for a severe day and a clean day |
| `covid_01`, `covid_02` | frozen model through the lockdown; same-date AQI by year |
