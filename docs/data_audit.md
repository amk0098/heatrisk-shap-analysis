# Data audit — source project heatrisk-deutschland

Read-only audit of [amk0098/heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland)
at commit `9974c0eb22b027d3a9ed66f89a7ecf3925d6b102`, done 11.11.2026.
Checks were run with pandas 3.0.5 on the raw DWD files shipped in that repo (`data/raw/*.txt`).

## Pipeline in the source project

| Step | Function | What it does |
|---|---|---|
| Load | `data_loader.load_station_data` | Reads `;`-separated DWD file, `-999` → NaN, keeps dates ≥ 1990-01-01 and the columns TMK, TXK, TNK, VPM, UPM |
| Combine | `data_loader.load_all_stations` | Concatenates the 5 cities, sorts by city and date |
| Target | `data_loader.add_target` | `heat_day_next` = 1 if the **next row's** TXK ≥ 30.0 within the same city; NA if next TXK is missing |
| Drop | `data_loader.remove_missing_targets` | Drops rows with missing target |
| Features | `features.add_lag_features` | `TXK_lag_1` = TXK of the previous row; `TXK_rolling_3` = **mean** of TXK over the 3 previous rows (t−1, t−2, t−3), both per city |
| Drop | `features.remove_missing_features` | Drops rows where the lag/rolling features are missing (first days of each city) |
| Split | `split.chronological_split` | Train ≤ 2020-12-31, validation 2021-01-01 … 2023-12-31, test ≥ 2024-01-01 |
| Preprocess | `models.py` pipelines | Mean imputation (+ scaling for LR), fitted inside the pipeline on training data only |
| Threshold | `evaluation.select_threshold` | Best F1 on validation among 0.20 … 0.50 |

## Checks

| Check | Result |
|---|---|
| `-999` removed? | **Yes.** No value ≤ −998 remains after loading (the leading spaces in the raw files do not break `na_values`). |
| Value ranges plausible? | Yes. TXK −12.8 … 40.3 °C, TNK −25.1 … 25.7 °C, UPM 25 … 100 %, VPM 1.2 … 24.9 hPa. |
| Date coverage | Berlin, Frankfurt, Hamburg, Köln: 1990-01-01 … 2025-12-31. **München: from 1992-05-17** (file starts there). |
| Gaps / duplicates | None. Every "previous row" is yesterday and every "next row" is tomorrow, so row-based `shift` equals calendar-day shift. |
| Target correct? | Yes. Spot check Berlin 2019-06-26: TXK 36.6, next day 26.0 → `heat_day_next` = 0. |
| Lags per city, past only? | Yes. `groupby("city")` + `shift(1)`; rolling uses `shift(1)` first, so today is not included. Spot check: rolling_3 on 2019-06-26 = (28.0 + 29.3 + 33.9) / 3 = 30.4. |
| Leakage | None found. Features use day t and earlier, target is day t+1, splits are chronological, imputation/scaling fitted on train. |
| Remaining missing values | TMK/TXK/TNK: 2 rows (train). VPM/UPM: 6 (train), 8 (validation), 26 (test). Must be imputed with training means. |

## Split sizes (after all drops)

| Split | Dates | Rows | Heat days | Rate |
|---|---|---:|---:|---:|
| Train | 1990-01-04 … 2020-12-31 | 55 543 | 1 682 | 3.03 % |
| Validation | 2021-01-01 … 2023-12-31 | 5 475 | 219 | 4.00 % |
| Test | 2024-01-01 … 2025-12-30 | 3 650 | 135 | 3.70 % |
| **All** | | 64 668 | 2 036 | 3.15 % |

Heat days per month (all years): May 69, June 424, July 781, August 687, September 73, April 2, other months 0.
June–August hold 1 892 of 2 036 heat days (93 %).

## Feature correlation (training period, Pearson)

The five temperature features correlate at 0.87–0.98 with each other (TMK–TXK 0.98, TXK_lag_1–TXK_rolling_3 0.97).
Vapour pressure (VPM) also correlates strongly with them (0.83–0.92).
UPM is negatively correlated with temperature (−0.29 … −0.55). This matters for the Shapley interpretation:
the attribution will be spread across the temperature features, and KernelSHAP will evaluate the model on
unrealistic combinations.

## Existing baseline results (test 2024–2025, from `outputs/model_comparison.csv`)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Always negative | 0.00 | 0.00 | 0.00 | 0.037 |
| Persistence (today ≥ 30 °C) | 0.415 | 0.415 | 0.415 | 0.194 |
| Seasonal (June–August) | 0.135 | 0.919 | 0.235 | 0.127 |
| Logistic Regression | 0.252 | 0.956 | 0.399 | 0.411 |
| Random Forest | 0.429 | 0.607 | 0.503 | 0.428 |

## Consequences for this project

1. The source loader, target, lag features and split are reused unchanged, so the MLP is comparable with the baselines above.
2. Missing values are imputed with training means (as in the source pipeline); standardisation uses training statistics.
3. `TXK_rolling_3` is the mean of the maximum temperatures of the three previous days, not a 3-day maximum.
4. The positive rate is ≈ 3 % (3.15 % overall, 3.7 % in the test period).
5. The München series starts in 1992.
