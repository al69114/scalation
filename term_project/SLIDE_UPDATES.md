# Presentation updates after the new repository work

Copy-ready wording for the current implementation. The original datasets and 1/6/24-hour targets are unchanged.

## Dataset Overview

- **Global Air Quality:** Hourly measurements of six pollutants from Brasilia, Cairo, Dubai, London, New York, and Sydney during 2023.
- **Beijing Air Quality:** Hourly pollutant and weather measurements from 12 monitoring stations during March 2013–February 2017.
- **Prediction target:** PM2.5 concentration 1, 6, and 24 hours ahead.
- **Comparisons:** Model accuracy across forecast horizons and individual cities/stations, using a separate model for each location and horizon.

## Data Preprocessing and Forecasting

- Cleaned missing/invalid observations and encoded Beijing wind direction.
- Matched future PM2.5 targets and 1-, 2-, and 3-hour historical measurements by location and timestamp.
- Added a trailing 24-hour PM2.5 mean and cyclical hour/month features.
- Used chronological train/test periods with a 24-hour separation; NN tuning uses an additional chronological validation block.

Speaker notes: The trailing mean requires at least 12 observations in the 24-hour window. Current inputs total 14 features for Global and 35 for Beijing. Training/test rows are 39,102/13,104 and 236,547/79,830 respectively.

## Exploratory Data Analysis and Feature Importance

- PM2.5 distributions are right-skewed, and concentrations vary across locations and calendar periods.
- PM10 is strongly associated with current PM2.5; current PM2.5 is strongly associated with short-horizon future values.
- Added historical PM2.5 and time features to represent recent pollution patterns and calendar cycles.
- NN permutation importance identifies features the fitted models rely on; importance changes with forecast horizon.

Speaker notes: Replace the earlier heatmap with the refreshed `eda/figures/beijing_forecast_correlations.png`. Optionally use `results/nn/fig_permutation_importance.png` on a separate interpretation slide. Current PM2.5 has the largest average permutation importance at 1h and 6h in both datasets; pressure ranks highest at 24h in Beijing. These are model-specific test-set interpretations, not causal effects or a validated feature-selection procedure. If retaining “Feature Selection” in the title, label selection as pending. Global training covers January–September only.

## Modeling Techniques and Evaluation

- **Reference models:** Training-mean prediction and persistence (future PM2.5 equals current PM2.5).
- **Learned models:** Multiple linear regression, random forest, and a 3-layer MLP with one ReLU hidden layer.
- **NN tuning:** Hidden-layer size and weight decay, chronological validation, early stopping, and five random seeds.
- **Evaluation:** Test R², RMSE, and MAE for every location and forecast horizon.

Speaker notes: RF uses 100 trees, min_samples_leaf=5, max_features=0.5, and five random seeds; scores are averaged over seeds. Transformed regression was proposed but has not been implemented. RMSE/MAE have PM2.5 units; R² is dimensionless.

## Current Results — RMSE (µg/m³)

| Dataset | Horizon | Persistence | Linear regression | Random forest | Neural network |
|---|---:|---:|---:|---:|---:|
| Global | 1h | 2.402 | 2.189 | 2.247 | 2.357 |
| Global | 6h | 7.341 | 6.213 | 6.161 | 6.367 |
| Global | 24h | 8.215 | 7.476 | 7.782 | 7.491 |
| Beijing | 1h | 18.874 | 17.799 | 17.833 | 17.500 |
| Beijing | 6h | 54.154 | 49.335 | 48.970 | 48.966 |
| Beijing | 24h | 86.476 | 71.020 | 71.705 | 71.461 |

Footer: Equal-weight averages over locations; RF and NN metrics averaged over five seeds. Matching current test samples; NN reserves part of training for validation.

## Conclusions Supported by the Current Comparison

- Forecast errors increase with the forecast horizon in these experiments.
- Linear regression, random forest, and the NN all improve on persistence in average RMSE at all three horizons in both datasets.
- Linear regression has lower average RMSE than the NN for Global at all horizons; the NN has lower average RMSE for Beijing at 1h and 6h, with linear regression slightly lower at 24h.
- Random forest has the lowest average RMSE for Global at 6h but the highest of the three learned models for Global at 24h; in Beijing the three learned models are within about 1 µg/m³ at every horizon.
- No single learned model is best across all settings. These results compare known locations over time and do not establish generalization to unseen cities.

Source: `results/integration/CURRENT_RESULTS.md`. Random forest was rerun on the current data in the baseline notebook; full NN training was not rerun during integration; NN evidence and saved-score aggregation were checked, and simple baselines were recomputed.
