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

## Beijing EDA and feature selection

- The training median PM2.5 was 56.0 µg/m³ and the mean was 79.64 µg/m³. Occasional high readings produced a long right tail.
- PM10 had the strongest same-hour correlation with PM2.5 among the other pollutants (r = 0.88). Current PM2.5 correlated with future PM2.5 at 0.97, 0.76 and 0.41 for the 1-, 6- and 24-hour horizons.
- In a separate linear-regression feature-selection experiment, the combination of pollutants, recent PM2.5, time and weather had the lowest average validation RMSE at all three horizons. At 24 hours its score was 88.966 versus 88.977 µg/m³ using only current and historical PM2.5.
- Neural-network permutation importance ranked current PM2.5 first at 1 and 6 hours and atmospheric pressure first at 24 hours, averaged across stations.

[Beijing training correlations](eda/figures/beijing_forecast_correlations.png) · [Beijing neural-network importance](results/nn/fig_permutation_importance.png)

## Global EDA and feature selection

- The training median PM2.5 was 14.10 µg/m³ and the mean was 19.08 µg/m³. Training covered January through September 2023.
- PM10 had the strongest same-hour correlation with PM2.5 among the other pollutants (r = 0.89). Current PM2.5 correlated with future PM2.5 at 0.98, 0.82 and 0.77 for the three horizons.
- Linear regression had the lowest average validation RMSE with all 14 predictors at 1 and 6 hours. At 24 hours the five PM2.5 measurements performed best (8.42 versus 8.93 µg/m³ with all 14). These five are current PM2.5, the three hourly lags and the trailing 24-hour mean.
- Neural-network permutation importance ranked current PM2.5 first at 1 and 24 hours and hour of day first at 6 hours, averaged across cities.

[Global training correlations](eda/figures/global_forecast_correlations.png) · [Global neural-network importance](results/integration/global_feature_importance.png)

**Speaker notes for both datasets.** For linear regression we compared pre-specified groups of features. Each location contributed its last 15% of training observations for validation, with a 24-hour gap before validation. We selected the group with the lowest average validation RMSE across locations for each dataset and horizon. The five-model test comparison below uses the original model runs and full inputs for the learned models. Permutation importance describes the seed-0 neural network on the test period. [Feature-set validation results](results/feature_selection/validation_summary.csv)

## Modeling Techniques and Evaluation

- **Reference models:** Training-mean prediction and persistence (future PM2.5 equals current PM2.5).
- **Learned models:** Multiple linear regression, random forest, and a 3-layer MLP with one ReLU hidden layer.
- **NN tuning:** Hidden-layer size and weight decay, chronological validation, early stopping, and five random seeds.
- **Evaluation:** Test R², RMSE, and MAE for every location and forecast horizon.

Models were evaluated on the same later observations for each location and horizon. RMSE emphasizes larger misses, MAE measures the average absolute miss, and R² compares squared error with the variation around the test mean.

Speaker notes: RF uses 100 trees, min_samples_leaf=5, max_features=0.5, and five random seeds; scores are averaged over seeds. Transformed regression was proposed but has not been implemented. RMSE/MAE have PM2.5 units; R² is dimensionless.

## Current test results — RMSE (µg/m³)

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
- Linear regression has lower average RMSE than the NN for Global at all horizons; the NN has lower average RMSE for Beijing at 1h and 6h, with linear regression slightly lower at 24h. At Beijing 6h, NN and RF are nearly tied (48.966 and 48.970 µg/m³).
- Random forest has the lowest average RMSE for Global at 6h but the highest of the three learned models for Global at 24h; in Beijing the three learned models are within about 1 µg/m³ at every horizon.
- No single learned model is best across all settings. These results compare known locations over time and do not establish generalization to unseen cities.

Source: `results/integration/CURRENT_RESULTS.md`. Random forest was rerun on the current data in the baseline notebook; full NN training was not rerun during integration; NN evidence and saved-score aggregation were checked, and simple baselines were recomputed.
