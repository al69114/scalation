# Current model comparison

Current exported data; metrics averaged equally over locations after averaging NN seeds within each location. RMSE and MAE are in µg/m³; R² is dimensionless. Linear regression, mean, and persistence were recalculated. NN scores are the saved per-seed results; their summary aggregation, targets, and baseline predictions were checked. The NN was not retrained.

NN uses a chronological validation block and trains on the preceding portion; the baselines fit the full exported training period. The test observations match, but training budgets differ. The saved NN prediction file averages five seed predictions; its ensemble metrics are exported separately and must not be confused with mean per-seed scores.

the `nn_nolag` outputs describe the earlier data version and are not part of this table.

The `nn_nolag` outputs describe the earlier data version and are not part of this table.

| Dataset | Horizon | Model | R² | RMSE | MAE |
|---|---:|---|---:|---:|---:|
| beijing | 1h | Linear Regression | 0.952 | 17.799 | 9.974 |
| beijing | 1h | Mean | -0.003 | 81.781 | 59.625 |
| beijing | 1h | Neural Network | 0.954 | 17.500 | 9.923 |
| beijing | 1h | Persistence | 0.946 | 18.874 | 10.047 |
| beijing | 1h | Random Forest | 0.952 | 17.833 | 9.562 |
| beijing | 6h | Linear Regression | 0.637 | 49.335 | 31.050 |
| beijing | 6h | Mean | -0.003 | 81.948 | 59.560 |
| beijing | 6h | Neural Network | 0.642 | 48.966 | 30.069 |
| beijing | 6h | Persistence | 0.563 | 54.154 | 31.675 |
| beijing | 6h | Random Forest | 0.643 | 48.970 | 29.368 |
| beijing | 24h | Linear Regression | 0.234 | 71.020 | 50.429 |
| beijing | 24h | Mean | -0.003 | 81.357 | 59.464 |
| beijing | 24h | Neural Network | 0.225 | 71.461 | 50.076 |
| beijing | 24h | Persistence | -0.136 | 86.476 | 57.255 |
| beijing | 24h | Random Forest | 0.219 | 71.705 | 49.560 |
| global | 1h | Linear Regression | 0.912 | 2.189 | 1.411 |
| global | 1h | Mean | -0.082 | 8.122 | 6.593 |
| global | 1h | Neural Network | 0.898 | 2.357 | 1.580 |
| global | 1h | Persistence | 0.896 | 2.402 | 1.534 |
| global | 1h | Random Forest | 0.912 | 2.247 | 1.406 |
| global | 6h | Linear Regression | 0.308 | 6.213 | 4.700 |
| global | 6h | Mean | -0.083 | 8.126 | 6.599 |
| global | 6h | Neural Network | 0.316 | 6.367 | 4.744 |
| global | 6h | Persistence | 0.074 | 7.341 | 5.318 |
| global | 6h | Random Forest | 0.335 | 6.161 | 4.586 |
| global | 24h | Linear Regression | 0.012 | 7.476 | 5.692 |
| global | 24h | Mean | -0.087 | 8.146 | 6.617 |
| global | 24h | Neural Network | 0.032 | 7.491 | 5.686 |
| global | 24h | Persistence | -0.202 | 8.215 | 5.884 |
| global | 24h | Random Forest | -0.053 | 7.782 | 6.036 |
