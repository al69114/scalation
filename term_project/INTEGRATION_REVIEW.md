# Repository integration — November 17, 2026

Updated local `main` by fast-forward from `56a101bd` to `de819d26`, preserving the local EDA work and unrelated untracked `SimpleTransformer.scala`. No commits were created or pushed, and no Google Slides document was edited.

## What arrived

- Six commits: `f1e09d46`, `f793e554`, `221c6e4b`, `53e5ae58`, `c834e6a8`, `de819d26`.
- Preprocessing adds exact timestamp-matched PM2.5 lags (1/2/3 hours), a trailing mean over `(t−24h, t]` requiring at least 12 valid observations, and sine/cosine encodings for hour and month. These eight predictors increase Global from 6 to 14 and Beijing from 27 to 35 inputs.
- Lag-history requirements reduce training rows from 39,168 to 39,102 for Global and from 253,140 to 236,547 for Beijing. Global test rows remain 13,104; Beijing test rows fall from 86,342 to 79,830. Raw datasets and forecast horizons are unchanged.
- Both training location/timestamp files are now exported.
- Baseline notebook: per-location/per-horizon mean, persistence, linear regression, and random forest; five RF seeds.
- NN notebook: one ReLU hidden layer, Adam, training-only scaling, final 15% chronological validation with a 24-hour gap, early stopping, shared dataset-level tuning on representative locations, and five seeds. Models are fit separately for all 18 locations and three horizons.
- NN artifacts include scores, tuning tables, seed-averaged predictions, learning curves, residuals, and grouped permutation importance. `nn_nolag` retains results from the earlier dataset.
- Outside the term project: an earlier Project 3 Scala NN implementation was copied into the default source tree, a blank markdown cell was added to its 2L notebook, and tracked build artifacts changed. The Scala file uses Airfoil/Auto MPG/Concrete; it is not the new air-quality model. Adding `project3/` as an extra source directory may now introduce duplicate `NeuralNet3L` definitions and needs checking before using the old README command.

## Local fixes and integration

- Fixed the truncated `X_beij` entry in the preprocessing validation cell; it was a syntax error.
- Replaced the categorical inspection's `include="str"` with portable object/string dtype selection.
- Fixed the baseline demonstration's test mask to use `info_te`, not `info_tr`. The main baseline loop already used the correct mask.
- Cleared stale outputs only in modified code cells.
- Extended the EDA reconstruction to verify all eight new predictors and both training-metadata files. Regenerated all 14 figures, statistics, and the report on the new training rows. Future-PM2.5 heatmaps now also include lag/time features.
- Added `review_current_results.py`, current result summaries, and slide wording. These preserve all original modeling results.

## Evidence and remaining limits

The full corrected preprocessing notebook was run in a temporary directory and all 12 generated CSVs compared with the committed exports. All notebook code cells were syntax checked. EDA checks all training values and metadata before plotting. The result audit checks NN test timestamps/targets, mean/persistence predictions and scores, the five seeds, and the aggregation in `nn_summary.csv`. Simple baselines were refit on the current exported training data. Full NN tuning/training and random forest fitting were not repeated.

The historical baseline scores match the older no-lag experiment's mean/persistence scores. In particular, the Beijing test sample differs from the current NN sample, so putting the old RF numbers beside the new NN numbers would mix evaluation populations. `results/integration/CURRENT_RESULTS.md` supplies current mean/persistence/linear-regression/NN comparisons. Rerun the baseline notebook before including a current RF result. Do not claim a controlled lag ablation from `nn` versus `nn_nolag`: the test samples differ.

The NN uses only the fitting portion preceding its validation block; simple baselines use the full exported training period. Test rows match in the new comparison, but training budgets differ. Metrics are averaged over locations, not pooled over observations. NN summary scores average individual seed metrics; saved `nn_pred` values average predictions, whose ensemble metrics are different and are exported separately.

Permutation importance uses the test set and the seed-0 model. It is post-fit interpretation, not validated feature selection, and correlated predictors complicate attribution. A final feature subset still requires training/validation-only selection and a genuinely untouched evaluation period if choices are informed by test results.

The proposal's transformed regression remains absent. Random forest is now an implemented additional nonlinear model; the presentation should explicitly reflect this methodological change rather than imply transformed regression was run.
