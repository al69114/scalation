# Air-quality term project

The project forecasts PM2.5 at 1-, 6-, and 24-hour horizons using Global and Beijing air-quality data.

## Integrated updates — November 17, 2026

The local branch includes the six upstream commits through `de819d26`. See [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md) for changes and verification, and [SLIDE_UPDATES.md](SLIDE_UPDATES.md) for presentation wording.

- Inputs now include PM2.5 lags at 1, 2, and 3 hours, a trailing 24-hour mean, and cyclical hour/month features.
- Global has 14 predictors, 39,102 training rows, and 13,104 test rows. Beijing has 35 predictors, 236,547 training rows, and 79,830 test rows.
- `baseline_model.ipynb` implements mean, persistence, linear regression, and random forest per location/horizon.
- `3lnn_model.ipynb` implements a one-hidden-layer PyTorch MLP with chronological validation, tuning, five seeds, and permutation importance.
- [Current comparison](results/integration/CURRENT_RESULTS.md) combines recalculated simple baselines with checked saved NN results on the current test rows. It includes the current random-forest rerun integrated on November 18.

## Preprocessing

`data_processing.ipynb` contains the existing cleaning and chronological train/test export workflow. Its outputs are in `processed/`. Run that notebook from `term_project/` if regenerating the inputs. The EDA workflow does not change these files.

## Exploratory data analysis

- **Read the completed analysis:** [eda/EDA_REPORT.md](eda/EDA_REPORT.md).
- **Explore in Jupyter:** [eda.ipynb](eda.ipynb); run its cells in order. The notebook uses the same script and displays the generated figures and tables.
- **Presentation figures:** [eda/figures](eda/figures), with seven PNGs per dataset.
- **Numerical evidence:** [eda/tables](eda/tables), including descriptive statistics, location summaries, Pearson/Spearman correlations, within-location-centered correlations, future-target correlations, and time/calendar means with observation counts.

From the repository root:

```bash
python3 -m pip install -r term_project/requirements-eda.txt
python3 term_project/eda_analysis.py
```

The script also works from other working directories when invoked with its absolute path. Jupyter/IPython is additionally required only for the notebook viewer. Installed analysis-library versions are recorded in `eda/environment.json`.

The script reconstructs current measurements, lag/time features, timestamps, and location labels, then compares every input and target against all four training CSVs and both exported training-metadata files before analyzing them. Analytical plots use precisely those training rows, including the current preprocessing's complete-case and history restrictions and 24-hour boundary gap. Test X/Y files are never loaded by the EDA script. Full raw-data counts in the overview table describe cleaning only.

Global's training period is January–September 2023, so its monthly plot is an incomplete annual profile. Global hourly patterns use UTC rather than each city's local time. Beijing's monthly plot pools multiple training years. All calendar averages describe retained observations; counts are saved to show unequal coverage.

The EDA is descriptive. Run `python3 term_project/feature_selection.py` for the separate linear-regression feature-set comparison. It selects among pre-specified groups on chronological validation data within training. The NN permutation rankings summarize test-set model reliance. See [validation scores](results/feature_selection/validation_summary.csv) and [slide notes](SLIDE_UPDATES.md).

## Checking current model results

With scikit-learn installed, run `python3 term_project/review_current_results.py`. This recalculates mean, persistence, and linear-regression scores, checks saved NN targets/baselines and summary aggregation, and writes `results/integration/`. It does not retrain the NN or random forest. The original result files are preserved.
