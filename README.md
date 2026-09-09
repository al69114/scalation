# Project 1: EDA and Simple Regression

This project provides an exploratory data analysis (EDA) and simple linear regression study across the three required UCI Machine Learning Repository datasets:

1. **Auto MPG**: 392 rows, 8 columns (target: `mpg`)
2. **Concrete Compressive Strength**: 1,030 rows, 9 columns (target: `compressive_strength_mpa`)
3. **Airfoil Self-Noise**: 1,503 rows, 6 columns (target: `scaled_sound_pressure_db`)

---

## Directory Structure

- `project1/data/processed/`: Clean numeric CSV files for each dataset:
  - `auto_mpg.csv` (392 rows × 8 columns)
  - `concrete.csv` (1,030 rows × 9 columns)
  - `airfoil.csv` (1,503 rows × 6 columns)
- `project1/data/raw/`: Raw source files from UCI repository.
- `project1/results/`: Summary statistics, correlation matrices, and regression tables.
- `project1/results/figures/`: High-resolution correlation heatmaps and observed-vs-fitted regression plots.
- `project1/report/full_report.tex`: Report containing both tools' fit reports and references to the heatmap and regression figures.
- `project1/report/project1_report.tex`: Comprehensive LaTeX report with dedicated sections for each dataset.
- `project1/analysis_statsmodels.py`: Python script performing preprocessing, statsmodels OLS regressions, plot generation, and LaTeX report authoring.

---

## Where to Find the Results

From the repository root, numerical results are in `project1/results/` and PNG images are in `project1/results/figures/`. The links below are relative to this README (at the repository root) and open the corresponding files.

### Dataset CSV Files

Use the processed CSV files for the EDA and regression analyses. They contain the numeric datasets after the documented preprocessing steps.

| Dataset | Processed analysis CSV | Raw source files |
| --- | --- | --- |
| Auto MPG | [project1/data/processed/auto_mpg.csv](project1/data/processed/auto_mpg.csv) | [auto-mpg.data](project1/data/raw/auto-mpg.data), [auto_mpg_raw.csv](project1/data/raw/auto_mpg_raw.csv) |
| Concrete Compressive Strength | [project1/data/processed/concrete.csv](project1/data/processed/concrete.csv) | [Concrete_Data.xls](project1/data/raw/Concrete_Data.xls), [concrete_raw.csv](project1/data/raw/concrete_raw.csv) |
| Airfoil Self-Noise | [project1/data/processed/airfoil.csv](project1/data/processed/airfoil.csv) | [airfoil_self_noise.dat](project1/data/raw/airfoil_self_noise.dat), [airfoil_raw.csv](project1/data/raw/airfoil_raw.csv) |

The processed files are the CSVs loaded by the ScalaTion workflow and analyzed by the regression scripts.

### Correlation Matrices and Heatmaps

Each CSV contains the full Pearson correlation matrix for all numeric variables, including the target. Open it in Excel or another spreadsheet viewer to inspect the values. Each heatmap PNG displays the same matrix with rounded values inside the cells.

| Dataset | Correlation matrix (CSV) | Heatmap (PNG) |
| --- | --- | --- |
| Auto MPG | [auto_mpg_correlation.csv](project1/results/auto_mpg_correlation.csv) | [auto_mpg_heatmap.png](project1/results/figures/auto_mpg_heatmap.png) |
| Concrete Compressive Strength | [concrete_correlation.csv](project1/results/concrete_correlation.csv) | [concrete_heatmap.png](project1/results/figures/concrete_heatmap.png) |
| Airfoil Self-Noise | [airfoil_correlation.csv](project1/results/airfoil_correlation.csv) | [airfoil_heatmap.png](project1/results/figures/airfoil_heatmap.png) |

`project1/report/full_report.tex` includes the annotated heatmaps in each dataset's "Correlation Analysis: Matrix and HeatMap" subsection. Immediately below each heatmap, a table ranks every predictor by absolute correlation with the target, shows an Absolute correlation column, and explains the selection of the top two predictors. A second table lists distinct predictor pairs with absolute correlation greater than 0.75, retaining the signs in its Correlation column, with an explanation of the result. If no pairs qualify, the table states this explicitly. These tables summarize selected entries of each full matrix; separate numeric tables of the complete pairwise matrices are not currently included in this report. The ScalaTion `project1EDA` command also prints the full matrices to the terminal.

### Regression Result Plots (PNG)

Each plot shows observed target values (`y`) and fitted values (`y-hat`) against one predictor (`x`). There are two plots per dataset, using the predictors with the largest absolute correlation with the target.

| Dataset | Predictor | Result plot (PNG) |
| --- | --- | --- |
| Auto MPG | Weight | [auto_mpg_weight_regression.png](project1/results/figures/auto_mpg_weight_regression.png) |
| Auto MPG | Displacement | [auto_mpg_displacement_regression.png](project1/results/figures/auto_mpg_displacement_regression.png) |
| Concrete Compressive Strength | Cement | [concrete_cement_regression.png](project1/results/figures/concrete_cement_regression.png) |
| Concrete Compressive Strength | Superplasticizer | [concrete_superplasticizer_regression.png](project1/results/figures/concrete_superplasticizer_regression.png) |
| Airfoil Self-Noise | Frequency | [airfoil_frequency_hz_regression.png](project1/results/figures/airfoil_frequency_hz_regression.png) |
| Airfoil Self-Noise | Suction displacement thickness | [airfoil_suction_displacement_thickness_m_regression.png](project1/results/figures/airfoil_suction_displacement_thickness_m_regression.png) |

These nine PNGs (three heatmaps and six regression plots) are generated by `analysis_statsmodels.py` using seaborn and Matplotlib, with regression fits from Statsmodels. The ScalaTion visualization commands below open interactive windows rather than saving these PNG files.

All result images are in [project1/results/figures](project1/results/figures):

- Heatmaps: `auto_mpg_heatmap.png`, `concrete_heatmap.png`, and `airfoil_heatmap.png`.
- Regression plots: `*_regression.png`, one for each selected dataset--predictor pair.

### Numerical Summary and Regression Results

| Dataset | Descriptive statistics (CSV) | Regression coefficients and fit metrics (CSV) |
| --- | --- | --- |
| Auto MPG | [auto_mpg_summary.csv](project1/results/auto_mpg_summary.csv) | [auto_mpg_simple_regression.csv](project1/results/auto_mpg_simple_regression.csv) |
| Concrete Compressive Strength | [concrete_summary.csv](project1/results/concrete_summary.csv) | [concrete_simple_regression.csv](project1/results/concrete_simple_regression.csv) |
| Airfoil Self-Noise | [airfoil_summary.csv](project1/results/airfoil_summary.csv) | [airfoil_simple_regression.csv](project1/results/airfoil_simple_regression.csv) |

[simple_regression_all.csv](project1/results/simple_regression_all.csv) combines the results for all six regressions.

---

## Running the Analyses

### 1. Python / Statsmodels Analysis
From the repository root, run:
```bash
python3 project1/analysis_statsmodels.py
```
*(or `python3 project1_eda/analysis_statsmodels.py`)*

### 2. ScalaTion Workflow
From the repository root, run:
```bash
sbt "runMain scalation.modeling.project1EDA"
```

The Scala program loads the datasets, prints statistical summaries, displays correlation matrices, fits `SimpleRegression` models for the top two predictors of each dataset, prints formatted summary statements, and demonstrates `Table.load` alongside `MatrixD.load` and `MatrixD.loadStr`.

Two additional entry points open ScalaTion's GUI visualizations for all three datasets:
```bash
sbt "runMain scalation.modeling.project1HeatMaps"          # correlation HeatMap per dataset
sbt "runMain scalation.modeling.project1RegressionPlots"   # y and y-hat vs. x plots for the top two predictors
```

---

## Data Loading Mechanisms in ScalaTion

- `MatrixD.load`: Reads numeric CSV matrices directly into `MatrixD`.
- `MatrixD.loadStr`: Converts text files with categorical string attributes to ordinal integers.
- `Table.load`: Reads CSV files into a relational `Table` object with schema typing.
