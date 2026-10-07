# Data Science Projects: ScalaTion, statsmodels and PySR

This repository contains Project 1 (EDA and simple regression) and Project 2 (regression and symbolic regression). The Project 2 submission and run instructions are below; the existing Project 1 documentation follows.

## Project 2: Regression and Symbolic Regression

Project 2 covers **20 required cases** using ScalaTion, statsmodels and PySR. The full report includes model setup, numerical results, evaluation, interpretation and symbolic expressions.

### Datasets

The Project 2 programs use the numeric CSV files in the repository's `data/` directory. Column counts include the target.

| Dataset | Rows | Columns | Target | CSV |
| --- | ---: | ---: | --- | --- |
| Auto MPG | 392 | 8 | Miles per gallon (`mpg`) | [auto_mpg.csv](data/auto_mpg.csv) |
| Concrete Compressive Strength | 1,030 | 9 | Compressive strength in MPa (`compressive_strength_mpa`) | [concrete.csv](data/concrete.csv) |
| Airfoil Self-Noise | 1,503 | 6 | Sound pressure level in decibels (`scaled_sound_pressure_db`) | [airfoil.csv](data/airfoil.csv) |

Auto MPG excludes the six rows with missing horsepower and the car-name field. These datasets originate from the UCI Machine Learning Repository.

### Required Models

| Method | Datasets | Tools | Cases |
| --- | --- | --- | ---: |
| Multiple linear regression | All three | ScalaTion and statsmodels | 6 |
| Ridge and Lasso regression | Auto MPG | ScalaTion and statsmodels | 4 |
| Forward feature selection | Auto MPG and Concrete | ScalaTion | 2 |
| Transformed regression | Auto MPG and Concrete | ScalaTion | 2 |
| Symbolic regression | All three | ScalaTion and PySR | 6 |
| **Total** | | | **20** |

Transformed regression evaluates log, square-root and Box–Cox transformations of the target, with predictions evaluated on the original target scale. Reciprocal transformation is also explored, including domain checks. Box–Cox tuning uses five-fold cross-validation within the training data.

ScalaTion symbolic regression constructs nonlinear basis terms and selects terms using training data. PySR searches for mathematical expressions. The report provides the resulting expressions, fit metrics and interpretations for both tools. ScalaTion regularized and symbolic workflows reserve test rows before fitting preprocessing or tuning; regularization leaves the intercept unpenalized. Forward-selection results include a full-data refit, and their validation is not nested feature-selection validation. Splits and procedures differ across tools, so cross-tool scores should be read with their stated evaluation setup.

### Report, Code and Results

- **Final report:** [project2_full_report.pdf](project2/report/project2_full_report.pdf).
- **LaTeX source:** [project2_full_report.tex](project2/report/project2_full_report.tex), with report images in [overleaf_images](project2/report/overleaf_images) and [overleaf_images.zip](project2/report/overleaf_images.zip).
- **ScalaTion models:** [regression](src/main/scala/scalation/modeling/Project2_Regression.scala), [Ridge/Lasso](src/main/scala/scalation/modeling/Project2_Regular_Regression.scala), [forward selection](src/main/scala/scalation/modeling/Project2_Forward.scala), [transformed regression](src/main/scala/scalation/modeling/Project2_Transformed_Regression.scala), and [symbolic regression](src/main/scala/scalation/modeling/Project2_Symbolic_Regression.scala).
- **statsmodels:** [OLS analysis script](project2/analysis_statsmodels_regression.py) and [Ridge/Lasso notebook](project2/Project2_Regular_Regression.ipynb).
- **PySR:** [symbolic regression notebook](project2/symbolic_regression-2.ipynb), including saved outputs for all three datasets.
- **Exported results:** [project2/results](project2/results), including [regularized results](project2/results/regularized), [symbolic results](project2/results/symbolic) and [plots](project2/results/plots). Regularized and symbolic exports include split assignments, coefficients, test predictions and evaluation metrics; symbolic coefficient tables also include training means and scales needed to evaluate the standardized expressions.

### Running Project 2

Run the commands from the repository root. Scala commands require a JDK and sbt; `build.sbt` specifies the Scala version. Some analyses open plot windows.

```bash
# Multiple linear regression: all three datasets
sbt "runMain scalation.modeling.Project2_Regression"

# Auto MPG Ridge and Lasso, including tuning and result exports
sbt "runMain scalation.modeling.project2RegularizedRegression"

# Auto MPG and Concrete forward selection
sbt "runMain scalation.modeling.project2_Forward"

# Auto MPG and Concrete transformed regression
sbt "runMain scalation.modeling.project2TransformedRegression"

# Symbolic regression: all three datasets
sbt "runMain scalation.modeling.project2SymbolicRegression"
```

For a single ScalaTion symbolic dataset, use `scalation.modeling.project2Symbolic` (Auto MPG), `scalation.modeling.project2ConcreteSymbolic`, or `scalation.modeling.project2AirfoilSymbolic` with `sbt "runMain ..."`.

Python analyses use NumPy, pandas, Matplotlib, statsmodels and scikit-learn. The notebooks also require Jupyter; the symbolic notebook requires PySR and its Julia backend.

```bash
# Generate statsmodels OLS results and plots
python3 project2/analysis_statsmodels_regression.py

# Open the regularized regression and PySR notebooks
jupyter notebook project2/Project2_Regular_Regression.ipynb
jupyter notebook project2/symbolic_regression-2.ipynb
```

Running the Scala export workflows or Python OLS script regenerates the corresponding result files. Repeating the PySR searches may produce different expressions, particularly for multithreaded runs; the report documents the saved runs.

### Verifying the Report Results

```bash
# Independently check ScalaTion symbolic and regularized exports
python3 project2/verify_corrected_results.py

# Recalculate OLS, forward-selection, transformed and statsmodels
# regularized results, and evaluate the printed PySR equations
python3 project2/verify_full_report.py
```

These checks compare saved results with independently calculated predictions, metrics and model fits. Evaluating the printed PySR expressions does not repeat the evolutionary searches.

---

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
