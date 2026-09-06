# Project 1: EDA and Simple Regression

This project provides an exploratory data analysis (EDA) and simple linear regression study across the three required UCI Machine Learning Repository datasets:

1. **Auto MPG**: 392 rows, 8 columns (target: `mpg`)
2. **Concrete Compressive Strength**: 1,030 rows, 9 columns (target: `compressive_strength_mpa`)
3. **Airfoil Self-Noise**: 1,503 rows, 6 columns (target: `scaled_sound_pressure_db`)

---

## Directory Structure

- `data/processed/`: Clean numeric CSV files for each dataset:
  - `auto_mpg.csv` (392 rows × 8 columns)
  - `concrete.csv` (1,030 rows × 9 columns)
  - `airfoil.csv` (1,503 rows × 6 columns)
- `data/raw/`: Raw source files from UCI repository.
- `results/`: Summary statistics, correlation matrices, and regression tables.
- `results/figures/`: High-resolution correlation heatmaps and observed-vs-fitted regression plots.
- `report/project1_report.tex`: Comprehensive LaTeX report with dedicated sections for each dataset.
- `analysis_statsmodels.py`: Python script performing preprocessing, statsmodels OLS regressions, plot generation, and LaTeX report authoring.

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
