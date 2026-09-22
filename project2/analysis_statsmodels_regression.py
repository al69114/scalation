from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


datasets = [
    ("auto_mpg", DATA_DIR / "auto_mpg.csv", "mpg"),
    ("concrete", DATA_DIR / "concrete.csv", "compressive_strength_mpa"),
    ("airfoil", DATA_DIR / "airfoil.csv", "scaled_sound_pressure_db"),
]


for name, path, y_col in datasets:

    print(f"\n{'=' * 70}")
    print(f"Project 2 - Regression (statsmodels) - {name}")
    print(f"{'=' * 70}")

    df = pd.read_csv(path)

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    print(df.describe())

    summary_path = RESULTS_DIR / f"{name}_descriptive_statistics.txt"

    with open(summary_path, "w") as f:
        f.write(df.describe().to_string())

    correlation = df.corr(numeric_only=True)

    print("\nCorrelation matrix:")
    print(correlation)

    corr_path = RESULTS_DIR / f"{name}_correlation_matrix.csv"
    correlation.to_csv(corr_path)

    plt.figure(figsize=(10, 8))

    plt.imshow(correlation, aspect="auto")

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.index)),
        correlation.index
    )

    plt.title(f"{name}: Correlation Heatmap")
    plt.tight_layout()

    heatmap_path = PLOTS_DIR / f"{name}_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    predictor_correlations = (
        correlation[y_col]
        .drop(labels=[y_col])
        .abs()
        .sort_values(ascending=False)
    )

    top_two = predictor_correlations.head(2).index.tolist()

    print("\nTop two predictors based on absolute correlation with response:")

    for predictor in top_two:
        signed_corr = correlation.loc[predictor, y_col]
        print(
            f"  {predictor}: "
            f"correlation = {signed_corr:.6f}, "
            f"|correlation| = {abs(signed_corr):.6f}"
        )

    top_path = RESULTS_DIR / f"{name}_top_two_predictors.txt"

    with open(top_path, "w") as f:
        f.write("Top two predictors based on absolute correlation:\n\n")

        for predictor in top_two:
            signed_corr = correlation.loc[predictor, y_col]

            f.write(
                f"{predictor}: "
                f"correlation = {signed_corr:.6f}, "
                f"|correlation| = {abs(signed_corr):.6f}\n"
            )

    y = df[y_col]
    X = df.drop(columns=[y_col])

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("\nFull multiple regression:")
    print(model.summary())

    out_path = RESULTS_DIR / f"{name}_regression_statsmodels.txt"

    with open(out_path, "w") as f:
        f.write(model.summary().as_text())

    for predictor in top_two:

        print("\n" + "-" * 70)
        print(f"Simple Regression: {y_col} ~ {predictor}")
        print("-" * 70)

        x = df[[predictor]]

        x_with_constant = sm.add_constant(x)

        simple_model = sm.OLS(y, x_with_constant).fit()

        print(simple_model.summary())

        simple_path = (
            RESULTS_DIR /
            f"{name}_{predictor}_simple_regression.txt"
        )

        with open(simple_path, "w") as f:
            f.write(simple_model.summary().as_text())

        y_hat = simple_model.predict(x_with_constant)

        sorted_indices = x[predictor].argsort()

        x_sorted = x.iloc[sorted_indices][predictor]
        y_hat_sorted = y_hat.iloc[sorted_indices]

        plt.figure(figsize=(8, 6))

        plt.scatter(
            x[predictor],
            y,
            label="Actual y"
        )

        plt.plot(
            x_sorted,
            y_hat_sorted,
            label="Predicted y-hat"
        )

        plt.xlabel(predictor)
        plt.ylabel(y_col)

        plt.title(
            f"{name}: {y_col} vs. {predictor}"
        )

        plt.legend()
        plt.tight_layout()

        plot_path = (
            PLOTS_DIR /
            f"{name}_{predictor}_simple_regression.png"
        )

        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Plot saved to: {plot_path}")

print("\n" + "=" * 70)
print("Project 2 Statsmodels analysis complete.")
print("=" * 70)

