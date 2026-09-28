"""Plot the ScalaTion transformed-regression results.

Run the ScalaTion program first (it writes the CSVs this script reads):
    sbt "runMain scalation.modeling.project2TransformedRegression"
    python3 project2/plot_transformed_regression.py
"""

import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = BASE / "results"
PLOTS = RESULTS / "plots"

DATASETS = {
    "auto_mpg": ("Auto MPG", "MPG"),
    "concrete": ("Concrete", "Compressive strength (MPa)"),
}


def best_lambda(cv: pd.DataFrame) -> float:
    return float(cv.loc[cv["cv_rmse"].idxmin(), "lambda"])


def plot_dataset(key: str, title: str, ylabel: str) -> None:
    cv = pd.read_csv(RESULTS / f"{key}_boxcox_lambda_cv.csv")
    pred = pd.read_csv(RESULTS / f"{key}_transformed_predictions.csv")
    lam = best_lambda(cv)
    y, yp_reg, yp_bc = pred["y"], pred["yp_regression"], pred["yp_boxcox"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    valid = cv.dropna()
    ax = axes[0]
    ax.plot(cv["lambda"], cv["cv_rmse"], "o-", color="#1f77b4")
    ax.axvline(1.0, color="gray", linestyle=":", label=r"$\lambda=1$ (no transform)")
    ax.axvline(lam, color="#b2182b", linestyle="--", label=rf"best $\lambda={lam:g}$")
    ax.set_xlabel(r"Box-Cox $\lambda$")
    ax.set_ylabel("5-fold CV RMSE (original scale)")
    ax.set_title("Box-Cox $\\lambda$ tuning (training set)")
    if (valid["cv_rmse"].max() / valid["cv_rmse"].min()) > 5:
        ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    lo, hi = y.min(), y.max()
    ax.scatter(y, yp_reg, s=12, alpha=0.45, label="Regression")
    ax.scatter(y, yp_bc, s=12, alpha=0.45, label=rf"Box-Cox $\lambda={lam:g}$")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect fit")
    ax.set_xlabel(f"Actual {ylabel}")
    ax.set_ylabel(f"Predicted {ylabel}")
    ax.set_title("Actual vs. predicted (in-sample)")
    ax.legend()

    ax = axes[2]
    ax.scatter(yp_reg, y - yp_reg, s=12, alpha=0.45, label="Regression")
    ax.scatter(yp_bc, y - yp_bc, s=12, alpha=0.45, label=rf"Box-Cox $\lambda={lam:g}$")
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel(f"Fitted {ylabel}")
    ax.set_ylabel("Residual (original scale)")
    ax.set_title("Residuals vs. fitted")
    ax.legend()

    for a in axes:
        a.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"{title}: Transformed Regression (ScalaTion TranRegression)")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"{key}_transformed_regression.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    for key, (title, ylabel) in DATASETS.items():
        plot_dataset(key, title, ylabel)
