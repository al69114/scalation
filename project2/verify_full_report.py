"""Recalculate Project 2 report results without changing the report or model outputs.

Requires numpy, pandas, statsmodels and scikit-learn. Run from any directory.
The PySR check evaluates the printed equations; it does not repeat their search.
"""
from pathlib import Path
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from verify_transformed_regression import verify

ROOT = Path(__file__).resolve().parents[1]


def check_regression_and_equation(key):
    data = pd.read_csv(ROOT / "data" / f"{key}.csv")
    x = sm.add_constant(data.iloc[:, :-1])
    y = data.iloc[:, -1]
    fit = sm.OLS(y, x).fit()
    print(f"\n{key}: full-data OLS")
    print(f"R2={fit.rsquared:.10f}, adjusted R2={fit.rsquared_adj:.10f}, "
          f"residual SE={np.sqrt(fit.mse_resid):.10f}, F={fit.fvalue:.10f}")
    print(pd.DataFrame({"coefficient": fit.params, "SE": fit.bse,
                        "t": fit.tvalues, "p": fit.pvalues}).to_string())
    _, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
    a = data.iloc[test, :-1].to_numpy().T
    if key == "auto_mpg":
        pred = 2149.9915 * (a[5] - 46.281776) / a[3]
    elif key == "concrete":
        cement, slag, fly, water, sp, _, _, age = a
        pred = (-0.200393 * water + np.sqrt(age * sp ** (1 / 16))
                + 1.201766 * np.sqrt(np.sqrt(fly) + slag - 2.197127 * slag / age)
                + 6.627109 * np.log(age)
                + 1.201766 * np.log(cement) ** 2
                * np.log(np.log(np.sqrt(cement))) ** 2)
    else:
        f, _, c, v, d = a
        pred = (-36.989017 * f / (f + (-195.3614 * d**2 * v + v)**2
                                + 128.70335 + 2.3002949 / (c * d))
                + 136.76428 - 530.60527 / (c * f * np.log(195.3614 * d / c)))
    actual = y.iloc[test].to_numpy()
    assert np.isfinite(pred).all()
    error = actual - pred
    print("Printed PySR equation on stated test split:")
    print(f"MSE={np.mean(error**2):.10f}, RMSE={np.sqrt(np.mean(error**2)):.10f}, "
          f"MAE={np.mean(abs(error)):.10f}, "
          f"R2={1 - error @ error / np.sum((actual - actual.mean())**2):.10f}")


def check_forward(key):
    data = pd.read_csv(ROOT / "data" / f"{key}.csv")
    split = pd.read_csv(ROOT / "project2/results" / f"{key}_transformed_split.csv")
    # The saved split records the same seed-42 Scala shuffle used by loadFS.
    shuffled = data.iloc[split.row_index.to_numpy()]
    x = sm.add_constant(shuffled.iloc[:, :-1]).to_numpy()
    y = shuffled.iloc[:, -1].to_numpy()
    n = len(y)
    # Mirrors Model.trSize, not the transformed-regression split size.
    n_train = n - int(np.floor(n * 0.2 + 1))
    selected = [0]
    print(f"\n{key}: forward selection, selection n={n_train}, full n={n}")
    print("step, added, R2 %, reported-formula adjusted R2 %, correct adjusted R2 %, sMAPE")
    best = None
    for step in range(1, x.shape[1]):
        _, j = max((sm.OLS(y[:n_train], x[:n_train, selected + [j]]).fit().rsquared_adj, j)
                   for j in range(1, x.shape[1]) if j not in selected)
        selected.append(j)
        fit = sm.OLS(y[:n_train], x[:n_train, selected]).fit()
        wrong_adj = 1 - (1 - fit.rsquared) * (n - 1) / (n - len(selected))
        smape = np.mean(200 * abs(fit.resid) / (abs(y[:n_train]) + abs(fit.fittedvalues)))
        print(f"{step}, {data.columns[j-1]}, {100*fit.rsquared:.3f}, "
              f"{100*wrong_adj:.3f}, {100*fit.rsquared_adj:.3f}, {smape:.3f}")
        if best is None or fit.rsquared_adj > best[0]:
            best = (fit.rsquared_adj, selected.copy())
    fit = sm.OLS(y, x[:, best[1]]).fit()
    print("Selected columns including intercept:", best[1])
    print("Full-data refit R2, adjusted R2, RMSE:", fit.rsquared,
          fit.rsquared_adj, np.sqrt(np.mean(fit.resid**2)))
    print("Refit coefficients:", fit.params)


def check_regularized_notebook():
    notebook = json.loads((ROOT / "project2/Project2_Regular_Regression.ipynb").read_text())
    previous = Path.cwd()
    env = {}
    try:
        os.chdir(ROOT / "project2")
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                code = "".join(cell["source"])
                if cell["cell_type"] == "code" and not code.lstrip().startswith("%"):
                    exec(compile(code, "project2_regular_notebook", "exec"), env)
    finally:
        os.chdir(previous)
    print("\nStatsmodels regularized notebook rerun:")
    for method in ("ridge", "lasso"):
        print(method, "alpha", env[f"best_{method}_alpha"], "RMSE", env[f"{method}_rmse"],
              "R2", env[f"{method}_r2"])
        print("CV RMSE:", env[f"{method}_cv_scores"])


if __name__ == "__main__":
    for key in ("auto_mpg", "concrete", "airfoil"):
        check_regression_and_equation(key)
    for key in ("auto_mpg", "concrete"):
        check_forward(key)
        verify(key)
    check_regularized_notebook()
