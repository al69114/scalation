"""Independently verify ScalaTion predictions, metrics, and training-only CV with NumPy.

Run after the Scala program: python3 project2/verify_transformed_regression.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "project2/results"


def predict(x_train, y_train, x_eval, kind, lam=1.0):
    if kind == "regression":
        target = y_train
    elif kind == "log" or (kind == "boxcox" and lam == 0):
        target = np.log(y_train)
    elif kind == "sqrt":
        target = np.sqrt(y_train)
    elif kind == "recip":
        target = 1 / y_train
    else:
        target = (y_train**lam - 1) / lam
    z = x_eval @ np.linalg.lstsq(x_train, target, rcond=None)[0]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if kind == "regression":
            return z
        if kind == "log" or (kind == "boxcox" and lam == 0):
            return np.exp(z)
        if kind == "sqrt":
            return np.where(z >= 0, z**2, np.nan)
        if kind == "recip":
            return np.where(z > 0, 1 / z, np.nan)
        base = lam * z + 1
        return np.where(base > 0, base ** (1 / lam), np.nan)


def metrics(y, yp):
    if not np.isfinite(yp).all():
        return [np.nan] * 3
    err = y - yp
    return [1 - err @ err / np.sum((y - y.mean()) ** 2),
            np.sqrt(np.mean(err**2)), np.mean(np.abs(err))]


def verify(key):
    data = pd.read_csv(ROOT / "data" / f"{key}.csv").to_numpy()
    x = np.column_stack([np.ones(len(data)), data[:, :-1]])
    y = data[:, -1]
    split = pd.read_csv(RESULTS / f"{key}_transformed_split.csv")
    train = split[split.split == "train"]
    test = split[split.split == "test"]
    tr, te = train.row_index.to_numpy(), test.row_index.to_numpy()
    assert len(set(tr) & set(te)) == 0
    assert sorted(split.row_index) == list(range(len(y)))
    assert len(te) == int(len(y) * 0.2)
    cv = pd.read_csv(RESULTS / f"{key}_boxcox_lambda_cv.csv")
    scores = []
    for lam in cv["lambda"]:
        errors = []
        for fold in sorted(train.cv_fold.unique()):
            fit = train.loc[train.cv_fold != fold, "row_index"].to_numpy()
            val = train.loc[train.cv_fold == fold, "row_index"].to_numpy()
            errors.extend(y[val] - predict(x[fit], y[fit], x[val], "boxcox", lam))
        errors = np.array(errors)
        scores.append(np.sqrt(np.mean(errors**2)) if np.isfinite(errors).all() else np.nan)
    np.testing.assert_allclose(scores, cv.cv_rmse, rtol=1e-7, atol=1e-7, equal_nan=True)
    lam = float(cv.loc[cv.cv_rmse.idxmin(), "lambda"])
    reported = pd.read_csv(RESULTS / f"{key}_transformed_metrics.csv")
    for label, fit_idx, eval_idx, suffix in [
        ("full_fit", np.arange(len(y)), np.arange(len(y)), "predictions"),
        ("test", tr, te, "test_predictions"),
    ]:
        saved = pd.read_csv(RESULTS / f"{key}_transformed_{suffix}.csv")
        np.testing.assert_array_equal(saved.y, y[eval_idx])
        if label == "test":
            np.testing.assert_array_equal(saved.row_index, te)
        rows = reported[reported.split == label].reset_index(drop=True)
        for i, kind in enumerate(["regression", "log", "sqrt", "recip", "boxcox"]):
            yp = predict(x[fit_idx], y[fit_idx], x[eval_idx], kind, lam)
            np.testing.assert_allclose(yp, saved[f"yp_{kind}"], rtol=1e-7, atol=1e-7, equal_nan=True)
            np.testing.assert_allclose(metrics(y[eval_idx], yp), rows.loc[i, ["r2", "rmse", "mae"]].to_numpy(dtype=float),
                                       rtol=1e-7, atol=1e-7, equal_nan=True)
            assert rows.loc[i, "invalid_predictions"] == (~np.isfinite(yp)).sum()
            assert rows.loc[i, "n"] == len(eval_idx)
    print(f"PASS {key}: split, 13 CV candidates, 5 models, full/test predictions and metrics; lambda={lam:g}")


if __name__ == "__main__":
    for dataset in ["auto_mpg", "concrete"]:
        verify(dataset)
