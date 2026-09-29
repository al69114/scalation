"""Independent Statsmodels checks of Project 2 symbolic regression.

Run: python project2/verify_symbolic_statsmodels.py
Needs numpy, pandas, scipy, statsmodels, scikit-learn; no sbt or PySR.
Keeps the existing verify_symbolic_regression.py and report unchanged.
"""
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split

from verify_symbolic_regression import REPORTED, N_FULL, SETUPS, design

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "project2/results"
DATA = ROOT / "data"


def interval(token):
    """Preserve the number of decimal places printed in the report."""
    d = Decimal(token)
    half = Decimal("0.5") * Decimal(10) ** d.as_tuple().exponent
    return float(d - half), float(d + half)


def overlaps(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def check_summary():
    print("A. Exact rounding-interval consistency (not proof of holdout independence)")
    matches = {}
    for key, (r, adj, rmse, rse, f) in REPORTED.items():
        found = []
        h = 0.00005  # all five summary quantities have four decimal places
        for n in range(3, N_FULL[key] + 1):
            p = np.arange(1, n - 1)
            df = n - p - 1
            # All three formulas must admit the SAME unrounded R-squared.
            lower = np.maximum.reduce([
                np.full(len(p), r - h),
                1 - (1 - (adj - h)) * df / (n - 1),
                (f - h) * p / (df + (f - h) * p)])
            upper = np.minimum.reduce([
                np.full(len(p), r + h),
                1 - (1 - (adj + h)) * df / (n - 1),
                (f + h) * p / (df + (f + h) * p)])
            # RMSE and RSE must admit the SAME unrounded SSE.
            valid = (lower <= upper) & (
                np.maximum(rmse - h, (rse - h) * np.sqrt(df / n)) <=
                np.minimum(rmse + h, (rse + h) * np.sqrt(df / n)))
            found.extend((n, int(k), n - int(k) - 1) for k in p[valid])
        matches[key] = found
        print(f"  {key}: feasible (n, predictors, residual df) = {found}")
    return matches


def check_coefficients(matches):
    print("\nB. Printed coefficient arithmetic, allowing rounding of BOTH estimate and SE")
    tex = (ROOT / "project2/report/project2_full_report.tex").read_text()
    section = tex.split(r"\section{Symbolic Regression}", 1)[1].split(r"\subsection{PySR}")[0]
    tables = re.findall(r"\\begin\{tabular\}\{lrrrr\}(.*?)\\end\{tabular\}", section, re.S)[1:]
    assert len(tables) == 3, "Expected three ScalaTion symbolic coefficient tables"
    for key, table in zip(REPORTED, tables):
        n, p, test_df = matches[key][0]
        train_df = N_FULL[key] - n - p - 1
        total = t_ok = normal_ok = train_t_ok = test_t_ok = 0
        for row in table.split(r"\\"):
            fields = [s.strip() for s in row.split("&")]
            if len(fields) != 5 or not re.fullmatch(r"-?\d+\.\d+", fields[1]):
                continue
            est, se, printed_t = map(interval, fields[1:4])
            quotients = [b / s for b in est for s in se]
            ratio = (min(quotients), max(quotients))
            compatible = overlaps(ratio, printed_t)
            total += 1
            t_ok += compatible
            if not compatible:
                print(f"    t mismatch: {fields[0]} {fields[1:4]}")
                continue
            lo, hi = max(ratio[0], printed_t[0]), min(ratio[1], printed_t[1])
            abs_lo = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            abs_hi = max(abs(lo), abs(hi))
            p_interval = (0, 0.001) if "<" in fields[4] else interval(fields[4])
            normal_ok += overlaps((2 * stats.norm.sf(abs_hi), 2 * stats.norm.sf(abs_lo)), p_interval)
            train_t_ok += overlaps((2 * stats.t.sf(abs_hi, train_df),
                                    2 * stats.t.sf(abs_lo, train_df)), p_interval)
            test_t_ok += overlaps((2 * stats.t.sf(abs_hi, test_df),
                                   2 * stats.t.sf(abs_lo, test_df)), p_interval)
        print(f"  {key}: t arithmetic {t_ok}/{total}; p compatibility: "
              f"normal {normal_ok}/{total}, training Student-t {train_t_ok}/{total}, "
              f"evaluation Student-t {test_t_ok}/{total}")
    print("  P-value compatibility does not verify SE calculation or post-selection inference.")


def score(y, prediction):
    e = np.asarray(y) - np.asarray(prediction)
    return (1 - e @ e / np.sum((y - np.mean(y)) ** 2),
            np.sqrt(np.mean(e**2)), np.mean(abs(e)))


def refit_recovered_terms():
    print("\nC. Independent Statsmodels refit of recovered ScalaTion terms and test rows")
    for key in ("auto_mpg", "concrete"):
        d = pd.read_csv(DATA / f"{key}.csv")
        prefix = RESULTS / "symbolic_audit"
        coefficients = pd.read_csv(prefix / f"{key}_selected_coefficients.csv")
        saved = pd.read_csv(prefix / f"{key}_selected_predictions.csv")
        columns = []
        for term in coefficients.term:
            if term == "one":
                columns.append(np.ones(len(d)))
            elif "^" in term:
                name, power = term.rsplit("^", 1)
                columns.append(d[name].to_numpy() ** float(power))
            else:
                columns.append(d[term].to_numpy())
        x = np.column_stack(columns)
        y = d.iloc[:, -1].to_numpy()
        test = saved.row_index.to_numpy()
        train = np.setdiff1d(np.arange(len(d)), test)
        np.testing.assert_array_equal(saved.y, y[test])
        # Column scaling is essential for inverse powers with huge coefficients.
        scales = np.linalg.norm(x[train], axis=0)
        fit = sm.OLS(y[train], x[train] / scales, hasconst=True).fit()
        prediction = fit.predict(x[test] / scales)
        np.testing.assert_allclose(prediction, saved.prediction, atol=1e-5, rtol=1e-6)
        print(f"  {key}: R2, RMSE, MAE = {score(y[test], prediction)}")
        print(f"    max absolute difference from Scala predictions: "
              f"{np.max(abs(prediction - saved.prediction)):.3g}")
        selection_n = len(d) - int(np.floor(0.2 * len(d) + 1))
        print(f"    final test rows also used by reconstructed feature selection: "
              f"{np.sum(test < selection_n)}/{len(test)}")
    print("  Airfoil exact refit unavailable: custom term and complete coefficients are missing.")


def independent_models():
    print("\nD. Independent models: split FIRST, select terms only on training data")
    print("  These are new models, not validation of the report's exact model.")
    rows = []
    for key, setup in SETUPS.items():
        d = pd.read_csv(DATA / f"{key}.csv")
        assert d.shape == (N_FULL[key], {"auto_mpg": 8, "concrete": 9, "airfoil": 6}[key])
        assert np.isfinite(d.to_numpy()).all()
        x, y = d.iloc[:, :-1], d.iloc[:, -1].to_numpy()
        z = design(x, **setup)
        train, test = train_test_split(np.arange(len(d)), test_size=0.2, random_state=42)
        # Fix the split before fitting any scaling or selecting any terms.
        mu, sd = z.iloc[train].mean(), z.iloc[train].std(ddof=0).replace(0, 1)
        a = sm.add_constant(((z - mu) / sd).to_numpy(), has_constant="add")
        chosen = [0]
        fit = sm.OLS(y[train], a[train][:, chosen], hasconst=True).fit()
        while len(chosen) < a.shape[1]:
            best = None
            for j in range(1, a.shape[1]):
                if j in chosen:
                    continue
                candidate_x = a[train][:, chosen + [j]]
                if np.linalg.matrix_rank(candidate_x) != len(chosen) + 1:
                    continue
                candidate = sm.OLS(y[train], candidate_x, hasconst=True).fit()
                if candidate.model.rank != len(chosen) + 1:
                    continue  # do not add algebraically redundant symbolic terms
                if best is None or candidate.rsquared_adj > best[0]:
                    best = candidate.rsquared_adj, j, candidate
            if best is None or best[0] <= fit.rsquared_adj + 1e-6:
                break
            chosen.append(best[1])
            fit = best[2]
        prediction = fit.predict(a[test][:, chosen])
        r2, rmse, mae = score(y[test], prediction)
        plain_x = sm.add_constant(x).to_numpy()
        plain = sm.OLS(y[train], plain_x[train]).fit()
        baseline = score(y[test], plain.predict(plain_x[test]))[0]
        print(f"  {key}: {len(chosen)-1} terms, n_test={len(test)}, R2={r2:.6f}, "
              f"RMSE={rmse:.6f}, MAE={mae:.6f}, plain OLS R2={baseline:.6f}")
        selected = [z.columns[j - 1] for j in chosen[1:]]
        rows.append(dict(dataset=key, test_n=len(test), terms=len(selected),
                         r2=r2, rmse=rmse, mae=mae, plain_r2=baseline,
                         selected_terms=";".join(selected)))
    pd.DataFrame(rows).to_csv(RESULTS / "symbolic_statsmodels_independent.csv", index=False)
    print("  Similar or higher accuracy does not prove the reported models are unbiased or correct.")


if __name__ == "__main__":
    output = StringIO()
    with redirect_stdout(output):
        matches = check_summary()
        check_coefficients(matches)
        refit_recovered_terms()
        independent_models()
    text = output.getvalue()
    (RESULTS / "symbolic_statsmodels_verification.txt").write_text(text)
    print(text, end="")
