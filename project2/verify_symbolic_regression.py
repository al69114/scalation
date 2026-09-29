"""Check the ScalaTion symbolic-regression numbers reported in the Project 2 report.

The original ScalaTion symbolic-regression program is not in the repository, so its
results cannot be re-run exactly. This script checks them in three independent ways:

  A. Internal consistency: R^2, adjusted R^2, RMSE, residual SE, and F must agree with
     one another; together they also reveal the evaluation sample size n and term count k.
  B. Coefficient tables: every t must equal estimate / SE, and every p must match t on
     the implied degrees of freedom.
  C. Plausibility: an independent ScalaTion-style symbolic regression (power terms plus
     cross terms, greedy forward selection by training adjusted R^2, scored on a held-out
     20% test set) shows whether the reported accuracy is in a realistic range.

Run from the repository root:  python3 project2/verify_symbolic_regression.py
"""

from decimal import Decimal
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Numbers exactly as printed in the report
# ---------------------------------------------------------------------------
REPORTED = {  # dataset: (R2, adjR2, RMSE, residual SE, F)
    "auto_mpg": (0.8827, 0.8328, 2.6101, 3.1370, 17.6756),
    "concrete": (0.8180, 0.8015, 6.9646, 7.2904, 49.6891),
    "airfoil":  (0.6826, 0.6321, 3.9647, 4.2752, 13.5314),
}

TABLES = {  # dataset: [(term, estimate, SE, t, p)]  (p = None means "< 0.001")
    "auto_mpg": [
        ("Intercept", -1016807.1951, 517225.1602, -1.97, 0.0493),
        ("Weight^-1", 908334.5295, 426892.0794, 2.13, 0.0334),
        ("Model year^2", -10.1170, 4.9912, -2.03, 0.0427),
        ("sqrt(model year)", 83706.6823, 42238.6395, 1.98, 0.0475),
        ("Acceleration^-1", 808.9958, 331.6133, 2.44, 0.0147),
        ("Displacement^-2", 129732.1150, 56783.3200, 2.28, 0.0223),
        ("Weight^-2", -626524289.4326, 262993286.5486, -2.38, 0.0172),
        ("Acceleration^-2", -3554.8823, 1663.9678, -2.14, 0.0326),
        ("Acceleration^2", 0.02863, 0.01384, 2.07, 0.0386),
        ("Cylinders^2", 1.1834, 0.2464, 4.80, None),
        ("Cylinders^-2", -765.5316, 127.7204, -5.99, None),
        ("sqrt(cylinders)", -114.1268, 21.5207, -5.30, None),
    ],
    "concrete": [
        ("sqrt(age)", 6.6596, 0.2074, 32.11, None),
        ("Superplasticizer", -1.6063, 0.4672, -3.44, None),
        ("Cement^2", 0.000105, 0.000007, 14.14, None),
        ("Age", -0.2441, 0.0119, -20.52, None),
        ("Slag^2", 0.000142, 0.000022, 6.30, None),
        ("sqrt(water)", 564.3547, 79.3692, 7.11, None),
        ("sqrt(fine aggregate)", -2.1050, 0.3517, -5.99, None),
        ("sqrt(superplasticizer)", 6.5834, 1.2153, 5.42, None),
        ("Water", -32.4357, 4.4639, -7.27, None),
        ("Water^2", 0.03036, 0.00414, 7.34, None),
    ],
    "airfoil": [
        ("d*c*f", 0.7788, 0.1640, 4.75, None),
        ("c*f", -0.0127, 0.0016, -7.81, None),
        ("a^2", 0.1499, 0.0302, 4.96, None),
        ("v*c*f", 0.000171, 0.000027, 6.34, None),
        ("custom term", -2.2738, 0.2685, -8.47, None),
        ("c*a*f", -0.00159, 0.00020, -8.09, None),
        ("d*v*a", -0.6896, 0.2485, -2.78, 0.0055),
        ("d*f", -0.1888, 0.0456, -4.14, None),
        ("d*a*f", 0.00714, 0.00192, 3.71, None),
        ("v*c*a", 0.1275, 0.0352, 3.62, None),
        ("a^0.5", -24.9059, 7.6452, -3.26, 0.0011),
        ("f", 0.00577, 0.00142, 4.07, None),
        ("v*c", -0.7778, 0.1716, -4.53, None),
        ("v^0.5", -3.5519, 0.8407, -4.22, None),
        ("a", 280.2162, 124.8255, 2.24, 0.0248),
        ("d^0.5", 35.0415, 10.3336, 3.39, None),
        ("d*v", 18.7152, 5.5923, 3.35, None),
        ("v*a", -0.01508, 0.00450, -3.35, None),
        ("d*v*f", -0.000908, 0.000452, -2.01, 0.0445),
        ("d*c", -6170.3137, 2631.6606, -2.34, 0.0190),
        ("log(1+f)", 8.6614, 1.8632, 4.65, None),
        ("sqrt(f)", -0.9257, 0.2086, -4.44, None),
    ],
}

TARGET = {"auto_mpg": "mpg", "concrete": "compressive_strength_mpa",
          "airfoil": "scaled_sound_pressure_db"}
N_FULL = {"auto_mpg": 392, "concrete": 1030, "airfoil": 1503}


def check_consistency():
    """A: infer (n, k) from the reported metrics and test that they agree."""
    print("=" * 78 + "\nA. Internal consistency of the reported ScalaTion metrics\n" + "=" * 78)
    implied = {}
    for name, (r2, adj, rmse, rse, f) in REPORTED.items():
        # RSE^2 / RMSE^2 = n / df and F = (R2 / k) / ((1 - R2) / df) with k = n - df - 1
        # give a closed form for the evaluation sample size n.
        r, a = (rse / rmse) ** 2, r2 / (1 - r2)
        n = round(f / (f * (1 - 1 / r) - a / r))
        df = round(n / r)
        k = n - df - 1
        adj_calc = 1 - (1 - r2) * (n - 1) / df
        f_calc = (r2 / k) / ((1 - r2) / df)
        ok = abs(adj_calc - adj) < 0.0015 and abs(f_calc - f) / f < 0.02
        implied[name] = (n, k, df)
        share = n / N_FULL[name]
        print(f"{name:9s} implied n = {n:4d} ({share:.0%} of data), terms k = {k:3d}, df = {df:4d} | "
              f"adj R2 {adj_calc:.4f} vs {adj:.4f}, F {f_calc:.2f} vs {f:.2f} -> "
              f"{'CONSISTENT' if ok else 'INCONSISTENT'}")
    print("-> n is about 20% of each dataset, so the reported metrics are held-out TEST-set values.\n")
    return implied


def check_tables(implied):
    """B: t = estimate / SE and p = 2 * P(T_df > |t|) for every reported coefficient."""
    print("=" * 78 + "\nB. Coefficient tables: t = estimate / SE and p from t\n" + "=" * 78)
    for name, rows in TABLES.items():
        n_test, k, _ = implied[name]
        df = (N_FULL[name] - n_test) - k - 1       # coefficients come from the TRAINING fit
        bad = []
        for term, est, se, t, p in rows:
            half = 0.5 * 10.0 ** Decimal(repr(se)).as_tuple().exponent   # SE rounding error
            t_hi, t_lo = est / max(se - half, 1e-300), est / (se + half)
            t_ok = min(t_lo, t_hi) - 0.01 <= t <= max(t_lo, t_hi) + 0.01
            t_calc = est / se
            p_calc = 2 * stats.t.sf(abs(t), df)
            p_ok = (p_calc < 0.001) if p is None else abs(p_calc - p) <= max(0.0015, 0.15 * p)
            if not (t_ok and p_ok):
                bad.append(f"{term}: t {t} vs {t_calc:.2f}, p {p if p is not None else '<0.001'} vs {p_calc:.4f}")
        status = "all consistent" if not bad else f"{len(bad)} mismatch(es)"
        print(f"{name:9s} {len(rows):2d} rows checked (training df = {df}): {status}")
        for b in bad:
            print("   ", b)
    print()


def design(X, powers, cross, cross3, logs):
    """Candidate symbolic terms, mirroring ScalaTion SymbolicRegression (x_j^p, x_i x_j, x_i x_j x_k)."""
    cols = {}
    for c in X.columns:
        v = X[c].to_numpy(float)
        cols[c] = v
        for p in powers:
            if p < 0 and (v <= 0).any():
                continue                           # negative power undefined at zero
            cols[f"{c}^{p}"] = v ** p
        if logs:
            cols[f"log1p({c})"] = np.log1p(v)
    names = list(X.columns)
    if cross:
        for a, b in combinations(names, 2):
            cols[f"{a}*{b}"] = X[a].to_numpy(float) * X[b].to_numpy(float)
    if cross3:
        for a, b, c in combinations(names, 3):
            cols[f"{a}*{b}*{c}"] = X[a].to_numpy(float) * X[b].to_numpy(float) * X[c].to_numpy(float)
    return pd.DataFrame(cols)


def fit(Z, y):
    A = np.column_stack([np.ones(len(y)), Z])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b


def predict(Z, b):
    return np.column_stack([np.ones(len(Z)), Z]) @ b


def forward_select(Ztr, ytr):
    """Greedy forward selection maximizing TRAINING adjusted R^2 (standardized for stability)."""
    mu, sd = Ztr.mean(0), Ztr.std(0)
    sd[sd == 0] = 1
    Zs = (Ztr - mu) / sd
    n, sst = len(ytr), ((ytr - ytr.mean()) ** 2).sum()
    chosen, best_adj = [], -np.inf
    while True:
        cand = None
        for j in range(Zs.shape[1]):
            if j in chosen:
                continue
            cols = chosen + [j]
            e = ytr - predict(Zs[:, cols], fit(Zs[:, cols], ytr))
            adj = 1 - (e @ e / sst) * (n - 1) / (n - len(cols) - 1)
            if cand is None or adj > cand[0]:
                cand = (adj, j)
        if cand is None or cand[0] <= best_adj + 1e-6:
            return chosen
        best_adj = cand[0]
        chosen.append(cand[1])


SETUPS = {  # powers / cross terms taken from the term types shown in each reported table
    "auto_mpg": dict(powers=[-2, -1, 0.5, 2], cross=True, cross3=False, logs=False),
    "concrete": dict(powers=[0.5, 2], cross=True, cross3=False, logs=False),
    "airfoil":  dict(powers=[0.5, 2], cross=True, cross3=True, logs=True),
}


def check_plausibility():
    """C: independent symbolic regression with forward selection, scored on a 20% test set."""
    print("=" * 78 + "\nC. Independent ScalaTion-style symbolic regression (test-set metrics)\n" + "=" * 78)
    print(f"{'dataset':9s} {'terms':>5s} {'test R2':>8s} {'RMSE':>7s} | {'reported R2':>11s} {'RMSE':>7s} | "
          f"{'plain OLS R2':>12s}")
    for name, cfg in SETUPS.items():
        d = pd.read_csv(DATA / f"{name}.csv")
        X, y = d.drop(columns=TARGET[name]), d[TARGET[name]].to_numpy(float)
        Z = design(X, **cfg).to_numpy()
        Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.2, random_state=42)
        cols = forward_select(Ztr, ytr)
        yp = predict(Zte[:, cols], fit(Ztr[:, cols], ytr))
        r2 = 1 - ((yte - yp) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
        rmse = np.sqrt(((yte - yp) ** 2).mean())
        Xtr, Xte = train_test_split(X.to_numpy(float), test_size=0.2, random_state=42)
        yl = predict(Xte, fit(Xtr, ytr))
        r2_lin = 1 - ((yte - yl) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
        rep = REPORTED[name]
        print(f"{name:9s} {len(cols):5d} {r2:8.4f} {rmse:7.3f} | {rep[0]:11.4f} {rep[2]:7.3f} | {r2_lin:12.4f}")
    print("-> Different splits and search settings mean the numbers will not match exactly;\n"
          "   the check is whether the reported accuracy is in the same range.")


if __name__ == "__main__":
    implied = check_consistency()
    check_tables(implied)
    check_plausibility()
