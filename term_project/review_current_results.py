"""Check saved NN evidence and build comparisons on the current exported test rows.

Does not retrain the NN/random forest or overwrite the team's historical results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/integration"


def score(y, p):
    return dict(r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=mean_absolute_error(y, p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(ROOT / "results/nn/nn_all_runs.csv")
    pred = pd.read_csv(ROOT / "results/nn/nn_test_predictions.csv")
    rows, ensembles, counts = [], [], []
    for key, loc_col in [("global", "City"), ("beijing", "station")]:
        xtr, xte, ytr, yte = [pd.read_csv(ROOT / f"processed/{v}_{key}_{s}.csv")
                             for v, s in [("X", "train"), ("X", "test"), ("Y", "train"), ("Y", "test")]]
        itr, ite = [pd.read_csv(ROOT / f"processed/{key}_{s}_info.csv") for s in ["train", "test"]]
        counts.append(dict(dataset=key, train_rows=len(xtr), test_rows=len(xte), features=xtr.shape[1]))
        assert len(xtr) == len(ytr) == len(itr) and len(xte) == len(yte) == len(ite)
        assert len(pred.loc[pred.dataset == key]) == 3 * len(xte)
        for loc in sorted(itr[loc_col].unique()):
            tr, te = itr[loc_col] == loc, ite[loc_col] == loc
            for h in [1, 6, 24]:
                target = f"PM25_{h}hr_ahead"
                actual = yte.loc[te, target].to_numpy()
                evidence = pred.loc[(pred.dataset == key) & (pred.location == loc) & (pred.horizon == h)].copy()
                evidence["Date"] = pd.to_datetime(evidence.Date)
                evidence = evidence.set_index("Date").reindex(pd.to_datetime(ite.loc[te, "Date"]))
                assert len(evidence) == len(actual) and not evidence.isna().any().any()
                np.testing.assert_allclose(evidence.actual, actual, rtol=1e-12, atol=1e-12)
                predictions = {
                    "Mean": np.full(len(actual), ytr.loc[tr, target].mean()),
                    "Persistence": xte.loc[te, "PM2.5"].to_numpy(),
                    "Linear Regression": LinearRegression().fit(xtr.loc[tr], ytr.loc[tr, target]).predict(xte.loc[te]),
                }
                for model, p in predictions.items():
                    metrics = score(actual, p)
                    rows.append(dict(dataset=key, location=loc, horizon=h, model=model, seed=0, **metrics))
                    if model in ["Mean", "Persistence"]:
                        col = "mean_pred" if model == "Mean" else "persistence_pred"
                        np.testing.assert_allclose(evidence[col], p, rtol=1e-12, atol=1e-12)
                        saved = runs.loc[(runs.dataset == key) & (runs.location == loc) &
                                         (runs.horizon == h) & (runs.model == model)]
                        assert len(saved) == 1
                        np.testing.assert_allclose(saved[["r2", "rmse", "mae"]].iloc[0],
                                                   list(metrics.values()), rtol=1e-10, atol=1e-10)
                ensembles.append(dict(dataset=key, location=loc, horizon=h,
                                      **score(actual, evidence.nn_pred)))
    simple = pd.DataFrame(rows)
    simple.to_csv(OUT / "current_baselines.csv", index=False)
    nn = runs.loc[runs.model == "Neural Network", simple.columns]
    assert len(nn) == 18 * 3 * 5
    assert nn.groupby(["dataset", "location", "horizon"]).seed.apply(lambda s: set(s) == set(range(5))).all()
    combined = pd.concat([simple, nn], ignore_index=True)
    # Average seeds within locations first, then give each location equal weight.
    per_location = combined.groupby(["dataset", "location", "horizon", "model"])[["r2", "rmse", "mae"]].mean()
    summary = per_location.groupby(["dataset", "horizon", "model"]).mean().reset_index()
    summary.to_csv(OUT / "current_model_comparison.csv", index=False)
    pd.DataFrame(ensembles).to_csv(OUT / "nn_ensemble_metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "current_data_dimensions.csv", index=False)
    # Saved NN summary is a mean of per-seed scores, not a score of mean predictions.
    saved_summary = pd.read_csv(ROOT / "results/nn/nn_summary.csv")
    recomputed = runs.groupby(["dataset", "location", "horizon", "model"])[["r2", "rmse", "mae"]].mean().sort_index()
    saved_summary = saved_summary.set_index(["dataset", "location", "horizon", "model"]).sort_index()
    np.testing.assert_allclose(saved_summary[["r2", "rmse", "mae"]], recomputed, rtol=1e-10, atol=1e-10)
    historical = pd.read_csv(ROOT / "baseline_results.csv")
    historical.groupby(["dataset", "horizon", "model"])[["r2", "rmse", "mae"]].mean().to_csv(OUT / "historical_baseline_summary.csv")
    lines = ["# Current model comparison", "",
             "Current exported data; metrics averaged equally over locations after averaging NN seeds within each location. RMSE and MAE are in µg/m³; R² is dimensionless. Linear regression, mean, and persistence were recalculated. NN scores are the saved per-seed results; their summary aggregation, targets, and baseline predictions were checked. The NN was not retrained.", "",
             "NN uses a chronological validation block and trains on the preceding portion; the baselines fit the full exported training period. The test observations match, but training budgets differ. The saved NN prediction file averages five seed predictions; its ensemble metrics are exported separately and must not be confused with mean per-seed scores.", "",
             "The original `baseline_results.csv` and `nn_nolag` outputs describe the earlier data version. Their Beijing test row counts differ from the current NN run, so the old Random Forest scores are excluded from this table. Rerun the baseline notebook for a current Random Forest comparison.", "",
             "| Dataset | Horizon | Model | R² | RMSE | MAE |", "|---|---:|---|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.dataset} | {r.horizon}h | {r.model} | {r.r2:.3f} | {r.rmse:.3f} | {r.mae:.3f} |")
    (OUT / "CURRENT_RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Mean": "tab:gray", "Persistence": "tab:orange",
              "Linear Regression": "tab:blue", "Neural Network": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key in zip(axes, ["global", "beijing"]):
        for model, color in colors.items():
            part = summary.loc[(summary.dataset == key) & (summary.model == model)].sort_values("horizon")
            ax.plot(part.horizon, part.rmse, marker="o", color=color, label=model)
        ax.set(title=key.title(), xlabel="Forecast horizon (hours)", ylabel="Test RMSE (µg/m³)", xticks=[1, 6, 24])
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle("Current test data: RMSE averaged over locations")
    fig.text(.5, .01, "NN: mean over five seeds; validation reserved from training. Other models fit full training. RF excluded pending rerun.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(OUT / "current_rmse_comparison.png", dpi=170)
    plt.close(fig)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Verified NN test targets, baseline predictions/metrics, seeds and summary aggregation. Wrote", OUT)


if __name__ == "__main__":
    main()
