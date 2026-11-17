"""Create the Global feature-importance slide visual from saved NN results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/integration"
LABELS = {
    "PM2.5": "Current PM2.5", "PM10": "PM10", "SO2": "SO₂", "NO2": "NO₂",
    "CO": "CO", "O3": "O₃", "PM25_lag1": "PM2.5: 1-hour lag",
    "PM25_lag2": "PM2.5: 2-hour lag", "PM25_lag3": "PM2.5: 3-hour lag",
    "PM25_roll24": "PM2.5: 24-hour mean", "hour_of_day": "Hour of day",
    "month": "Month",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "results/nn/nn_permutation_importance.csv")
    data = data.loc[data.dataset == "global"]
    assert data.location.nunique() == 6
    assert not data.duplicated(["location", "horizon", "feature"]).any()
    assert data.groupby(["horizon", "feature"]).location.nunique().eq(6).all()
    means = data.groupby(["horizon", "feature"]).rmse_increase.mean()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1), facecolor="white")
    selected = []
    for ax, horizon in zip(axes, [1, 6, 24]):
        top = means.loc[horizon].nlargest(5).sort_values()
        colors = ["#e8b9b9"] * (len(top) - 1) + ["#b51f2d"]
        bars = ax.barh([LABELS.get(k, k) for k in top.index], top.values, color=colors, height=.62)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in top.values], padding=6, fontsize=12)
        ax.set_xlim(0, top.max() * 1.24)
        ax.set_title(f"{horizon}-hour forecast", fontsize=19, fontweight="bold", pad=18)
        ax.set_xlabel("Increase in RMSE (µg/m³)", labelpad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, alpha=.16)
        ax.set_axisbelow(True)
        for feature, value in top.sort_values(ascending=False).items():
            selected.append(dict(horizon=horizon, feature=feature, mean_rmse_increase=value))
    fig.suptitle("Global: Which features does the neural network rely on?", fontsize=23,
                 fontweight="bold", color="#991c26", y=.98)
    fig.text(.5, .89, "Top five feature groups at each horizon · average across six cities",
             ha="center", fontsize=14, color="#555555")
    fig.text(.5, .035, "Higher bars = greater error after shuffling. Test-set permutation importance (seed 0; five shuffles).\n"
             "Each panel uses a different x-axis scale. Importance describes model reliance, not causation.",
             ha="center", fontsize=11, color="#555555", linespacing=1.5)
    fig.tight_layout(rect=(.005, .13, .995, .86), w_pad=2.4)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"global_feature_importance.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    pd.DataFrame(selected).to_csv(OUT / "global_feature_importance_top5.csv", index=False)
    print(OUT / "global_feature_importance.png")
    print(pd.DataFrame(selected).to_string(index=False))


if __name__ == "__main__":
    main()
