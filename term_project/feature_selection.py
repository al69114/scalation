"""Compare pre-specified feature sets for linear regression on chronological validation data."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
POLLUTANTS = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
HISTORY = ["PM25_lag1", "PM25_lag2", "PM25_lag3", "PM25_roll24"]
CLOCK = ["hour_sin", "hour_cos", "month_sin", "month_cos"]


def main():
    scores = []
    for dataset, location in [("global", "City"), ("beijing", "station")]:
        x = pd.read_csv(ROOT / f"processed/X_{dataset}_train.csv")
        y = pd.read_csv(ROOT / f"processed/Y_{dataset}_train.csv")
        info = pd.read_csv(ROOT / f"processed/{dataset}_train_info.csv", parse_dates=["Date"])
        assert len(x) == len(y) == len(info)
        candidates = {
            "Current PM2.5": ["PM2.5"],
            "PM2.5 with history": ["PM2.5"] + HISTORY,
            "Pollutants and history": POLLUTANTS + HISTORY,
            "Pollutants, history and time": POLLUTANTS + HISTORY + CLOCK,
        }
        if dataset == "beijing":
            # Use one reference direction with the intercept.
            candidates["Pollutants, history, time and weather"] = x.columns.drop("wd_E").tolist()
        for site, frame in info.groupby(location):
            dates = frame.Date.sort_values()
            val_start = dates.iloc[-round(0.15 * len(dates))]
            train = dates.index[dates + pd.Timedelta(hours=24) < val_start]
            val = dates.index[dates >= val_start]
            for name, cols in candidates.items():
                model = LinearRegression().fit(x.loc[train, cols], y.loc[train])
                residuals = y.loc[val] - model.predict(x.loc[val, cols])
                for h in [1, 6, 24]:
                    rmse = np.sqrt(np.mean(residuals[f"PM25_{h}hr_ahead"] ** 2))
                    scores.append((dataset, site, h, name, len(cols), rmse))
    rows = pd.DataFrame(scores, columns=["dataset", "location", "horizon", "feature_set", "predictors", "validation_rmse"])
    summary = rows.groupby(["dataset", "horizon", "feature_set", "predictors"], as_index=False).validation_rmse.mean()
    summary["selected"] = False
    winners = summary.sort_values(["validation_rmse", "predictors"]).groupby(["dataset", "horizon"]).head(1)
    summary.loc[winners.index, "selected"] = True
    out = ROOT / "results/feature_selection"
    out.mkdir(exist_ok=True, parents=True)
    rows.to_csv(out / "by_location.csv", index=False)
    summary.to_csv(out / "validation_summary.csv", index=False)
    print(summary.loc[summary.selected].to_string(index=False))


if __name__ == "__main__":
    main()
