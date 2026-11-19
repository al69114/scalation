"""Reproduce training-period air-quality EDA without changing processed datasets.

Run from any directory: python3 /path/to/term_project/eda_analysis.py
"""

from pathlib import Path
import calendar
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda"
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
HORIZONS = [1, 6, 24]
TARGETS = [f"PM25_{h}hr_ahead" for h in HORIZONS]
ENGINEERED = ["PM25_lag1", "PM25_lag2", "PM25_lag3", "PM25_roll24",
              "hour_sin", "hour_cos", "month_sin", "month_cos"]
CONFIG = {
    "global": {"location": "City", "cutoff": pd.Timestamp("2023-10-01", tz="UTC"),
               "title": "Global cities", "clock": "UTC hour"},
    "beijing": {"location": "station", "cutoff": pd.Timestamp("2016-03-01"),
                "title": "Beijing stations", "clock": "Recorded hour"},
}
FIGURES = [
    ("distributions", "Pollutant distributions"),
    ("location_distributions", "PM2.5 by location"),
    ("correlations", "Current-measurement correlations"),
    ("forecast_correlations", "Correlations with future PM2.5"),
    ("pollutant_relationships", "Pollutant relationships"),
    ("time_patterns", "Time trends and observation counts"),
    ("calendar_patterns", "Monthly and hourly patterns"),
]


def training_data(key):
    """Mirror existing cleaning/target logic, then verify exact exported training rows.

    Reconstruction checks both original measurements and the newly added lag/time
    features against the exports. Training metadata is also checked row by row.
    This intentionally preserves the existing complete-case target restriction.
    """
    cfg = CONFIG[key]
    loc = cfg["location"]
    if key == "global":
        raw = pd.read_csv(ROOT / "datasets/Global_Air_Quality.csv")
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", utc=True)
        work = raw.drop(columns="AQI")
    else:
        paths = sorted(p for p in (ROOT / "datasets").glob("*.csv")
                       if p.name != "Global_Air_Quality.csv")
        raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        raw["Date"] = pd.to_datetime(raw[["year", "month", "day", "hour"]], errors="coerce")
        work = raw.drop(columns="No")
    counts = {"dataset": key, "raw_rows": len(raw), "locations": raw[loc].nunique(),
              "raw_start": str(raw.Date.min()), "raw_end": str(raw.Date.max())}
    work = work.drop_duplicates([loc, "Date"])
    counts["duplicate_location_times_removed"] = len(raw) - len(work)
    n = len(work)
    work = work.dropna().copy()
    counts["missing_rows_removed"] = n - len(work)
    n = len(work)
    work = work.loc[(work[POLLUTANTS] >= 0).all(axis=1)].copy()
    counts["negative_pollutant_rows_removed"] = n - len(work)
    counts["clean_rows"] = len(work)
    clean = work.sort_values([loc, "Date"]).reset_index(drop=True)
    work = clean.copy()
    for h, target in zip(HORIZONS, TARGETS):
        future = clean[[loc, "Date", "PM2.5"]].copy()
        future["Date"] -= pd.Timedelta(hours=h)
        future = future.rename(columns={"PM2.5": target})
        work = work.merge(future, on=[loc, "Date"], how="left", validate="one_to_one")
    work = work.dropna(subset=TARGETS)
    counts["rows_with_all_three_targets"] = len(work)
    base = clean[[loc, "Date", "PM2.5"]]
    for k in [1, 2, 3]:
        lag = base.copy()
        lag["Date"] += pd.Timedelta(hours=k)
        work = work.merge(lag.rename(columns={"PM2.5": f"PM25_lag{k}"}),
                          on=[loc, "Date"], how="left", validate="one_to_one")
    rolling = (base.set_index("Date").groupby(loc)["PM2.5"].rolling("24h", min_periods=12)
               .mean().rename("PM25_roll24").reset_index())
    work = work.merge(rolling, on=[loc, "Date"], how="left", validate="one_to_one")
    for field, period, offset in [("hour", 24, 0), ("month", 12, 1)]:
        angle = 2 * np.pi * (getattr(work.Date.dt, field) - offset) / period
        work[f"{field}_sin"], work[f"{field}_cos"] = np.sin(angle), np.cos(angle)
    work = work.dropna(subset=ENGINEERED)
    counts["rows_with_lag_history"] = len(work)
    train = work.loc[work.Date + pd.Timedelta(hours=24) < cfg["cutoff"]]
    train = train.sort_values(["Date", loc]).reset_index(drop=True)
    encoded = pd.get_dummies(train, columns=["wd"], dtype=int) if key == "beijing" else train
    for prefix, cols in [("X", None), ("Y", TARGETS)]:
        saved = pd.read_csv(ROOT / f"processed/{prefix}_{key}_train.csv")
        actual = encoded[list(saved.columns) if cols is None else cols]
        pd.testing.assert_frame_equal(actual, saved, check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
    assert train.Date.max() + pd.Timedelta(hours=24) < cfg["cutoff"]
    assert not train[POLLUTANTS + TARGETS].isna().any().any()
    metadata = pd.read_csv(ROOT / f"processed/{key}_train_info.csv")
    metadata["Date"] = pd.to_datetime(metadata["Date"])
    pd.testing.assert_frame_equal(train[[loc, "Date"]], metadata)
    counts["input_features"] = len(pd.read_csv(ROOT / f"processed/X_{key}_train.csv", nrows=0).columns)
    counts.update(training_rows=len(train), training_start=str(train.Date.min()),
                  training_end=str(train.Date.max()), test_cutoff=str(cfg["cutoff"]))
    train[[loc, "Date"]].to_csv(OUT / f"tables/{key}_training_metadata.csv", index=False)
    return train, counts


def save_figure(fig, key, name):
    fig.savefig(OUT / f"figures/{key}_{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def analyze(key, df):
    cfg = CONFIG[key]
    loc, title = cfg["location"], cfg["title"]
    numeric = POLLUTANTS + (WEATHER if key == "beijing" else [])
    table_dir = OUT / "tables"
    summary = df[numeric].describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
    summary["skewness"] = df[numeric].skew()
    summary["zero_count"] = (df[numeric] == 0).sum()
    q1, q3 = df[numeric].quantile(.25), df[numeric].quantile(.75)
    summary["outside_1.5_iqr_count"] = ((df[numeric] < q1 - 1.5 * (q3-q1)) |
                                         (df[numeric] > q3 + 1.5 * (q3-q1))).sum()
    summary.to_csv(table_dir / f"{key}_descriptive_statistics.csv")
    by_location = df.groupby(loc)["PM2.5"].agg(count="size", mean="mean", median="median", std="std", max="max")
    by_location.to_csv(table_dir / f"{key}_location_summary.csv")
    correlation = df[numeric].corr()
    correlation.to_csv(table_dir / f"{key}_pearson_correlations.csv")
    df[numeric].corr(method="spearman").to_csv(table_dir / f"{key}_spearman_correlations.csv")
    future_corr = df[numeric + ENGINEERED + TARGETS].corr().loc[numeric + ENGINEERED, TARGETS]
    future_corr.to_csv(table_dir / f"{key}_forecast_correlations.csv")
    within = df[numeric] - df.groupby(loc)[numeric].transform("mean")
    within.corr().to_csv(table_dir / f"{key}_within_location_correlations.csv")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, POLLUTANTS):
        ax.hist(df[col], bins=65, color="#16877c", alpha=.85)
        ax.axvline(df[col].median(), color="#d45d33", linestyle="--", label="Median")
        ax.set(title=col, xlabel="Recorded concentration", ylabel="Count (log scale)", yscale="log")
        ax.legend(fontsize=8)
    fig.suptitle(f"{title}: training-period pollutant distributions\nLog counts show sparse tails; extreme values retained")
    fig.tight_layout(rect=(0, 0, 1, .92))
    save_figure(fig, key, "distributions")

    fig, ax = plt.subplots(figsize=(11, 6))
    order = by_location.sort_values("median").index
    sns.boxplot(data=df, x="PM2.5", y=loc, order=order, showfliers=False, color="#80bdb5", ax=ax)
    ax.set(title=f"{title}: training PM2.5 by location", xlabel="PM2.5 (µg/m³)", ylabel="")
    fig.text(.5, .01, "Boxes: median and quartiles. Whiskers: 1.5×IQR. Outlier points hidden here, not removed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, key, "location_distributions")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title(f"{title}: pooled training Pearson correlations\nCurrent measurements; association does not establish causation")
    fig.tight_layout()
    save_figure(fig, key, "correlations")

    fig, ax = plt.subplots(figsize=(7, 10))
    sns.heatmap(future_corr, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=ax,
                xticklabels=["1 hour", "6 hours", "24 hours"])
    ax.set(title=f"{title}: current features vs. future PM2.5\nTraining data only", xlabel="Forecast horizon")
    fig.tight_layout()
    save_figure(fig, key, "forecast_correlations")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    others = [p for p in POLLUTANTS if p != "PM2.5"]
    for ax, col in zip(axes.flat, others):
        hb = ax.hexbin(df[col], df["PM2.5"], gridsize=40, bins="log", mincnt=1, cmap="viridis")
        ax.set(xlabel=col + " (recorded concentration)", ylabel="PM2.5 (µg/m³)",
               title=f"Pearson r = {correlation.loc[col, 'PM2.5']:.2f}")
        fig.colorbar(hb, ax=ax, label="Observation count")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .6, "All training observations included.\nColor shows density on a log scale.\nRelationships pool locations.\nThese are contemporaneous measurements,\nnot forecast accuracy scores.", va="center", fontsize=10)
    fig.suptitle(f"{title}: relationships with current PM2.5")
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, key, "pollutant_relationships")

    temporal = df.assign(day=df.Date.dt.floor("D"), month=df.Date.dt.month, hour=df.Date.dt.hour,
                         month_start=df.Date.dt.strftime("%Y-%m"))
    grouped = {}
    for field in ["day", "month_start", "month", "hour"]:
        stat = temporal.groupby([loc, field])["PM2.5"].agg(mean="mean", median="median", count="size").reset_index()
        stat.to_csv(table_dir / f"{key}_{field}_patterns.csv", index=False)
        grouped[field] = stat
    locations = sorted(df[loc].unique())
    colors = dict(zip(locations, sns.color_palette("husl", len(locations))))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 1]})
    for name in locations:
        part = grouped["day"].loc[grouped["day"][loc] == name].set_index("day")
        # Reindex so missing full days appear as gaps, rather than connecting across them.
        dates = pd.date_range(df.Date.min().floor("D"), df.Date.max().floor("D"), freq="D")
        axes[0].plot(dates, part["mean"].reindex(dates), linewidth=.8, alpha=.8, color=colors[name], label=name)
    axes[0].set(title=f"{title}: daily mean PM2.5 in the training sample", ylabel="PM2.5 (µg/m³)")
    axes[0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    daily_n = temporal.groupby("day").size()
    axes[1].plot(daily_n.index, daily_n.values, color="#555555")
    axes[1].set(ylabel="Rows/day", xlabel="Forecast-origin date")
    fig.tight_layout()
    save_figure(fig, key, "time_patterns")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in locations:
        for ax, field, domain in [(axes[0], "month", range(1, 13)), (axes[1], "hour", range(24))]:
            part = grouped[field].loc[grouped[field][loc] == name].set_index(field)["mean"].reindex(domain)
            ax.plot(part.index, part, marker="o", markersize=3, color=colors[name], label=name)
    axes[0].set(xticks=range(1, 13), xticklabels=list(calendar.month_abbr)[1:], ylabel="Mean PM2.5 (µg/m³)", title="Month-of-year profile")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].set(xlabel=cfg["clock"], xticks=range(0, 24, 3), ylabel="Mean PM2.5 (µg/m³)", title="Hour-of-day profile")
    axes[1].legend(ncol=1, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    note = ("Global training covers January–September only; October–December are not estimated. Months are not shared climate seasons across cities."
            if key == "global" else "Month profiles pool the training years. Unequal complete-case coverage can influence averages; consult exported counts.")
    fig.suptitle(f"{title}: training-period calendar patterns")
    fig.text(.5, .01, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save_figure(fig, key, "calendar_patterns")
    return summary, by_location, correlation, future_corr, grouped


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    overview = []
    sections = []
    for key in CONFIG:
        df, counts = training_data(key)
        overview.append(counts)
        summary, locations, corr, future, grouped = analyze(key, df)
        strongest = corr.loc[[p for p in POLLUTANTS if p != "PM2.5"], "PM2.5"].abs().idxmax()
        highest, lowest = locations["mean"].idxmax(), locations["mean"].idxmin()
        monthly = df.groupby(df.Date.dt.month)["PM2.5"].mean()
        section = [f"## {CONFIG[key]['title']}", "",
                   f"The verified training sample contains **{len(df):,} rows**, from {counts['training_start']} through {counts['training_end']}.", "",
                   f"- Current PM2.5: mean **{summary.loc['PM2.5', 'mean']:.2f}**, median **{summary.loc['PM2.5', '50%']:.2f}**, maximum **{summary.loc['PM2.5', 'max']:.2f} µg/m³**; skewness **{summary.loc['PM2.5', 'skewness']:.2f}**.",
                   f"- Highest location mean: **{highest} ({locations.loc[highest, 'mean']:.2f} µg/m³)**. Lowest: **{lowest} ({locations.loc[lowest, 'mean']:.2f} µg/m³)**. These describe the retained training sample.",
                   f"- Strongest pooled current-PM2.5 correlation among the other pollutants: **{strongest} (Pearson r = {corr.loc[strongest, 'PM2.5']:.3f})**.",
                   "- Current PM2.5 correlation with future PM2.5: " + "; ".join(f"**{h}h: {future.loc['PM2.5', t]:.3f}**" for h, t in zip(HORIZONS, TARGETS)) + ". These are correlations, not predictive performance metrics.",
                   f"- Across retained observations, the highest pooled month-of-year mean is **{calendar.month_name[monthly.idxmax()]} ({monthly.max():.2f} µg/m³)** and the lowest is **{calendar.month_name[monthly.idxmin()]} ({monthly.min():.2f} µg/m³)**. Per-location profiles and observation counts are exported.", ""]
        if key == "global":
            section += ["Only January–September are available for training-period calendar EDA. This cannot establish a complete annual seasonal cycle or its repeatability. Global hours are UTC, not local city hours; do not interpret a shared UTC peak as a shared rush-hour effect.", ""]
        else:
            section += ["Month-of-year averages pool multiple training years. They describe calendar variation rather than proving a stable seasonal mechanism. The original complete-case filtering is preserved, including its requirement for complete future rows when constructing targets.", ""]
        for name, caption in FIGURES:
            section += [f"### {caption}", "", f"![{caption}](figures/{key}_{name}.png)", ""]
        sections += section
        print(f"{key}: verified {len(df):,} training rows; saved {len(FIGURES)} figures")
    pd.DataFrame(overview).to_csv(OUT / "tables/dataset_overview.csv", index=False)
    report = ["# Air-quality exploratory data analysis", "",
              "## Scope and reproducibility", "",
              "This analysis covers distributions, pollutant relationships, correlations, and calendar/time patterns for both datasets. All analytical figures and correlations use exactly the rows in the existing training exports. The four training X/Y files are checked against reconstructed rows before plotting. Test X/Y files are never loaded. Full-source counts in `dataset_overview.csv` describe cleaning only; they are not used to select features or models.", "",
              "The analysis preserves existing preprocessing and does not overwrite `processed/`. It verifies the new 1–3-hour PM2.5 lags, rolling 24-hour mean (at least 12 observations), cyclical hour/month features, and exported training metadata. A copy of the metadata is saved under `eda/tables/` for traceability. This EDA script does not train models or select a final feature subset; separate baseline and neural-network notebooks now provide modeling results.", "",
              "Run `python3 term_project/eda_analysis.py` from the repository root. The script resolves paths relative to itself, so it also works from other directories. Figures are PNGs and all statistics are exported as CSVs.", "",
              "## Interpretation limits", "",
              "- Current pollutant correlations describe association, not causation. Pooled correlations can reflect location differences; compare the exported within-location-centered correlations and Spearman correlations.",
              "- IQR flags identify unusual observations for inspection; they do not establish sensor error. No additional outlier filtering or clipping is performed.",
              "- Calendar averages use retained complete cases and are observation-weighted. Missingness and target availability can alter their composition. Daily/monthly counts are exported alongside means.",
              "- Global and Beijing differ in period, locations, available predictors, and sampling. EDA does not isolate a geographic cause for differences or demonstrate transfer to unseen locations.",
              "- PM2.5 is labeled in µg/m³ following the project definition (and UCI metadata for Beijing). Other pollutant axes retain source values; confirm Global source units before making cross-dataset concentration comparisons.",
              "- The separate linear-regression feature-set comparison uses chronological validation within training, with a 24-hour gap. See the feature-selection results linked in the project README.", ""]
    report += sections
    report += ["## Implications for modeling", "",
               "1. Compare the implemented mean and persistence baselines with learned models at each horizon. Use matching data versions and test rows; see `../INTEGRATION_REVIEW.md` for the current result audit.",
               "2. Examine correlated predictors jointly before interpreting coefficients; marginal correlation alone is not feature selection.",
               "3. Evaluate transformation choices on chronological validation data. Global contains zero PM2.5 values, so a plain log transform is not defined everywhere.",
               "4. Report errors by city/station as well as overall. Keep chronological validation distinct from any experiment intended to test unseen locations.", "",
               "## Dataset references", "",
               "- [Global Air Quality (2023), six cities — Kaggle](https://www.kaggle.com/datasets/youssefelebiary/global-air-quality-2023-6-cities)",
               "- [Beijing Multi-Site Air Quality — UCI](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)", ""]
    (OUT / "EDA_REPORT.md").write_text("\n".join(report))
    (OUT / "environment.json").write_text(json.dumps({"pandas": pd.__version__, "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "seaborn": sns.__version__}, indent=2) + "\n")
    print(f"Report: {OUT / 'EDA_REPORT.md'}")


if __name__ == "__main__":
    main()
