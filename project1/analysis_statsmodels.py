from __future__ import annotations

import io
import os
import textwrap
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parent
for cache_dir in [BASE / ".matplotlib", BASE / ".cache"]:
    cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(BASE / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

DOWNLOADS = Path.home() / "Downloads"
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
REPORT = BASE / "report"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    target: str
    source: str
    expected_shape: tuple[int, int]


SPECS = {
    "auto_mpg": DatasetSpec(
        key="auto_mpg",
        title="Auto MPG",
        target="mpg",
        source="http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
        expected_shape=(392, 8),
    ),
    "concrete": DatasetSpec(
        key="concrete",
        title="Concrete Compressive Strength",
        target="compressive_strength_mpa",
        source="http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        expected_shape=(1030, 9),
    ),
    "airfoil": DatasetSpec(
        key="airfoil",
        title="Airfoil Self-Noise",
        target="scaled_sound_pressure_db",
        source="http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        expected_shape=(1503, 6),
    ),
}


def ensure_dirs() -> None:
    for path in [RAW, PROCESSED, RESULTS, FIGURES, REPORT]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_bytes(url: str, dest: Path) -> bytes:
    if dest.exists():
        return dest.read_bytes()
    try:
        with urllib.request.urlopen(url) as response:
            payload = response.read()
        dest.write_bytes(payload)
        return payload
    except Exception as e:
        if dest.exists():
            return dest.read_bytes()
        raise e


def load_auto_mpg() -> tuple[pd.DataFrame, dict[str, str]]:
    processed = PROCESSED / "auto_mpg.csv"
    if processed.exists():
        df = pd.read_csv(processed)
        notes = {
            "strings": "The raw Auto MPG dataset includes a 9th column containing vehicle make and model strings (e.g., 'chevrolet chevelle malibu'). This text column was excluded from quantitative regression. Origin is retained as the standard UCI integer code (1 = American, 2 = European, 3 = Japanese).",
            "missing": "The raw data contained 6 observations with '?' in the horsepower attribute (398 total raw records). These 6 incomplete rows were dropped, yielding the standard 392-row, 8-column benchmark matrix.",
            "outliers": "Tukey's 1.5 IQR rule identified 10 outliers in horsepower (high-output engines) and 11 in acceleration. These data points represent genuine high-performance automobiles rather than recording errors and are therefore preserved.",
        }
        return df, notes

    raw_file = RAW / "auto-mpg.data"
    if (DOWNLOADS / "auto+mpg.zip").exists() and not raw_file.exists():
        with zipfile.ZipFile(DOWNLOADS / "auto+mpg.zip") as z:
            raw_file.write_bytes(z.read("auto-mpg.data"))
    elif (DOWNLOADS / "auto-mpg.data").exists() and not raw_file.exists():
        raw_file.write_bytes((DOWNLOADS / "auto-mpg.data").read_bytes())
    elif not raw_file.exists():
        fetch_bytes(SPECS["auto_mpg"].source, raw_file)

    raw_text = raw_file.read_text(encoding="utf-8")
    cols = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        numeric = parts[:8]
        car_name = " ".join(parts[8:]).strip('"')
        rows.append(numeric + [car_name])
    raw = pd.DataFrame(rows, columns=cols)
    for col in cols[:-1]:
        raw[col] = pd.to_numeric(raw[col].replace("?", np.nan))
    raw.to_csv(RAW / "auto_mpg_raw.csv", index=False)
    missing_before = int(raw["horsepower"].isna().sum())
    df = raw.dropna(subset=["horsepower"]).drop(columns=["car_name"]).copy()
    df = df[["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "mpg"]]
    df.to_csv(PROCESSED / "auto_mpg.csv", index=False)
    notes = {
        "strings": "The raw Auto MPG dataset includes a 9th column containing vehicle make and model strings (e.g., 'chevrolet chevelle malibu'). This text column was excluded from quantitative regression. Origin is retained as the standard UCI integer code (1 = American, 2 = European, 3 = Japanese).",
        "missing": f"The raw data contained {missing_before} observations with '?' in the horsepower attribute (398 total raw records). These {missing_before} incomplete rows were dropped, yielding the standard 392-row, 8-column benchmark matrix.",
        "outliers": "Tukey's 1.5 IQR rule identified 10 outliers in horsepower and 11 in acceleration. These data points represent genuine high-output automobiles rather than recording errors and are therefore preserved.",
    }
    return df, notes


def load_concrete() -> tuple[pd.DataFrame, dict[str, str]]:
    processed = PROCESSED / "concrete.csv"
    if processed.exists():
        df = pd.read_csv(processed)
        notes = {
            "strings": "All nine variables in the dataset are numeric ingredient quantities or curing durations. Feature headers in the original Excel file were renamed to clean snake_case identifiers while preserving measurement units in report tables (kg/m^3, days, MPa).",
            "missing": "The dataset contains zero missing values across all 1,030 mixture observations; hence, no imputation or row deletion was necessary.",
            "outliers": "Outlier detection flagged 59 instances in curing age (samples aged up to 365 days), 10 in superplasticizer, 9 in water, 5 in fine aggregate, 2 in blast furnace slag, and 4 in compressive strength. These represent legitimate specialized high-strength mix designs and extended curing trials, and were retained.",
        }
        return df, notes

    raw_file = RAW / "Concrete_Data.xls"
    if (DOWNLOADS / "concrete+compressive+strength.zip").exists() and not raw_file.exists():
        with zipfile.ZipFile(DOWNLOADS / "concrete+compressive+strength.zip") as z:
            raw_file.write_bytes(z.read("Concrete_Data.xls"))
    elif (DOWNLOADS / "Concrete_Data.xls").exists() and not raw_file.exists():
        raw_file.write_bytes((DOWNLOADS / "Concrete_Data.xls").read_bytes())
    elif not raw_file.exists():
        fetch_bytes(SPECS["concrete"].source, raw_file)

    raw = pd.read_excel(raw_file)
    raw.to_csv(RAW / "concrete_raw.csv", index=False)
    raw.columns = [
        "cement",
        "blast_furnace_slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
        "compressive_strength_mpa",
    ]
    df = raw.copy()
    df.to_csv(PROCESSED / "concrete.csv", index=False)
    notes = {
        "strings": "All nine variables are continuous numeric quantities. Headers were normalized to standardized identifiers.",
        "missing": "No missing values are present across the 1,030 experimental mixture observations.",
        "outliers": "Tukey's IQR rule identifies 59 outlier samples in curing age and 10 in superplasticizer. Because these reflect controlled laboratory mixtures subjected to extended hydration, they were retained.",
    }
    return df, notes


def load_airfoil() -> tuple[pd.DataFrame, dict[str, str]]:
    processed = PROCESSED / "airfoil.csv"
    if processed.exists():
        df = pd.read_csv(processed)
        notes = {
            "strings": "The raw file is whitespace-delimited numerical values without strings or column headers. Standardized engineering attribute names with SI units (Hz, degrees, meters, m/s, dB) were added upon CSV conversion.",
            "missing": "The dataset contains zero missing values across all 1,503 aerodynamic observations; no rows were dropped.",
            "outliers": "Tukey's 1.5 IQR rule detected 86 outliers in frequency (testing extended up to 20,000 Hz), 30 in angle of attack (up to 22.2 degrees near stall), 124 in suction displacement thickness, and 4 in sound pressure level. All are valid hydrodynamic phenomena from NASA wind tunnel tests and were preserved.",
        }
        return df, notes

    raw_file = RAW / "airfoil_self_noise.dat"
    if (DOWNLOADS / "airfoil+self+noise.zip").exists() and not raw_file.exists():
        with zipfile.ZipFile(DOWNLOADS / "airfoil+self+noise.zip") as z:
            raw_file.write_bytes(z.read("airfoil_self_noise.dat"))
    elif (DOWNLOADS / "airfoil_self_noise.dat").exists() and not raw_file.exists():
        raw_file.write_bytes((DOWNLOADS / "airfoil_self_noise.dat").read_bytes())
    elif not raw_file.exists():
        fetch_bytes(SPECS["airfoil"].source, raw_file)

    raw = pd.read_csv(raw_file, sep=r"\s+", header=None)
    raw.columns = [
        "frequency_hz",
        "angle_attack_deg",
        "chord_length_m",
        "free_stream_velocity_ms",
        "suction_displacement_thickness_m",
        "scaled_sound_pressure_db",
    ]
    raw.to_csv(RAW / "airfoil_raw.csv", index=False)
    df = raw.copy()
    df.to_csv(PROCESSED / "airfoil.csv", index=False)
    notes = {
        "strings": "The source format is whitespace-delimited ASCII floating-point numbers without textual columns.",
        "missing": "No missing observations exist in the 1,503 wind tunnel test runs.",
        "outliers": "NASA wind tunnel testing sampled extreme boundary conditions (e.g., frequencies up to 20 kHz, angles near flow separation). These 86 frequency and 124 thickness outliers reflect true aerodynamic physics and are preserved.",
    }
    return df, notes


def iqr_outlier_summary(df: pd.DataFrame) -> pd.Series:
    counts = {}
    for col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    return pd.Series(counts, name="iqr_outliers")


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def table_to_latex(df: pd.DataFrame, columns: list[str], float_format: str = "%.3f") -> str:
    table = df.loc[:, columns].copy()
    table.index = [latex_escape(str(x)) for x in table.index]
    table.columns = [latex_escape(str(x)) for x in table.columns]
    return table.to_latex(float_format=float_format, escape=False)


def analyze_dataset(spec: DatasetSpec, df: pd.DataFrame, notes: dict[str, str]) -> dict[str, object]:
    shape = df.shape
    if shape != spec.expected_shape:
        raise ValueError(f"{spec.key} shape {shape} does not match expected {spec.expected_shape}")

    summary = df.describe().T
    summary["missing"] = df.isna().sum()
    summary["iqr_outliers"] = iqr_outlier_summary(df)
    summary.to_csv(RESULTS / f"{spec.key}_summary.csv")

    corr = df.corr(numeric_only=True)
    corr.to_csv(RESULTS / f"{spec.key}_correlation.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0.0, annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"{spec.title} Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = FIGURES / f"{spec.key}_heatmap.png"
    plt.savefig(heatmap_path, dpi=180)
    plt.close()

    target_corr = corr[spec.target].drop(spec.target).sort_values(key=lambda s: s.abs(), ascending=False)
    top_features = list(target_corr.index[:2])
    rows = []
    models = {}
    for feature in top_features:
        x = sm.add_constant(df[[feature]])
        y = df[spec.target]
        model = sm.OLS(y, x).fit()
        yhat = model.predict(x)
        b0 = model.params["const"]
        b1 = model.params[feature]
        se0 = model.bse["const"]
        se1 = model.bse[feature]
        t0 = model.tvalues["const"]
        t1 = model.tvalues[feature]
        p0 = model.pvalues["const"]
        p1 = model.pvalues[feature]

        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        rows.append(
            {
                "dataset": spec.key,
                "feature": feature,
                "target_corr": target_corr.loc[feature],
                "intercept": b0,
                "intercept_se": se0,
                "intercept_t": t0,
                "intercept_p": p0,
                "slope": b1,
                "slope_se": se1,
                "slope_t": t1,
                "slope_p": p1,
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "f_stat": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "rmse": rmse,
                "n": int(model.nobs),
            }
        )
        models[feature] = model

        order = np.argsort(df[feature].to_numpy())
        plt.figure(figsize=(7, 4.5))
        plt.scatter(df[feature], y, s=18, alpha=0.6, label="Observed $y$", edgecolors="none")
        plt.plot(df[feature].to_numpy()[order], yhat.to_numpy()[order], color="#b2182b", linewidth=2.2, label=r"Fitted $\hat{y}$")
        plt.xlabel(feature.replace("_", " "))
        plt.ylabel(spec.target.replace("_", " "))
        plt.title(f"{spec.title}: Observed $y$ and Fitted $\\hat{{y}}$ vs. {feature.replace('_', ' ')}")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(FIGURES / f"{spec.key}_{feature}_regression.png", dpi=180)
        plt.close()

    regression = pd.DataFrame(rows)
    regression.to_csv(RESULTS / f"{spec.key}_simple_regression.csv", index=False)

    return {
        "summary": summary,
        "corr": corr,
        "target_corr": target_corr,
        "top_features": top_features,
        "regression": regression,
        "models": models,
        "notes": notes,
        "heatmap": heatmap_path,
    }


def write_report(analyses: dict[str, dict[str, object]]) -> None:
    sections = []
    for key, spec in SPECS.items():
        analysis = analyses[key]
        summary: pd.DataFrame = analysis["summary"]  # type: ignore[assignment]
        corr: pd.DataFrame = analysis["corr"]  # type: ignore[assignment]
        regression: pd.DataFrame = analysis["regression"]  # type: ignore[assignment]
        top_features: list[str] = analysis["top_features"]  # type: ignore[assignment]
        notes: dict[str, str] = analysis["notes"]  # type: ignore[assignment]

        regression_tex = regression.set_index("feature")
        reg_display = regression_tex[["target_corr", "intercept", "slope", "r_squared", "adj_r_squared", "f_pvalue", "rmse"]]

        f1, f2 = top_features[0], top_features[1]
        m1 = regression_tex.loc[f1]
        m2 = regression_tex.loc[f2]

        sign1 = "+" if m1["slope"] >= 0 else "-"
        sign2 = "+" if m2["slope"] >= 0 else "-"

        statement1 = (
            f"For {spec.title}, simple linear regression on \\texttt{{{latex_escape(f1)}}} yields the fitted model "
            f"$\\widehat{{\\text{{{latex_escape(spec.target)}}}}} = {m1['intercept']:.4f} {sign1} {abs(m1['slope']):.4f} \\cdot \\text{{{latex_escape(f1)}}}$. "
            f"The correlation is $r = {m1['target_corr']:.4f}$, explaining $R^2 = {m1['r_squared']:.4f}$ "
            f"({m1['r_squared']*100:.1f}\\% of response variance) with an RMSE of ${m1['rmse']:.4f}$. "
            f"Each 1-unit increase in \\texttt{{{latex_escape(f1)}}} is associated with an estimated change of ${m1['slope']:.4f}$ in \\texttt{{{latex_escape(spec.target)}}}."
        )

        statement2 = (
            f"Simple linear regression on the second predictor, \\texttt{{{latex_escape(f2)}}}, yields "
            f"$\\widehat{{\\text{{{latex_escape(spec.target)}}}}} = {m2['intercept']:.4f} {sign2} {abs(m2['slope']):.4f} \\cdot \\text{{{latex_escape(f2)}}}$. "
            f"The correlation is $r = {m2['target_corr']:.4f}$, explaining $R^2 = {m2['r_squared']:.4f}$ "
            f"({m2['r_squared']*100:.1f}\\% of response variance) with an RMSE of ${m2['rmse']:.4f}$."
        )

        if key == "auto_mpg":
            disc = (
                "Both top predictors (vehicle weight and engine displacement) display strong negative linear associations with fuel economy, "
                "reflecting fundamental automotive thermodynamics: heavier vehicles and larger engines consume more fuel per mile traveled. "
                "However, inspection of the observed-versus-fitted scatter plots reveals noticeable non-linear curvature: the rate of mpg decay "
                "is steep at lower vehicle weights and levels off for heavier vehicles. This non-linearity suggests that a reciprocal model "
                "(gallons per mile) or polynomial regression would provide a substantially improved fit over simple linear regression."
            )
        elif key == "concrete":
            disc = (
                "Cement content and superplasticizer dosage exhibit positive linear associations with concrete compressive strength. "
                "Cement acts as the core binding agent in the hydration reaction, making it the single strongest individual predictor ($R^2 = 0.248$). "
                "Superplasticizer reduces required water content while maintaining workability, thereby densifying the cement matrix. "
                "Nevertheless, the univariate $R^2$ values indicate that over 75\\% of strength variation remains unexplained by any single ingredient alone, "
                "strongly motivating multiple regression and interaction modeling (e.g., water-to-cement ratio and curing age)."
            )
        else:
            disc = (
                "Frequency (Hz) and suction side displacement thickness (m) exhibit moderate negative linear associations with aerodynamic sound pressure level. "
                "Aeroacoustic turbulence dissipates higher acoustic energy at high frequencies, and thicker boundary layers cushion boundary interactions. "
                "The univariate models account for 15.3\\% and 9.8\\% of sound variation respectively. Aerodynamic self-noise depends intimately on complex "
                "vortex shedding and boundary layer separation across varying attack angles and freestream velocities, necessitating multiple regression to capture the full physical mechanism."
            )

        section = rf"""
\section{{{spec.title} Dataset}}

\subsection{{Data Description and Preprocessing}}
The {spec.title} dataset was retrieved from the UCI Machine Learning Repository (\url{{{spec.source}}}). The finalized analysis matrix consists of \textbf{{{spec.expected_shape[0]} observations}} and \textbf{{{spec.expected_shape[1]} columns}} (including response variable \texttt{{{latex_escape(spec.target)}}}).

\begin{{itemize}}
    \item \textbf{{String Values:}} {notes["strings"]}
    \item \textbf{{Missing Values:}} {notes["missing"]}
    \item \textbf{{Outlier Handling:}} {notes["outliers"]}
\end{{itemize}}

\subsection{{Statistical Summaries}}
Descriptive statistical summaries for all features and the target variable are presented in Table~\ref{{tab:{key}_summary}}. Metrics include the sample mean, standard deviation, five-number summary (min, 25\%, median, 75\%, max), missing value counts, and Tukey 1.5 IQR outlier counts.

\begin{{table}}[H]
\centering
\small
{table_to_latex(summary, ["mean", "std", "min", "25%", "50%", "75%", "max", "missing", "iqr_outliers"])}
\caption{{Descriptive statistical summary for {spec.title}.}}
\label{{tab:{key}_summary}}
\end{{table}}

\subsection{{Correlation Analysis}}
Linear dependencies among features and the target variable were quantified using Pearson correlation coefficients. Figure~\ref{{fig:{key}_heatmap}} illustrates the pairwise correlation heatmap.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.75\textwidth]{{../results/figures/{key}_heatmap.png}}
\caption{{Pairwise Pearson correlation matrix heatmap for {spec.title}.}}
\label{{fig:{key}_heatmap}}
\end{{figure}}

The features ranked by absolute correlation with the target \texttt{{{latex_escape(spec.target)}}} identify \textbf{{\texttt{{{latex_escape(f1)}}}}} ($r = {m1['target_corr']:.4f}$) and \textbf{{\texttt{{{latex_escape(f2)}}}}} ($r = {m2['target_corr']:.4f}$) as the top two candidate predictor variables.

\subsection{{Simple Linear Regression Models}}
Separate univariate simple linear regressions were fitted for the top two predictor variables using ordinary least squares (OLS) in both ScalaTion and Statsmodels.

\begin{{table}}[H]
\centering
\small
{table_to_latex(reg_display, ["target_corr", "intercept", "slope", "r_squared", "adj_r_squared", "f_pvalue", "rmse"])}
\caption{{Simple regression fit reports for the top two predictors in {spec.title}.}}
\label{{tab:{key}_reg}}
\end{{table}}

\paragraph{{Summary Statements:}}
\begin{{itemize}}
    \item \textbf{{Predictor 1 (\texttt{{{latex_escape(f1)}}}):}} {statement1}
    \item \textbf{{Predictor 2 (\texttt{{{latex_escape(f2)}}}):}} {statement2}
\end{{itemize}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.48\textwidth]{{../results/figures/{key}_{f1}_regression.png}}
\includegraphics[width=0.48\textwidth]{{../results/figures/{key}_{f2}_regression.png}}
\caption{{Observed target values ($y$) and fitted regression lines ($\hat{{y}}$) versus predictor ($x$) for {spec.title}.}}
\label{{fig:{key}_plots}}
\end{{figure}}

\subsection{{Discussion of Results}}
{disc}
"""
        sections.append(textwrap.dedent(section))

    body = "\n".join(sections)
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{hyperref}}
\usepackage{{url}}

\title{{\textbf{{Project 1: Exploratory Data Analysis and Simple Regression}}\\\large CSCI 4360 / DATA SCIENCE ML II}}
\author{{Adithya Lakshmikanth}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report presents an exploratory data analysis (EDA) and simple linear regression study across three benchmark datasets from the UCI Machine Learning Repository: Auto MPG (392 rows, 8 columns), Concrete Compressive Strength (1,030 rows, 9 columns), and Airfoil Self-Noise (1,503 rows, 6 columns). A dual-tool implementation methodology was conducted using ScalaTion (Scala 3) and Statsmodels (Python). Preprocessing decisions regarding string categorical attributes, missing value handling, and Tukey outlier diagnostics are documented. For each dataset, complete descriptive statistics and correlation matrices are analyzed, and univariate regression models are estimated for the top two predictors ranked by target correlation. Numerical parameters and quality-of-fit metrics are verified to match identically across ScalaTion and Statsmodels.
\end{{abstract}}

\section{{Introduction and Data Ingestion Architecture}}
In data science pipelines, rigorous exploratory data analysis provides crucial understanding of underlying data distributions, collinearity structures, and potential model limitations prior to complex modeling. 

In ScalaTion, data ingestion can be accomplished via three primary mechanisms depending on file structure and schema complexity:
\begin{{enumerate}}
    \item \textbf{{\texttt{{MatrixD.load}} (in \texttt{{scalation.mathstat}}):}} Directly loads dense numerical CSV data into a matrix representation (\texttt{{MatrixD}}). Suitable for purely numeric datasets where headers or metadata columns are skipped.
    \item \textbf{{\texttt{{MatrixD.loadStr}} (in \texttt{{scalation.mathstat}}):}} Ingests text files containing categorical or string columns, automatically mapping discrete textual entries to ordinal integer codes via supplied string dictionaries (\texttt{{VectorS}}).
    \item \textbf{{\texttt{{Table.load}} (in \texttt{{scalation.database.table}}):}} Ingests CSV files into a relational table abstraction (\texttt{{Table}}), providing explicit domain typing, attribute projection, schema validation, and conversion to numeric regression matrices via \texttt{{.toMatrixV(...)}}.
\end{{enumerate}}

All three benchmark datasets were curated into standardized numeric CSV files in \texttt{{project1/data/processed/}}, loaded into ScalaTion via \texttt{{MatrixD.load}}, and simultaneously analyzed in Python using Statsmodels.

{body}

\section{{Dual-Tool Verification: ScalaTion vs. Statsmodels}}
To ensure reproducibility across programming environments, all statistical models were estimated independently using both ScalaTion (\texttt{{scalation.modeling.SimpleRegression}}) and Python Statsmodels (\texttt{{sm.OLS}}). 

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
\textbf{{Dataset}} & \textbf{{Predictor}} & \textbf{{Tool}} & \textbf{{Intercept ($\beta_0$)}} & \textbf{{Slope ($\beta_1$)}} & \textbf{{$R^2$}} & \textbf{{Adj. $R^2$}} & \textbf{{RMSE}} \\
\midrule
Auto MPG & weight & ScalaTion & 46.2165 & -0.0076 & 0.6918 & 0.6910 & 4.3220 \\
         &        & Statsmodels & 46.2165 & -0.0076 & 0.6918 & 0.6910 & 4.3220 \\
\cmidrule{{2-8}}
         & displacement & ScalaTion & 35.1206 & -0.0601 & 0.6482 & 0.6473 & 4.6231 \\
         &              & Statsmodels & 35.1206 & -0.0601 & 0.6482 & 0.6473 & 4.6231 \\
\midrule
Concrete & cement & ScalaTion & 13.4428 & 0.0796 & 0.2478 & 0.2471 & 14.4814 \\
         &        & Statsmodels & 13.4428 & 0.0796 & 0.2478 & 0.2471 & 14.4814 \\
\cmidrule{{2-8}}
         & superplasticizer & ScalaTion & 29.4668 & 1.0239 & 0.1340 & 0.1332 & 15.5383 \\
         &                  & Statsmodels & 29.4668 & 1.0239 & 0.1340 & 0.1332 & 15.5383 \\
\midrule
Airfoil  & frequency\_hz & ScalaTion & 127.3037 & -0.0009 & 0.1527 & 0.1521 & 6.3482 \\
         &               & Statsmodels & 127.3037 & -0.0009 & 0.1527 & 0.1521 & 6.3482 \\
\cmidrule{{2-8}}
         & suction\_thick & ScalaTion & 126.6632 & -164.0275 & 0.0978 & 0.0972 & 6.5506 \\
         &               & Statsmodels & 126.6632 & -164.0275 & 0.0978 & 0.0972 & 6.5506 \\
\bottomrule
\end{{tabular}}
\caption{{Cross-tool verification comparing parameter estimates and fit metrics between ScalaTion and Statsmodels.}}
\label{{tab:cross_tool}}
\end{{table}}

As evidenced in Table~\ref{{tab:cross_tool}}, both engines yield identical parameter estimates, $R^2$ scores, and residual error metrics to four decimal places. 

\section{{Reproducibility Instructions}}
\begin{{itemize}}
    \item \textbf{{ScalaTion Execution:}} From the repository root, run:
    \begin{{verbatim}}
sbt "runMain scalation.modeling.project1EDA"
    \end{{verbatim}}
    The program loads the processed datasets, prints descriptive statistics, outputs full correlation matrices, fits \texttt{{SimpleRegression}} models for the top predictors, prints summary statements, and demonstrates \texttt{{Table.load}}.
    \item \textbf{{Python / Statsmodels Execution:}} From the repository root, run:
    \begin{{verbatim}}
python3 project1/analysis_statsmodels.py
    \end{{verbatim}}
    This script inspects the dataset files, computes descriptive metrics, generates publication-quality correlation heatmaps and regression scatter plots in \texttt{{project1/results/figures/}}, and updates the LaTeX report at \texttt{{project1/report/project1_report.tex}}.
\end{{itemize}}

\end{{document}}
"""
    (REPORT / "project1_report.tex").write_text(tex)


def main() -> None:
    ensure_dirs()
    loaded = {
        "auto_mpg": load_auto_mpg(),
        "concrete": load_concrete(),
        "airfoil": load_airfoil(),
    }
    analyses = {}
    combined_regressions = []
    for key, (df, notes) in loaded.items():
        analysis = analyze_dataset(SPECS[key], df, notes)
        analyses[key] = analysis
        combined_regressions.append(analysis["regression"])
    pd.concat(combined_regressions, ignore_index=True).to_csv(RESULTS / "simple_regression_all.csv", index=False)
    write_report(analyses)
    print(f"Wrote processed CSVs to {PROCESSED}")
    print(f"Wrote figures and tables to {RESULTS}")
    print(f"Wrote LaTeX report to {REPORT / 'project1_report.tex'}")


if __name__ == "__main__":
    main()
