# Symbolic regression: data and result verification

This follow-up resolves part of the uncertainty in the initial report audit.
The report itself has not been edited.

## Additional independent Statsmodels verification

Run `python project2/verify_symbolic_statsmodels.py` in an environment with
NumPy, pandas, SciPy, Statsmodels and scikit-learn. No sbt is needed. Its
output is saved in `project2/results/symbolic_statsmodels_verification.txt`.
The earlier `verify_symbolic_regression.py` is preserved unchanged; the new
script reuses its declared candidate-feature definitions but performs its
own fitting and numerical checks.

- Exact rounding-interval checks yield unique (evaluation rows, predictors,
  residual degrees of freedom): Auto MPG (78, 23, 54), Concrete (206, 17,
  188), Airfoil (300, 41, 258). The earlier approximate inversion yielding
  77, 207, and 303 was sensitive to rounding. These identities establish
  arithmetic compatibility, not independent test-set provenance.
- All 44 printed coefficient ratios pass when rounding uncertainty in both
  estimates and standard errors is included. Slag² is compatible with its
  printed t=6.30. All 44 p-values are compatible with a normal approximation;
  only 31/44 are compatible with exact training-df Student-t calculations
  at the displayed precision. Broad tolerances in the earlier script hide
  this distinction. Neither check proves that the SEs or post-selection
  significance claims are statistically valid.
- Independently refitting the recovered Auto MPG and Concrete terms with
  Statsmodels reproduces ScalaTion's saved test predictions with maximum
  absolute differences below 1e-8. This confirms the exact Auto MPG match
  and the remaining Concrete discrepancy described below.
- A separate independent model for each dataset splits before feature
  selection and scales features using training rows only. Complete selected
  term lists and scores are written to
  `project2/results/symbolic_statsmodels_independent.csv`. These provide
  reproducible benchmarks; their accuracy cannot certify another model's
  results or establish that lower reported scores are unbiased.

## Input data

The Project 2 programs use the shared `data/` directory:

| File | Observations | Predictors | Response |
|---|---:|---:|---|
| auto_mpg.csv | 392 | 7 | mpg |
| concrete.csv | 1030 | 8 | compressive_strength_mpa |
| airfoil.csv | 1503 | 5 | scaled_sound_pressure_db |

All columns are numeric and all values are finite; no missing values were
found. The built-in 392-by-8 matrix in `Example_AutoMPG.scala` equals the
Auto MPG CSV element-for-element. Its convenience variable `x` drops
origin, but passing all seven CSV predictors to symbolic regression
reproduces the report's model.

This establishes consistency with the local project data. It is not a new
comparison against downloaded upstream dataset versions.

## ScalaTion Auto MPG: numerical results reproduced

A diagnostic runner was compiled against the installed ScalaTion classes,
without changing library source or enabling GUI plots. The reconstruction
uses seven predictors, powers {-2, -1, 0.5, 2}, an intercept, no pairwise
interactions, and forward selection with the library's default sMAPE_IC
criterion (`QoF.smapeC`). Selection uses the first 313 rows. Cross-validation
consumes the first native permutation, and final validation uses the second.

This reproduces all of the displayed Auto MPG coefficient estimates, not
just similar performance. It selects 23 terms plus the intercept.

| Metric | Reproduced | Report |
|---|---:|---:|
| R² | 0.8827457900 | 0.8827 |
| Adjusted R² | 0.8328041820 | 0.8328 |
| RMSE | 2.6101105290 | 2.6101 |
| MAE | 1.9913616996 | 1.9914 |
| Residual SE | 3.136962 | 3.1370 |
| F | 17.675558 | 17.6756 |

For example, the intercept is -1016807.1950886175, inverse-weight coefficient
908334.5294565515, and squared-model-year coefficient -10.1169714517.
Complete coefficients and row-level test predictions are saved under
`project2/results/symbolic_audit/`.

**Evaluation limitation:** 61 of the 78 final test rows were among the first
313 rows used to select terms. Although final coefficient fitting excludes
those test rows, feature selection already used their responses. Therefore
the reproduced scores are not an independent held-out evaluation of the
whole modeling procedure. Freeze a test split before selecting terms and
restrict selection and tuning to its training rows to obtain that evaluation.
The report's claim that this uses Scala's ordinary random shuffle is also
unsupported by this exact reproduction: it uses ScalaTion's native
permutation generator, stream 0, on the original CSV order.

Coefficient estimates and aggregate metrics have been verified here; the
reported coefficient standard errors and p-values have not been independently
certified as valid post-selection inference.

## ScalaTion Concrete: matching subset, small unresolved differences

Using all eight predictors, powers {1, 0.5, 2}, no interaction terms, and
the same default selection workflow yields 17 selected terms plus the
intercept, including all the significant terms listed in the report.

| Metric | Current CSV reconstruction | Report |
|---|---:|---:|
| R² | 0.8180567462 (0.8181) | 0.8180 |
| Adjusted R² | 0.801604 | 0.8015 |
| RMSE | 6.9625783543 (6.9626) | 6.9646 |
| MAE | 5.6184200084 (5.6184) | 5.6196 |
| Residual SE | 7.288276 | 7.2904 |
| F | 49.722894 | 49.6891 |

The superplasticizer coefficient is -1.6161603 versus -1.6063 in the report;
the square-root-water coefficient is 564.8709806 versus 564.3547. These
differences exceed rounding of the reported coefficients. The full 24-term
candidate also differs slightly: RMSE 6.4865967 versus 6.4876 in the report.

The original model runner or exact input file is still needed to identify
the cause. Rounding predictor values to one/two decimal places and the
response to two decimal places did not recover an exact match. Do not
describe a different dataset version as established fact.

In this reconstruction, 156 of 206 final test rows were used during feature
selection. The candidate and selected models also use different native
test splits, so their test scores are not a comparison on identical rows.

## ScalaTion Airfoil: partial consistency only

The CSV has the correct reported dimensions and variables. The response
variance of the second native split is 49.5196252. The report's R²=0.6826
and RMSE=3.9647 imply a variance between 49.5147255 and 49.5328267 after
allowing four-decimal rounding. These are consistent, as are the analogous
second-split checks for Auto MPG and Concrete.

This does not verify Airfoil's predictions or model coefficients. The report
leaves its custom transformed term undefined and prints only some of the
coefficients. No matching Airfoil experiment runner was found. The custom
term definition and original code are required for a complete reproduction.

## PySR: printed equations evaluated on the stated splits

With the local CSV files and scikit-learn's 80/20 split, random_state=42:

| Dataset | R² | RMSE | MSE | MAE |
|---|---:|---:|---:|---:|
| Auto MPG | 0.851891216 | 2.749465170 | 7.559558723 | 2.055022308 |
| Concrete | 0.835725677 | 6.506180618 | 42.330386236 | 5.360601707 |
| Airfoil | 0.815874245 | 3.037176071 | 9.224438484 | 2.352891151 |

Auto MPG and Airfoil agree at the displayed precision. Concrete's printed
equation gives MSE 42.3304 and MAE 5.3606, versus 42.3301 and 5.3603 in the
report. Float32 evaluation gives essentially the same mismatch; original
unrounded equation coefficients could still explain it. RMSE and R² agree.
The model search itself was not rerun.

## Evidence

- `project2/audit/SymbolicDataCheck.scala`: diagnostic reconstruction; not
  claimed to be the missing original experiment code.
- `project2/results/symbolic_audit/`: native second-split row indices,
  complete selected coefficients, test predictions, and Scala execution log.
- Prediction-file R², RMSE, and MAE were independently recalculated with
  NumPy, and actual responses checked against the CSV by row index.
- `project2/verify_full_report.py`: reproduces the PySR equation calculations.
