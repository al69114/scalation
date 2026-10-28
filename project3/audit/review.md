# Project 3 review — October 27, 2026

Reviewed the attached `Data_Science_2_Project_3 (3).pdf` (22 pages), the rubric supplied in chat, all six Project 3 implementation files, saved tuning/QoF tables, and the relevant local ScalaTion implementation. The PDF is the reviewed report; local LaTeX edits made during this review may differ. Original notebooks, model code, and report were not edited by this audit.

## Overall assessment

The report covers every substantive modeling and reporting requirement in the supplied rubric. The main PyTorch results are correct and reproducible. Scala code compiles and runs, but exact Scala results cannot be certified from these files because training randomness is not fully controlled and original predictions/run logs are absent. Address the reproducibility and wording issues below before submission.

## Rubric coverage

| Requirement | Evidence | Assessment |
|---|---|---|
| 2L, ScalaTion and PyTorch | Section 1; NeuralNet_2L.scala and .ipynb | Present; zero hidden layers and identity output |
| 3L, ScalaTion and PyTorch | Section 2; NeuralNet_3L.scala and .ipynb | Present; one hidden layer |
| 4L, PyTorch | Section 3; NeuralNet_4L.ipynb | Present; two hidden layers |
| Pick an architecture, PyTorch | Section 5; NeuralNet_Custom.ipynb | Present; three hidden blocks with batch normalization and tunable dropout/weight decay |
| Hidden nodes, activation, batch size, learning-rate tuning | Code and report grids | Present for hidden-layer models; 2L appropriately has no hidden widths to tune and fixes identity output for regression |
| Same datasets | data/airfoil.csv, auto_mpg.csv, concrete.csv | 1,503 / 392 / 1,030 rows; 5 / 7 / 8 predictors; no missing numeric values |
| In-sample and 80/20 validation QoF | Tables 2, 4, 6, 8, 11, 14, 17 | R², MSE, RMSE reported for both sets |
| Discussion | Individual sections and Section 6 | Present, with several interpretation corrections below |
| PDF and code ZIP | Attached PDF; local files | PDF present. No Project 3 code-submission ZIP found in this workspace; overleaf_images.zip is not a code archive |

The extra 5L experiment is permitted supplementary work; it is not a separate requirement in the rubric supplied in chat. Cross-validation and five-seed studies are also supplementary, not rubric requirements.

## Numerical verification

`verify_results.py` loads the model/training definitions from the actual submitted files, reconstructs the 80/20 splits, refits scalers on training data, and retrains selected models. It avoids full search reruns and leaves original result files intact. All 15 final PyTorch models reproduce all six reported QoF metrics to four decimal places, including Tables 4, 8, 11, 14 and 17.

| PyTorch model | Airfoil validation R² | Auto MPG validation R² | Concrete validation R² |
|---|---:|---:|---:|
| 2L | 0.5528 | 0.7829 | 0.6236 |
| 3L | 0.9072 | 0.8526 | 0.8923 |
| 4L | 0.9643 | 0.8317 | 0.8691 |
| 5L | 0.9618 | 0.8813 | 0.8723 |
| Picked | 0.8688 | 0.8717 | 0.8677 |

Saved complete grids have 180 configurations per dataset for 3L, 144 for 4L, and 144 for the picked architecture. Their minimum-MSE configurations agree with the selections used for the reruns. The Scala 3L grid has 36 combinations, the 2L grids 21, and the 5L grid 96, as specified by the code. Full grids were not rerun, so this audit verifies the saved selections and final fits, not every historical tuning score.

The supplementary 4L seed sweep also reproduced the report's five-seed statistics; see `recomputed_4L_seeds.csv`. The other architectures' full seed sweeps were inspected but not rerun.

Both Scala project sources compiled against the local ScalaTion classes using the cached Scala 3.8.4 compiler. A headless wrapper retrained the six Scala models at the report's selected parameters. It confirmed Adam for 2L and SGDM for 3L. The observed validation R² values in this run were 0.4882 / 0.8039 / 0.6074 for 2L and 0.9114 / 0.8389 / 0.8865 for 3L. These differ from the report and do not establish that the reported historical values were wrong: the training uses random batch streams that are not fixed by `splitData(seed=42)`. See `scala_verification.log`.

## Corrections before submission

1. **Make Scala results reproducible and identify its optimizers.** `Optimizer.permGenerator` uses `RNGStream.ranStream`; that stream selector uses an unseeded Java RNG (`src/main/scala/scalation/random/RNG.scala`). The split seed controls observations, not batch order. Record/control training randomness, save final predictions and tuning logs, and regenerate report values if necessary. The active import in the local `NeuralNet_3L.scala` library selects `Optimizer_SGDM`; state momentum SGD in Section 2.1. The 2L library uses Adam.

2. **Include the missing 4L five-seed experiment.** The submitted 4L notebook ends with residual plots and has no seed sweep. Its `train_final_model` hardcodes `torch.manual_seed(42)`, so repeatedly calling it would train the same initialization. Yet pages 12 and 21 report five-seed statistics. Add a seed argument and an explicit sweep over seeds 0–4, and save its results. The separate `verify_4l_seeds.py` audit makes precisely this seed-parameter adaptation without editing the notebook; its rerun reproduced all three reported means and standard deviations to four decimal places: Airfoil 0.9610 ± 0.0082, Auto MPG 0.8525 ± 0.0255, Concrete 0.8738 ± 0.0111. The numbers are supported; the submission is missing the code that generates them. The other hidden-layer implementations already contain seed sweeps.

3. **Correct epoch and validation descriptions.** In Sections 1.1, 1.3 and 2.3, replace the unconditional statement “trained for 400 epochs” with “trained for up to 400 epochs using ScalaTion's training-loss stopping rule,” unless an original log establishes a full 400 epochs for that run. `Optimizer.hp` sets a maximum of 400, and both optimizers can stop early. Explain the inner 80/20 tuning split in both 2L methods and label Tables 1 and 3 “Inner-validation MSE.” Figures 4 and 14 are tuning histories, not the outer validation learning curves for the final retrained model; label them consistently with Figures 10, 18 and 24.

4. **Make the code submission runnable.** `project3/*.scala` is outside sbt's default source directories, and `build.sbt` does not include it. Include documented run commands that add those sources, or place them under the normal source tree. For example, from the repository root: `sbt 'set Compile / unmanagedSourceDirectories += baseDirectory.value / "project3"' 'runMain scalation.modeling.neuralnet.runNeuralNet2L'` (use `runNeuralNet3L` for 3L). This command is a proposed submission instruction, not a command verified through sbt in this audit; sbt itself encountered a sandbox lock-file permission restriction, so compilation was checked directly instead. Python notebooks expect their working directory to be `project3`; the 5L script's dataset paths expect the repository root. Document these different working directories and dependencies. Include the data or precise data setup instructions in the code ZIP. No code ZIP has been created by this review.

5. **Restore LaTeX figure assets if submitting/rebuilding the source.** Both local report directories are missing all 18 referenced `2L/*.png` files. The attached PDF contains the figures, so this does not invalidate that PDF, but the local LaTeX source cannot rebuild the full report as packaged. The rubric requires the PDF, not necessarily the LaTeX source; treat this as a reproducibility/packaging issue.

## Discussion changes

- Page 9: the 3L Auto MPG seed-42 score, 0.8526, is **above** the seeds-0–4 range [0.7959, 0.8453], rather than within its “high end.” Page 16 similarly has seed-42 5L Auto MPG 0.8813 above the reported five-seed maximum 0.8758. State that seed 42 is a separate run from seeds 0–4.
- Page 20: lower variability of the picked model is an observed result for the whole configuration. It does not isolate batch normalization and weight decay as the cause because width, activation, learning rate and training duration also vary. Suggested replacement: “The selected custom configuration had lower variability across these five training seeds for Auto MPG and Concrete; an ablation study would be needed to attribute the change to individual regularizers.”
- Likewise, lower Airfoil accuracy does not by itself demonstrate that batch normalization caused the drop. Present that as a possible explanation.
- Page 21: similar Concrete scores do not establish that data noise/size, rather than architecture, limits performance. Suggested replacement: “Within the tested architectures and grids, extra depth did not materially improve Concrete validation performance.”
- Page 3 compares residual spreads across targets with different units. Avoid treating raw dB, mpg, and MPa residual spreads as directly comparable performance measures.
- Five training seeds on one split measure training randomness (initialization and batch order), not uncertainty across different data splits. The report's final caveat about a single split is appropriate.
- The custom architecture was motivated by held-out performance of prior architectures. Its evaluation therefore has some exploratory reuse of that holdout; describe model rankings as exploratory rather than an untouched final benchmark. This is not a failure of the rubric's requested 80/20 approach.

## Additional dataset limitation

Concrete has 25 exact duplicate rows. With the PyTorch split, 6 of the 206 held-out rows have identical full rows in the training set. This is a limitation of the requested random row split, not evidence that scalers or tuning use validation targets. Mention it if discussing generalization; a duplicate-grouped split would be a separate robustness experiment and would change the reported results.

## What is correct in the code

All reviewed pipelines reserve the outer 20% before tuning and use inner validation on the remaining training data. Scaling is fit on the relevant training portion. Hidden-layer PyTorch models standardize the target and invert that transformation for QoF. Metrics use residual SSE over the correct target variance, MSE is averaged per observation, and RMSE is its square root. The network depths match the assignment's layer-count convention. Final PyTorch fits use the epoch selected during inner tuning; not restoring the tuning model's best checkpoint is therefore not a final-fit bug, because those tuning weights are discarded before retraining.

The report already discloses that Scala and PyTorch use different observations despite both using a seed value of 42. This is appropriate; identical integer seeds do not imply identical split algorithms across languages.
