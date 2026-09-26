# Beyond Aggregate Performance: Individual Prediction Stability Under Resampling in WDBC

> **Research question:** If the same prediction procedure is redeveloped on slightly different training samples, does the same observation receive approximately the same predicted probability even when aggregate predictive performance remains stable?

Two models can appear highly stable when judged by ROC-AUC or Brier score while still assigning substantially different predicted probabilities to some individual observations after modest changes in the data used for model development.

Aggregate performance metrics cannot reveal this behavior by themselves. They summarize performance across a sample; they do not directly measure how sensitive the prediction assigned to a particular observation is to repeated model redevelopment.

This repository studies that distinction using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset as a controlled methodological case study.

The aim is not to build a better breast-cancer classifier. It is to examine a narrower question about **prediction stability, reproducibility, and what aggregate metrics can and cannot tell us about individual predictions**.

---

## Overview

Biomedical machine-learning studies are commonly summarized using population-level quantities such as ROC-AUC, accuracy, sensitivity, specificity, or Brier score. These measures are important, but they answer questions about average predictive behavior across a dataset.

They do not answer a different question:

> If the model-development process were repeated using a slightly different training sample from the same observed cohort, how much would the predicted probability assigned to the same observation change?

This project separates these two levels of analysis.

The primary objectives are to characterize:

1. **aggregate predictive performance under repeated model redevelopment**, and
2. **observation-level stability of genuine out-of-sample predicted probabilities under that same redevelopment process**.

Individual prediction stability is summarized using

\[
W_i = Q_{0.90}(p_{ir}) - Q_{0.10}(p_{ir}),
\]

where \(p_{ir}\) is the genuine out-of-sample predicted probability assigned to observation \(i\) during redevelopment repetition \(r\).

\(W_i\) therefore describes the central 80% spread of the probabilities assigned to the same fixed observation across repeated redevelopment.

The study is **descriptive and methodological**. It is not intended to optimize predictive accuracy, develop a clinical classifier, identify biomarkers, or establish clinical validity.

---

## Scientific motivation

A prediction procedure can show little variation in aggregate performance across repeated development samples while still producing heterogeneous prediction stability at the level of individual observations.

In other words,

\[
\text{stable aggregate performance}
\]

does not logically imply

\[
\text{stable predictions for every observation}.
\]

The distinction matters because many biomedical ML evaluations stop at aggregate performance. A procedure may appear reproducible when judged by ROC-AUC or Brier score while the probability assigned to a particular case remains sensitive to which observations happened to be available for model development.

This project tests whether that distinction is consequential in WDBC rather than assuming that it is.

Two contrasting learning procedures are examined:

- **L2-regularized Logistic Regression**
- **Random Forest**

A **prevalence-only predictor** is included as a reference control.

The objective is not to rank algorithms or identify a clinically preferred model. The objective is to study how aggregate performance stability and individual prediction stability behave under different learning procedures when the redevelopment process is explicitly repeated.

---

## Dataset

The analysis uses the **Wisconsin Diagnostic Breast Cancer dataset** distributed through `scikit-learn`.

The frozen analysis dataset contains:

- **569 observations**
- **30 quantitative nuclear-morphology predictors**
- **212 malignant observations**
- **357 benign observations**

The outcome is explicitly recoded as

\[
Y = 1 \quad \text{malignant}
\]

and

\[
Y = 0 \quad \text{benign}.
\]

All 30 predefined predictors are retained.

No outcome-driven feature selection or dimensionality reduction is performed.

The analysis concerns **tabular cytomorphometric features derived from digitized fine-needle aspirate images**. It is not a raw medical-image classification study.

The dataset is intentionally modest. That is a feature of the design rather than a limitation to be hidden: the goal is to isolate and audit the behavior of a redevelopment procedure in a setting where the full prediction pipeline can be inspected, reproduced, and stress-tested.

---

## Study design

The primary experiment uses:

**Repeated stratified 5-fold cross-validation with 50 repetitions.**

For each repetition:

- the dataset is partitioned into five stratified folds;
- each observation appears exactly once in a held-out test fold;
- every observation receives one genuine out-of-sample predicted probability;
- the same fold assignments are used for all learning procedures.

Across 50 repetitions, each observation therefore receives exactly

\[
50
\]

genuine out-of-sample predicted probabilities per procedure.

This yields

\[
569 \times 50 = 28{,}450
\]

out-of-sample predictions for each learning procedure.

The complete frozen resampling assignment is stored in:

```text
artifacts/splits/resampling_manifest.csv
```

The resampling manifest was validated so that, within every repetition, each observation appears exactly once as an out-of-sample observation.

The repetitions are interpreted as repeated realizations of a specified redevelopment procedure **conditional on the observed WDBC dataset**.

They are not treated as 50 independent patient cohorts.

---

## Modeling procedures

### Prevalence-only control

For each held-out observation, the prediction is the malignancy prevalence estimated from the corresponding training fold.

This procedure uses no morphological predictors and serves as:

- a pipeline sanity check;
- a reference for probabilistic performance;
- a demonstration that predictive usefulness and prediction stability are distinct properties.

Because the training-fold prevalence changes slightly across resampling realizations, even this simple control produces a small non-zero resampling width.

---

### Logistic Regression

The prespecified Logistic Regression procedure is:

```text
StandardScaler
    ↓
L2-regularized LogisticRegression
```

with:

```text
C = 1.0
solver = "lbfgs"
max_iter = 5000
class_weight = None
```

Under the `scikit-learn` version used for the frozen analysis, the default `l1_ratio = 0.0` corresponds to L2 regularization.

Standardization is learned exclusively from the corresponding training fold.

No feature selection or hyperparameter tuning is performed.

The full analysis comprises 250 fitted Logistic Regression models:

\[
50 \text{ repetitions} \times 5 \text{ folds}.
\]

Optimizer diagnostics are retained for all fits.

---

### Random Forest

The Random Forest procedure and its finite-forest randomness diagnostic were specified before inspection of the final Random Forest results.

The structural model settings are:

```text
criterion = "gini"
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = "sqrt"
bootstrap = True
class_weight = None
```

A fixed primary random seed is used so that the main analysis isolates variation caused by changing the training sample rather than deliberately mixing redevelopment variability with Random Forest seed variability.

#### Finite-forest randomness diagnostic

A separate prespecified diagnostic assessed whether the number of trees was large enough for residual seed-induced prediction variability to be small relative to the redevelopment signal of interest.

The diagnostic began at 2,000 trees and escalated only if the prespecified criterion was not satisfied.

The criterion was:

\[
P90(\text{seed-induced prediction SD}) \leq 0.005.
\]

The observed diagnostic values were:

| Trees | Median seed SD | P90 | P95 | Maximum |
|---:|---:|---:|---:|---:|
| 2,000 | 0.001405 | 0.008233 | 0.010109 | 0.015602 |
| 4,000 | 0.001141 | 0.005394 | 0.007312 | 0.010564 |
| 8,000 | 0.000631 | 0.003893 | 0.004655 | 0.006756 |

The criterion was not met at 2,000 or 4,000 trees and was met at 8,000 trees.

The final primary Random Forest therefore uses:

```text
n_estimators = 8000
criterion = "gini"
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = "sqrt"
bootstrap = True
class_weight = None
```

This escalation was part of the prespecified diagnostic logic; it was not chosen after comparing Random Forest predictive performance.

---

## Primary individual-level estimand

For observation \(i\), the primary prediction-stability measure is:

\[
W_i =
Q_{0.90}(p_{ir})
-
Q_{0.10}(p_{ir}).
\]

This is termed the:

**individual prediction resampling width**.

It describes the central 80% spread of the predicted probabilities assigned to the same fixed observation across repeated model redevelopment.

Quantiles are computed using NumPy's `method="linear"` convention in the primary analysis.

For each observation, the analysis also retains:

- median predicted probability;
- P10 predicted probability;
- P90 predicted probability;
- prediction variance across repetitions;
- number of genuine out-of-sample predictions.

\(W_i\) is **not** interpreted as:

- a confidence interval;
- a prediction interval;
- a Bayesian credible interval;
- uncertainty in a patient's true malignancy probability;
- an intrinsic biological property of the observation.

It is a property of the **specified redevelopment procedure applied repeatedly to the observed dataset**.

---

## Aggregate predictive performance

Two global predictive quantities are evaluated.

### Brier score

For each repetition, the five held-out folds are reconstructed into one complete vector of 569 genuine out-of-sample predictions.

The Brier score is then calculated across the complete out-of-sample vector:

\[
B_r =
\frac{1}{569}
\sum_{i=1}^{569}
(y_i-p_{ir})^2.
\]

This produces one Brier score per redevelopment repetition.

---

### ROC-AUC

ROC-AUC is calculated independently within each held-out fold.

The five fold-specific AUC values within a repetition are then combined using positive-negative pair weights:

\[
w_{rk}=n_{+,rk}n_{-,rk},
\]

giving

\[
AUC_r =
\frac{
\sum_k w_{rk}AUC_{rk}
}{
\sum_k w_{rk}
}.
\]

This avoids treating rankings produced by different fold-specific fitted models as though they came from a single common model.

One weighted ROC-AUC is therefore retained per repetition.

---

## Statistical interpretation

This project does not treat cross-validation repetitions as independent population samples.

Accordingly:

- folds are not treated as independent observations;
- repetitions are not interpreted as newly recruited cohorts;
- the \(569 \times 50\) stored predictions are not analyzed as independent patient observations;
- P10–P90 ranges are not labeled confidence intervals;
- conventional hypothesis tests across CV repetitions are not part of the primary analysis.

Results are interpreted as **conditional resampling sensitivity** of the specified learning procedures within the observed finite WDBC cohort.

This distinction is central to the project.

The analysis characterizes what happened when the defined model-development procedure was repeatedly perturbed through resampling. It does not estimate a population-level distribution of prediction uncertainty for future patients.

---

## Primary results

### Aggregate performance remained narrowly distributed across redevelopment repetitions

The primary aggregate results were:

| Procedure | Mean ROC-AUC | SD ROC-AUC | Mean Brier | SD Brier |
|---|---:|---:|---:|---:|
| Prevalence control | 0.500000 | 0.000000 | 0.233774 | ~0 |
| Logistic Regression | 0.994556 | 0.001299 | 0.020547 | 0.001677 |
| Random Forest | 0.990640 | 0.000924 | 0.031696 | 0.001008 |

Both feature-based procedures therefore showed little repetition-to-repetition variation in aggregate predictive performance under the specified redevelopment experiment.

These values are descriptive. They are not used to declare one learning procedure clinically superior to the other.

---

### Individual prediction stability was more heterogeneous

The observation-level resampling-width distributions were:

| Procedure | Median \(W_i\) | P90 \(W_i\) | P95 \(W_i\) | P99 \(W_i\) | Maximum \(W_i\) |
|---|---:|---:|---:|---:|---:|
| Prevalence control | 0.002198 | 0.002198 | 0.002198 | 0.002198 | 0.002198 |
| Logistic Regression | 0.001004 | 0.072620 | 0.157334 | 0.271392 | 0.851826 |
| Random Forest | 0.010450 | 0.107858 | 0.128518 | 0.181158 | 0.347762 |

The principal observation is therefore not that the entire prediction distribution was unstable.

For most WDBC observations, redevelopment-induced changes were modest.

Instead, the distribution was heterogeneous: a subset of observations received materially different out-of-sample predicted probabilities depending on the particular training realization, despite narrow distributions of aggregate ROC-AUC and Brier score.

The most extreme Logistic Regression case reached a resampling width of approximately 0.85, whereas the maximum Random Forest width was approximately 0.35.

These extreme values are descriptive properties of this experiment and are not interpreted as patient-level uncertainty estimates.

---

## Cross-procedure comparison

The cross-procedure comparison was conducted after the primary procedure-specific analyses and is treated as **exploratory/descriptive**.

Across the 569 observations, Logistic Regression and Random Forest resampling widths were positively rank-correlated:

\[
\rho_{\mathrm{Spearman}} = 0.728.
\]

This indicates that observations that were relatively sensitive to redevelopment under one procedure also tended, to a substantial degree, to be relatively sensitive under the other.

However, the magnitude of instability was not identical across procedures.

Random Forest produced a larger \(W_i\) than Logistic Regression for:

\[
463/569 = 81.4\%
\]

of observations.

Conversely, the largest Logistic Regression widths were more extreme than the largest Random Forest widths.

Among the 57 observations in the highest 10% of \(W_i\) for each procedure:

- 32 observations were shared;
- 56.1% of the Logistic Regression upper-tail set also appeared in the Random Forest upper-tail set;
- the Jaccard index was 0.390.

These findings support a cautious interpretation: some redevelopment sensitivity appears to be shared across learning procedures, while procedure-specific behavior also remains important.

They do **not** establish a biological mechanism or a single intrinsic category of "unstable patients."

---

## Robustness analyses

Several secondary checks were used to determine whether the main descriptive pattern depended strongly on arbitrary analytic choices.

### Number of retained repetitions

For Logistic Regression, Monte Carlo approximation error of the empirical width distribution was assessed by repeatedly resampling subsets of the 50 repetitions.

For the median \(W_i\):

```text
Estimate: 0.001004
MC SE:    0.000036
95% Monte Carlo interval:
0.000931 to 0.001068
```

For the P90 of \(W_i\):

```text
Estimate: 0.072620
MC SE:    0.004159
95% Monte Carlo interval:
0.067244 to 0.083122
```

This diagnostic was designed to assess numerical approximation from using 50 repetitions, not biological or population uncertainty.

---

### Upper-tail membership stability

The stability of the highest-instability 10% was evaluated by repeatedly estimating the top-57 set using subsets of the 50 redevelopment repetitions.

At \(m=40\) retained repetitions:

- all 57 full-analysis Logistic Regression upper-tail observations had membership frequency \(\geq 0.80\);
- 54 of 57 had membership frequency \(\geq 0.90\);
- 46 of 57 Random Forest upper-tail observations had membership frequency \(\geq 0.80\);
- 41 of 57 had membership frequency \(\geq 0.90\).

The cross-procedure Jaccard overlap remained centered near the full-analysis value under these subsampling checks.

---

### Quantile-definition sensitivity

The primary \(W_i\) definition uses the linear quantile convention.

Sensitivity analyses repeated the cross-procedure upper-tail comparison using:

- `linear`
- `nearest`
- `lower`
- `higher`
- `midpoint`

Across all five conventions:

- the number of observations shared between the two top-57 sets remained exactly 32;
- the Jaccard index remained 0.390244;
- Spearman correlation remained approximately 0.726–0.730.

The Logistic Regression top-57 set was unchanged across methods. Random Forest showed only one or two membership substitutions depending on the quantile convention.

The rank-57/rank-58 boundaries were also checked for numerical indistinguishability at tolerance \(10^{-12}\); none was observed.

The main upper-tail comparison is therefore not an artifact of one specific quantile interpolation convention.

---

## Figures

The repository contains the primary figures in both PNG and PDF format.

### Figure 1 — Experimental design and stability framework

```text
artifacts/figures/experimental_design_stability_framework.png
artifacts/figures/experimental_design_stability_framework.pdf
```

This figure summarizes the redevelopment design and the distinction between aggregate performance and observation-level prediction stability.

### Figure 2 — Aggregate performance stability

```text
artifacts/figures/aggregate_performance_stability.png
artifacts/figures/aggregate_performance_stability.pdf
```

This figure shows ROC-AUC and Brier score across the 50 redevelopment repetitions for Logistic Regression and Random Forest.

### Figure 3 — Individual prediction stability

```text
artifacts/figures/individual_prediction_stability.png
artifacts/figures/individual_prediction_stability.pdf
```

This figure relates each observation's median out-of-sample predicted probability to its resampling width.

The vertical probability value of 0.5 is included only as a geometric reference. It is not presented as a clinically validated decision threshold.

### Supplementary Figure S1 — Cross-procedure resampling width

```text
artifacts/figures/cross_procedure_resampling_width.png
artifacts/figures/cross_procedure_resampling_width.pdf
```

This exploratory figure compares Logistic Regression and Random Forest \(W_i\) values observation by observation.

---

## Prespecified and exploratory analyses

The repository deliberately distinguishes between analyses defined before inspection of the relevant results and analyses developed after observing the primary findings.

The operational analysis plan was consolidated after completion of the prevalence-control and Logistic Regression stages but **before inspection of the Random Forest results**.

The Random Forest specification, finite-forest diagnostic, primary estimands, aggregate metrics, resampling design, and interpretation rules were therefore frozen before the final Random Forest analysis.

Subsequent analyses examining:

- cross-procedure \(W_i\) association;
- upper-tail overlap;
- top-decile membership stability;
- quantile-method sensitivity;

are labeled as **post-hoc or exploratory robustness analyses**.

They are included because they help characterize the observed phenomenon, not because they were part of the original primary estimand.

No attempt is made to retroactively present them as preregistered.

---

## Repository structure

```text
WDBC-prediction-stability/
│
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml
│
├── 01_wdbc_prediction_stability.ipynb
│
├── scripts/
│   └── run_wdbc_prediction_stability.py
│
├── protocol/
│   ├── research_proposal.md
│   └── analysis_plan_v1.md
│
├── artifacts/
│   ├── splits/
│   │   └── resampling_manifest.csv
│   │
│   ├── predictions/
│   │   ├── prevalence_oos_predictions.csv
│   │   ├── logistic_regression_oos_predictions.csv
│   │   └── random_forest_oos_predictions.csv
│   │
│   ├── diagnostics/
│   │   ├── logistic_regression_fit_diagnostics.csv
│   │   ├── logistic_regression_mc_precision_diagnostic.csv
│   │   ├── random_forest_finite_forest_diagnostic.csv
│   │   ├── random_forest_fit_diagnostics.csv
│   │   └── random_forest_fit_diagnostics_deterministic.csv
│   │
│   ├── results/
│   │   ├── main_results_table.csv
│   │   ├── aggregate_performance_stability_summary.csv
│   │   ├── prevalence_control_aggregate_metrics.csv
│   │   ├── logistic_regression_aggregate_metrics.csv
│   │   ├── random_forest_aggregate_metrics.csv
│   │   ├── prevalence_individual_stability.csv
│   │   ├── logistic_regression_individual_stability.csv
│   │   ├── random_forest_individual_stability.csv
│   │   ├── cross_procedure_individual_comparison.csv
│   │   ├── upper_tail_robustness_summary.csv
│   │   ├── upper_tail_membership_stability_summary.csv
│   │   ├── upper_tail_membership_m40_summary.csv
│   │   ├── upper_tail_tie_audit_summary.csv
│   │   ├── quantile_method_sensitivity.csv
│   │   └── quantile_method_membership_sensitivity.csv
│   │
│   └── figures/
│       ├── experimental_design_stability_framework.png
│       ├── experimental_design_stability_framework.pdf
│       ├── aggregate_performance_stability.png
│       ├── aggregate_performance_stability.pdf
│       ├── individual_prediction_stability.png
│       ├── individual_prediction_stability.pdf
│       ├── cross_procedure_resampling_width.png
│       └── cross_procedure_resampling_width.pdf
│
└── metadata/
    └── notebook_cleanup_20260926/
        ├── AUDIT.md
        └── verification.json
```

Generated runtime caches and notebook checkpoints are excluded from version control.

---

## Key repository components

### `protocol/research_proposal.md`

Defines the scientific motivation, research question, scope, and intended contribution.

### `protocol/analysis_plan_v1.md`

Contains the frozen operational analysis contract, including:

- data representation;
- outcome coding;
- resampling design;
- seeds;
- model specifications;
- primary estimands;
- aggregate metrics;
- statistical interpretation;
- Random Forest finite-forest diagnostic;
- protocol-change rules.

The document explicitly records the timing of protocol consolidation relative to completed and uninspected analyses.

### `01_wdbc_prediction_stability.ipynb`

Contains the complete narrative analysis, diagnostic checks, primary figures, robustness analyses, and methodological audit trail.

### `scripts/run_wdbc_prediction_stability.py`

Provides a non-interactive canonical implementation of the primary analysis pipeline.

It reconstructs the frozen dataset representation and resampling design, generates the primary out-of-sample predictions, and produces deterministic primary result artifacts.

### `artifacts/splits/resampling_manifest.csv`

Contains the complete frozen mapping between observation, redevelopment repetition, and held-out fold.

### `artifacts/predictions/`

Contains the genuine out-of-sample prediction records used to calculate the reported primary results.

### `artifacts/diagnostics/`

Contains model-fitting and numerical-stability diagnostics.

### `artifacts/results/`

Contains the tabular outputs used in the primary and robustness analyses.

---

## Reproducibility

Reproducibility is treated as part of the analysis rather than as an afterthought.

The project preserves:

- dataset source and representation;
- dataset fingerprint;
- software-version provenance;
- explicit malignancy-positive outcome coding;
- predictor ordering;
- stable internal observation IDs;
- master resampling seed;
- all repetition-specific fold assignments;
- fixed model configurations;
- model-specific random seeds;
- optimization diagnostics;
- genuine out-of-sample predictions;
- primary metric definitions;
- analysis decisions;
- protocol boundaries.

The goal is for every reported primary result to be traceable to a specific dataset representation, model procedure, and resampling realization.

---

### Environment

The frozen computational environment is defined in:

```text
environment.yml
```

Create it with:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate wdbc-prediction-stability
```

The validated environment used for the canonical analysis included:

```text
Python       3.13.12
NumPy        2.4.4
pandas       3.0.2
scikit-learn 1.8.0
```

Additional dependencies are specified directly in `environment.yml`.

---

### Canonical execution

From the repository root, run:

```bash
python scripts/run_wdbc_prediction_stability.py
```

The canonical script reproduces the primary computational pipeline, including:

- frozen repeated-stratified resampling assignments;
- prevalence-control out-of-sample predictions;
- Logistic Regression out-of-sample predictions;
- Random Forest out-of-sample predictions;
- repetition-level ROC-AUC and Brier metrics;
- observation-level prediction-stability summaries;
- model fit diagnostics;
- deterministic result artifacts.

The notebook remains the reference for the full narrative analysis and the later robustness/exploratory analyses.

---

### Byte-level verification

The final canonical Random Forest out-of-sample prediction file is:

```text
artifacts/predictions/random_forest_oos_predictions.csv
```

Its SHA-256 checksum is:

```text
7b3570fa69710f8e7bdf73031858e319b26992663a51cd5a92c367b864339420
```

The canonical analysis script reproduced this file byte-for-byte.

This provides a stronger reproducibility check than agreement of summary statistics alone.

Runtime-dependent quantities such as wall-clock fit time are intentionally excluded from deterministic hash-based verification.

---

## Analysis integrity checks

Several invariants are explicitly checked rather than assumed.

### Resampling integrity

The frozen manifest verifies that:

- every observation receives exactly one held-out prediction per repetition;
- no observation is assigned to more than one held-out fold within a repetition;
- the same fold assignments are used across learning procedures.

### Outcome orientation

The positive class is explicitly frozen as:

```text
malignant = 1
benign    = 0
```

This prevents silent reversal of ROC-AUC or probability interpretation.

### Prevalence-control sanity check

The prevalence-only procedure produced:

```text
ROC-AUC = 0.5
```

for all fold-level and repetition-level evaluations, as expected for a predictor that does not discriminate between observations within a fold.

This served as an early end-to-end check of outcome coding, prediction assembly, and metric computation.

### Logistic Regression fitting

All 250 Logistic Regression fits converged under the frozen configuration.

Observed optimizer iterations ranged from 14 to 22, with a mean of 18.36.

### Random Forest stochasticity

Residual finite-forest randomness was quantified separately before finalizing the number of trees.

The final 8,000-tree specification satisfied the prespecified precision criterion.

---

## What the results support

Within the observed WDBC dataset and the specified redevelopment process, the results support the following descriptive conclusions:

1. **Aggregate performance can remain narrowly distributed across repeated redevelopment while individual prediction stability is heterogeneous.**

2. **Most observations can be highly stable while a smaller subset shows much larger redevelopment-induced probability changes.**

3. **The identity of relatively sensitive observations shows substantial, but incomplete, agreement across Logistic Regression and Random Forest.**

4. **The magnitude and shape of redevelopment sensitivity depend in part on the learning procedure.**

5. **The main upper-tail comparison is not strongly dependent on the specific quantile interpolation convention used to define \(W_i\).**

These conclusions apply to the observed experiment.

They should not be generalized automatically to other datasets, clinical populations, tasks, or model-development pipelines.

---

## What the results do not establish

The analysis does **not** establish that:

- either model is clinically validated;
- predicted probabilities are calibrated patient cancer risks;
- WDBC represents a contemporary clinical target population;
- prediction instability represents biological heterogeneity;
- unstable predictions identify intrinsically uncertain patients;
- stable predictions imply clinical trustworthiness;
- any predictor is a validated biomarker;
- any predictor causes malignancy;
- one learning algorithm is clinically superior to another;
- redevelopment sensitivity observed here will have the same magnitude in another cohort;
- the resampling distribution represents uncertainty across independent future patient cohorts.

The project is deliberately careful about the distinction between **empirical redevelopment sensitivity** and broader forms of statistical, biological, or clinical uncertainty.

---

## Scope boundaries

This first project deliberately does **not** include:

- feature selection;
- SHAP;
- permutation importance;
- feature-reliance analysis;
- grouped importance analyses;
- PCA;
- neural networks;
- XGBoost;
- training-size experiments;
- external validation;
- probability recalibration;
- clinical threshold optimization;
- decision-curve analysis.

These are not omitted because they are unimportant.

They are omitted because each introduces a different scientific question.

The purpose of this project is to answer one narrow question cleanly rather than combine multiple partially developed analyses into a single repository.

---

## Limitations

Several limitations should be kept explicit.

### Single, modest-sized dataset

The study uses one historical benchmark dataset with 569 observations.

The results therefore characterize this particular experimental setting and should not be treated as evidence that the same instability distribution occurs in contemporary clinical datasets.

### Conditional resampling experiment

All redevelopment repetitions reuse the same finite cohort.

The resulting distributions describe sensitivity to training-set composition within that cohort; they do not reproduce the full uncertainty that would arise from recruiting independent future patient populations.

### No external validation

The project does not estimate transportability to another institution, scanner, laboratory, population, or time period.

### No clinical probability interpretation

Although the models output values between 0 and 1, those values are not treated as validated patient-level malignancy risks.

### Limited model family

The analysis compares a prevalence reference, regularized Logistic Regression, and Random Forest.

The project does not claim that these procedures span the full range of possible biomedical ML behavior.

### Mechanisms are not identified

The experiment can reveal which observations are more sensitive to redevelopment under the specified procedures.

It does not, by itself, establish why an observation is sensitive.

Any explanation involving geometry, correlation structure, local data density, leverage-like behavior, or biological heterogeneity would require a separate analysis specifically designed to test that mechanism.

---

## Project objective

The project is considered successful if it rigorously answers:

> **Can apparently stable aggregate predictive performance coexist with heterogeneous stability of individual out-of-sample predicted probabilities?**

No particular result is required.

The objective is to characterize the behavior of the modeling procedures transparently and reproducibly rather than to demonstrate that instability must exist.

More broadly, the repository is intended as a small example of a principle that is easy to lose in applied machine learning:

> A model-development pipeline should not be judged only by how well it performs on average. It can also be useful to ask how reproducibly it behaves for the individual observations to which those averages ultimately refer.

---

## License

Code and repository materials are released under the **MIT License**.

The Wisconsin Diagnostic Breast Cancer dataset is not authored by this repository and remains subject to the terms of its original source and distribution.
