# Beyond Aggregate Performance: Individual Prediction Stability Under Resampling in WDBC

> **Research question:** If we retrain the same model on slightly different samples, how much do its predictions for individual cases change, even when overall performance stays similar? **

## Overview

Biomedical machine-learning models are commonly evaluated using aggregate metrics such as ROC-AUC, accuracy, or Brier score. These quantities summarize predictive performance across a sample, but they do not directly describe how sensitive the prediction assigned to a particular observation is to changes in the data used to develop the model.

This project investigates that distinction using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset as a controlled methodological case study.

The primary objective is to characterize two different properties of a prediction procedure:

1. **aggregate predictive performance under repeated model redevelopment**, and
2. **observation-level stability of out-of-sample predicted probabilities under the same redevelopment process**.

Individual prediction stability is summarized using:

\[
W_i = Q_{0.90}(p_{ir}) - Q_{0.10}(p_{ir}),
\]

where \(p_{ir}\) is the genuine out-of-sample predicted probability assigned to observation \(i\) during redevelopment repetition \(r\).

The project is **descriptive and methodological**. It is not intended to optimize predictive accuracy, develop a new breast-cancer classifier, identify biomarkers, or establish clinical validity.

---

## Scientific motivation

A modeling procedure can produce similar aggregate performance across repeated development samples while still assigning meaningfully different predicted probabilities to some individual observations.

Therefore:

\[
\text{stable aggregate performance}
\]

does not logically imply:

\[
\text{stable predictions for every observation}.
\]

This project empirically examines whether that distinction is consequential in WDBC rather than assuming that it is.

Two contrasting learning procedures are examined:

- L2-regularized Logistic Regression;
- Random Forest.

A prevalence-only predictor is included as a reference control.

The objective is not to determine which algorithm is the "best" classifier, but to study how aggregate and individual-level stability behave under different learning procedures.

---

## Dataset

The analysis uses the **Wisconsin Diagnostic Breast Cancer dataset** distributed through `scikit-learn`.

The frozen analysis dataset contains:

- **569 observations**
- **30 quantitative nuclear-morphology predictors**
- **212 malignant observations**
- **357 benign observations**

The outcome is explicitly recoded as:

\[
Y = 1 \quad \text{malignant}
\]

\[
Y = 0 \quad \text{benign}.
\]

All 30 predefined predictors are retained.

No outcome-driven feature selection or dimensionality reduction is performed.

The analysis concerns **tabular cytomorphometric features derived from digitized fine-needle aspirate images**. It is not a raw medical-image classification study.

---

## Study design

The primary experiment uses:

**Repeated stratified 5-fold cross-validation with 50 repetitions.**

For each repetition:

- the dataset is partitioned into five stratified folds;
- each observation appears exactly once in a held-out test fold;
- every observation receives one genuine out-of-sample predicted probability;
- the same fold assignments are used for all learning procedures.

Across 50 repetitions, each observation therefore receives:

\[
50
\]

genuine out-of-sample predicted probabilities.

The complete frozen resampling assignment is stored in:

```text
artifacts/splits/resampling_manifest.csv
```

The repetitions are interpreted as repeated realizations of a specified redevelopment procedure **conditional on the observed WDBC dataset**.

They are not treated as 50 independent patient cohorts.

---

## Modeling procedures

### Prevalence-only control

Each held-out observation receives the malignancy prevalence estimated from its corresponding training fold.

This model uses no morphological predictors and serves as:

- a pipeline sanity check;
- a reference for probabilistic performance;
- an illustration that predictive usefulness and prediction stability are different properties.

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

Standardization is learned exclusively from the corresponding training fold.

No feature selection or hyperparameter tuning is performed.

### Random Forest

The Random Forest procedure is frozen in the analysis protocol before inspection of any Random Forest results.

The primary specification is:

```text
n_estimators = 2000
criterion = "gini"
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = "sqrt"
bootstrap = True
class_weight = None
```

A fixed primary random seed is used so that intentional variation in Random Forest construction is not confounded with variation caused by changing the training sample.

A separate prespecified diagnostic assesses whether residual finite-forest randomness is sufficiently small.

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

It describes the central 80% spread of predicted probabilities assigned to the same fixed observation across repeated model redevelopment.

For each observation, the analysis also retains:

- median predicted probability;
- P10 predicted probability;
- P90 predicted probability;
- prediction variance across repetitions.

\(W_i\) is **not** interpreted as:

- a confidence interval;
- a prediction interval;
- a Bayesian credible interval;
- uncertainty in a patient's true malignancy probability;
- an intrinsic biological property of the observation.

---

## Aggregate performance

Two global predictive quantities are evaluated.

### Brier score

For each repetition, the five held-out folds are reconstructed into one complete vector of 569 genuine out-of-sample predictions.

The Brier score is calculated across this complete OOS vector:

\[
B_r =
\frac{1}{569}
\sum_{i=1}^{569}
(y_i-p_{ir})^2.
\]

### ROC-AUC

ROC-AUC is calculated independently within each held-out fold.

The five fold-specific AUC values within a repetition are then combined using positive-negative pair weights:

\[
w_{rk}=n_{+,rk}n_{-,rk},
\]

giving:

\[
AUC_r =
\frac{
\sum_k w_{rk}AUC_{rk}
}{
\sum_k w_{rk}
}.
\]

This avoids treating rankings produced by different fold-specific models as though they were generated by one common fitted model.

---

## Statistical interpretation

This project does not treat cross-validation repetitions as independent population samples.

Accordingly:

- folds are not treated as independent observations;
- repetitions are not interpreted as newly recruited cohorts;
- the \(569 \times 50\) stored predictions are not analyzed as independent patient observations;
- P10–P90 ranges are not labeled confidence intervals;
- conventional hypothesis tests across CV repetitions are not part of the primary analysis.

Results are interpreted as **conditional resampling sensitivity** of the specified learning procedure within the observed finite WDBC cohort.

---

## Current status

Stages 1–4 are complete:

- dataset provenance and structural audit;
- frozen analysis data contract;
- frozen repeated stratified 5-fold resampling design;
- prevalence-only control;
- Logistic Regression procedure;
- 28,450 genuine OOS Logistic Regression predictions;
- aggregate ROC-AUC and Brier analysis;
- individual prediction-stability analysis;
- Monte Carlo approximation diagnostic;
- descriptive inspection of highly resampling-sensitive observations.

The repository is currently frozen at the **pre-Random-Forest checkpoint**.

The complete Random Forest specification and its algorithmic-randomness diagnostic were committed before inspection of any Random Forest results.

---

## Preliminary Logistic Regression result

Under the specified repeated redevelopment procedure, Logistic Regression showed little variation in aggregate predictive performance:

```text
Mean ROC-AUC: 0.9946
SD ROC-AUC:   0.0013

Mean Brier:   0.0205
SD Brier:     0.0017
```

Observation-level prediction stability was more heterogeneous.

The individual resampling width distribution was:

```text
Median W_i: 0.0010
P90 W_i:    0.0726
P95 W_i:    0.1573
P99 W_i:    0.2714
Maximum:    0.8518
```

These results are currently interpreted only for the prespecified Logistic Regression procedure.

No claim of generality is made until the Random Forest analysis is completed.

---

## Repository structure

```text
WDBC-prediction-stability/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── protocol/
│   ├── research_proposal.md
│   └── analysis_plan_v1.md
│
├── 01_wdbc_prediction_stability.ipynb
│
└── artifacts/
    └── splits/
        └── resampling_manifest.csv
```

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
- Random Forest randomness diagnostic;
- protocol-change rules.

The document explicitly records that it was consolidated after completion of the prevalence-control and Logistic Regression stages but **before inspection of Random Forest results**.

### `01_wdbc_prediction_stability.ipynb`

Contains the executable analysis and methodological audit trail.

### `artifacts/splits/resampling_manifest.csv`

Contains the complete frozen mapping of observation, repetition, and held-out fold used in the primary experiment.

---

## Reproducibility

The project preserves:

- dataset source and representation;
- dataset fingerprint;
- software-version provenance;
- explicit malignancy-positive outcome coding;
- predictor ordering;
- stable internal observation IDs;
- master resampling seed;
- all repetition-specific seeds;
- complete cross-validation assignments;
- fixed model configurations;
- model-specific random seeds;
- optimization diagnostics;
- genuine out-of-sample predictions;
- primary metric definitions;
- analysis decisions and protocol boundaries.

The goal is for every reported primary result to be traceable to a specific dataset representation, model procedure, and resampling realization.

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

These may become separate studies only if they address a distinct scientific question.

---

## Claims not supported by this project

The analysis does not establish that:

- either model is clinically validated;
- predicted probabilities are calibrated patient cancer risks;
- WDBC represents a contemporary clinical target population;
- prediction instability represents biological heterogeneity;
- unstable predictions identify intrinsically uncertain patients;
- stable predictions imply clinical trustworthiness;
- any predictor is a validated biomarker;
- any predictor causes malignancy;
- one learning algorithm is clinically superior to another;
- the resampling distribution represents uncertainty across independent future patient cohorts.

---

## Project objective

The project is considered successful if it rigorously answers:

> **Can apparently stable aggregate predictive performance coexist with heterogeneous stability of individual out-of-sample predicted probabilities?**

No particular result is required.

The objective is to characterize the behavior of the modeling procedures transparently and reproducibly rather than to demonstrate that instability must exist.

---

## License

Code and repository materials are released under the **MIT License**.

The Wisconsin Diagnostic Breast Cancer dataset is not authored by this repository and remains subject to the terms of its original source and distribution.
