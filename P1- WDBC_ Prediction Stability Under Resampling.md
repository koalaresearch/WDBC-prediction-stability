# Project 1 — Prediction Stability Under Resampling in WDBC

## Working title

**Beyond Aggregate Performance: Individual Prediction Stability Under Resampling in the Wisconsin Diagnostic Breast Cancer Dataset**

## 1. Background

Biomedical machine-learning models are commonly evaluated using aggregate performance metrics such as ROC-AUC, accuracy, or Brier score. These quantities describe average predictive performance across a sample, but they do not directly describe how sensitive the prediction assigned to a particular observation is to changes in the data used to develop the model.

This distinction matters because a modeling procedure may produce similar aggregate performance across repeated development samples while assigning meaningfully different predicted probabilities to some individual observations.

Prediction instability under resampling is an established methodological problem and should not be interpreted as uncertainty in a patient's true disease probability. Instead, it reflects sensitivity of a specified model-development procedure to changes in the observations available for fitting.

The Wisconsin Diagnostic Breast Cancer dataset (WDBC) provides a useful controlled case study for examining this distinction. It contains 569 observations and 30 quantitative nuclear-morphology predictors derived from digitized fine-needle aspirate images. The dataset is widely studied and generally permits high discrimination, making it suitable for asking whether apparently strong and stable aggregate performance necessarily implies stable predictions at the observation level.

This project is therefore not intended to develop a new breast-cancer classifier, identify novel biomarkers, or claim clinical validity. Its purpose is methodological: to examine what conventional aggregate predictive performance does and does not reveal about the stability of individual out-of-sample predictions.

## 2. Primary research question

**How stable are out-of-sample predicted probabilities for the same WDBC observations when the model is repeatedly redeveloped using slightly different training samples, relative to the stability of aggregate predictive performance?**

In simpler terms:

> If we rebuild the same model using slightly different data, does the same observation receive approximately the same prediction even when overall model performance remains similar?

## 3. Study objective

The primary objective is to quantify and compare two distinct properties of a biomedical prediction procedure:

1. **Aggregate predictive performance across repeated model redevelopment.**
2. **Observation-level stability of out-of-sample predicted probabilities across the same redevelopment process.**

The study will determine whether these two properties exhibit similar or different sensitivity to resampling.

The aim is descriptive and methodological rather than inferential or clinical.

## 4. Working hypothesis

A modeling procedure may exhibit relatively stable aggregate predictive performance while predictions for some individual observations remain sensitive to changes in the training sample.

This is a hypothesis to be tested, not an expected or desired result.

The study remains scientifically informative if individual predictions are highly stable.

## 5. Dataset

The study will use the Wisconsin Diagnostic Breast Cancer dataset.

The dataset contains:

- 569 observations;
- 30 quantitative predictors;
- nuclear-morphology measurements derived from digitized fine-needle aspirate images of breast masses;
- malignant and benign diagnostic labels.

Before modeling, dataset provenance, feature definitions, missingness, identifiers, duplicate observations, and class coding will be audited.

The outcome will be explicitly recoded as:

\[
Y=1 \quad \text{malignant}
\]

\[
Y=0 \quad \text{benign}.
\]

This is necessary because the scikit-learn representation uses malignant and benign class ordering that can otherwise produce probability-direction and metric errors.

## 6. Modeling procedures

The study will deliberately use a small model set.

### Prevalence-only predictor

A constant prediction equal to the prevalence of malignancy in the corresponding training sample.

This provides a reference model that contains no morphological information.

### Logistic regression

L2-regularized logistic regression with training-only standardization.

The procedure will be deterministic once the training observations are specified.

### Random forest

A prespecified random-forest procedure will provide a nonlinear comparison.

A sufficiently large number of trees and a fixed primary random seed will be used so that avoidable Monte Carlo variability from forest construction does not dominate the resampling experiment.

The purpose is not to determine which algorithm is the best classifier.

Instead, the two learning procedures provide different modeling assumptions under which prediction stability can be examined.

Model configurations will be frozen before inspection of the primary results.

## 7. Resampling design

The primary experiment will use:

\[
\text{Repeated stratified 5-fold cross-validation}.
\]

An initial 50 repetitions will be generated using independently specified resampling seeds.

The same folds will be used for all modeling procedures.

Within each repetition, every observation will appear exactly once in an outer test fold and therefore receive one genuine out-of-sample predicted probability.

After 50 repetitions, each observation will consequently have a set of repeated out-of-sample predictions:

\[
p_{i1},p_{i2},...,p_{i50}.
\]

The repetitions represent repeated realizations of a prespecified redevelopment procedure conditional on the observed WDBC dataset.

They are not 50 independent patient cohorts.

The number of repetitions may be increased only according to a prespecified numerical-precision criterion, not in response to whether the observed scientific result appears interesting.

## 8. Primary outcome: individual prediction stability

For observation \(i\), let:

\[
p_{ir}
\]

denote its out-of-sample predicted probability of malignancy in repetition \(r\).

The primary observation-level stability measure will be:

\[
W_i =
Q_{0.90}(p_{ir}) -
Q_{0.10}(p_{ir}).
\]

\(W_i\) measures the width of the central 80% of predictions produced for that observation under repeated redevelopment.

It will be described as an:

**individual prediction resampling width**

or

**descriptive resampling variation width**.

It is not a confidence interval, prediction interval, Bayesian credible interval, or estimate of uncertainty in a patient's true malignancy probability.

For each observation, the median predicted probability will always be reported alongside its resampling width because the same width can have very different interpretations depending on where on the probability scale it occurs.

Prediction variance across repetitions will also be retained as a secondary mathematical summary.

At the dataset level, the primary summaries will be the median and upper-tail distribution of \(W_i\), together with visualization of the relationship between median predicted probability and prediction instability.

## 9. Aggregate predictive performance

Two aggregate quantities will form the minimal global evaluation.

### Brier score

Brier score will represent overall probabilistic predictive accuracy.

For every repetition, the five held-out portions will collectively provide one out-of-sample prediction for each of the 569 observations.

The Brier score will be calculated over this complete out-of-sample set.

### ROC-AUC

ROC-AUC will represent discrimination.

Because pooling predictions produced by different fold-specific models can create cross-model ranking comparisons, ROC-AUC will first be calculated independently within each outer test fold.

The five fold-specific AUCs will then be combined using a positive-negative-pair-weighted average to obtain one AUC estimate per repetition.

Folds and repetitions will not be treated as independent clinical samples.

The resampling distributions of AUC and Brier score will therefore be interpreted descriptively as sensitivity of aggregate model performance to the specified redevelopment procedure.

## 10. Primary comparison

The main analysis will examine whether aggregate performance and observation-level predictions respond similarly to repeated redevelopment.

The study will not numerically compare an AUC spread with a probability spread as though they were quantities on the same scale.

Instead, it will examine their respective sensitivity profiles.

For example, the study may find that aggregate AUC and Brier score change little across repetitions while a subset of observations receives substantially different predicted probabilities.

Alternatively, both global and individual quantities may prove highly stable.

Both outcomes are scientifically valid.

## 11. Statistical interpretation

Repeated cross-validation does not generate new independent patient populations.

Therefore:

- folds will not be treated as independent observations;
- repetitions will not be interpreted as newly sampled clinical cohorts;
- the \(569\times R\) predictions will not be analyzed as independent patient predictions;
- P10–P90 intervals will not be labelled confidence intervals;
- conventional significance tests across CV repetitions will not form part of the primary analysis.

The primary results will consist of effect magnitudes, empirical distributions and visualizations of conditional resampling sensitivity.

Any numerical uncertainty arising solely from finite Monte Carlo repetition will be distinguished from scientific variability of the modeling procedure.

## 12. Planned outputs

The core project should require only a small number of principal results.

### Figure 1 — Experimental design

A schematic showing repeated model redevelopment and repeated out-of-sample predictions for the same observation.

### Figure 2 — Aggregate performance stability

Distributions of repeated:

\[
ROC\text{-}AUC
\]

and

\[
Brier\ score
\]

for Logistic Regression and Random Forest.

### Figure 3 — Individual prediction stability

Each observation shown according to:

\[
median(p_i)
\]

and

\[
P90(p_i)-P10(p_i).
\]

Selected observations may illustrate qualitatively different prediction patterns, but these examples will not be interpreted as clinical phenotypes.

### Main table

For each modeling procedure:

- central aggregate AUC;
- resampling spread of AUC;
- central Brier score;
- resampling spread of Brier score;
- median individual prediction width;
- 90th percentile of individual prediction width.

## 13. Reproducibility

The analysis will preserve:

- dataset source and version;
- data provenance;
- dataset hash where appropriate;
- environment and package versions;
- random seeds;
- all cross-validation assignments;
- model configurations;
- complete out-of-sample predictions;
- generated summary tables;
- analysis decisions.

The canonical model-fitting procedures will live in scripts or source modules rather than being hidden inside exploratory notebooks.

The complete experiment should run end-to-end from the original dataset to the final tables and figures.

## 14. Claims supported by the study

The study may support statements such as:

> Under this specified redevelopment procedure, aggregate predictive performance was relatively insensitive to resampling while some observation-level predicted probabilities showed greater variation.

or:

> Both aggregate predictive performance and individual predicted probabilities were highly stable under the examined redevelopment procedure.

The interpretation will remain conditional on:

- WDBC;
- the chosen learning procedures;
- the chosen resampling mechanism;
- and the observed finite cohort.

## 15. Claims explicitly outside the scope

This study will not claim that:

- the models are clinically validated;
- the predicted probabilities are calibrated patient cancer risks;
- WDBC represents a contemporary target clinical population;
- prediction instability represents biological heterogeneity;
- a stable model is clinically trustworthy;
- an unstable prediction identifies an intrinsically uncertain patient;
- any predictor is a validated biomarker;
- any predictor causes malignancy;
- one algorithm is clinically superior to another.

## 16. Expected contribution

The contribution is not a new classifier or a new statistical method.

The contribution is a transparent and reproducible methodological case study demonstrating an important distinction in biomedical machine learning:

\[
\boxed{\text{stable aggregate performance}}
\]

does not logically imply

\[
\boxed{\text{stable predictions for every observation}}.
\]

The project will empirically determine whether that distinction is consequential in WDBC rather than assuming that it is.

Its value therefore lies in scientific reasoning, careful validation, explicit estimands, reproducibility and appropriately constrained biomedical interpretation.

## 17. Scope boundary

This first project stops after answering the global-versus-individual prediction-stability question.

Feature-reliance stability, grouped permutation importance and training-size experiments are **not part of the core research question**.

They may be developed later as separate extensions or subsequent projects if the core study provides a strong scientific reason to investigate them.

The project is considered successful once the primary question has been answered rigorously, regardless of whether substantial instability is observed.