# Analysis Plan v1.0

## Project

**Beyond Aggregate Performance: Individual Prediction Stability Under Resampling in the Wisconsin Diagnostic Breast Cancer Dataset**

## Protocol status

**Protocol consolidation checkpoint: after completion of the prevalence-control
and Logistic Regression analyses, and before inspection of any Random Forest
results.**

This document consolidates the design decisions recorded sequentially in the
analysis notebook.

It should not be interpreted as a retrospective claim that the complete
document was preregistered before all analyses.

Stages 1–4 were developed and documented sequentially before Random Forest
analysis. The Random Forest procedure and its algorithmic-randomness diagnostic
are frozen in this version before any Random Forest results are inspected.

Any subsequent change to the analysis described here must be documented as a
protocol amendment rather than silently incorporated into the primary analysis.

---

# 1. Scientific question

The primary research question is:

> **How stable are out-of-sample predicted probabilities for the same WDBC
> observations when the model is repeatedly redeveloped using slightly
> different training samples, relative to the stability of aggregate predictive
> performance?**

In simpler terms:

> If we rebuild the same model using slightly different training data, does the
> same observation receive approximately the same predicted probability even
> when overall model performance remains similar?

The study is descriptive and methodological.

It is not intended to develop a new breast-cancer classifier, identify novel
biomarkers, establish clinical utility, or estimate causal biological effects.

---

# 2. Dataset

The study uses the Wisconsin Diagnostic Breast Cancer dataset distributed
through:

`sklearn.datasets.load_breast_cancer`

The frozen analysis dataset contains:

- 569 observations;
- 30 quantitative nuclear-morphology predictors;
- 212 malignant observations;
- 357 benign observations.

The predictors are derived from digitized fine-needle aspirate images of breast
masses.

The 30 predictors correspond to 10 underlying morphological characteristics,
each represented by three summaries:

- mean;
- standard error;
- worst.

The "worst" representation refers to the mean of the three largest measurements
for the corresponding characteristic.

All 30 predefined predictors are retained.

No predictor is removed based on association with the outcome or subsequent
model performance.

---

# 3. Observational unit

The analysis unit is one WDBC breast-mass case represented by a 30-dimensional
cytomorphometric feature vector.

The analysis is therefore a tabular biomedical machine-learning study based on
image-derived morphology variables.

It is not a raw medical-image classification study.

Project-specific identifiers:

`WDBC_000` through `WDBC_568`

are used only to track observations across resampling repetitions.

They are not original patient identifiers and are never used as model inputs.

---

# 4. Outcome coding

The scikit-learn representation uses:

- `0 = malignant`
- `1 = benign`

For this project the outcome is explicitly recoded as:

\[
Y = 1 \quad \text{malignant}
\]

\[
Y = 0 \quad \text{benign}
\]

All predicted probabilities reported in the study therefore refer to the
predicted probability of malignancy.

The positive-class probability column is verified explicitly from
`model.classes_` rather than assumed from column position.

---

# 5. Data-quality checks

Before model development, the following properties were verified:

- predictor matrix shape: 569 × 30;
- all 30 expected predictors are present;
- expected 10 × 3 feature structure is present;
- no missing predictor values;
- no missing outcomes;
- no exact duplicate predictor rows;
- no constant predictors;
- 212 malignant and 357 benign outcomes;
- unique internal observation IDs.

The absence of exact duplicate predictor rows does not establish statistical
independence of the source observations.

No imputation is required.

---

# 6. Frozen analysis data contract

All subsequent analyses operate on the same data representation:

\[
X \in \mathbb{R}^{569 \times 30}
\]

and:

\[
y \in \{0,1\}^{569},
\]

where `1` denotes malignancy.

The following are never included in the predictor matrix:

- internal observation identifiers;
- the diagnostic outcome.

Predictor order is preserved exactly as distributed in the frozen analysis
representation.

The dataset fingerprint and software versions are recorded in the analysis
notebook.

Any future modification of the analysis dataset must be treated as a protocol
change.

---

# 7. Primary resampling design

The primary experiment uses:

\[
\text{Repeated stratified 5-fold cross-validation}
\]

with:

- 5 folds;
- stratification by outcome;
- shuffling enabled;
- 50 repetitions;
- one reproducible random seed per repetition;
- the same partitions for all modeling procedures.

The master resampling seed is:

`20260923`

The 50 repetition seeds are deterministically derived from this master seed
using NumPy `SeedSequence`.

The complete resampling assignment is frozen in:

`artifacts/splits/resampling_manifest.csv`

The manifest contains:

\[
569 \times 50 = 28,450
\]

observation–repetition out-of-sample assignments.

For every repetition:

- each observation appears exactly once in a test fold;
- each observation is absent from the corresponding training set;
- all 569 observations therefore receive exactly one genuine out-of-sample
  prediction.

Across 50 repetitions, every observation receives exactly 50 genuine
out-of-sample predictions.

No train/test overlap is permitted.

---

# 8. Interpretation of the resampling experiment

The 50 repetitions are repeated realizations of a prespecified redevelopment
procedure conditional on the observed WDBC dataset.

They are not 50 independent patient cohorts.

Repeated use of the same observations means that folds and repetitions are not
treated as independent population samples.

The resulting variability therefore characterizes sensitivity of the specified
model-development procedure to this finite-cohort resampling mechanism.

It does not estimate uncertainty that would necessarily arise from independently
recruiting new development cohorts.

---

# 9. Modeling procedures

Three procedures are examined:

1. prevalence-only control;
2. L2-regularized Logistic Regression;
3. Random Forest.

The objective is not to identify the best-performing algorithm.

The procedures provide different reference learning mechanisms under which
aggregate and observation-level stability can be examined.

No hyperparameter optimization is part of the primary experiment.

---

# 10. Prevalence-only control

For each outer training fold, the prevalence-only predictor assigns every
corresponding test observation:

\[
\hat p =
\frac{\text{number of malignant training observations}}
{\text{number of training observations}}
\]

No predictor variables are used.

This model serves as:

- a probability-performance reference with no morphological information;
- a pipeline sanity check;
- an example showing that prediction stability and discriminative ability are
  distinct properties.

Within any single held-out fold, all observations receive the same probability,
so within-fold ROC-AUC must equal 0.5.

---

# 11. Logistic Regression procedure

The prespecified Logistic Regression procedure is:

`StandardScaler → LogisticRegression`

with:

- training-only standardization;
- L2 regularization;
- `C = 1.0`;
- `solver = "lbfgs"`;
- `max_iter = 5000`;
- no class weighting;
- no feature selection;
- no hyperparameter tuning.

Under the scikit-learn version used in the analysis, the default
`l1_ratio = 0.0` corresponds to pure L2 regularization. The deprecated
`penalty` parameter is therefore not explicitly specified.

A fresh pipeline is constructed for every fold.

The scaler is fitted exclusively to the corresponding training observations.

The Logistic Regression procedure is deterministic conditional on the
training data.

`C = 1.0` is treated as a prespecified reference configuration and is not
claimed to be optimal for WDBC.

---

# 12. Logistic Regression optimization audit

Optimizer convergence is checked for every Logistic Regression fit.

A fit is considered operationally acceptable if the optimizer terminates before
the prespecified:

`max_iter = 5000`

limit and does not generate a convergence failure.

Optimization diagnostics are retained separately from predictive results so
that model instability is not confused with failure of numerical optimization.

---

# 13. Random Forest procedure

The Random Forest procedure is frozen before inspection of any Random Forest
results.

The prespecified configuration is:

- `n_estimators = 2000`;
- `criterion = "gini"`;
- `max_depth = None`;
- `min_samples_split = 2`;
- `min_samples_leaf = 1`;
- `max_features = "sqrt"`;
- `bootstrap = True`;
- `max_samples = None`;
- `class_weight = None`;
- no feature selection;
- no hyperparameter tuning.

The primary Random Forest seed is:

`20260925`

The same seed is used for every primary Random Forest fit.

Fixing the seed makes the forest reproducible conditional on a given training
set and prevents intentional variation of algorithmic randomness from being
confounded with the primary training-sample perturbation.

Parallel computation settings such as `n_jobs` are computational choices and
are not treated as statistical hyperparameters.

---

# 14. Random Forest algorithmic-randomness diagnostic

Before running the complete Random Forest resampling analysis, finite-forest
randomness will be evaluated on the first prespecified outer train/test split.

The training and test observations will remain fixed while Random Forest is
refitted using the following seeds:

`101, 202, 303, 404, 505`

with all other Random Forest parameters unchanged.

For every held-out observation, the standard deviation of predicted malignancy
probabilities across the five seeds will be calculated.

The diagnostic summary will include:

- median prediction SD;
- P90 prediction SD;
- P95 prediction SD;
- P99 prediction SD.

A pragmatic computational tolerance of:

\[
P90(\text{seed-induced prediction SD}) \leq 0.005
\]

will be used to judge whether finite-forest Monte Carlo variability is small
relative to the probability scale.

This is a computational tolerance, not a biological or clinical threshold.

If the tolerance is exceeded:

1. `n_estimators` will be increased uniformly to 4000 and the diagnostic
   repeated;
2. if still exceeded, `n_estimators` will be increased to at most 8000;
3. if the tolerance remains exceeded at 8000 trees, the residual algorithmic
   variability will be reported rather than continuing to increase model size.

No seed will be selected on the basis of predictive performance.

The primary seed remains fixed independently of diagnostic performance.

---

# 15. Primary individual-level estimand

For observation \(i\) and repetition \(r\), let:

\[
p_{ir}
\]

denote the genuine out-of-sample predicted probability of malignancy.

For every observation, the primary stability measure is:

\[
W_i =
Q_{0.90}(p_{ir})
-
Q_{0.10}(p_{ir}).
\]

This quantity is termed:

**individual prediction resampling width**.

Quantiles are calculated using:

`method = "linear"`

with exactly 50 out-of-sample probabilities per observation.

\(W_i\) measures the central 80% spread of predictions assigned to the same
fixed observation across repeated realizations of the specified redevelopment
procedure.

It is not:

- a confidence interval;
- a Bayesian credible interval;
- a prediction interval;
- uncertainty around a patient's true malignancy probability;
- an intrinsic property of the patient.

---

# 16. Individual-level secondary summaries

For each observation, the following are retained:

- median out-of-sample predicted probability;
- P10 predicted probability;
- P90 predicted probability;
- prediction variance across repetitions;
- number of OOS predictions.

The median probability is always interpreted alongside \(W_i\), because a
given probability-scale width may have different implications depending on
where it occurs on the bounded `[0,1]` scale.

---

# 17. Cohort-level prediction-stability summaries

The primary cohort summaries are:

\[
\operatorname{Median}(W_i)
\]

and:

\[
P_{90}(W_i).
\]

The complete empirical distribution of \(W_i\) is also retained.

Particular observations may be inspected descriptively to understand how a
large P90–P10 width can arise, but individual cases will not be converted into
new clinical subgroups or phenotypes.

The study will not rank observations as though the ordering of individual
instability were known precisely.

---

# 18. Aggregate probabilistic performance

Brier score is the primary measure of overall probabilistic predictive
accuracy.

For repetition \(r\), the five held-out folds are reconstructed into one
complete vector containing one genuine OOS probability for each of the 569
observations.

The repetition-level Brier score is:

\[
B_r =
\frac{1}{569}
\sum_{i=1}^{569}
(y_i-p_{ir})^2.
\]

One Brier score is therefore produced per repetition.

The 50 Brier values describe resampling sensitivity of aggregate probabilistic
performance.

---

# 19. Aggregate discrimination

ROC-AUC represents discrimination.

A pooled AUC across predictions produced by different fold-specific models is
not used as the primary AUC estimator.

Instead, ROC-AUC is calculated independently within each outer test fold.

For repetition \(r\), the five fold-specific AUCs are combined using
positive-negative pair weights:

\[
w_{rk}
=
n_{+,rk}n_{-,rk}.
\]

The repetition-level AUC is:

\[
AUC_r =
\frac{
\sum_k w_{rk} AUC_{rk}
}{
\sum_k w_{rk}
}.
\]

This produces one aggregate AUC value per repetition without treating folds as
independent inferential observations.

---

# 20. Comparison of global and individual stability

Aggregate metrics and individual resampling widths are different quantities
measured on different scales.

The study therefore will not numerically compare, for example:

`SD(AUC)` versus `Median(W_i)`

as though one could determine which is "more stable" from their raw numerical
magnitudes.

Instead, the analysis asks whether aggregate performance displays a narrow or
broad resampling distribution while individual prediction sensitivity is
homogeneous or heterogeneous across observations.

The scientific comparison is therefore qualitative and distributional rather
than a direct ratio of incompatible stability measures.

---

# 21. Reference-threshold sensitivity

A probability threshold of:

\[
0.5
\]

may be used descriptively to identify observations whose central P10–P90
resampling range crosses that value.

This analysis is secondary and descriptive.

The value 0.5 is treated only as a mathematical reference threshold.

No clinical decision meaning is assigned to it.

Probability stability remains the primary object of study rather than
classification-label stability.

---

# 22. Monte Carlo approximation precision

The number of redevelopment repetitions determines how accurately the empirical
resampling distributions are numerically approximated.

For Logistic Regression, numerical precision of the two primary cohort-level
stability summaries was evaluated by resampling complete repetition columns of
the 569 × 50 probability matrix.

The diagnostic used:

- 2000 Monte Carlo bootstrap resamples;
- complete repetitions as the resampling unit;
- Monte Carlo seed `20260924`.

This diagnostic quantifies sensitivity to having only a finite number of
resampling realizations.

It is not a population bootstrap and does not produce clinical or population
confidence intervals.

The 50-repetition design was retained after this numerical diagnostic.

This stopping decision is documented transparently as a numerical diagnostic
performed during the sequential analysis and is not presented as a
preregistered population-inference rule.

Random Forest will use the same frozen set of 50 repetitions to preserve paired
comparison across learning procedures.

---

# 23. Statistical interpretation

Primary results are descriptive.

The following are not treated as independent observations:

- CV folds;
- CV repetitions;
- the 28,450 stored prediction rows;
- repeated predictions of the same WDBC observation.

No conventional population-level:

- t-tests;
- Wilcoxon tests;
- KS tests;
- significance tests across CV repetitions

form part of the primary analysis.

Resampling distributions are interpreted conditionally on:

- the observed WDBC dataset;
- the specified learning procedure;
- the specified resampling mechanism.

---

# 24. Exploratory diagnostic inspection

Highly resampling-sensitive observations may be inspected descriptively by
examining their complete set of 50 OOS probabilities.

The purpose is to determine whether a large \(W_i\) arises from patterns such
as:

- broad gradual variation;
- apparent multi-regime behavior;
- asymmetric tails;
- a small number of unusual realizations.

These inspections do not create new formal endpoints.

No post-hoc instability categories, clinical phenotypes, or model
hyperparameters will be defined from these cases.

---

# 25. Analyses excluded from Project 1

The primary project does not include:

- feature selection;
- SHAP;
- feature-reliance analysis;
- permutation importance;
- grouped permutation importance;
- training-size experiments;
- PCA;
- neural networks;
- XGBoost;
- conformal prediction;
- automatic probability recalibration;
- external validation;
- clinical threshold optimization;
- decision-curve analysis.

These analyses may be considered in later projects only if they answer a
separate scientific question.

---

# 26. Reproducibility requirements

The project preserves:

- dataset source;
- dataset representation;
- dataset fingerprint;
- software versions;
- outcome recoding;
- feature ordering;
- internal observation IDs;
- master resampling seed;
- repetition-specific resampling seeds;
- complete frozen resampling manifest;
- model configurations;
- model-specific random seeds;
- all genuine OOS predictions;
- fit diagnostics;
- generated metric tables;
- analysis decisions.

Canonical computation must be reproducible from the original dataset and the
frozen protocol.

---

# 27. Planned primary outputs

The final project will contain three principal figures.

### Figure 1 — Experimental design

Repeated redevelopment and repeated genuine out-of-sample prediction of the
same observations.

### Figure 2 — Aggregate performance stability

Resampling distributions of:

- ROC-AUC;
- Brier score

for Logistic Regression and Random Forest.

### Figure 3 — Individual prediction stability

Relationship between:

\[
\operatorname{Median}(p_i)
\]

and:

\[
W_i = P90_i-P10_i
\]

for the primary learning procedures.

A primary summary table will report, for each procedure:

- central AUC;
- AUC resampling spread;
- central Brier score;
- Brier resampling spread;
- median \(W_i\);
- P90 \(W_i\).

---

# 28. Supported claims

Depending on the observed results, the study may support statements such as:

> Under the specified redevelopment procedure, aggregate predictive performance
> showed relatively little resampling variation while observation-level
> prediction sensitivity was heterogeneous.

or:

> Aggregate performance and individual predicted probabilities were both highly
> stable under the specified redevelopment procedure.

Any conclusion will remain conditional on the observed WDBC dataset and the
prespecified modeling and resampling procedures.

---

# 29. Claims outside the scope

The study will not claim that:

- either model is clinically validated;
- predicted probabilities are calibrated patient cancer risks;
- WDBC represents a contemporary target clinical population;
- prediction instability represents biological heterogeneity;
- stable predictions imply clinical trustworthiness;
- unstable observations are intrinsically uncertain patients;
- any predictor is a validated biomarker;
- any predictor causes malignancy;
- one learning algorithm is clinically superior to another;
- the resampling distribution estimates independent-cohort uncertainty;
- Project 1 establishes external generalizability.

---

# 30. Success criterion

The project is successful if the prespecified experiment answers the primary
question rigorously and reproducibly.

No particular result is required.

The project remains informative if:

- both models are highly stable;
- one procedure is more sensitive to redevelopment than the other;
- aggregate and individual stability behave similarly;
- aggregate performance appears stable while individual stability is
  heterogeneous.

The objective is to characterize the behavior of the modeling procedure, not
to demonstrate instability.

---

# 31. Protocol freeze

After this file is committed at the pre-Random-Forest checkpoint:

- the frozen WDBC dataset is not changed;
- the 50 resampling repetitions are not changed;
- the Logistic Regression specification is not changed;
- the definition of \(W_i\) is not changed;
- the definitions of AUC and Brier score are not changed;
- the Random Forest specification is not changed in response to its predictive
  results.

Only the explicitly prespecified finite-forest randomness diagnostic may alter
the number of trees, according to the rule stated in Section 14.

Any other methodological modification requires a new protocol version with a
written rationale and changelog.