# Analysis Plan v1

## Analysis dataset

WDBC will be loaded using `sklearn.datasets.load_breast_cancer`.

All 569 observations and all 30 quantitative predictors will be retained.

The target will be explicitly recoded as:

- malignant = 1
- benign = 0

Dataset audit performed before model fitting confirmed:

- 30 expected predictors;
- expected 10 × 3 feature structure;
- no missing values;
- no exact duplicate predictor rows.

No outcome-driven feature exclusion or feature selection will be performed.
