# Beyond Aggregate Performance: Individual Prediction Stability Under Resampling in WDBC

> **Question:** When the same biomedical prediction model is redeveloped on slightly different training samples, does the same observation receive approximately the same predicted probability even when aggregate performance remains stable?

---

## Overview

A biomedical machine-learning model can show highly stable aggregate performance (e.g., ROC-AUC or Brier score) while still assigning substantially different predicted probabilities to the same observation when redeveloped on slightly different training samples.

This project investigates that distinction using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset as a controlled methodological case study.

The primary objective is to study **prediction stability under repeated model redevelopment** and to distinguish observation-level stability ($W_i = P_{90}(p_i) - P_{10}(p_i)$) from conventional population-level performance metrics.

The project is descriptive and methodological: it does not seek to optimize predictive accuracy or develop a new clinical classifier.

---

## Scientific Motivation

Biomedical machine-learning studies commonly report aggregate metrics such as ROC-AUC, accuracy, or Brier score.

These metrics describe average predictive performance across a cohort, but they do not directly answer a critical individual-level question:

> How sensitive is the prediction assigned to a specific observation to the particular sample used to train the model?

A narrow distribution of aggregate performance across model redevelopment realizations does not logically guarantee that every observation receives a stable predicted probability. This project evaluates both properties simultaneously across competing model architectures (Logistic Regression and Random Forest).

---

## Dataset

The analysis uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset distributed through `scikit-learn`:

- **569 observations** ($N = 569$)
- **30 quantitative nuclear-morphology predictors**
- **212 malignant cases** ($Y = 1$)
- **357 benign cases** ($Y = 0$)

All 30 continuous features are retained without feature selection or dimensionality reduction.

---

## Repository Architecture

```text
WDBC-prediction-stability/
│
├── README.md
├── .gitignore
│
├── protocol/
│   ├── research_proposal.md      # Conceptual and scientific rationale
│   └── analysis_plan_v1.md       # Prespecified frozen analysis contract
│
├── 01_wdbc_prediction_stability.ipynb   # Primary analysis notebook (Stages 1–5)
│
└── artifacts/
    └── splits/
        └── resampling_manifest.csv  # Frozen 5-fold CV x 50 repetition manifest