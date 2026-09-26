from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd

from numpy.random import SeedSequence

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Frozen analysis configuration
# ==========================================================

N_SPLITS = 5
N_REPEATS = 50

MASTER_RESAMPLING_SEED = 20260923

LR_C = 1.0
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 5000

RF_N_ESTIMATORS = 8000

QUANTILE_METHOD = "linear"


# ==========================================================
# Output directories
# ==========================================================

ARTIFACT_DIR = Path("artifacts")

SPLIT_DIR = ARTIFACT_DIR / "splits"
PREDICTION_DIR = ARTIFACT_DIR / "predictions"
DIAGNOSTIC_DIR = ARTIFACT_DIR / "diagnostics"
RESULTS_DIR = ARTIFACT_DIR / "results"

for directory in [
    SPLIT_DIR,
    PREDICTION_DIR,
    DIAGNOSTIC_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

def load_analysis_data():
    """Load WDBC and construct the frozen analysis representation."""

    data = load_breast_cancer(as_frame=True)

    X = data.data.copy()

    # sklearn original coding:
    # 0 = malignant, 1 = benign
    #
    # Frozen study coding:
    # malignant = 1
    y_original = data.target.copy()
    y = (y_original == 0).astype(int)
    y.name = "malignant"

    observation_id = pd.Series(
        [f"WDBC_{i:03d}" for i in range(len(X))],
        name="observation_id",
    )

    analysis_df = X.copy()
    analysis_df.insert(
        0,
        "observation_id",
        observation_id.values,
    )
    analysis_df["malignant"] = y.values

    feature_columns = tuple(X.columns)

    X_analysis = analysis_df.loc[:, feature_columns].copy()
    y_analysis = analysis_df["malignant"].copy()
    ids_analysis = analysis_df["observation_id"].copy()

    # Frozen data contract
    assert analysis_df.shape == (569, 32)
    assert analysis_df["observation_id"].is_unique
    assert len(feature_columns) == 30
    assert X_analysis.shape == (569, 30)
    assert X_analysis.isna().sum().sum() == 0
    assert set(y_analysis.unique()) == {0, 1}
    assert y_analysis.sum() == 212
    assert (y_analysis == 0).sum() == 357

    return (
        analysis_df,
        X_analysis,
        y_analysis,
        ids_analysis,
        feature_columns,
    )

def compute_dataset_fingerprint(
    X,
    y,
    feature_names,
):
    """Compute deterministic SHA-256 fingerprint of the analysis dataset."""

    hasher = hashlib.sha256()

    schema = "\n".join(feature_names).encode("utf-8")
    hasher.update(schema)

    X_bytes = np.asarray(
        X,
        dtype="<f8",
        order="C",
    ).tobytes(order="C")

    hasher.update(X_bytes)

    y_bytes = np.asarray(
        y,
        dtype=np.uint8,
    ).tobytes(order="C")

    hasher.update(y_bytes)

    return hasher.hexdigest()
def build_resampling_manifest(
    X_analysis,
    y_analysis,
    ids_analysis,
):
    """Construct the frozen repeated stratified 5-fold resampling manifest."""

    seed_sequence = SeedSequence(
        MASTER_RESAMPLING_SEED
    )

    repeat_seeds = [
        int(child.generate_state(1)[0])
        for child in seed_sequence.spawn(N_REPEATS)
    ]

    assert len(repeat_seeds) == N_REPEATS
    assert len(set(repeat_seeds)) == N_REPEATS

    records = []

    for repeat, seed in enumerate(repeat_seeds):

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        for fold, (_, test_idx) in enumerate(
            cv.split(
                X_analysis,
                y_analysis,
            )
        ):
            for idx in test_idx:

                records.append({
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "observation_id":
                        ids_analysis.iloc[idx],
                    "row_index":
                        int(idx),
                    "malignant":
                        int(y_analysis.iloc[idx]),
                })

    manifest = pd.DataFrame(records)

    # Frozen manifest contract
    assert len(manifest) == (
        len(y_analysis) * N_REPEATS
    )

    assert (
        manifest
        .groupby("observation_id")
        .size()
        .eq(N_REPEATS)
        .all()
    )

    assert (
        manifest
        .groupby("repeat")
        .size()
        .eq(len(y_analysis))
        .all()
    )

    assert (
        manifest
        .duplicated(
            ["repeat", "observation_id"]
        )
        .sum()
        == 0
    )

    return manifest

def make_lr_pipeline():
    """Return a fresh Logistic Regression pipeline for each fold."""
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                C=LR_C,
                solver=LR_SOLVER,
                max_iter=LR_MAX_ITER,
            ),
        ),
    ])


MASTER_RF_SEED = 20260925


def make_rf_model(
    random_state=MASTER_RF_SEED,
    n_estimators=RF_N_ESTIMATORS,
):
    """Return a fresh Random Forest classifier for each fold."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        max_samples=None,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def get_train_test_indices(
    manifest,
    repeat,
    fold,
    n_observations,
):
    """Recover train/test indices for one frozen repeat-fold split."""

    fold_rows = manifest.loc[
        (manifest["repeat"] == repeat)
        & (manifest["fold"] == fold)
    ]

    test_idx = (
        fold_rows["row_index"]
        .to_numpy(dtype=int)
    )

    all_idx = np.arange(
        n_observations,
        dtype=int,
    )

    train_idx = np.setdiff1d(
        all_idx,
        test_idx,
        assume_unique=False,
    )

    # Split integrity checks
    assert len(
        np.intersect1d(train_idx, test_idx)
    ) == 0

    assert (
        len(train_idx)
        + len(test_idx)
        == n_observations
    )

    assert len(test_idx) > 0
    assert len(train_idx) > 0

    return train_idx, test_idx


def run_prevalence_predictions(
    manifest,
    y_analysis,
    ids_analysis,
):
    """Generate out-of-sample predictions for the prevalence-only control."""

    records = []
    n_observations = len(y_analysis)

    for repeat in range(N_REPEATS):
        for fold in range(N_SPLITS):

            train_idx, test_idx = get_train_test_indices(
                manifest=manifest,
                repeat=repeat,
                fold=fold,
                n_observations=n_observations,
            )

            y_train = y_analysis.iloc[train_idx]

            train_prevalence = float(
                y_train.mean()
            )

            for idx in test_idx:
                records.append({
                    "model": "prevalence",
                    "repeat": repeat,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "train_prevalence": train_prevalence,
                    "observation_id": ids_analysis.iloc[idx],
                    "row_index": int(idx),
                    "y_true": int(y_analysis.iloc[idx]),
                    "predicted_probability": train_prevalence,
                })

    predictions = pd.DataFrame(records)

    assert len(predictions) == (
        n_observations * N_REPEATS
    )

    assert (
        predictions
        .groupby(["repeat", "observation_id"])
        .size()
        .eq(1)
        .all()
    )

    assert predictions[
        "predicted_probability"
    ].between(0, 1).all()

    return predictions


def run_logistic_regression_predictions(
    manifest,
    X_analysis,
    y_analysis,
    ids_analysis,
):
    """Generate OOS Logistic Regression predictions and fit diagnostics."""

    prediction_records = []
    diagnostic_records = []

    n_observations = len(y_analysis)

    for repeat in range(N_REPEATS):
        for fold in range(N_SPLITS):

            train_idx, test_idx = get_train_test_indices(
                manifest=manifest,
                repeat=repeat,
                fold=fold,
                n_observations=n_observations,
            )

            X_train = X_analysis.iloc[train_idx]
            X_test = X_analysis.iloc[test_idx]

            y_train = y_analysis.iloc[train_idx]
            y_test = y_analysis.iloc[test_idx]

            model = make_lr_pipeline()

            model.fit(
                X_train,
                y_train,
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            n_iter = int(
                model.named_steps["model"].n_iter_[0]
            )

            diagnostic_records.append({
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_iter": n_iter,
            })

            for local_position, idx in enumerate(test_idx):
                prediction_records.append({
                    "model": "logistic_regression",
                    "repeat": repeat,
                    "fold": fold,
                    "observation_id": ids_analysis.iloc[idx],
                    "row_index": int(idx),
                    "y_true": int(y_test.iloc[local_position]),
                    "predicted_probability": float(
                        probabilities[local_position]
                    ),
                })

    predictions = pd.DataFrame(
        prediction_records
    )

    diagnostics = pd.DataFrame(
        diagnostic_records
    )

    # OOS prediction contract
    assert len(predictions) == (
        n_observations * N_REPEATS
    )

    assert (
        predictions
        .groupby(["repeat", "observation_id"])
        .size()
        .eq(1)
        .all()
    )

    assert predictions[
        "predicted_probability"
    ].between(0, 1).all()

    # Fit diagnostics contract
    assert len(diagnostics) == (
        N_REPEATS * N_SPLITS
    )

    return predictions, diagnostics

def run_random_forest_predictions(
    manifest,
    X_analysis,
    y_analysis,
    ids_analysis,
):
    """Generate OOS Random Forest predictions and fit diagnostics."""

    prediction_records = []
    diagnostic_records = []

    n_observations = len(y_analysis)

    for repeat in range(N_REPEATS):
        for fold in range(N_SPLITS):

            train_idx, test_idx = get_train_test_indices(
                manifest=manifest,
                repeat=repeat,
                fold=fold,
                n_observations=n_observations,
            )

            X_train = X_analysis.iloc[train_idx]
            X_test = X_analysis.iloc[test_idx]

            y_train = y_analysis.iloc[train_idx]
            y_test = y_analysis.iloc[test_idx]

            model = make_rf_model()

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            fit_seconds = (
                time.perf_counter() - start
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            diagnostic_records.append({
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_estimators": model.n_estimators,
                "fit_seconds": fit_seconds,
            })

            for local_position, idx in enumerate(test_idx):
                prediction_records.append({
                    "model": "random_forest",
                    "repeat": repeat,
                    "fold": fold,
                    "observation_id": ids_analysis.iloc[idx],
                    "row_index": int(idx),
                    "y_true": int(y_test.iloc[local_position]),
                    "predicted_probability": float(
                        probabilities[local_position]
                    ),
                })

    predictions = pd.DataFrame(
        prediction_records
    )

    diagnostics = pd.DataFrame(
        diagnostic_records
    )

    # OOS prediction contract
    assert len(predictions) == (
        n_observations * N_REPEATS
    )

    assert (
        predictions
        .groupby(["repeat", "observation_id"])
        .size()
        .eq(1)
        .all()
    )

    assert predictions[
        "predicted_probability"
    ].between(0, 1).all()

    # Fit diagnostics contract
    assert len(diagnostics) == (
        N_REPEATS * N_SPLITS
    )

    return predictions, diagnostics

def compute_aggregate_metrics(predictions):
    """Compute repetition-level Brier score and pair-weighted ROC-AUC."""

    records = []

    for repeat in range(N_REPEATS):

        repeat_df = predictions.loc[
            predictions["repeat"] == repeat
        ]

        assert len(repeat_df) == 569

        brier = brier_score_loss(
            repeat_df["y_true"],
            repeat_df["predicted_probability"],
        )

        auc_values = []
        auc_weights = []

        for fold in range(N_SPLITS):

            fold_df = repeat_df.loc[
                repeat_df["fold"] == fold
            ]

            n_positive = int(
                (fold_df["y_true"] == 1).sum()
            )

            n_negative = int(
                (fold_df["y_true"] == 0).sum()
            )

            assert n_positive > 0
            assert n_negative > 0

            fold_auc = roc_auc_score(
                fold_df["y_true"],
                fold_df["predicted_probability"],
            )

            pair_weight = (
                n_positive * n_negative
            )

            auc_values.append(fold_auc)
            auc_weights.append(pair_weight)

        weighted_auc = np.average(
            auc_values,
            weights=auc_weights,
        )

        records.append({
            "repeat": int(repeat),
            "weighted_auc": weighted_auc,
            "brier_score": brier,
        })

    result = (
        pd.DataFrame(records)
        .sort_values("repeat")
        .reset_index(drop=True)
    )

    assert len(result) == N_REPEATS
    assert result["weighted_auc"].between(0, 1).all()
    assert result["brier_score"].between(0, 1).all()

    return result


def compute_individual_stability(predictions):
    """Compute observation-level redevelopment stability summaries."""

    def q10(values):
        return np.quantile(
            values,
            0.10,
            method=QUANTILE_METHOD,
        )

    def q90(values):
        return np.quantile(
            values,
            0.90,
            method=QUANTILE_METHOD,
        )

    summary = (
        predictions
        .groupby("observation_id")
        .agg(
            y_true=(
                "y_true",
                "first",
            ),
            median_probability=(
                "predicted_probability",
                "median",
            ),
            p10=(
                "predicted_probability",
                q10,
            ),
            p90=(
                "predicted_probability",
                q90,
            ),
            prediction_variance=(
                "predicted_probability",
                "var",
            ),
            n_oos_predictions=(
                "predicted_probability",
                "size",
            ),
        )
        .reset_index()
    )

    summary["resampling_width"] = (
        summary["p90"]
        - summary["p10"]
    )

    assert len(summary) == 569

    assert (
        summary["n_oos_predictions"]
        .eq(N_REPEATS)
        .all()
    )

    assert (
        summary["resampling_width"]
        >= 0
    ).all()

    assert summary[
        "median_probability"
    ].between(0, 1).all()

    return summary

def save_artifacts(
    manifest,
    prevalence_predictions,
    lr_predictions,
    lr_diagnostics,
    rf_predictions,
    rf_diagnostics,
    prevalence_metrics,
    lr_metrics,
    rf_metrics,
    prevalence_individual,
    lr_individual,
    rf_individual,
):
    """Save deterministic primary analysis artifacts."""

    # ------------------------------------------------------
    # Frozen resampling manifest
    # ------------------------------------------------------

    manifest.to_csv(
        SPLIT_DIR / "resampling_manifest.csv",
        index=False,
    )

    # ------------------------------------------------------
    # OOS predictions
    # ------------------------------------------------------

    prevalence_predictions.to_csv(
        PREDICTION_DIR
        / "prevalence_oos_predictions.csv",
        index=False,
    )

    lr_predictions.to_csv(
        PREDICTION_DIR
        / "logistic_regression_oos_predictions.csv",
        index=False,
    )

    rf_predictions.to_csv(
        PREDICTION_DIR
        / "random_forest_oos_predictions.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Fit diagnostics
    # ------------------------------------------------------

    lr_diagnostics.to_csv(
        DIAGNOSTIC_DIR
        / "logistic_regression_fit_diagnostics.csv",
        index=False,
    )

    rf_diagnostics.to_csv(
        DIAGNOSTIC_DIR
        / "random_forest_fit_diagnostics.csv",
        index=False,
    )

    # Deterministic RF diagnostic subset:
    # fit_seconds is intentionally excluded.
    rf_diagnostics[
        [
            "repeat",
            "fold",
            "n_train",
            "n_test",
            "n_estimators",
        ]
    ].to_csv(
        DIAGNOSTIC_DIR
        / "random_forest_fit_diagnostics_deterministic.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------

    prevalence_metrics.to_csv(
        RESULTS_DIR
        / "prevalence_control_aggregate_metrics.csv",
        index=False,
    )

    lr_metrics.to_csv(
        RESULTS_DIR
        / "logistic_regression_aggregate_metrics.csv",
        index=False,
    )

    rf_metrics.to_csv(
        RESULTS_DIR
        / "random_forest_aggregate_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Observation-level stability
    # ------------------------------------------------------

    prevalence_individual.to_csv(
        RESULTS_DIR
        / "prevalence_individual_stability.csv",
        index=False,
    )

    lr_individual.to_csv(
        RESULTS_DIR
        / "logistic_regression_individual_stability.csv",
        index=False,
    )

    rf_individual.to_csv(
        RESULTS_DIR
        / "random_forest_individual_stability.csv",
        index=False,
    )


def dataframe_sha256(df):
    """Return SHA-256 of a DataFrame serialized deterministically as CSV."""
    csv_bytes = df.to_csv(
        index=False
    ).encode("utf-8")

    return hashlib.sha256(
        csv_bytes
    ).hexdigest()


def main():
    """Run the complete canonical WDBC prediction-stability analysis."""

    print("Loading analysis data...")

    (
        analysis_df,
        X_analysis,
        y_analysis,
        ids_analysis,
        feature_columns,
    ) = load_analysis_data()

    dataset_hash = compute_dataset_fingerprint(
        X_analysis,
        y_analysis,
        feature_columns,
    )

    print("Dataset SHA256:")
    print(dataset_hash)

    print("\nBuilding frozen resampling manifest...")

    manifest = build_resampling_manifest(
        X_analysis,
        y_analysis,
        ids_analysis,
    )

    manifest_hash = dataframe_sha256(
        manifest
    )

    print("Manifest SHA256:")
    print(manifest_hash)

    # ------------------------------------------------------
    # Prevalence control
    # ------------------------------------------------------

    print("\nRunning prevalence control...")

    prevalence_predictions = (
        run_prevalence_predictions(
            manifest,
            y_analysis,
            ids_analysis,
        )
    )

    prevalence_metrics = (
        compute_aggregate_metrics(
            prevalence_predictions
        )
    )

    prevalence_individual = (
        compute_individual_stability(
            prevalence_predictions
        )
    )

    # ------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------

    print("\nRunning Logistic Regression...")

    (
        lr_predictions,
        lr_diagnostics,
    ) = run_logistic_regression_predictions(
        manifest,
        X_analysis,
        y_analysis,
        ids_analysis,
    )

    lr_metrics = compute_aggregate_metrics(
        lr_predictions
    )

    lr_individual = (
        compute_individual_stability(
            lr_predictions
        )
    )

    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------

    print("\nRunning Random Forest...")
    print(
        f"Using {RF_N_ESTIMATORS} trees "
        f"across {N_REPEATS * N_SPLITS} fits."
    )

    (
        rf_predictions,
        rf_diagnostics,
    ) = run_random_forest_predictions(
        manifest,
        X_analysis,
        y_analysis,
        ids_analysis,
    )

    rf_metrics = compute_aggregate_metrics(
        rf_predictions
    )

    rf_individual = (
        compute_individual_stability(
            rf_predictions
        )
    )

    # ------------------------------------------------------
    # Save canonical artifacts
    # ------------------------------------------------------

    print("\nSaving artifacts...")

    save_artifacts(
        manifest=manifest,
        prevalence_predictions=prevalence_predictions,
        lr_predictions=lr_predictions,
        lr_diagnostics=lr_diagnostics,
        rf_predictions=rf_predictions,
        rf_diagnostics=rf_diagnostics,
        prevalence_metrics=prevalence_metrics,
        lr_metrics=lr_metrics,
        rf_metrics=rf_metrics,
        prevalence_individual=prevalence_individual,
        lr_individual=lr_individual,
        rf_individual=rf_individual,
    )

    # ------------------------------------------------------
    # Reproducibility fingerprints
    # ------------------------------------------------------

    prevalence_hash = dataframe_sha256(
        prevalence_predictions
    )

    lr_hash = dataframe_sha256(
        lr_predictions
    )

    rf_hash = dataframe_sha256(
        rf_predictions
    )

    print("\nPrediction SHA256 fingerprints:")

    print(
        "Prevalence control:",
        prevalence_hash,
    )

    print(
        "Logistic Regression:",
        lr_hash,
    )

    print(
        "Random Forest:",
        rf_hash,
    )

    # ------------------------------------------------------
    # Compact numerical summary
    # ------------------------------------------------------

    def print_summary(
        name,
        metrics,
        individual,
    ):
        print(f"\n{name}")

        print(
            "Mean ROC-AUC:",
            f"{metrics['weighted_auc'].mean():.6f}",
        )

        print(
            "SD ROC-AUC:",
            f"{metrics['weighted_auc'].std(ddof=1):.6f}",
        )

        print(
            "Mean Brier:",
            f"{metrics['brier_score'].mean():.6f}",
        )

        print(
            "SD Brier:",
            f"{metrics['brier_score'].std(ddof=1):.6f}",
        )

        print(
            "Median W:",
            f"{individual['resampling_width'].median():.6f}",
        )

        print(
            "P90 W:",
            f"{individual['resampling_width'].quantile(
                0.90,
                interpolation='linear',
            ):.6f}",
        )

        print(
            "Max W:",
            f"{individual['resampling_width'].max():.6f}",
        )

    print("\nPrimary analysis summary")

    print_summary(
        "Prevalence control",
        prevalence_metrics,
        prevalence_individual,
    )

    print_summary(
        "Logistic Regression",
        lr_metrics,
        lr_individual,
    )

    print_summary(
        "Random Forest",
        rf_metrics,
        rf_individual,
    )

    print("\nCanonical analysis completed successfully.")

if __name__ == "__main__":
    main()
