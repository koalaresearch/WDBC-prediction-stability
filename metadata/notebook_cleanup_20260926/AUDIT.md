# WDBC notebook cleanup and reproducibility audit — 2026-09-26

Status: **Restart Kernel + Run All passed in an isolated working copy.** The validated executed notebook has been applied to the repository. No commit, stage, or push was performed.

## Scope and reference preservation

`protocol/analysis_plan_v1.md` was authoritative. Both protocol documents were read and remain unchanged. The reference is the actual pre-edit working tree, including all pre-existing modifications and untracked artifacts, rather than Git HEAD. `reproduction_evidence.zip` contains the complete pre-edit non-Git snapshot, the initial Git status and diff, per-file SHA-256 hashes, the notebook source diff, and all regenerated artifacts. The existing Git repository was not modified.

All pre-existing files were checked against their initial hashes before applying the notebook changes. All pre-existing artifacts remain byte-for-byte untouched in the repository, including both old RF diagnostic files under `artifacts/predictions/`. New runs now save RF diagnostics under `artifacts/diagnostics/`; the old copies were retained to respect the instruction not to delete or overwrite user artifacts.

## Exact files changed or added

- `01_wdbc_prediction_stability.ipynb`
- `artifacts/diagnostics/random_forest_fit_diagnostics.csv`
- `artifacts/diagnostics/random_forest_fit_diagnostics_deterministic.csv`
- `metadata/notebook_cleanup_20260926/AUDIT.md`
- `metadata/notebook_cleanup_20260926/verification.json`
- `metadata/notebook_cleanup_20260926/reproduction_evidence.zip`

## Complete editorial/reproducibility changelog

- Kept the experimental-design diagram and its canonical Figure 1 caption; removed the duplicate Figure 1 caption and placeholder block (original cell 20).
- Replaced the misplaced individual-stability caption after the aggregate plot with its Figure 2 caption; renumbered the actual individual-stability caption from Figure 4 to Figure 3.
- Designated the final two-panel cross-procedure scatter as supplementary Figure S1 in its heading, code comment, and caption. Filenames and plotted values remain unchanged.
- Added P99 to both finite-forest comparison tables by reading the existing `rf_seed_sd_summary*['99%']` values; added the existing rounded P99 values 0.014923, 0.008500, and 0.005913 to the narrative table. No refitting, quantile definition, or diagnostic calculation was introduced by this edit.
- Clarified Stage 5A's starting specification and the frozen 2,000 → 4,000 → 8,000 escalation rule, tolerance, upper limit, and performance-independent decision.
- Redirected the full and deterministic RF fit-diagnostic saves to `artifacts/diagnostics/`; prediction paths and fingerprint definitions remain unchanged.
- Removed the residual display depending on the loop variable `procedure` (original cell 142).
- Consolidated `primary_summary` into one definition and display, retaining the previously calculated P95 and maximum columns. Removed duplicate top-10/top-3 LR displays, the duplicated finite-forest sanity assertion and its heading (the same assertion remains in the diagnostic cell), the preliminary duplicate cross-procedure scatter and heading, the repeated main-table save, the repeated RF-prediction save, and the empty trailing code cell.
- Explicitly labeled Stages 5G–5J and Figure S1 as post-hoc/exploratory; corrected Stage 5I's misleading reference to its cross-procedure result as primary. Distinguished the restatement of primary endpoints from post-hoc comparisons and clarified the return to primary outputs in 5K–5M. Stage 5N remains explicitly post-hoc.
- Refreshed notebook outputs and execution metadata through the clean sequential rerun. Scientific code for fitting, resampling, seeds, metrics, statistical procedures, and quantile definitions is unchanged.

Original cell indices above are zero-based and refer to the preserved pre-edit notebook. The full source diff and index mapping are inside the evidence archive.

## Execution and assertions

- Existing Python environment: Python 3.13.12, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0. No dependencies or user-level configuration were changed.
- Session-only writable `IPYTHONDIR`, Jupyter runtime, and Matplotlib cache paths were temporary. The kernel used local IPC transport and was explicitly restarted before execution.
- 94 non-empty code cells executed in order with consecutive execution counts, without skipped cells or reused model predictions.
- 148 source-level `assert` statements executed, including repetitions inside loops, plus the notebook's NumPy/pandas testing assertions. All passed. All 129 distinct original assert statements were retained.
- Final Run All produced no analysis warnings or errors. Temporary font-cache initialization during setup was informational; it did not change the scientific environment or analysis. After the completed run, the kernel supervisor logged `Parent appears to have exited, shutting down`; this was a process-teardown message, after all notebook cells and numerical checks had completed.
- Final Run All elapsed time: 20.91 minutes.
- Artifact output directory in the disposable execution copy started empty; all 29 expected analysis files were regenerated successfully. The pre-existing `.DS_Store` file is Finder metadata and was preserved in the reference, not treated as an analysis output. The evidence archive preserves those new files.

## Deterministic hashes

| Object | Before SHA-256 | After |
|---|---|---|
| dataset_sha256 | `be06e11b816f830afa615fe2296762d77eb9e29429b057484484931e7b06ff61` | identical |
| artifacts/splits/resampling_manifest.csv | `d2bd8ab9e86c1d05dd4291954f873f13553068bcf02271f7fb873bd35089d499` | identical |
| artifacts/predictions/random_forest_oos_predictions.csv | `7b3570fa69710f8e7bdf73031858e319b26992663a51cd5a92c367b864339420` | identical |
| artifacts/diagnostics/random_forest_fit_diagnostics_deterministic.csv | `7ab807d0f803fcc6ad905f530428849a657d8aefc25480ef62a4995b86636701` | identical |

Every deterministic CSV was compared byte-for-byte, with no numerical tolerance. This includes all three complete 28,450-row OOS prediction tables, the frozen manifest, LR/RF aggregate metrics, all saved individual summaries, primary result tables, deterministic fit diagnostics, and existing exploratory result tables. Existing printed finite-forest and Monte Carlo diagnostic results also match the pre-edit outputs exactly. Full hashes and artifact checks are recorded in `verification.json`.

## Primary numerical results before and after

| Procedure | Metric | Before | After | Comparison |
|---|---|---:|---:|---|
| Prevalence-only control | mean_auc | 0.5 | 0.5 | exact |
| Prevalence-only control | sd_auc | 0.0 | 0.0 | exact |
| Prevalence-only control | mean_brier | 0.2337738358769459 | 0.2337738358769459 | exact |
| Prevalence-only control | sd_brier | 2.803736533363884e-17 | 2.803736533363884e-17 | exact |
| Prevalence-only control | median_W | 0.00219780219780219 | 0.00219780219780219 | exact |
| Prevalence-only control | P90_W | 0.00219780219780219 | 0.00219780219780219 | exact |
| Logistic Regression | mean_auc | 0.9945560253699788 | 0.9945560253699788 | exact |
| Logistic Regression | sd_auc | 0.0012989152875897596 | 0.0012989152875897596 | exact |
| Logistic Regression | mean_brier | 0.020547257795615344 | 0.020547257795615344 | exact |
| Logistic Regression | sd_brier | 0.001676794818571691 | 0.001676794818571691 | exact |
| Logistic Regression | median_W | 0.001003938059742554 | 0.001003938059742554 | exact |
| Logistic Regression | P90_W | 0.07262021934736318 | 0.07262021934736318 | exact |
| Random Forest | mean_auc | 0.9906401955602537 | 0.9906401955602537 | exact |
| Random Forest | sd_auc | 0.0009240946632887024 | 0.0009240946632887024 | exact |
| Random Forest | mean_brier | 0.03169631534215729 | 0.03169631534215729 | exact |
| Random Forest | sd_brier | 0.0010077788286968698 | 0.0010077788286968698 | exact |
| Random Forest | median_W | 0.01045000000000007 | 0.01045000000000007 | exact |
| Random Forest | P90_W | 0.10785750000000002 | 0.10785750000000002 | exact |

## Remaining reproducibility concerns and intentional differences

- Full RF diagnostics contain `fit_seconds`, which changes with hardware/load. Every other column matches exactly, and the separate deterministic diagnostic hash is unchanged.
- All four PNG figures are byte-for-byte identical. The four PDFs differ only in date metadata; after excluding that metadata they are also byte-for-byte identical. Regenerated files are preserved without overwriting the pre-existing figures.
- The README still describes a pre-Random-Forest checkpoint. It was left untouched because README revision was outside the authorized corrections.
- The repository records core software versions in the notebook but has no dependency lockfile. Exact reproduction was established in the current existing environment; no environment was rebuilt or upgraded.
- The two legacy RF diagnostics in `artifacts/predictions/` remain deliberately preserved. Future notebook runs write their canonical copies in `artifacts/diagnostics/`.
