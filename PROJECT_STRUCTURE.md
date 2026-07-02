# Project Structure

## Purpose

This repository is a thesis handoff package for redundancy-aware visual token
selection in SparseVLM / LLaVA-1.5. It contains modified source code, analysis
scripts, selected raw results, final tables, visualizations, and thesis-facing
documentation.

The project compares Dense / Vanilla, SparseVLM-Original, Ours, Threshold-Fixed-k,
and Threshold-Adaptive on GQA, POPE, and a failure-mining set.

## Top-level files

- `README.md`: short project overview and setup summary.
- `REPRODUCIBILITY.md`: command-level guide for rebuilding outputs.
- `PROJECT_STRUCTURE.md`: this structure map.
- `ARTIFACTS.md`: committed versus external artifact policy.
- `evaluation_protocol_updated.md`: final experiment checklist and run matrix.
- `paper_experiment_summary.md`: thesis experiment background and scope.
- `failure_mining_set.csv`: locked 100-case failure-mining set.
- `final_report.pdf`: final thesis report.
- `KAGGLE_NOTEBOOK_CELLS.md`: Kaggle full-run notebook cells.
- `KAGGLE_STAGE8_NOTEBOOK_CELLS.md`: Stage 8 Kaggle run plan.

Development notes and early manual files are present at the repository root:

- `notes.txt`
- `manual_review.csv`
- `failure_mining_sparse_pruned_outputs.jsonl`

They are useful historical context, but they are not the canonical final
reproduction path.

## Source code

`SparseVLMs/` is the modified SparseVLM / LLaVA source tree. It also contains
the upstream SparseVLM README and docs.

Core thesis modifications are concentrated in:

- `SparseVLMs/llava/model/language_model/score.py`
- `SparseVLMs/llava/model/language_model/modelling_sparse_llama.py`
- `SparseVLMs/llava/model/language_model/sparse_llava_llama.py`
- `SparseVLMs/llava/eval/model_vqa_loader.py`
- `SparseVLMs/llava/eval/run_llava.py`
- `SparseVLMs/llava/model/builder.py`

Tests:

- `SparseVLMs/tests/test_sparse_selection.py`

Upstream or mostly inherited areas:

- `SparseVLMs/docs/`
- `SparseVLMs/scripts/`
- `SparseVLMs/llava/train/`
- `SparseVLMs/llava/serve/`
- `SparseVLMs/assests/`

## Analysis scripts

`scripts/` contains local analysis, aggregation, visualization, and packaging
scripts. These scripts assume repo-relative paths and write under `outputs/` or
`data/`.

Canonical final-output scripts:

- `scripts/aggregate_evaluation_results.py`
- `scripts/build_stage6_failure_analysis.py`
- `scripts/build_stage7_sparsevlm_style_visualizations.py`
- `scripts/build_stage8_token_metrics.py`
- `scripts/build_stage8_ablation_results.py`
- `scripts/build_stage8_analysis.py`
- `scripts/build_report_insert_figures.py`

Supporting scripts:

- `scripts/build_stage7_expanded_visual_candidates.py`
- `scripts/build_stage7_visualizations.py`
- `scripts/download_full_eval_splits.py`
- `scripts/package_kaggle_eval_datasets.py`

See `scripts/README.md` for commands and input/output details.

## Notebooks

`notebooks/` contains exploratory and Kaggle notebooks:

- `1-image-smoke-test-kaggle.ipynb`
- `2-100-image-qualitative-check-dense.ipynb`
- `2-100-image-qualitative-check-sparse.ipynb`
- `3-all-visualize-results.ipynb`
- `3-failure-mining-visualize-results.ipynb`
- `4-full-vis.ipynb`
- `4-full-vis-mmr.ipynb`
- `5-experiments.ipynb`
- `6-ablation-experiments.ipynb`

Several notebooks contain executed outputs and are large. The scripts and
Markdown docs are the preferred reproducibility path for final handoff.

## Data

`data/sample_images/` contains 100 small images used for qualitative inspection,
failure mining, and visualization.

Local external data directories are intentionally excluded by `.gitignore`:

- `data/full_eval_downloads/`
- `data/kaggle_datasets/`

Those directories may exist on the local machine, but they are not required for
local analysis if `outputs/raw_results/` is present.

## Outputs

`outputs/summary/` contains thesis-level summary tables:

- `final_evaluation_table.csv`
- `gqa_summary.csv`
- `pope_summary.csv`
- `failure_mining_summary.csv`
- `adaptive_threshold_summary.csv`
- `latency_summary.csv`
- `figures/`

`outputs/raw_results/` contains run-level provenance: prediction JSONL files,
manifests, logs, and per-run CSVs. It is currently committed in this repository
snapshot and must not be untracked without first archiving it.

`outputs/stage6/` contains failure-analysis deliverables:

- `stage6_failure_analysis.md`
- `failure_pattern_review.csv`
- `failure_pattern_cases.csv`
- `failure_pattern_summary.csv`
- `visualization_cases.txt`

`outputs/stage7/` contains final visualization deliverables:

- `README.md`
- `figures/`
- `stage7_visualization_manifest.csv`
- `stage7_visualization_summary.md`
- `stage7_selected_visual_support_cases.csv`
- `stage7_expanded_recovery_candidates.csv`
- `stage7_expanded_recovery_metrics.csv`

`outputs/stage8/` contains ablation and selected-token metrics:

- `stage8_analysis.md`
- `stage8_ablation_results.csv`
- `stage8_selected_token_metrics.csv`
- `stage8_pairwise_similarity_summary.csv`
- `stage8_spatial_coverage_summary.csv`
- `stage8_overlap_jaccard_summary.csv`
- `stage8_failure_recovery_summary.csv`
- `figures/`

`outputs/report_figures/` contains report-insert figures.

## Local-only clutter to keep out of release zips

Do not include these in a final source-only handoff archive:

- `.git/`
- `.venv/`
- `data/full_eval_downloads/`
- `data/kaggle_datasets/`
- Python bytecode and cache folders
- downloaded model checkpoints and Hugging Face caches

