# Reproducibility Guide

## Thesis objective

This repository supports a thesis on redundancy-aware visual token selection for
SparseVLM / LLaVA-1.5. The main question is whether a redundancy-aware,
MMR-style selector can preserve useful visual evidence better than the original
SparseVLM top-k selector under tight visual-token budgets.

The compared methods are:

- Dense / Vanilla: full 576-token LLaVA-1.5 baseline.
- SparseVLM-Original: original text-guided top-k sparse selector.
- Ours: redundancy-aware MMR-style token selector.
- Threshold-Fixed-k: fixed-budget similarity-threshold baseline.
- Threshold-Adaptive: variable-token similarity-threshold baseline.

The main evaluation uses GQA, POPE, and a 100-case failure-mining set. The main
fixed sparse budgets are 128 and 64 visual tokens; Dense uses 576 tokens.

## Environment policy

Use two environment tiers:

- Full inference runs: use Kaggle or another CUDA GPU environment with the
  LLaVA-1.5 model weights and full benchmark data. Full reruns are not expected
  to work from this repository alone because full GQA, POPE, COCO images, and
  model checkpoints are external artifacts.
- Local analysis runs: use a normal local Python environment to rebuild CSV
  summaries, Stage 6 analysis, Stage 7 figures, Stage 8 token metrics, and report
  figures from the committed or restored raw result files.

Minimal local setup for analysis:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e SparseVLMs
```

Lightweight validation of the core selection code:

```powershell
python -m pytest SparseVLMs\tests\test_sparse_selection.py -q
```

The full model path, dataset paths, CUDA version, and Kaggle runtime are outside
the local analysis contract. See `KAGGLE_NOTEBOOK_CELLS.md` and
`KAGGLE_STAGE8_NOTEBOOK_CELLS.md` for the notebook-based full-run workflow.

## Core modified files

Reviewers should inspect these files first:

- `SparseVLMs/llava/model/language_model/score.py`: top-k, MMR,
  threshold-fixed, threshold-adaptive selection, pairwise similarity metrics.
- `SparseVLMs/llava/model/language_model/modelling_sparse_llama.py`: sparse
  inference integration and sparse metadata capture.
- `SparseVLMs/llava/model/language_model/sparse_llava_llama.py`: LLaVA sparse
  model wrapper and generation argument forwarding.
- `SparseVLMs/llava/eval/model_vqa_loader.py`: benchmark inference entry point
  with `selection_method`, `threshold_tau`, `candidate_pool_factor`,
  `lambda_relevance`, and `record_selection_similarity`.
- `SparseVLMs/llava/model/builder.py`: sparse model loading path.
- `SparseVLMs/tests/test_sparse_selection.py`: unit tests for the selector
  behavior.

## Required inputs for local regeneration

Local regeneration assumes the following are present:

- `outputs/raw_results/`: raw prediction JSONL, manifests, logs, and per-run CSVs.
- `outputs/stage7/stage7_selected_visual_support_cases.csv`: manually selected
  Stage 7 visualization cases.
- `failure_mining_set.csv`: locked 100-case failure-mining source set.
- `data/sample_images/`: small image subset used by failure mining and
  visualization.

External data that is intentionally not part of the source handoff:

- `data/full_eval_downloads/`
- `data/kaggle_datasets/`
- full GQA / POPE / COCO data
- LLaVA / SparseVLM model weights and Hugging Face caches
- `.venv/`

## Regenerate final evaluation tables

This rebuilds the thesis summary CSVs from `outputs/raw_results/`.

```powershell
python scripts\aggregate_evaluation_results.py
```

Primary outputs:

- `outputs/summary/final_evaluation_table.csv`
- `outputs/summary/gqa_summary.csv`
- `outputs/summary/pope_summary.csv`
- `outputs/summary/failure_mining_summary.csv`
- `outputs/summary/adaptive_threshold_summary.csv`

The final table should contain 30 rows: 7 fixed-budget rows plus 3 adaptive rows
for each of GQA, POPE, and failure-mining.

## Regenerate Stage 6 failure analysis

```powershell
python scripts\build_stage6_failure_analysis.py --strict-locked-data
```

Primary outputs:

- `outputs/stage6/failure_pattern_review.csv`
- `outputs/stage6/failure_pattern_cases.csv`
- `outputs/stage6/failure_pattern_summary.csv`
- `outputs/stage6/visualization_cases.txt`
- `outputs/stage6/stage6_failure_analysis.md`

Note: the script can read optional seed annotations from
`scripts/stage6_failure_annotations.csv`. In the current repository snapshot,
the curated annotations are preserved in the generated Stage 6 review CSV.

## Regenerate Stage 7 visualizations

The final Stage 7 handoff uses the SparseVLM-paper-style layer-wise retained
patch renderer:

```powershell
python scripts\build_stage7_sparsevlm_style_visualizations.py
```

Primary outputs:

- `outputs/stage7/figures/`
- `outputs/stage7/stage7_visualization_manifest.csv`
- `outputs/stage7/stage7_visualization_summary.md`

Supporting Stage 7 scripts:

```powershell
python scripts\build_stage7_expanded_visual_candidates.py
python scripts\build_stage7_visualizations.py
```

Those scripts are useful for candidate discovery and earlier overlay-style
visualization workflows. The consolidated final Stage 7 output directory is
`outputs/stage7/`.

## Regenerate Stage 8 ablation and token metrics

Normal local rebuild:

```powershell
python scripts\build_stage8_token_metrics.py
python scripts\build_stage8_ablation_results.py
python scripts\build_stage8_analysis.py
```

Strict validation after all required Stage 8 GQA/POPE subset runs are present:

```powershell
python scripts\build_stage8_token_metrics.py --require-pairwise
python scripts\build_stage8_ablation_results.py --require-ablation
python scripts\build_stage8_analysis.py --require-pairwise --require-ablation
```

Primary outputs:

- `outputs/stage8/stage8_selected_token_metrics.csv`
- `outputs/stage8/stage8_pairwise_similarity_summary.csv`
- `outputs/stage8/stage8_spatial_coverage_summary.csv`
- `outputs/stage8/stage8_overlap_jaccard_summary.csv`
- `outputs/stage8/stage8_failure_recovery_summary.csv`
- `outputs/stage8/stage8_ablation_results.csv`
- `outputs/stage8/stage8_analysis.md`

Stage 8 subset results are auxiliary mechanism and hyperparameter evidence.
They do not replace the full Stage 5 benchmark tables in `outputs/summary/`.

## Regenerate report figures

```powershell
python scripts\build_report_insert_figures.py
```

Primary outputs:

- `outputs/report_figures/method_pipeline_ours_selection.png`
- `outputs/report_figures/method_pipeline_ours_selection.pdf`
- `outputs/report_figures/stage8_ablation_accuracy_similarity.png`
- `outputs/report_figures/stage8_ablation_accuracy_similarity.pdf`

The latency figure files live under `outputs/summary/figures/`.

## Recommended local validation commands

```powershell
git status --short --branch
git ls-files | Measure-Object
git count-objects -vH
python -m pytest SparseVLMs\tests\test_sparse_selection.py -q
Import-Csv outputs\summary\final_evaluation_table.csv | Group-Object evaluation_group,dataset
```

