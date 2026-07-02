# Scripts

This directory contains local data preparation, aggregation, analysis, and
visualization scripts. Scripts use repo-relative paths and generally write under
`outputs/` or `data/`.

Run scripts from the repository root:

```powershell
cd D:\Thesis\thesis
```

## Canonical final-output order

```powershell
python scripts\aggregate_evaluation_results.py
python scripts\build_stage6_failure_analysis.py --strict-locked-data
python scripts\build_stage7_sparsevlm_style_visualizations.py
python scripts\build_stage8_token_metrics.py --require-pairwise
python scripts\build_stage8_ablation_results.py --require-ablation
python scripts\build_stage8_analysis.py --require-pairwise --require-ablation
python scripts\build_report_insert_figures.py
```

These commands assume `outputs/raw_results/` is present and complete.

## Summary aggregation

### `aggregate_evaluation_results.py`

Purpose: rebuild final benchmark summary tables from raw run folders.

Inputs:

- `outputs/raw_results/`

Outputs:

- `outputs/summary/gqa_summary.csv`
- `outputs/summary/pope_summary.csv`
- `outputs/summary/failure_mining_summary.csv`
- `outputs/summary/adaptive_threshold_summary.csv`
- `outputs/summary/final_evaluation_table.csv`

Command:

```powershell
python scripts\aggregate_evaluation_results.py
```

## Stage 6

### `build_stage6_failure_analysis.py`

Purpose: join Dense, SparseVLM-Original-64, Ours-64, and Threshold-Fixed-64
failure-mining predictions; compute recovery groups; emit curated failure
analysis files.

Inputs:

- `outputs/raw_results/`
- `failure_mining_set.csv`
- optional `scripts/stage6_failure_annotations.csv`
- existing `outputs/stage6/failure_pattern_review.csv` if preserving current
  manual annotations

Outputs:

- `outputs/stage6/failure_pattern_review.csv`
- `outputs/stage6/failure_pattern_cases.csv`
- `outputs/stage6/failure_pattern_summary.csv`
- `outputs/stage6/visualization_cases.txt`
- `outputs/stage6/stage6_failure_analysis.md`

Command:

```powershell
python scripts\build_stage6_failure_analysis.py --strict-locked-data
```

## Stage 7

### `build_stage7_sparsevlm_style_visualizations.py`

Purpose: render final SparseVLM-paper-style layer-wise retained-patch figures
for manually selected visual-support cases.

Inputs:

- `outputs/stage7/stage7_selected_visual_support_cases.csv`
- `outputs/raw_results/`
- `data/sample_images/` and any image paths referenced by selected cases

Outputs:

- `outputs/stage7/figures/`
- `outputs/stage7/stage7_visualization_manifest.csv`
- `outputs/stage7/stage7_visualization_summary.md`

Command:

```powershell
python scripts\build_stage7_sparsevlm_style_visualizations.py
```

### `build_stage7_expanded_visual_candidates.py`

Purpose: build an expanded candidate pool for visual support inspection. This is
a supporting discovery script, not the final renderer.

Default outputs:

- `outputs/stage7_expanded/`

Command:

```powershell
python scripts\build_stage7_expanded_visual_candidates.py
```

### `build_stage7_visualizations.py`

Purpose: older overlay-style Stage 7 visualization workflow. Keep for reference
and diagnostics; the final handoff figures are produced by
`build_stage7_sparsevlm_style_visualizations.py`.

Command:

```powershell
python scripts\build_stage7_visualizations.py
```

## Stage 8

### `build_stage8_token_metrics.py`

Purpose: extract selected-token spatial, overlap, failure-recovery, and pairwise
similarity metrics from saved predictions and metadata.

Inputs:

- `outputs/raw_results/`

Outputs:

- `outputs/stage8/stage8_selected_token_metrics.csv`
- `outputs/stage8/stage8_pairwise_similarity_summary.csv`
- `outputs/stage8/stage8_spatial_coverage_summary.csv`
- `outputs/stage8/stage8_overlap_jaccard_summary.csv`
- `outputs/stage8/stage8_failure_recovery_summary.csv`

Commands:

```powershell
python scripts\build_stage8_token_metrics.py
python scripts\build_stage8_token_metrics.py --require-pairwise
```

### `build_stage8_ablation_results.py`

Purpose: build Stage 8 GQA/POPE subset ablation rows and optional
failure-mining stress-test rows.

Inputs:

- Stage 8 run folders under `outputs/raw_results/`
- Stage 8 metric summaries from `build_stage8_token_metrics.py`

Output:

- `outputs/stage8/stage8_ablation_results.csv`

Commands:

```powershell
python scripts\build_stage8_ablation_results.py
python scripts\build_stage8_ablation_results.py --require-ablation
```

### `build_stage8_analysis.py`

Purpose: build the Stage 8 Markdown interpretation from metric and ablation
tables.

Inputs:

- `outputs/stage8/stage8_selected_token_metrics.csv`
- `outputs/stage8/stage8_pairwise_similarity_summary.csv`
- `outputs/stage8/stage8_spatial_coverage_summary.csv`
- `outputs/stage8/stage8_overlap_jaccard_summary.csv`
- `outputs/stage8/stage8_failure_recovery_summary.csv`
- `outputs/stage8/stage8_ablation_results.csv`

Output:

- `outputs/stage8/stage8_analysis.md`

Commands:

```powershell
python scripts\build_stage8_analysis.py
python scripts\build_stage8_analysis.py --require-pairwise --require-ablation
```

## Report figures

### `build_report_insert_figures.py`

Purpose: build report-ready figures from final output tables.

Inputs:

- `outputs/stage8/stage8_ablation_results.csv`
- summary outputs under `outputs/summary/`

Outputs:

- `outputs/report_figures/method_pipeline_ours_selection.png`
- `outputs/report_figures/method_pipeline_ours_selection.pdf`
- `outputs/report_figures/stage8_ablation_accuracy_similarity.png`
- `outputs/report_figures/stage8_ablation_accuracy_similarity.pdf`

Command:

```powershell
python scripts\build_report_insert_figures.py
```

## Data packaging

### `download_full_eval_splits.py`

Purpose: download or assemble full evaluation splits locally.

Outputs:

- `data/full_eval_downloads/`

Command:

```powershell
python scripts\download_full_eval_splits.py
```

### `package_kaggle_eval_datasets.py`

Purpose: package downloaded evaluation data into Kaggle dataset archives.

Inputs:

- `data/full_eval_downloads/`

Outputs:

- `data/kaggle_datasets/`

Command:

```powershell
python scripts\package_kaggle_eval_datasets.py
```

## Notes

- Analysis scripts are intended for local use after raw predictions have been
  produced on Kaggle or another GPU machine.
- Full inference is controlled by the Kaggle runner docs, not by these local
  analysis scripts.
- Most scripts overwrite their target output files. Check `git status` and
  `git diff` before committing regenerated outputs.

