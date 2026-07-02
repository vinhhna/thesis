# Artifact Policy

## Artifact classes

The repository has four artifact classes:

1. Source and documentation: code, scripts, Markdown docs, protocol files.
2. Small committed data: `failure_mining_set.csv` and `data/sample_images/`.
3. Generated thesis outputs: final CSV tables, Stage 6/7/8 outputs, report
   figures, and `final_report.pdf`.
4. External or bulky artifacts: full datasets, model weights, Kaggle packages,
   local virtual environments, and full-download caches.

## Committed artifacts in the current snapshot

These are part of the current repository state:

- `SparseVLMs/`: modified SparseVLM / LLaVA source.
- `scripts/`: local aggregation, analysis, visualization, and packaging scripts.
- `outputs/summary/`: final thesis tables.
- `outputs/stage6/`: failure-analysis outputs.
- `outputs/stage7/`: final visualization outputs.
- `outputs/stage8/`: ablation and token-metric outputs.
- `outputs/report_figures/`: report-ready figures.
- `outputs/raw_results/`: raw run-level prediction and manifest files.
- `data/sample_images/`: small image subset.
- `failure_mining_set.csv`: locked failure-mining set.
- `final_report.pdf`: final report artifact.
- Kaggle runner documentation: `KAGGLE_NOTEBOOK_CELLS.md` and
  `KAGGLE_STAGE8_NOTEBOOK_CELLS.md`.

Important: `outputs/raw_results/` is currently tracked and should not be
untracked until it has been archived or replaced by an explicit external
artifact bundle.

## External artifacts

These are expected to be local or external, not source-controlled:

- `data/full_eval_downloads/`
- `data/kaggle_datasets/`
- full GQA data
- full POPE / COCO image data
- LLaVA-1.5 model weights
- SparseVLM or Hugging Face cache directories
- Kaggle output zip downloads
- `.venv/`
- Python caches and notebook checkpoints

The root `.gitignore` already excludes `data/full_eval_downloads/`,
`data/kaggle_datasets/`, `.venv/`, and common caches. Future cleanup should also
ignore newly downloaded raw result zips, transient logs, and local editor files.

## Raw result policy

`outputs/raw_results/` is the provenance layer for local reproduction. It
contains prediction JSONL files, manifests, logs, and run-specific summaries.
The analysis scripts use this folder to rebuild:

- `outputs/summary/*.csv`
- `outputs/stage6/*`
- `outputs/stage8/*`

For a source-only repository, this folder is a candidate for external archival.
For a thesis-review handoff, keeping it available is useful because reviewers
can regenerate tables without rerunning expensive GPU inference.

Recommended final handoff options:

- Source-only package: exclude `outputs/raw_results/` and provide an external
  artifact link or zip.
- Review package: include `outputs/raw_results/` so all summaries can be
  rebuilt locally.
- Minimal report package: include only `outputs/summary/`, `outputs/stage6/`,
  `outputs/stage7/`, `outputs/stage8/`, `outputs/report_figures/`, and
  `final_report.pdf`.

## Final thesis outputs

Main thesis table:

- `outputs/summary/final_evaluation_table.csv`

Dataset summaries:

- `outputs/summary/gqa_summary.csv`
- `outputs/summary/pope_summary.csv`
- `outputs/summary/failure_mining_summary.csv`
- `outputs/summary/adaptive_threshold_summary.csv`
- `outputs/summary/latency_summary.csv`

Stage outputs:

- Stage 6: `outputs/stage6/stage6_failure_analysis.md`
- Stage 7: `outputs/stage7/stage7_visualization_summary.md`
- Stage 8: `outputs/stage8/stage8_analysis.md`

Report figures:

- `outputs/report_figures/method_pipeline_ours_selection.*`
- `outputs/report_figures/stage8_ablation_accuracy_similarity.*`
- `outputs/summary/figures/main_latency_by_method.*`

## Files that are not canonical final artifacts

The following root-level files are historical or intermediate context rather
than canonical final artifacts:

- `notes.txt`
- `manual_review.csv`
- `failure_mining_sparse_pruned_outputs.jsonl`

They should not be used as the primary reproduction path. The canonical final
outputs are under `outputs/`, and the canonical scripts are under `scripts/`.

## Archive checklist before untracking large outputs

Before removing or untracking any bulky artifact, confirm:

```powershell
git status --short --branch
python scripts\aggregate_evaluation_results.py
python scripts\build_stage8_token_metrics.py --require-pairwise
python scripts\build_stage8_ablation_results.py --require-ablation
python scripts\build_stage8_analysis.py --require-pairwise --require-ablation
```

Then create an external archive containing at least:

- `outputs/raw_results/`
- `outputs/summary/`
- `outputs/stage6/`
- `outputs/stage7/`
- `outputs/stage8/`
- `outputs/report_figures/`

Do not include `.git/`, `.venv/`, full downloaded datasets, or model caches in a
review archive unless explicitly requested.

