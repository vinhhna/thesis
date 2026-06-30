# Redundancy-Aware SparseVLM Thesis

Minimal project package for the thesis experiment on visual-token sparsification in
LLaVA-1.5 / SparseVLM.

## Overview

This project evaluates whether redundancy-aware visual-token selection can improve
SparseVLM-style inference under tight token budgets. The main comparison is:

- Dense / Vanilla: full 576-token LLaVA-1.5 baseline
- SparseVLM-Original: original text-guided top-k token selection
- Ours: redundancy-aware MMR-style token selection
- Threshold Filtering: simpler similarity-based redundancy baseline

The focused thesis evaluation uses GQA, POPE, and a small failure-mining set at
128-token and 64-token settings.

## Package Contents

- `SparseVLMs/`: modified SparseVLM/LLaVA source code and tests
- `scripts/`: aggregation, analysis, visualization, and packaging scripts
- `notebooks/`: experiment notebooks with outputs stripped in the minimal zip
- `outputs/summary/`: final CSV summaries used for thesis reporting
- `outputs/report_figures/`: lightweight report figures
- `data/sample_images/`: small image subset for qualitative inspection
- `final_report.pdf`: final thesis report
- `evaluation_protocol_updated.md`: detailed experiment checklist
- `paper_experiment_summary.md`: background and reproduction scope

Large local artifacts are intentionally excluded from the minimal package:

- `.git/`
- `.venv/`
- full Kaggle/GQA/POPE datasets
- raw prediction JSONL outputs
- generated caches and Python bytecode
- bulky visualization exports

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e SparseVLMs
```

Full benchmark reruns require external model weights and datasets that are not
included in the minimal archive.

## Key Files

- Main result table: `outputs/summary/final_evaluation_table.csv`
- GQA summary: `outputs/summary/gqa_summary.csv`
- POPE summary: `outputs/summary/pope_summary.csv`
- Failure-mining summary: `outputs/summary/failure_mining_summary.csv`
- Adaptive threshold summary: `outputs/summary/adaptive_threshold_summary.csv`

## Notes

The minimal `.zip` is designed for review and handoff, not for storing the full
experimental workspace. Use the summary files and report for the final results;
use the scripts and source tree to inspect or reproduce the analysis pipeline.
