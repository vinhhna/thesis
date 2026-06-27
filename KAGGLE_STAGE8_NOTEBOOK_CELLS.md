# Stage 8 Kaggle Run Cells

These cells describe the extra Kaggle runs needed for Stage 8. They reuse the same environment, model loading, prompt formatting, metadata validation, output folders, and download packaging style used by `KAGGLE_NOTEBOOK_CELLS.md`.

Stage 8 has two kinds of Kaggle runs:

1. Instrumented similarity probes for the already evaluated methods.
2. 64-token Ours ablations over candidate-pool size and MMR lambda.

The local Stage 8 scripts can already compute spatial coverage, overlap, and recovery from existing prediction files. These Kaggle runs are only needed for true selected-token hidden-state pairwise similarity and the Ours ablation table.

## Required code support

The SparseVLM code now accepts two extra generation arguments:

```python
lambda_relevance=0.8
record_selection_similarity=True
```

When `record_selection_similarity=True`, each pruning-layer entry in `metadata.layer_token_stats` stores aggregate pairwise similarity metrics:

```text
pairwise_similarity_available
pairwise_similarity_token_count
mean_pairwise_similarity
median_pairwise_similarity
max_pairwise_similarity
p90_pairwise_similarity
similarity_above_0.80_ratio
similarity_above_0.85_ratio
similarity_above_0.90_ratio
```

No hidden-state vectors are stored.

## Run configuration fields

Extend the run dataclass used in the Kaggle notebook with:

```python
lambda_relevance: float = 0.8
record_selection_similarity: bool = True
```

When calling `model.generate`, pass:

```python
output_ids = model.generate(
    inputs=input_ids,
    images=images_tensor,
    image_sizes=[image.size],
    retained_tokens=run.retained_tokens,
    selection_method=generate_selection_method,
    threshold_tau=run.threshold_tau,
    candidate_pool_factor=run.candidate_pool_factor,
    lambda_relevance=run.lambda_relevance,
    record_selection_similarity=run.record_selection_similarity,
    **generation_kwargs,
)
```

Add these values to the per-row metadata if they are not already present:

```python
metadata.setdefault("candidate_pool_factor", run.candidate_pool_factor)
metadata.setdefault("lambda_relevance", run.lambda_relevance)
metadata.setdefault("record_selection_similarity", run.record_selection_similarity)
```

## Instrumented similarity probe runs

Run these after updating the notebook to pass `record_selection_similarity=True`.

```python
STAGE8_SIMILARITY_RUNS = [
    # GQA
    ("GQA-STAGE8-SPARSE-ORIG-128", "GQA", "SparseVLM-Original", "topk", 128, 2, 0.8, 0.85),
    ("GQA-STAGE8-SPARSE-ORIG-64", "GQA", "SparseVLM-Original", "topk", 64, 2, 0.8, 0.85),
    ("GQA-STAGE8-OURS-128", "GQA", "Ours", "mmr", 128, 2, 0.8, 0.85),
    ("GQA-STAGE8-OURS-64", "GQA", "Ours", "mmr", 64, 2, 0.8, 0.85),
    ("GQA-STAGE8-THRESHOLD-FIXED-128", "GQA", "Threshold-Fixed-k", "threshold_fixed", 128, 2, 0.8, 0.85),
    ("GQA-STAGE8-THRESHOLD-FIXED-64", "GQA", "Threshold-Fixed-k", "threshold_fixed", 64, 2, 0.8, 0.85),

    # POPE
    ("POPE-STAGE8-SPARSE-ORIG-128", "POPE", "SparseVLM-Original", "topk", 128, 2, 0.8, 0.85),
    ("POPE-STAGE8-SPARSE-ORIG-64", "POPE", "SparseVLM-Original", "topk", 64, 2, 0.8, 0.85),
    ("POPE-STAGE8-OURS-128", "POPE", "Ours", "mmr", 128, 2, 0.8, 0.85),
    ("POPE-STAGE8-OURS-64", "POPE", "Ours", "mmr", 64, 2, 0.8, 0.85),
    ("POPE-STAGE8-THRESHOLD-FIXED-128", "POPE", "Threshold-Fixed-k", "threshold_fixed", 128, 2, 0.8, 0.85),
    ("POPE-STAGE8-THRESHOLD-FIXED-64", "POPE", "Threshold-Fixed-k", "threshold_fixed", 64, 2, 0.8, 0.85),

    # Failure mining
    ("FM-STAGE8-SPARSE-ORIG-128", "failure_mining", "SparseVLM-Original", "topk", 128, 2, 0.8, 0.85),
    ("FM-STAGE8-SPARSE-ORIG-64", "failure_mining", "SparseVLM-Original", "topk", 64, 2, 0.8, 0.85),
    ("FM-STAGE8-OURS-128", "failure_mining", "Ours", "mmr", 128, 2, 0.8, 0.85),
    ("FM-STAGE8-OURS-64", "failure_mining", "Ours", "mmr", 64, 2, 0.8, 0.85),
    ("FM-STAGE8-THRESHOLD-FIXED-128", "failure_mining", "Threshold-Fixed-k", "threshold_fixed", 128, 2, 0.8, 0.85),
    ("FM-STAGE8-THRESHOLD-FIXED-64", "failure_mining", "Threshold-Fixed-k", "threshold_fixed", 64, 2, 0.8, 0.85),
]
```

## Minimum Ours ablation runs

These are the required Stage 8 ablation runs.

```python
STAGE8_OURS_ABLATION_RUNS = [
    # GQA
    ("GQA-STAGE8-OURS-64-P2-L05", "GQA", "Ours", "mmr", 64, 2, 0.5, 0.85),
    ("GQA-STAGE8-OURS-64-P2-L07", "GQA", "Ours", "mmr", 64, 2, 0.7, 0.85),
    ("GQA-STAGE8-OURS-64-P3-L05", "GQA", "Ours", "mmr", 64, 3, 0.5, 0.85),
    ("GQA-STAGE8-OURS-64-P3-L07", "GQA", "Ours", "mmr", 64, 3, 0.7, 0.85),

    # POPE
    ("POPE-STAGE8-OURS-64-P2-L05", "POPE", "Ours", "mmr", 64, 2, 0.5, 0.85),
    ("POPE-STAGE8-OURS-64-P2-L07", "POPE", "Ours", "mmr", 64, 2, 0.7, 0.85),
    ("POPE-STAGE8-OURS-64-P3-L05", "POPE", "Ours", "mmr", 64, 3, 0.5, 0.85),
    ("POPE-STAGE8-OURS-64-P3-L07", "POPE", "Ours", "mmr", 64, 3, 0.7, 0.85),

    # Failure mining
    ("FM-STAGE8-OURS-64-P2-L05", "failure_mining", "Ours", "mmr", 64, 2, 0.5, 0.85),
    ("FM-STAGE8-OURS-64-P2-L07", "failure_mining", "Ours", "mmr", 64, 2, 0.7, 0.85),
    ("FM-STAGE8-OURS-64-P3-L05", "failure_mining", "Ours", "mmr", 64, 3, 0.5, 0.85),
    ("FM-STAGE8-OURS-64-P3-L07", "failure_mining", "Ours", "mmr", 64, 3, 0.7, 0.85),
]
```

## Importing results locally

After downloading the Kaggle outputs into `outputs/raw_results/`, run:

```powershell
python scripts\build_stage8_token_metrics.py --require-pairwise
python scripts\build_stage8_ablation_results.py --require-ablation
python scripts\build_stage8_analysis.py --require-pairwise --require-ablation
```

Before the Kaggle outputs are available, run without strict flags:

```powershell
python scripts\build_stage8_token_metrics.py
python scripts\build_stage8_ablation_results.py
python scripts\build_stage8_analysis.py
```
