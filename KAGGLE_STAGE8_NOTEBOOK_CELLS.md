# Stage 8 Kaggle Run Plan

Stage 8 no longer uses failure-mining as the main ablation dataset. Failure-mining is intentionally skewed toward difficult or SparseVLM-failure cases, so it is useful as a targeted stress test but not as general ablation evidence.

Use `KAGGLE_NOTEBOOK_CELLS.md` as the one-run Kaggle runner. For each Kaggle session, edit only:

```python
RUN_ID_TO_RUN = "..."
```

The notebook creates a run-specific output folder and a download zip:

```text
/kaggle/working/{RUN_ID_TO_RUN}_download.zip
```

## Fixed subset design

Primary Stage 8 ablations use deterministic 500-sample subsets:

```text
GQA_STAGE8_SUBSET_N = 500
POPE_STAGE8_SUBSET_N = 500
STAGE8_SUBSET_SEED = 20260610
```

Each run saves the exact subset JSONL inside the run output folder. All runs from the same dataset must use the same subset seed and subset size.

## Required comparison runs

These runs provide the same-subset context for ablation and overlap/recovery comparisons:

```text
GQA-STAGE8-SPARSE-ORIG-64
GQA-STAGE8-OURS-64-P2-L08
GQA-STAGE8-THRESHOLD-FIXED-64

POPE-STAGE8-SPARSE-ORIG-64
POPE-STAGE8-OURS-64-P2-L08
POPE-STAGE8-THRESHOLD-FIXED-64
```

`P2-L08` is the official Ours-64 baseline setting on the Stage 8 subset.

## Required Ours ablation runs

Run Ours at 64 tokens:

```text
GQA-STAGE8-OURS-64-P2-L05
GQA-STAGE8-OURS-64-P2-L07
GQA-STAGE8-OURS-64-P3-L05
GQA-STAGE8-OURS-64-P3-L07

POPE-STAGE8-OURS-64-P2-L05
POPE-STAGE8-OURS-64-P2-L07
POPE-STAGE8-OURS-64-P3-L05
POPE-STAGE8-OURS-64-P3-L07
```

Run-name meanings:

```text
P2 = candidate_pool_factor 2
P3 = candidate_pool_factor 3
L05 = lambda_relevance 0.5
L07 = lambda_relevance 0.7
L08 = lambda_relevance 0.8
```

## Optional secondary failure-mining runs

Use failure-mining only for targeted recovery/stress-test behavior:

```text
FM-STAGE8-OURS-64-P2-L05
FM-STAGE8-OURS-64-P2-L07
FM-STAGE8-OURS-64-P3-L05
FM-STAGE8-OURS-64-P3-L07
```

Do not use these as the main basis for general hyperparameter conclusions.

## Required code support

The SparseVLM code must accept these generation arguments:

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

## Importing results locally

After downloading each Kaggle zip, extract or copy each run folder under:

```text
outputs/raw_results/
```

Then rebuild Stage 8 outputs:

```powershell
python scripts\build_stage8_token_metrics.py
python scripts\build_stage8_ablation_results.py
python scripts\build_stage8_analysis.py
```

After all required GQA/POPE Stage 8 runs are imported, use strict validation:

```powershell
python scripts\build_stage8_token_metrics.py --require-pairwise
python scripts\build_stage8_ablation_results.py --require-ablation
python scripts\build_stage8_analysis.py --require-pairwise --require-ablation
```

Strict ablation validation requires the GQA/POPE Stage 8 subset comparison and ablation runs. It does not require the optional failure-mining runs.

## Reporting rule

Stage 8 subset results are auxiliary mechanism and hyperparameter evidence. They do not replace the full Stage 5 benchmark tables.

Report conclusions separately:

- GQA/POPE 500-sample subsets: general ablation behavior.
- Failure-mining: targeted recovery/stress-test behavior.
