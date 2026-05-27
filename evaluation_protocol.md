# Evaluation Protocol

This file lists the experiments to run for the thesis evaluation. Because of time constraints, the thesis will not evaluate the optional 192-token and 96-token settings. The required token settings are:

- Dense / Vanilla: 576 visual tokens
- Sparse methods: 128 and 64 visual tokens

For POPE, report both Accuracy and F1. Other printed POPE values such as Precision, Recall, and Yes ratio can be kept in logs, but Accuracy and F1 are the required thesis metrics.

## Experiment Table

| Run ID | Dataset | Method | Token setting | Metric | Prediction file | Output file | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GQA-DENSE-576 | GQA | Dense / Vanilla | 576 | Official GQA accuracy | `results/gqa/predictions/gqa_dense_576.jsonl` | `results/gqa/gqa_dense_576.csv` | Full-token reference setting. |
| GQA-SPARSE-ORIG-128 | GQA | SparseVLM-Original | 128 | Official GQA accuracy | `results/gqa/predictions/gqa_sparsevlm_original_128.jsonl` | `results/gqa/gqa_sparsevlm_original_128.csv` | Main SparseVLM baseline at the practical sparse setting. |
| GQA-SPARSE-ORIG-64 | GQA | SparseVLM-Original | 64 | Official GQA accuracy | `results/gqa/predictions/gqa_sparsevlm_original_64.jsonl` | `results/gqa/gqa_sparsevlm_original_64.csv` | Main SparseVLM baseline at the aggressive sparse setting. |
| GQA-OURS-128 | GQA | Ours | 128 | Official GQA accuracy | `results/gqa/predictions/gqa_ours_128.jsonl` | `results/gqa/gqa_ours_128.csv` | Tests whether the proposed selection method preserves reasoning evidence better than SparseVLM-Original. |
| GQA-OURS-64 | GQA | Ours | 64 | Official GQA accuracy | `results/gqa/predictions/gqa_ours_64.jsonl` | `results/gqa/gqa_ours_64.csv` | Main stress test for the proposed method under strong sparsification. |
| GQA-THRESHOLD-128 | GQA | Threshold Filtering | 128 | Official GQA accuracy | `results/gqa/predictions/gqa_threshold_128.jsonl` | `results/gqa/gqa_threshold_128.csv` | Simpler redundancy-aware baseline at the practical sparse setting. |
| GQA-THRESHOLD-64 | GQA | Threshold Filtering | 64 | Official GQA accuracy | `results/gqa/predictions/gqa_threshold_64.jsonl` | `results/gqa/gqa_threshold_64.csv` | Simpler redundancy-aware baseline at the aggressive sparse setting. |
| POPE-DENSE-576 | POPE | Dense / Vanilla | 576 | POPE Accuracy and F1 | `results/pope/predictions/pope_dense_576.jsonl` | `results/pope/pope_dense_576.csv` | Full-token reference for hallucination evaluation. |
| POPE-SPARSE-ORIG-128 | POPE | SparseVLM-Original | 128 | POPE Accuracy and F1 | `results/pope/predictions/pope_sparsevlm_original_128.jsonl` | `results/pope/pope_sparsevlm_original_128.csv` | Checks hallucination behavior of the original SparseVLM selection method. |
| POPE-SPARSE-ORIG-64 | POPE | SparseVLM-Original | 64 | POPE Accuracy and F1 | `results/pope/predictions/pope_sparsevlm_original_64.jsonl` | `results/pope/pope_sparsevlm_original_64.csv` | Aggressive sparse baseline for object hallucination. |
| POPE-OURS-128 | POPE | Ours | 128 | POPE Accuracy and F1 | `results/pope/predictions/pope_ours_128.jsonl` | `results/pope/pope_ours_128.csv` | Tests whether redundancy-aware selection improves reliable object grounding. |
| POPE-OURS-64 | POPE | Ours | 64 | POPE Accuracy and F1 | `results/pope/predictions/pope_ours_64.jsonl` | `results/pope/pope_ours_64.csv` | Main hallucination stress test under strong sparsification. |
| POPE-THRESHOLD-128 | POPE | Threshold Filtering | 128 | POPE Accuracy and F1 | `results/pope/predictions/pope_threshold_128.jsonl` | `results/pope/pope_threshold_128.csv` | Simpler redundancy-aware baseline for POPE. |
| POPE-THRESHOLD-64 | POPE | Threshold Filtering | 64 | POPE Accuracy and F1 | `results/pope/predictions/pope_threshold_64.jsonl` | `results/pope/pope_threshold_64.csv` | Aggressive sparse threshold baseline for POPE. |
| FM-DENSE-576 | Failure-mining | Dense / Vanilla | 576 | Correctness, qualitative failure label | `results/failure_mining/predictions/failure_mining_dense_576.jsonl` | `results/failure_mining/failure_mining_dense_576.csv` | Full-token reference for targeted failure cases. |
| FM-SPARSE-ORIG-128 | Failure-mining | SparseVLM-Original | 128 | Correctness, qualitative failure label | `results/failure_mining/predictions/failure_mining_sparsevlm_original_128.jsonl` | `results/failure_mining/failure_mining_sparsevlm_original_128.csv` | Identifies failures caused by redundant or missing selected visual evidence. |
| FM-SPARSE-ORIG-64 | Failure-mining | SparseVLM-Original | 64 | Correctness, qualitative failure label | `results/failure_mining/predictions/failure_mining_sparsevlm_original_64.jsonl` | `results/failure_mining/failure_mining_sparsevlm_original_64.csv` | Main source for aggressive-sparsification failure analysis. |
| FM-OURS-128 | Failure-mining | Ours | 128 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_ours_128.jsonl` | `results/failure_mining/failure_mining_ours_128.csv` | Measures whether the proposed method recovers SparseVLM-Original failures. |
| FM-OURS-64 | Failure-mining | Ours | 64 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_ours_64.jsonl` | `results/failure_mining/failure_mining_ours_64.csv` | Main targeted test for the proposed method under tight token budget. |
| FM-THRESHOLD-128 | Failure-mining | Threshold Filtering | 128 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_threshold_128.jsonl` | `results/failure_mining/failure_mining_threshold_128.csv` | Compares the proposed method with a simpler redundancy-aware strategy. |
| FM-THRESHOLD-64 | Failure-mining | Threshold Filtering | 64 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_threshold_64.jsonl` | `results/failure_mining/failure_mining_threshold_64.csv` | Aggressive sparse threshold baseline for failure analysis. |

## Required Summary Tables

After running the experiments, prepare these summary files:

| Summary | Purpose | Output file |
| --- | --- | --- |
| GQA benchmark summary | Compare official GQA accuracy across methods and token settings. | `results/summary/gqa_summary.csv` |
| POPE benchmark summary | Compare POPE Accuracy and F1 across methods and token settings. | `results/summary/pope_summary.csv` |
| Failure-mining summary | Compare correctness, recovery cases, and qualitative failure categories. | `results/summary/failure_mining_summary.csv` |
| Final thesis table | Combined table containing the main reported numbers for GQA, POPE, and failure mining. | `results/summary/final_evaluation_table.csv` |

## Reproduction Rules

- Do not run 192-token or 96-token experiments unless the thesis schedule changes.
- Always keep Dense / Vanilla as the 576-token reference.
- For sparse methods, run both 128-token and 64-token settings.
- For POPE, report both Accuracy and F1 in the final thesis table.
- For failure mining, preserve per-sample outputs so the answer, selected tokens, and failure category can be inspected later.
