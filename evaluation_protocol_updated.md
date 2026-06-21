# Evaluation Protocol

This file lists the experiments to run for the thesis evaluation. The thesis will use two types of evaluation:

1. **Main fixed-budget benchmark comparison**  
   This evaluates Dense / Vanilla, SparseVLM-Original, Ours, and Threshold-Fixed-k under controlled token budgets.

2. **Adaptive threshold trade-off analysis**  
   This evaluates Threshold-Adaptive without forcing the final number of retained tokens to be exactly 64 or 128. This analysis compares its accuracy-token trade-off against Ours-64 and Ours-128 from the main experiment.

Because of time constraints, the thesis will not evaluate the optional 192-token and 96-token settings. The required token settings are:

- Dense / Vanilla: 576 visual tokens
- Sparse methods: 128 and 64 visual tokens
- Threshold-Fixed-k: 128 and 64 visual tokens
- Threshold-Adaptive: variable number of retained tokens, controlled by similarity threshold

For POPE, report both Accuracy and F1. Other printed POPE values such as Precision, Recall, and Yes ratio can be kept in logs, but Accuracy and F1 are the required thesis metrics.

## Experiment Table

### A. Main Fixed-Budget Benchmark Comparison

| Run ID | Dataset | Method | Token setting | Metric | Prediction file | Output file | Notes | Checklist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA-DENSE-576 | GQA | Dense / Vanilla | 576 | Official GQA accuracy | `results/gqa/predictions/gqa_dense_576.jsonl` | `results/gqa/gqa_dense_576.csv` | Full-token reference setting. | [x] |
| GQA-SPARSE-ORIG-128 | GQA | SparseVLM-Original | 128 | Official GQA accuracy | `results/gqa/predictions/gqa_sparsevlm_original_128.jsonl` | `results/gqa/gqa_sparsevlm_original_128.csv` | Main SparseVLM baseline at the practical sparse setting. | [x] |
| GQA-SPARSE-ORIG-64 | GQA | SparseVLM-Original | 64 | Official GQA accuracy | `results/gqa/predictions/gqa_sparsevlm_original_64.jsonl` | `results/gqa/gqa_sparsevlm_original_64.csv` | Main SparseVLM baseline at the aggressive sparse setting. | [x] |
| GQA-OURS-128 | GQA | Ours | 128 | Official GQA accuracy | `results/gqa/predictions/gqa_ours_128.jsonl` | `results/gqa/gqa_ours_128.csv` | Tests whether the proposed selection method preserves reasoning evidence better than SparseVLM-Original. | [x] |
| GQA-OURS-64 | GQA | Ours | 64 | Official GQA accuracy | `results/gqa/predictions/gqa_ours_64.jsonl` | `results/gqa/gqa_ours_64.csv` | Main stress test for the proposed method under strong sparsification. | [x] |
| GQA-THRESHOLD-FIXED-128 | GQA | Threshold-Fixed-k | 128 | Official GQA accuracy | `results/gqa/predictions/gqa_threshold_fixed_128.jsonl` | `results/gqa/gqa_threshold_fixed_128.csv` | Simpler redundancy-aware baseline with the same token budget as Ours-128. | [x] |
| GQA-THRESHOLD-FIXED-64 | GQA | Threshold-Fixed-k | 64 | Official GQA accuracy | `results/gqa/predictions/gqa_threshold_fixed_64.jsonl` | `results/gqa/gqa_threshold_fixed_64.csv` | Simpler redundancy-aware baseline with the same token budget as Ours-64. | [x] |
| POPE-DENSE-576 | POPE | Dense / Vanilla | 576 | POPE Accuracy and F1 | `results/pope/predictions/pope_dense_576.jsonl` | `results/pope/pope_dense_576.csv` | Full-token reference for hallucination evaluation. | [x] |
| POPE-SPARSE-ORIG-128 | POPE | SparseVLM-Original | 128 | POPE Accuracy and F1 | `results/pope/predictions/pope_sparsevlm_original_128.jsonl` | `results/pope/pope_sparsevlm_original_128.csv` | Checks hallucination behavior of the original SparseVLM selection method. | [x] |
| POPE-SPARSE-ORIG-64 | POPE | SparseVLM-Original | 64 | POPE Accuracy and F1 | `results/pope/predictions/pope_sparsevlm_original_64.jsonl` | `results/pope/pope_sparsevlm_original_64.csv` | Aggressive sparse baseline for object hallucination. | [x] |
| POPE-OURS-128 | POPE | Ours | 128 | POPE Accuracy and F1 | `results/pope/predictions/pope_ours_128.jsonl` | `results/pope/pope_ours_128.csv` | Tests whether redundancy-aware selection improves reliable object grounding. | [x] |
| POPE-OURS-64 | POPE | Ours | 64 | POPE Accuracy and F1 | `results/pope/predictions/pope_ours_64.jsonl` | `results/pope/pope_ours_64.csv` | Main hallucination stress test under strong sparsification. | [x] |
| POPE-THRESHOLD-FIXED-128 | POPE | Threshold-Fixed-k | 128 | POPE Accuracy and F1 | `results/pope/predictions/pope_threshold_fixed_128.jsonl` | `results/pope/pope_threshold_fixed_128.csv` | Simpler redundancy-aware baseline with the same token budget as Ours-128. | [x] |
| POPE-THRESHOLD-FIXED-64 | POPE | Threshold-Fixed-k | 64 | POPE Accuracy and F1 | `results/pope/predictions/pope_threshold_fixed_64.jsonl` | `results/pope/pope_threshold_fixed_64.csv` | Simpler redundancy-aware baseline with the same token budget as Ours-64. | [x] |
| FM-DENSE-576 | Failure-mining | Dense / Vanilla | 576 | Correctness, qualitative failure label | `results/failure_mining/predictions/failure_mining_dense_576.jsonl` | `results/failure_mining/failure_mining_dense_576.csv` | Full-token reference for targeted failure cases. | [x] |
| FM-SPARSE-ORIG-128 | Failure-mining | SparseVLM-Original | 128 | Correctness, qualitative failure label | `results/failure_mining/predictions/failure_mining_sparsevlm_original_128.jsonl` | `results/failure_mining/failure_mining_sparsevlm_original_128.csv` | Identifies failures caused by redundant or missing selected visual evidence. | [x] |
| FM-SPARSE-ORIG-64 | Failure-mining | SparseVLM-Original | 64 | Correctness, qualitative failure label | `results/failure_mining/predictions/failure_mining_sparsevlm_original_64.jsonl` | `results/failure_mining/failure_mining_sparsevlm_original_64.csv` | Main source for aggressive-sparsification failure analysis. | [x] |
| FM-OURS-128 | Failure-mining | Ours | 128 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_ours_128.jsonl` | `results/failure_mining/failure_mining_ours_128.csv` | Measures whether the proposed method recovers SparseVLM-Original failures. | [x] |
| FM-OURS-64 | Failure-mining | Ours | 64 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_ours_64.jsonl` | `results/failure_mining/failure_mining_ours_64.csv` | Main targeted test for the proposed method under tight token budget. | [x] |
| FM-THRESHOLD-FIXED-128 | Failure-mining | Threshold-Fixed-k | 128 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_threshold_fixed_128.jsonl` | `results/failure_mining/failure_mining_threshold_fixed_128.csv` | Compares the proposed method with a simpler redundancy-aware strategy under the same token budget. | [x] |
| FM-THRESHOLD-FIXED-64 | Failure-mining | Threshold-Fixed-k | 64 | Correctness, recovery rate, qualitative failure label | `results/failure_mining/predictions/failure_mining_threshold_fixed_64.jsonl` | `results/failure_mining/failure_mining_threshold_fixed_64.csv` | Aggressive sparse threshold baseline for failure analysis. | [x] |

### B. Adaptive Threshold Trade-off Analysis

These runs evaluate Threshold-Adaptive. Unlike Threshold-Fixed-k, this method does not force the final retained token count to be exactly 64 or 128. It filters the candidate pool using a similarity threshold and reports the actual number of retained tokens.

Ours-64 and Ours-128 from the main experiment are reused as fixed-budget comparison points. They do not need to be rerun for this analysis.

| Run ID | Dataset | Method | Token setting | Metric | Prediction file | Output file | Notes | Checklist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA-THRESHOLD-ADAPT-080 | GQA | Threshold-Adaptive | Variable, tau=0.80 | Official GQA accuracy, avg/min/max retained tokens | `results/gqa/predictions/gqa_threshold_adaptive_tau080.jsonl` | `results/gqa/gqa_threshold_adaptive_tau080.csv` | Adaptive threshold run with stricter redundancy filtering. | [x] |
| GQA-THRESHOLD-ADAPT-085 | GQA | Threshold-Adaptive | Variable, tau=0.85 | Official GQA accuracy, avg/min/max retained tokens | `results/gqa/predictions/gqa_threshold_adaptive_tau085.jsonl` | `results/gqa/gqa_threshold_adaptive_tau085.csv` | Adaptive threshold run with default similarity threshold. | [x] |
| GQA-THRESHOLD-ADAPT-090 | GQA | Threshold-Adaptive | Variable, tau=0.90 | Official GQA accuracy, avg/min/max retained tokens | `results/gqa/predictions/gqa_threshold_adaptive_tau090.jsonl` | `results/gqa/gqa_threshold_adaptive_tau090.csv` | Adaptive threshold run with looser redundancy filtering. | [x] |
| POPE-THRESHOLD-ADAPT-080 | POPE | Threshold-Adaptive | Variable, tau=0.80 | POPE Accuracy, F1, avg/min/max retained tokens | `results/pope/predictions/pope_threshold_adaptive_tau080.jsonl` | `results/pope/pope_threshold_adaptive_tau080.csv` | Adaptive threshold run for hallucination evaluation. | [x] |
| POPE-THRESHOLD-ADAPT-085 | POPE | Threshold-Adaptive | Variable, tau=0.85 | POPE Accuracy, F1, avg/min/max retained tokens | `results/pope/predictions/pope_threshold_adaptive_tau085.jsonl` | `results/pope/pope_threshold_adaptive_tau085.csv` | Adaptive threshold run with default similarity threshold. | [x] |
| POPE-THRESHOLD-ADAPT-090 | POPE | Threshold-Adaptive | Variable, tau=0.90 | POPE Accuracy, F1, avg/min/max retained tokens | `results/pope/predictions/pope_threshold_adaptive_tau090.jsonl` | `results/pope/pope_threshold_adaptive_tau090.csv` | Adaptive threshold run with looser redundancy filtering. | [x] |
| FM-THRESHOLD-ADAPT-080 | Failure-mining | Threshold-Adaptive | Variable, tau=0.80 | Correctness, recovery rate, qualitative failure label, avg/min/max retained tokens | `results/failure_mining/predictions/failure_mining_threshold_adaptive_tau080.jsonl` | `results/failure_mining/failure_mining_threshold_adaptive_tau080.csv` | Adaptive threshold run for targeted failure cases. | [x] |
| FM-THRESHOLD-ADAPT-085 | Failure-mining | Threshold-Adaptive | Variable, tau=0.85 | Correctness, recovery rate, qualitative failure label, avg/min/max retained tokens | `results/failure_mining/predictions/failure_mining_threshold_adaptive_tau085.jsonl` | `results/failure_mining/failure_mining_threshold_adaptive_tau085.csv` | Adaptive threshold run with default similarity threshold. | [x] |
| FM-THRESHOLD-ADAPT-090 | Failure-mining | Threshold-Adaptive | Variable, tau=0.90 | Correctness, recovery rate, qualitative failure label, avg/min/max retained tokens | `results/failure_mining/predictions/failure_mining_threshold_adaptive_tau090.jsonl` | `results/failure_mining/failure_mining_threshold_adaptive_tau090.csv` | Adaptive threshold run with looser redundancy filtering. | [x] |

## Required Summary Tables

After running the experiments, prepare these summary files:

| Summary | Purpose | Output file | Checklist |
| --- | --- | --- | --- |
| GQA benchmark summary | Compare official GQA accuracy across Dense / Vanilla, SparseVLM-Original, Ours, and Threshold-Fixed-k under 128-token and 64-token settings. | `results/summary/gqa_summary.csv` | [ ] |
| POPE benchmark summary | Compare POPE Accuracy and F1 across Dense / Vanilla, SparseVLM-Original, Ours, and Threshold-Fixed-k under 128-token and 64-token settings. | `results/summary/pope_summary.csv` | [ ] |
| Failure-mining summary | Compare correctness, recovery cases, and qualitative failure categories across Dense / Vanilla, SparseVLM-Original, Ours, and Threshold-Fixed-k. | `results/summary/failure_mining_summary.csv` | [ ] |
| Adaptive threshold summary | Compare Threshold-Adaptive at tau=0.80, tau=0.85, and tau=0.90 against Ours-64 and Ours-128 using both performance and retained-token statistics. | `results/summary/adaptive_threshold_summary.csv` | [ ] |
| Final thesis table | Combined table containing the main reported numbers for GQA, POPE, failure mining, and adaptive threshold trade-off analysis. | `results/summary/final_evaluation_table.csv` | [ ] |

## Reproduction Rules

- Do not run 192-token or 96-token experiments.
- Always keep Dense / Vanilla as the 576-token reference.
- For SparseVLM-Original, run both 128-token and 64-token settings.
- For Ours, run both 128-token and 64-token settings.
- For Threshold-Fixed-k, run both 128-token and 64-token settings.
- For Threshold-Adaptive, run tau=0.80, tau=0.85, and tau=0.90.
- Threshold-Fixed-k must output exactly the target token budget, either 128 or 64.
- Threshold-Adaptive does not need to output exactly 128 or 64 tokens, but it must report the actual number of retained tokens for each sample.
- For Threshold-Adaptive, report average, minimum, and maximum retained token counts.
- For POPE, report both Accuracy and F1 in the final thesis table.
- For failure mining, preserve per-sample outputs so the answer, selected tokens, retained-token count, and failure category can be inspected later.
- Ours-64 and Ours-128 from the main experiment are reused as comparison points for the adaptive threshold trade-off analysis. They do not need to be rerun separately.
