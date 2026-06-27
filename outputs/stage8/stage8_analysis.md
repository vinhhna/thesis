# Stage 8 - Ablation and Auxiliary Metric Analysis

Stage 8 quantifies selected-token behavior because Stage 7 visualizations were mostly inconclusive. The revised ablation design uses deterministic 500-sample GQA and POPE subsets for general hyperparameter evidence. Failure-mining is treated only as a secondary targeted recovery/stress-test set because it is intentionally skewed toward difficult SparseVLM cases. Pairwise hidden-state similarity is reported only when instrumented Kaggle reruns provide the required metadata.

## Data status

- Detailed selected-token metric rows: 200400
- Pairwise similarity available: yes
- Pending required GQA/POPE Stage 8 subset runs: 0
- Optional failure-mining ablation rows tracked: 4

The full Stage 5 benchmark tables remain the official benchmark results. Stage 8 subset results are auxiliary mechanism and hyperparameter evidence, not replacement benchmark scores.
Spatial and overlap summaries may still include existing full-run metadata; the GQA/POPE subset ablation table is the place where Stage 8 subset accuracy and recovery are reported.

## Pairwise similarity

| dataset | token_setting | method | run_id | pairwise_available_count | mean_mean_pairwise_similarity | mean_p90_pairwise_similarity | mean_similarity_above_0.85_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| failure_mining | 128 | Ours | FM-OURS-128 | 0 |  |  |  |
| failure_mining | 128 | SparseVLM-Original | FM-SPARSE-ORIG-128 | 0 |  |  |  |
| failure_mining | 128 | Threshold-Fixed-k | FM-THRESHOLD-FIXED-128 | 0 |  |  |  |
| failure_mining | 64 | Ours | FM-OURS-64 | 0 |  |  |  |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P2-L05 | 100 | 0.211813 | 0.409538 | 0.012279 |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P2-L07 | 100 | 0.210999 | 0.470685 | 0.040735 |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P3-L05 | 100 | 0.198434 | 0.390037 | 0.010000 |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P3-L07 | 100 | 0.200996 | 0.447978 | 0.037574 |
| failure_mining | 64 | SparseVLM-Original | FM-SPARSE-ORIG-64 | 0 |  |  |  |
| failure_mining | 64 | Threshold-Fixed-k | FM-THRESHOLD-FIXED-64 | 0 |  |  |  |
| gqa | 128 | Ours | GQA-OURS-128 | 0 |  |  |  |
| gqa | 128 | SparseVLM-Original | GQA-SPARSE-ORIG-128 | 0 |  |  |  |
| gqa | 128 | Threshold-Fixed-k | GQA-THRESHOLD-FIXED-128 | 0 |  |  |  |
| gqa | 64 | Ours | GQA-OURS-64 | 0 |  |  |  |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P2-L05 | 500 | 0.213907 | 0.409688 | 0.009779 |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P2-L07 | 500 | 0.213340 | 0.459075 | 0.032059 |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P2-L08 | 500 | 0.220174 | 0.522689 | 0.046897 |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P3-L05 | 500 | 0.196494 | 0.387213 | 0.008147 |

Interpretation: lower similarity may support a redundancy-reduction claim only when it is paired with stable or improved answer-level behavior.

## Spatial coverage

The following final-layer spatial metrics are diagnostic only. Broader coverage is not automatically better, because some questions require focused local evidence.

| dataset | token_setting | method | run_id | mean_unique_patch_count | mean_bbox_area_ratio | mean_grid_occupancy_ratio | mean_distance_to_centroid | mean_quadrant_coverage_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| failure_mining | 128 | Ours | FM-OURS-128 | 32.810000 | 0.926875 | 0.056962 | 8.215401 | 3.990000 |
| failure_mining | 128 | SparseVLM-Original | FM-SPARSE-ORIG-128 | 33.780000 | 0.891979 | 0.058646 | 7.974661 | 3.980000 |
| failure_mining | 128 | Threshold-Fixed-k | FM-THRESHOLD-FIXED-128 | 32.420000 | 0.854288 | 0.056285 | 7.641772 | 3.950000 |
| failure_mining | 64 | Ours | FM-OURS-64 | 15.130000 | 0.783819 | 0.026268 | 8.096846 | 3.910000 |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P2-L05 | 14.810000 | 0.774462 | 0.025712 | 7.966747 | 3.900000 |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P2-L07 | 15.090000 | 0.770469 | 0.026198 | 8.037024 | 3.930000 |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P3-L05 | 14.980000 | 0.784705 | 0.026007 | 7.953900 | 3.910000 |
| failure_mining | 64 | Ours | FM-STAGE8-OURS-64-P3-L07 | 14.760000 | 0.778594 | 0.025625 | 8.019118 | 3.890000 |
| failure_mining | 64 | SparseVLM-Original | FM-SPARSE-ORIG-64 | 15.680000 | 0.775573 | 0.027222 | 8.109884 | 3.890000 |
| failure_mining | 64 | Threshold-Fixed-k | FM-THRESHOLD-FIXED-64 | 15.010000 | 0.637969 | 0.026059 | 7.166374 | 3.820000 |
| gqa | 128 | Ours | GQA-OURS-128 | 32.929000 | 0.927675 | 0.057169 | 8.187976 | 3.993200 |
| gqa | 128 | SparseVLM-Original | GQA-SPARSE-ORIG-128 | 33.701800 | 0.901398 | 0.058510 | 7.928293 | 3.989600 |
| gqa | 128 | Threshold-Fixed-k | GQA-THRESHOLD-FIXED-128 | 32.511000 | 0.849687 | 0.056443 | 7.541123 | 3.981400 |
| gqa | 64 | Ours | GQA-OURS-64 | 15.198800 | 0.785026 | 0.026387 | 8.015606 | 3.907000 |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P2-L05 | 14.858000 | 0.781187 | 0.025795 | 7.894294 | 3.904000 |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P2-L07 | 15.018000 | 0.783403 | 0.026073 | 7.916976 | 3.882000 |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P2-L08 | 15.182000 | 0.767903 | 0.026358 | 7.905069 | 3.894000 |
| gqa | 64 | Ours | GQA-STAGE8-OURS-64-P3-L05 | 15.032000 | 0.794465 | 0.026097 | 8.033977 | 3.904000 |

## Ours / Threshold overlap

Jaccard and overlap are computed from unique original patch IDs, not raw selected-index lists.

| dataset | token_setting | pair | sample_count | mean_overlap_count | mean_overlap_ratio_a | mean_overlap_ratio_b | mean_jaccard_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| failure_mining | 128 | ours_vs_sparse | 100 | 20.150000 | 0.612992 | 0.595337 | 0.439793 |
| failure_mining | 128 | threshold_fixed_vs_sparse | 100 | 15.210000 | 0.468245 | 0.450182 | 0.302472 |
| failure_mining | 128 | ours_vs_threshold_fixed | 100 | 16.120000 | 0.491364 | 0.495830 | 0.331794 |
| failure_mining | 64 | ours_vs_sparse | 100 | 9.760000 | 0.645338 | 0.622697 | 0.474031 |
| failure_mining | 64 | threshold_fixed_vs_sparse | 100 | 6.280000 | 0.417460 | 0.401720 | 0.262875 |
| failure_mining | 64 | ours_vs_threshold_fixed | 100 | 7.200000 | 0.475238 | 0.479767 | 0.320226 |
| gqa | 128 | ours_vs_sparse | 5000 | 20.037200 | 0.607982 | 0.594167 | 0.437204 |
| gqa | 128 | threshold_fixed_vs_sparse | 5000 | 15.527800 | 0.477267 | 0.460524 | 0.309641 |
| gqa | 128 | ours_vs_threshold_fixed | 5000 | 16.184000 | 0.491252 | 0.497530 | 0.331816 |
| gqa | 64 | ours_vs_sparse | 5000 | 9.507000 | 0.626037 | 0.616739 | 0.458070 |
| gqa | 64 | threshold_fixed_vs_sparse | 5000 | 6.442800 | 0.429126 | 0.416848 | 0.272789 |
| gqa | 64 | ours_vs_threshold_fixed | 5000 | 7.441000 | 0.489044 | 0.496198 | 0.332749 |
| pope | 128 | ours_vs_sparse | 1500 | 20.424667 | 0.613241 | 0.602702 | 0.443473 |
| pope | 128 | threshold_fixed_vs_sparse | 1500 | 15.677333 | 0.476872 | 0.462937 | 0.310020 |
| pope | 128 | ours_vs_threshold_fixed | 1500 | 16.555333 | 0.496894 | 0.503695 | 0.337343 |
| pope | 64 | ours_vs_sparse | 1500 | 9.472667 | 0.627108 | 0.613767 | 0.457619 |
| pope | 64 | threshold_fixed_vs_sparse | 1500 | 6.327333 | 0.421417 | 0.409460 | 0.267552 |
| pope | 64 | ours_vs_threshold_fixed | 1500 | 7.517333 | 0.496582 | 0.502283 | 0.339356 |

## Failure recovery

The table below summarizes recovery behavior from saved full-run metadata. These rows should be read separately from the Stage 8 GQA/POPE subset ablations.

| dataset | token_setting | run_id | sparse_wrong_count | recovered_baseline_failures | recovery_rate | regressions_vs_baseline | net_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| failure_mining | 64 | FM-OURS-64 | 30 | 10 | 0.333333 | 5 | 5 |
| failure_mining | 64 | FM-THRESHOLD-FIXED-64 | 30 | 7 | 0.233333 | 7 | 0 |
| gqa | 64 | GQA-OURS-64 | 1969 | 341 | 0.173184 | 248 | 93 |
| gqa | 64 | GQA-THRESHOLD-FIXED-64 | 1969 | 364 | 0.184865 | 235 | 129 |
| pope | 64 | POPE-OURS-64 | 285 | 55 | 0.192982 | 25 | 30 |
| pope | 64 | POPE-THRESHOLD-FIXED-64 | 285 | 70 | 0.245614 | 20 | 50 |

For failure-mining at 64 tokens, the known Stage 6 recovery grouping is preserved:

- SparseVLM-Original-64 wrong cases: 30
- Both Ours and Threshold recover: 7
- Ours-only recoveries: 3
- Threshold-only recoveries: 0
- Unresolved by both: 20

## Ours ablation status

General ablation conclusions should be drawn from the GQA/POPE 500-sample subset rows, not from failure-mining.

| dataset | run_id | run_available | ablation_role | row_role | candidate_pool_factor | lambda_relevance | subset_seed | subset_size | accuracy | f1 | mean_pairwise_similarity | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa | GQA-STAGE8-SPARSE-ORIG-64 | true | general_ablation_gqa | required_general_comparison | 2 | 0.8 | 20260610 | 500 | 0.616000 |  | 0.244429 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| gqa | GQA-STAGE8-OURS-64-P2-L08 | true | general_ablation_gqa | required_general_baseline | 2 | 0.8 | 20260610 | 500 | 0.626000 |  | 0.220174 | official Ours-64 baseline setting on the Stage 8 500-sample subset |
| gqa | GQA-STAGE8-THRESHOLD-FIXED-64 | true | general_ablation_gqa | required_general_comparison | 2 | 0.8 | 20260610 | 500 | 0.642000 |  | 0.290032 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| gqa | GQA-STAGE8-OURS-64-P2-L05 | true | general_ablation_gqa | required_general_ablation | 2 | 0.5 | 20260610 | 500 | 0.618000 |  | 0.213907 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| gqa | GQA-STAGE8-OURS-64-P2-L07 | true | general_ablation_gqa | required_general_ablation | 2 | 0.7 | 20260610 | 500 | 0.636000 |  | 0.213340 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| gqa | GQA-STAGE8-OURS-64-P3-L05 | true | general_ablation_gqa | required_general_ablation | 3 | 0.5 | 20260610 | 500 | 0.628000 |  | 0.196494 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| gqa | GQA-STAGE8-OURS-64-P3-L07 | true | general_ablation_gqa | required_general_ablation | 3 | 0.7 | 20260610 | 500 | 0.628000 |  | 0.200417 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| pope | POPE-STAGE8-SPARSE-ORIG-64 | true | general_ablation_pope | required_general_comparison | 2 | 0.8 | 20260610 | 500 | 0.828000 | 0.832031 | 0.243585 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| pope | POPE-STAGE8-OURS-64-P2-L08 | true | general_ablation_pope | required_general_baseline | 2 | 0.8 | 20260610 | 500 | 0.840000 | 0.846154 | 0.220071 | official Ours-64 baseline setting on the Stage 8 500-sample subset |
| pope | POPE-STAGE8-THRESHOLD-FIXED-64 | true | general_ablation_pope | required_general_comparison | 2 | 0.8 | 20260610 | 500 | 0.846000 | 0.852772 | 0.283629 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| pope | POPE-STAGE8-OURS-64-P2-L05 | true | general_ablation_pope | required_general_ablation | 2 | 0.5 | 20260610 | 500 | 0.842000 | 0.850095 | 0.212966 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| pope | POPE-STAGE8-OURS-64-P2-L07 | true | general_ablation_pope | required_general_ablation | 2 | 0.7 | 20260610 | 500 | 0.842000 | 0.848369 | 0.212920 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| pope | POPE-STAGE8-OURS-64-P3-L05 | true | general_ablation_pope | required_general_ablation | 3 | 0.5 | 20260610 | 500 | 0.838000 | 0.849162 | 0.197467 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |
| pope | POPE-STAGE8-OURS-64-P3-L07 | true | general_ablation_pope | required_general_ablation | 3 | 0.7 | 20260610 | 500 | 0.822000 | 0.831758 | 0.204471 | Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score |

### Optional failure-mining stress-test rows

These rows are secondary targeted recovery checks and should not be used as representative general ablation evidence.

| dataset | run_id | run_available | candidate_pool_factor | lambda_relevance | accuracy | recovered_baseline_failures | net_gain | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| failure_mining | FM-STAGE8-OURS-64-P2-L05 | true | 2 | 0.5 | 0.740000 | 9 | 4 | optional failure-mining targeted recovery/stress-test row; not representative |
| failure_mining | FM-STAGE8-OURS-64-P2-L07 | true | 2 | 0.7 | 0.720000 | 9 | 2 | optional failure-mining targeted recovery/stress-test row; not representative |
| failure_mining | FM-STAGE8-OURS-64-P3-L05 | true | 3 | 0.5 | 0.720000 | 7 | 2 | optional failure-mining targeted recovery/stress-test row; not representative |
| failure_mining | FM-STAGE8-OURS-64-P3-L07 | true | 3 | 0.7 | 0.730000 | 7 | 3 | optional failure-mining targeted recovery/stress-test row; not representative |

## Threshold-Adaptive interpretation

Threshold-Adaptive is treated as a variable-token trade-off baseline. It should not be described as beating or losing to Ours under the same fixed 64-token or 128-token budget, because its retained-token count varies by sample.

## Current thesis-safe conclusion

Stage 8 can compare answer-level behavior on the GQA/POPE 500-sample subsets with selected-token similarity, spatial coverage, and baseline overlap. Any redundancy-reduction claim should be conditioned on the actual pairwise similarity direction and whether it aligns with subset accuracy or targeted failure recovery.
Once all required GQA/POPE rows are available, the Stage 8 conclusion should separate general hyperparameter behavior from any optional failure-mining recovery behavior.
