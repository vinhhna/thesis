# Stage 8 — Ablation and Auxiliary Metric Analysis

Stage 8 quantifies selected-token behavior because Stage 7 visualizations were mostly inconclusive. The current outputs use all available saved prediction metadata for spatial coverage, selected-patch overlap, and failure recovery. Pairwise hidden-state similarity is reported only when instrumented Kaggle reruns provide the required metadata.

## Data status

- Detailed selected-token metric rows: 178200
- Pairwise similarity available: no
- Pending planned Ours ablation runs: 12

## Pairwise similarity

Pairwise similarity is not available in the current saved predictions because they do not contain instrumented selected-token hidden-state similarity aggregates. Therefore Stage 8 does not yet make a redundancy-reduction claim from pairwise similarity.

## Spatial coverage

The following final-layer spatial metrics are diagnostic only. Broader coverage is not automatically better, because some questions require focused local evidence.

| dataset | token_setting | method | run_id | mean_unique_patch_count | mean_bbox_area_ratio | mean_grid_occupancy_ratio | mean_distance_to_centroid | mean_quadrant_coverage_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| failure_mining | 128 | Ours | FM-OURS-128 | 32.810000 | 0.926875 | 0.056962 | 8.215401 | 3.990000 |
| failure_mining | 128 | SparseVLM-Original | FM-SPARSE-ORIG-128 | 33.780000 | 0.891979 | 0.058646 | 7.974661 | 3.980000 |
| failure_mining | 128 | Threshold-Fixed-k | FM-THRESHOLD-FIXED-128 | 32.420000 | 0.854288 | 0.056285 | 7.641772 | 3.950000 |
| failure_mining | 64 | Ours | FM-OURS-64 | 15.130000 | 0.783819 | 0.026268 | 8.096846 | 3.910000 |
| failure_mining | 64 | SparseVLM-Original | FM-SPARSE-ORIG-64 | 15.680000 | 0.775573 | 0.027222 | 8.109884 | 3.890000 |
| failure_mining | 64 | Threshold-Fixed-k | FM-THRESHOLD-FIXED-64 | 15.010000 | 0.637969 | 0.026059 | 7.166374 | 3.820000 |
| gqa | 128 | Ours | GQA-OURS-128 | 32.929000 | 0.927675 | 0.057169 | 8.187976 | 3.993200 |
| gqa | 128 | SparseVLM-Original | GQA-SPARSE-ORIG-128 | 33.701800 | 0.901398 | 0.058510 | 7.928293 | 3.989600 |
| gqa | 128 | Threshold-Fixed-k | GQA-THRESHOLD-FIXED-128 | 32.511000 | 0.849687 | 0.056443 | 7.541123 | 3.981400 |
| gqa | 64 | Ours | GQA-OURS-64 | 15.198800 | 0.785026 | 0.026387 | 8.015606 | 3.907000 |
| gqa | 64 | SparseVLM-Original | GQA-SPARSE-ORIG-64 | 15.442200 | 0.778135 | 0.026810 | 8.093133 | 3.888200 |
| gqa | 64 | Threshold-Fixed-k | GQA-THRESHOLD-FIXED-64 | 14.988800 | 0.640541 | 0.026022 | 7.002464 | 3.838000 |
| pope | 128 | Ours | POPE-OURS-128 | 33.321333 | 0.947807 | 0.057850 | 8.423348 | 3.998000 |
| pope | 128 | SparseVLM-Original | POPE-SPARSE-ORIG-128 | 33.875333 | 0.939264 | 0.058812 | 8.323659 | 3.996000 |
| pope | 128 | Threshold-Fixed-k | POPE-THRESHOLD-FIXED-128 | 32.878000 | 0.881514 | 0.057080 | 7.821930 | 3.980000 |
| pope | 64 | Ours | POPE-OURS-64 | 15.116667 | 0.794150 | 0.026244 | 8.093704 | 3.842667 |
| pope | 64 | SparseVLM-Original | POPE-SPARSE-ORIG-64 | 15.438000 | 0.805836 | 0.026802 | 8.310373 | 3.830000 |
| pope | 64 | Threshold-Fixed-k | POPE-THRESHOLD-FIXED-64 | 14.967333 | 0.650552 | 0.025985 | 7.200703 | 3.754667 |

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

| dataset | run_id | run_available | ablation_role | candidate_pool_factor | lambda_relevance | accuracy | f1 | mean_pairwise_similarity | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa | GQA-OURS-64 | true | official_baseline | 2 | 0.8 | 0.624800 |  |  | official Ours-64 baseline |
| gqa | GQA-STAGE8-OURS-64-P2-L05 | false | ablation_variant | 2 | 0.5 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| gqa | GQA-STAGE8-OURS-64-P2-L07 | false | ablation_variant | 2 | 0.7 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| gqa | GQA-STAGE8-OURS-64-P3-L05 | false | ablation_variant | 3 | 0.5 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| gqa | GQA-STAGE8-OURS-64-P3-L07 | false | ablation_variant | 3 | 0.7 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| pope | POPE-OURS-64 | true | official_baseline | 2 | 0.8 | 0.830000 | 0.840923 |  | official Ours-64 baseline |
| pope | POPE-STAGE8-OURS-64-P2-L05 | false | ablation_variant | 2 | 0.5 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| pope | POPE-STAGE8-OURS-64-P2-L07 | false | ablation_variant | 2 | 0.7 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| pope | POPE-STAGE8-OURS-64-P3-L05 | false | ablation_variant | 3 | 0.5 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| pope | POPE-STAGE8-OURS-64-P3-L07 | false | ablation_variant | 3 | 0.7 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| failure_mining | FM-OURS-64 | true | official_baseline | 2 | 0.8 | 0.750000 |  |  | official Ours-64 baseline |
| failure_mining | FM-STAGE8-OURS-64-P2-L05 | false | ablation_variant | 2 | 0.5 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| failure_mining | FM-STAGE8-OURS-64-P2-L07 | false | ablation_variant | 2 | 0.7 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| failure_mining | FM-STAGE8-OURS-64-P3-L05 | false | ablation_variant | 3 | 0.5 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |
| failure_mining | FM-STAGE8-OURS-64-P3-L07 | false | ablation_variant | 3 | 0.7 |  |  |  | planned Stage 8 ablation run; metrics populate after Kaggle output is imported |

## Threshold-Adaptive interpretation

Threshold-Adaptive is treated as a variable-token trade-off baseline. It should not be described as beating or losing to Ours under the same fixed 64-token or 128-token budget, because its retained-token count varies by sample.

## Current thesis-safe conclusion

At the current checkpoint, Stage 8 supports spatial/overlap/failure-recovery analysis from saved metadata, but it does not yet support a strong redundancy-reduction claim because true pairwise hidden-state similarity requires instrumented Kaggle reruns.
The Ours hyperparameter conclusion is also pending until the planned 64-token candidate-pool/lambda ablation runs are imported.
