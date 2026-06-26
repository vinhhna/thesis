# Stage 6 — Failure-Case Analysis

## Objective and protocol

This stage examines cases where SparseVLM-Original fails under the aggressive 64-token preset. The analysis is intentionally targeted: it asks whether Ours recovers failures that are plausibly connected to discarded visual evidence or redundant token allocation. It does not attempt to construct a complete taxonomy of VLM errors.

Dense-576, SparseVLM-Original-64, Ours-64, and Threshold-Fixed-64 predictions were joined by `case_id`. The source images and final selected original-patch indices were inspected for candidate cases. Answer changes and patch indices are treated as diagnostic evidence, not as causal proof.

## Full-set recovery results

| Group | Cases |
| --- | ---: |
| Original-64 failures | 30 |
| Recovered by Ours-64 | 10 |
| Recovered by Threshold-Fixed-64 | 7 |
| Recovered by both Ours and Threshold-Fixed | 7 |
| Recovered only by Ours | 3 |
| Recovered only by Threshold-Fixed | 0 |
| Unresolved by both | 20 |

These groups are computed directly from the current prediction files. The general builder does not assume that one method's recoveries must be a subset of another method's recoveries.

## Curated failure patterns

The curated table contains 8 cases. 8 are provisionally classified as `missed_relevant_visual_evidence`, and 0 as `redundant_or_dominant_region_selection`.

No case was assigned the redundancy/dominant-region label in the main table. Manual image inspection alone was not strong enough to support that mechanism. Stage 7 token overlays or feature-similarity analysis are required before using the label.

| Case | Group | Ground truth | Original | Ours | Evidence needed |
| --- | --- | --- | --- | --- | --- |
| GQA_VAL_003 | ours_and_threshold_recovery | yes | No | Yes | The gold dog, the nearby tent boundaries, and their relative horizontal positions. |
| GQA_VAL_010 | ours_only_recovery | keyboard | Screen | Keyboard | The keyboard or keyboard-like device beside the bag/laptop sleeve in the lower foreground. |
| GQA_VAL_028 | ours_only_recovery | black | Blue | Black | The small tie and its color on the right-side person. |
| GQA_VAL_029 | ours_and_threshold_recovery | yes | No | Yes | Partially visible chair structure near the lower or side boundaries of the image. |
| GQA_VAL_045 | ours_and_threshold_recovery | suit | Outfit | Suit | The black suit or suit-like clothing worn in the background rather than the central player's white clothing. |
| GQA_VAL_061 | ours_and_threshold_recovery | beach | Ocean | Beach | The shallow near-shore setting, surfboard, and beach-activity context. |
| VQAV2_VAL_013 | ours_and_threshold_recovery | 2 | 1 | 2 | Distinct head, body, and tusk evidence for both overlapping elephants. |
| VQAV2_VAL_015 | ours_only_recovery | taking pictures | Nothing | Taking picture | The camera in the person's hands and its orientation toward the sheep. |

## Interpretation

Across the selected recoveries, the recurring pattern is that the answer depends on evidence that is small, overlapping, peripheral, relational, or distributed across the scene. Examples include a small camera, a dark tie, a partially visible chair, two overlapping elephants, and the spatial relation between a dog and nearby tents. SparseVLM-Original gives an incorrect answer in these cases, while Ours produces the correct answer with a different retained patch set. Stage 7 visualization is needed to verify whether the changed token selection covers the hypothesized evidence region.

This pattern is consistent with improved preservation of relevant visual evidence under aggressive pruning. It does not establish that any particular retained patch caused the answer, and it does not yet establish reduced feature redundancy.

Since no defensible redundancy/dominant-region case was found at this stage, the Stage 6 evidence mainly supports the missed-evidence aspect of the thesis claim. The redundancy aspect will be evaluated through visualization and selected-token metrics in later stages.

## Stage 7 handoff

The high-priority visualization cases are: `GQA_VAL_010`, `GQA_VAL_028`, `GQA_VAL_029`, `VQAV2_VAL_013`, `VQAV2_VAL_015`.

For each case, Stage 7 should overlay the final original-patch selections for Original, Ours, and Threshold-Fixed on the same CLIP 24×24 image grid. The visualization should test whether Ours actually covers the hypothesized evidence region. Any future redundancy claim should additionally use pairwise feature similarity or another explicit redundancy measurement.

## Limitations

- The failure-mining set is targeted and small, so curated-case counts are not prevalence estimates.
- Exact-match recovery can reflect generation or language-model effects in addition to visual-token selection.
- Dense is not correct on every recovery, so it is a reference rather than an infallible upper bound.
- Ours regressions are intentionally deferred and must be discussed later for a balanced account.
- Redundancy/dominant-region explanations remain provisional until visualization or similarity analysis supports them.
