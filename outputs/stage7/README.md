# Stage 7 final output layout

This folder is the consolidated Stage 7 output directory.

Essential files:

- `figures/`: final SparseVLM-style layer-wise pruning visualizations.
- `stage7_visualization_manifest.csv`: one row per generated figure.
- `stage7_visualization_summary.md`: summary of the generated visualization set.
- `stage7_selected_visual_support_cases.csv`: the 141 manually selected expanded recovery cases used for visualization.
- `stage7_expanded_recovery_candidates.csv`: full expanded candidate pool before selecting the 141 cases.
- `stage7_expanded_recovery_metrics.csv`: diagnostic token-selection metrics for the expanded candidate pool.

Removed/obsolete files:

- Original five-case Stage 7 visualization metadata.
- Old final-overlay comparison figures and contact sheets.
- Stale selected-file lists pointing to old deleted figures.
- Duplicate `stage7_sparsevlm_style` output folder and zip.

Interpretation note: the figures visualize explicit retained original-patch traces at pruning layers 2, 6, and 15. Merged/recycled tokens are not shown because they do not map to a single original image patch.
