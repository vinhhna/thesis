# Stage 7 SparseVLM-Style Visualizations

These figures render the selected expanded Stage 7 cases in the same visual style as SparseVLM's qualitative pruning examples: original image first, followed by layer-wise faded masks where explicitly retained original patches remain visible.

- Selected cases rendered: 141
- Figures rendered: 423
- Figure folder: `outputs/stage7/figures`
- Manifest: `outputs/stage7/stage7_visualization_manifest.csv`

## Counts by method

- SparseVLM-Original-64: 141
- Ours-64: 141
- Threshold-Fixed-64: 141

## Selected cases by dataset

- gqa: 132
- pope: 9

## Figures by dataset

- gqa: 396
- pope: 27

## Interpretation note

The masks show explicit retained original-patch traces at pruning layers 2, 6, and 15. SparseVLM's merged/recycled tokens do not map to a single original patch, so they are intentionally not shown. Use these figures as qualitative evidence, not causal proof.
