from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE8_ROOT = REPO_ROOT / "outputs" / "stage8"

SELECTED_TOKEN_METRICS_PATH = STAGE8_ROOT / "stage8_selected_token_metrics.csv"
PAIRWISE_SUMMARY_PATH = STAGE8_ROOT / "stage8_pairwise_similarity_summary.csv"
SPATIAL_SUMMARY_PATH = STAGE8_ROOT / "stage8_spatial_coverage_summary.csv"
OVERLAP_SUMMARY_PATH = STAGE8_ROOT / "stage8_overlap_jaccard_summary.csv"
FAILURE_RECOVERY_PATH = STAGE8_ROOT / "stage8_failure_recovery_summary.csv"
ABLATION_RESULTS_PATH = STAGE8_ROOT / "stage8_ablation_results.csv"
ANALYSIS_PATH = STAGE8_ROOT / "stage8_analysis.md"

REQUIRED_INPUTS = [
    SELECTED_TOKEN_METRICS_PATH,
    PAIRWISE_SUMMARY_PATH,
    SPATIAL_SUMMARY_PATH,
    OVERLAP_SUMMARY_PATH,
    FAILURE_RECOVERY_PATH,
    ABLATION_RESULTS_PATH,
]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Missing required Stage 8 file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def md_table(rows: list[dict], fields: list[str], limit: int | None = None) -> list[str]:
    if limit is not None:
        rows = rows[:limit]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", r"\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def final_layer(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("layer_idx", "")) == "15"]


def rows_for_methods(rows: list[dict], methods: set[str]) -> list[dict]:
    return [row for row in rows if row.get("selection_method") in methods]


def has_pairwise(rows: list[dict]) -> bool:
    return any(int(row.get("pairwise_available_count") or 0) > 0 for row in rows)


def missing_ablation_rows(rows: list[dict]) -> list[str]:
    return [
        row["run_id"]
        for row in rows
        if row.get("ablation_role") == "ablation_variant"
        and row.get("run_available") != "true"
    ]


def compare_failure_row(rows: list[dict], run_id: str) -> dict:
    for row in rows:
        if row.get("run_id") == run_id:
            return row
    return {}


def build_markdown(
    token_rows: list[dict],
    pairwise_rows: list[dict],
    spatial_rows: list[dict],
    overlap_rows: list[dict],
    failure_rows: list[dict],
    ablation_rows: list[dict],
) -> str:
    final_spatial = final_layer(spatial_rows)
    final_pairwise = final_layer(pairwise_rows)
    final_overlap = final_layer(overlap_rows)
    pairwise_ready = has_pairwise(pairwise_rows)
    pending_ablation = missing_ablation_rows(ablation_rows)

    lines = [
        "# Stage 8 — Ablation and Auxiliary Metric Analysis",
        "",
        "Stage 8 quantifies selected-token behavior because Stage 7 visualizations were mostly inconclusive. "
        "The current outputs use all available saved prediction metadata for spatial coverage, selected-patch "
        "overlap, and failure recovery. Pairwise hidden-state similarity is reported only when instrumented "
        "Kaggle reruns provide the required metadata.",
        "",
        "## Data status",
        "",
        f"- Detailed selected-token metric rows: {len(token_rows)}",
        f"- Pairwise similarity available: {'yes' if pairwise_ready else 'no'}",
        f"- Pending planned Ours ablation runs: {len(pending_ablation)}",
        "",
    ]

    lines.extend([
        "## Pairwise similarity",
        "",
    ])
    if pairwise_ready:
        rows = rows_for_methods(final_pairwise, {"topk", "mmr", "threshold_fixed"})
        lines.extend(md_table(
            rows,
            [
                "dataset",
                "token_setting",
                "method",
                "run_id",
                "pairwise_available_count",
                "mean_mean_pairwise_similarity",
                "mean_p90_pairwise_similarity",
                "mean_similarity_above_0.85_ratio",
            ],
            limit=18,
        ))
        lines.append("")
        lines.append(
            "Interpretation: lower similarity may support a redundancy-reduction claim only when it is paired "
            "with stable or improved answer-level behavior."
        )
    else:
        lines.append(
            "Pairwise similarity is not available in the current saved predictions because they do not contain "
            "instrumented selected-token hidden-state similarity aggregates. Therefore Stage 8 does not yet make "
            "a redundancy-reduction claim from pairwise similarity."
        )
    lines.append("")

    lines.extend([
        "## Spatial coverage",
        "",
        "The following final-layer spatial metrics are diagnostic only. Broader coverage is not automatically better, "
        "because some questions require focused local evidence.",
        "",
    ])
    lines.extend(md_table(
        rows_for_methods(final_spatial, {"topk", "mmr", "threshold_fixed"}),
        [
            "dataset",
            "token_setting",
            "method",
            "run_id",
            "mean_unique_patch_count",
            "mean_bbox_area_ratio",
            "mean_grid_occupancy_ratio",
            "mean_distance_to_centroid",
            "mean_quadrant_coverage_count",
        ],
        limit=18,
    ))
    lines.append("")

    lines.extend([
        "## Ours / Threshold overlap",
        "",
        "Jaccard and overlap are computed from unique original patch IDs, not raw selected-index lists.",
        "",
    ])
    lines.extend(md_table(
        final_overlap,
        [
            "dataset",
            "token_setting",
            "pair",
            "sample_count",
            "mean_overlap_count",
            "mean_overlap_ratio_a",
            "mean_overlap_ratio_b",
            "mean_jaccard_overlap",
        ],
        limit=18,
    ))
    lines.append("")

    lines.extend([
        "## Failure recovery",
        "",
    ])
    focus_rows = [
        row for row in failure_rows
        if row.get("run_id") in {
            "GQA-OURS-64",
            "GQA-THRESHOLD-FIXED-64",
            "POPE-OURS-64",
            "POPE-THRESHOLD-FIXED-64",
            "FM-OURS-64",
            "FM-THRESHOLD-FIXED-64",
        }
    ]
    lines.extend(md_table(
        focus_rows,
        [
            "dataset",
            "token_setting",
            "run_id",
            "sparse_wrong_count",
            "recovered_baseline_failures",
            "recovery_rate",
            "regressions_vs_baseline",
            "net_gain",
        ],
    ))
    lines.append("")
    fm_ours = compare_failure_row(failure_rows, "FM-OURS-64")
    if fm_ours:
        lines.extend([
            "For failure-mining at 64 tokens, the known Stage 6 recovery grouping is preserved:",
            "",
            f"- SparseVLM-Original-64 wrong cases: {fm_ours.get('sparse_wrong_count')}",
            f"- Both Ours and Threshold recover: {fm_ours.get('both_recovered')}",
            f"- Ours-only recoveries: {fm_ours.get('ours_only_recovered')}",
            f"- Threshold-only recoveries: {fm_ours.get('threshold_only_recovered')}",
            f"- Unresolved by both: {fm_ours.get('unresolved_by_ours_and_threshold')}",
            "",
        ])

    lines.extend([
        "## Ours ablation status",
        "",
    ])
    lines.extend(md_table(
        ablation_rows,
        [
            "dataset",
            "run_id",
            "run_available",
            "ablation_role",
            "candidate_pool_factor",
            "lambda_relevance",
            "accuracy",
            "f1",
            "mean_pairwise_similarity",
            "notes",
        ],
        limit=30,
    ))
    lines.append("")

    lines.extend([
        "## Threshold-Adaptive interpretation",
        "",
        "Threshold-Adaptive is treated as a variable-token trade-off baseline. It should not be described as beating "
        "or losing to Ours under the same fixed 64-token or 128-token budget, because its retained-token count varies "
        "by sample.",
        "",
        "## Current thesis-safe conclusion",
        "",
    ])
    if pairwise_ready:
        lines.append(
            "Stage 8 can compare answer-level performance with selected-token similarity, spatial coverage, and "
            "baseline overlap. Any redundancy-reduction claim should be conditioned on the actual pairwise similarity "
            "direction and whether it aligns with accuracy or failure recovery."
        )
    else:
        lines.append(
            "At the current checkpoint, Stage 8 supports spatial/overlap/failure-recovery analysis from saved metadata, "
            "but it does not yet support a strong redundancy-reduction claim because true pairwise hidden-state "
            "similarity requires instrumented Kaggle reruns."
        )
    if pending_ablation:
        lines.append(
            "The Ours hyperparameter conclusion is also pending until the planned 64-token candidate-pool/lambda "
            "ablation runs are imported."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 8 markdown analysis.")
    parser.add_argument("--require-pairwise", action="store_true", help="Fail unless pairwise similarity is available.")
    parser.add_argument("--require-ablation", action="store_true", help="Fail unless all planned ablation runs are available.")
    args = parser.parse_args()

    for path in REQUIRED_INPUTS:
        if not path.is_file():
            raise RuntimeError(f"Missing required Stage 8 file: {path}")

    token_rows = read_csv(SELECTED_TOKEN_METRICS_PATH)
    pairwise_rows = read_csv(PAIRWISE_SUMMARY_PATH)
    spatial_rows = read_csv(SPATIAL_SUMMARY_PATH)
    overlap_rows = read_csv(OVERLAP_SUMMARY_PATH)
    failure_rows = read_csv(FAILURE_RECOVERY_PATH)
    ablation_rows = read_csv(ABLATION_RESULTS_PATH)

    if args.require_pairwise and not has_pairwise(pairwise_rows):
        raise RuntimeError("Pairwise similarity was required, but no pairwise metrics are available.")
    missing_ablation = missing_ablation_rows(ablation_rows)
    if args.require_ablation and missing_ablation:
        raise RuntimeError(f"Ablation completion was required, but these runs are missing: {missing_ablation}")

    ANALYSIS_PATH.write_text(
        build_markdown(token_rows, pairwise_rows, spatial_rows, overlap_rows, failure_rows, ablation_rows),
        encoding="utf-8",
    )
    print(f"Wrote Stage 8 analysis to {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
