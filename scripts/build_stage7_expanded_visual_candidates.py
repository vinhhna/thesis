from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from build_stage7_visualizations import (
    GRID_SIZE,
    RUN_SPECS,
    SelectionTrace,
    coordinate_stats,
    draw_overlay,
    draw_wrapped_text,
    extract_selection_trace,
    fit_on_white,
    font,
    repo_clip_input_view,
    unique_overlap_and_jaccard,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "stage7_expanded"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

FAILURE_MINING_SNAPSHOT = (
    REPO_ROOT / "outputs" / "raw_results" / "fm_sparse_64" / "failure_mining_set_snapshot.csv"
)

DATASET_SPECS = {
    "gqa": {
        "dense": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "gqa_dense"
        / "results"
        / "gqa"
        / "predictions"
        / "gqa_dense_576.jsonl",
        "sparse": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "gqa_sparse_64"
        / "results"
        / "gqa"
        / "predictions"
        / "gqa_sparsevlm_original_64.jsonl",
        "ours": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "gqa_ours_64"
        / "results"
        / "gqa"
        / "predictions"
        / "gqa_ours_64.jsonl",
        "threshold": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "gqa_thres_64"
        / "results"
        / "gqa"
        / "predictions"
        / "gqa_threshold_fixed_64.jsonl",
    },
    "pope": {
        "dense": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "pope_full_dense"
        / "results"
        / "pope"
        / "predictions"
        / "pope_dense_576.jsonl",
        "sparse": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "pope_sparse_64"
        / "results"
        / "pope"
        / "predictions"
        / "pope_sparsevlm_original_64.jsonl",
        "ours": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "pope_ours_64"
        / "results"
        / "pope"
        / "predictions"
        / "pope_ours_64.jsonl",
        "threshold": REPO_ROOT
        / "outputs"
        / "raw_results"
        / "pope_thres_64"
        / "results"
        / "pope"
        / "predictions"
        / "pope_threshold_fixed_64.jsonl",
    },
}

CANDIDATE_FIELDS = [
    "rank",
    "expanded_case_id",
    "dataset",
    "question_id",
    "image_id",
    "image_path",
    "local_image_path",
    "prompt",
    "ground_truth",
    "dense_prediction",
    "sparse_prediction",
    "ours_prediction",
    "threshold_prediction",
    "dense_is_correct",
    "is_sparse_wrong",
    "is_ours_correct",
    "is_threshold_correct",
    "case_group",
    "semantic_type",
    "structural_type",
    "detailed_type",
    "pope_category",
    "candidate_reason",
    "selection_note",
]

METRIC_FIELDS = [
    "expanded_case_id",
    "dataset",
    "question_id",
    "case_group",
    "sparse_raw_selected_patch_index_count",
    "sparse_unique_patch_count",
    "sparse_duplicate_patch_index_count",
    "sparse_bbox",
    "sparse_spatial_spread_mean_grid_distance",
    "ours_raw_selected_patch_index_count",
    "ours_unique_patch_count",
    "ours_duplicate_patch_index_count",
    "ours_bbox",
    "ours_spatial_spread_mean_grid_distance",
    "threshold_raw_selected_patch_index_count",
    "threshold_unique_patch_count",
    "threshold_duplicate_patch_index_count",
    "threshold_bbox",
    "threshold_spatial_spread_mean_grid_distance",
    "sparse_ours_unique_overlap_count",
    "sparse_ours_unique_jaccard",
    "sparse_threshold_unique_overlap_count",
    "sparse_threshold_unique_jaccard",
    "ours_threshold_unique_overlap_count",
    "ours_threshold_unique_jaccard",
]

REVIEW_FIELDS = [
    "rank",
    "expanded_case_id",
    "dataset",
    "question_id",
    "case_group",
    "comparison_figure",
    "prompt",
    "ground_truth",
    "dense_prediction",
    "sparse_prediction",
    "ours_prediction",
    "threshold_prediction",
    "manual_keep",
    "hypothesized_evidence_region",
    "sparse_covers_evidence",
    "ours_covers_evidence",
    "threshold_covers_evidence",
    "visual_support",
    "notes",
]

MANUAL_REVIEW_FIELDS = [
    "manual_keep",
    "hypothesized_evidence_region",
    "sparse_covers_evidence",
    "ours_covers_evidence",
    "threshold_covers_evidence",
    "visual_support",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine non-failure-mining SparseVLM-Original-64 failures recovered by "
            "Ours-64 and render expanded Stage 7 candidate visualizations."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gqa", "pope"],
        choices=sorted(DATASET_SPECS),
        help="Datasets to mine.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=0,
        help="Maximum ranked candidates to visualize; 0 means all candidates.",
    )
    parser.add_argument(
        "--expected-final-layer",
        type=int,
        default=15,
        help="Expected final sparse pruning layer.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write candidate/metric CSVs without rendering comparison figures.",
    )
    return parser.parse_args()


def read_jsonl_by_question_id(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise RuntimeError(f"Missing prediction file: {path}")
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id", "")).strip()
            if not question_id:
                raise RuntimeError(f"{path}:{line_number}: missing question_id")
            if question_id in rows:
                raise RuntimeError(f"{path}: duplicate question_id {question_id}")
            rows[question_id] = row
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_existing_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            case_id = str(row.get("expanded_case_id", "")).strip()
            if case_id:
                rows[case_id] = {key: str(value or "") for key, value in row.items()}
    return rows


def normalize_text(text: object) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pope_label(text: object) -> str:
    words = normalize_text(text).split()
    return "no" if "no" in words or "not" in words else "yes"


def is_correct(dataset: str, row: dict) -> bool:
    if dataset == "pope":
        return pope_label(row.get("text", "")) == normalize_text(row.get("ground_truth", ""))
    value = row.get("is_correct")
    if not isinstance(value, bool):
        raise RuntimeError(
            f"{dataset}/{row.get('question_id')}: missing boolean is_correct"
        )
    return value


def display_prediction(dataset: str, row: dict) -> str:
    if dataset == "pope":
        return pope_label(row.get("text", ""))
    return str(row.get("text", "")).strip()


def question_text(row: dict) -> str:
    return str(row.get("raw_question") or row.get("prompt") or "").strip()


def local_image_path(raw_path: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise RuntimeError("empty image_path")
    path = Path(raw)
    if path.is_file():
        return path
    if raw.startswith("/kaggle/input/datasets/tvtttv/gqa-thesis/GQA/images/"):
        mapped = REPO_ROOT / "data" / "kaggle_datasets" / "GQA" / "images" / path.name
        if mapped.is_file():
            return mapped
    if raw.startswith("/kaggle/input/datasets/tvtttv/pope-thesis/POPE/val2014/"):
        mapped = (
            REPO_ROOT
            / "data"
            / "kaggle_datasets"
            / "POPE"
            / "val2014"
            / path.name
        )
        if mapped.is_file():
            return mapped
    if not path.is_absolute():
        mapped = REPO_ROOT / path
        if mapped.is_file():
            return mapped
    raise RuntimeError(f"Could not resolve image path locally: {raw}")


def failure_mining_keys() -> set[tuple[str, str]]:
    if not FAILURE_MINING_SNAPSHOT.is_file():
        return set()
    keys: set[tuple[str, str]] = set()
    with FAILURE_MINING_SNAPSHOT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            basename = Path(str(row.get("image_path", ""))).name
            question = normalize_text(row.get("question", ""))
            if basename and question:
                keys.add((basename, question))
    return keys


def image_id(row: dict) -> str:
    return str(row.get("image_id") or row.get("image") or "").strip()


def expanded_case_id(dataset: str, question_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", question_id)
    return f"{dataset.upper()}_{safe_id}"


def load_dataset_runs(dataset: str) -> dict[str, dict[str, dict]]:
    return {
        method: read_jsonl_by_question_id(path)
        for method, path in DATASET_SPECS[dataset].items()
    }


def build_candidates_for_dataset(
    dataset: str,
    expected_final_layer: int,
    fm_keys: set[tuple[str, str]],
) -> tuple[list[dict], dict[str, dict[str, SelectionTrace]]]:
    runs = load_dataset_runs(dataset)
    common_ids = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    if not common_ids:
        raise RuntimeError(f"{dataset}: no common question_ids")

    candidates: list[dict] = []
    traces_by_case: dict[str, dict[str, SelectionTrace]] = {}

    for question_id in common_ids:
        dense = runs["dense"][question_id]
        sparse = runs["sparse"][question_id]
        ours = runs["ours"][question_id]
        threshold = runs["threshold"][question_id]

        basename = Path(str(sparse.get("image_path", ""))).name
        fm_key = (basename, normalize_text(question_text(sparse)))
        if fm_key in fm_keys:
            continue

        sparse_correct = is_correct(dataset, sparse)
        ours_correct = is_correct(dataset, ours)
        if sparse_correct or not ours_correct:
            continue

        threshold_correct = is_correct(dataset, threshold)
        case_group = (
            "ours_and_threshold_recovery"
            if threshold_correct
            else "ours_only_recovery"
        )
        local_path = local_image_path(str(sparse.get("image_path", "")))
        case_id = expanded_case_id(dataset, question_id)

        traces_by_case[case_id] = {
            "sparse": extract_selection_trace(
                case_id, "sparse", sparse, expected_final_layer
            ),
            "ours": extract_selection_trace(case_id, "ours", ours, expected_final_layer),
            "threshold": extract_selection_trace(
                case_id, "threshold", threshold, expected_final_layer
            ),
        }

        candidates.append(
            {
                "expanded_case_id": case_id,
                "dataset": dataset,
                "question_id": question_id,
                "image_id": image_id(sparse),
                "image_path": str(sparse.get("image_path", "")),
                "local_image_path": local_path.relative_to(REPO_ROOT).as_posix(),
                "prompt": question_text(sparse),
                "ground_truth": str(sparse.get("ground_truth", "")).strip(),
                "dense_prediction": display_prediction(dataset, dense),
                "sparse_prediction": display_prediction(dataset, sparse),
                "ours_prediction": display_prediction(dataset, ours),
                "threshold_prediction": display_prediction(dataset, threshold),
                "dense_is_correct": str(is_correct(dataset, dense)),
                "is_sparse_wrong": "True",
                "is_ours_correct": "True",
                "is_threshold_correct": str(threshold_correct),
                "case_group": case_group,
                "semantic_type": str(sparse.get("semantic_type", "")),
                "structural_type": str(sparse.get("structural_type", "")),
                "detailed_type": str(sparse.get("detailed_type", "")),
                "pope_category": str(sparse.get("pope_category", "")),
                "candidate_reason": (
                    "SparseVLM-Original-64 is wrong and Ours-64 is correct in a "
                    "non-failure-mining benchmark case."
                ),
                "selection_note": (
                    "Candidate for manual Stage 7 inspection only; not a prevalence "
                    "claim and not causal proof."
                ),
            }
        )

    return candidates, traces_by_case


def trace_metric_prefix(trace: SelectionTrace) -> dict[str, str]:
    stats = coordinate_stats(trace.unique_indices)
    return {
        "raw_selected_patch_index_count": str(len(trace.indices)),
        "unique_patch_count": str(len(trace.unique_indices)),
        "duplicate_patch_index_count": str(len(trace.indices) - len(trace.unique_indices)),
        "bbox": stats["bbox"],
        "spatial_spread_mean_grid_distance": stats["spread"],
    }


def build_metric_rows(
    candidates: list[dict],
    traces_by_case: dict[str, dict[str, SelectionTrace]],
) -> list[dict]:
    rows: list[dict] = []
    for candidate in candidates:
        case_id = candidate["expanded_case_id"]
        traces = traces_by_case[case_id]
        row = {
            "expanded_case_id": case_id,
            "dataset": candidate["dataset"],
            "question_id": candidate["question_id"],
            "case_group": candidate["case_group"],
        }
        for method in ["sparse", "ours", "threshold"]:
            metrics = trace_metric_prefix(traces[method])
            for key, value in metrics.items():
                row[f"{method}_{key}"] = value

        for left, right, prefix in [
            ("sparse", "ours", "sparse_ours"),
            ("sparse", "threshold", "sparse_threshold"),
            ("ours", "threshold", "ours_threshold"),
        ]:
            overlap, jaccard = unique_overlap_and_jaccard(
                traces[left].unique_indices,
                traces[right].unique_indices,
            )
            row[f"{prefix}_unique_overlap_count"] = str(overlap)
            row[f"{prefix}_unique_jaccard"] = jaccard
        rows.append(row)
    return rows


def sort_candidates(
    candidates: list[dict],
    metric_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    metrics_by_case = {row["expanded_case_id"]: row for row in metric_rows}

    def key(candidate: dict) -> tuple:
        metric = metrics_by_case[candidate["expanded_case_id"]]
        return (
            0 if candidate["case_group"] == "ours_only_recovery" else 1,
            0 if candidate["dense_is_correct"] == "True" else 1,
            float(metric["sparse_ours_unique_jaccard"]),
            candidate["dataset"],
            candidate["question_id"],
        )

    sorted_candidates = sorted(candidates, key=key)
    ranks = {
        candidate["expanded_case_id"]: idx
        for idx, candidate in enumerate(sorted_candidates, start=1)
    }
    for candidate in sorted_candidates:
        candidate["rank"] = str(ranks[candidate["expanded_case_id"]])
    sorted_metrics = sorted(metric_rows, key=lambda row: ranks[row["expanded_case_id"]])
    return sorted_candidates, sorted_metrics


def file_safe_case_id(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)


def make_comparison_figure(
    candidate: dict,
    traces: dict[str, SelectionTrace],
    metric: dict,
) -> Image.Image:
    raw = Image.open(REPO_ROOT / candidate["local_image_path"]).convert("RGB")
    clip = repo_clip_input_view(raw, image_aspect_ratio=None)
    overlays = {
        method: draw_overlay(
            clip,
            traces[method].unique_indices,
            RUN_SPECS[method]["color"],
        )
        for method in ["sparse", "ours", "threshold"]
    }

    panel_size = 336
    margin = 18
    gap = 14
    header_height = 158
    label_height = 46
    footer_height = 58
    panel_count = 5
    width = margin * 2 + panel_count * panel_size + (panel_count - 1) * gap
    height = header_height + label_height + panel_size + footer_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = (
        f"Rank {candidate['rank']} - {candidate['expanded_case_id']} - "
        f"{candidate['case_group']}"
    )
    draw.text((margin, 14), title, fill=(0, 0, 0), font=font())
    y = draw_wrapped_text(draw, (margin, 34), f"Question: {candidate['prompt']}", 160)
    y = draw_wrapped_text(
        draw,
        (margin, y + 3),
        (
            f"GT: {candidate['ground_truth']} | Dense: {candidate['dense_prediction']} | "
            f"Original: {candidate['sparse_prediction']} | Ours: {candidate['ours_prediction']} | "
            f"Threshold: {candidate['threshold_prediction']}"
        ),
        160,
    )
    draw_wrapped_text(
        draw,
        (margin, y + 3),
        (
            "Diagnostic: Sparse/Ours unique-set Jaccard="
            f"{metric['sparse_ours_unique_jaccard']}; "
            "inspect manually before using this case."
        ),
        160,
        fill=(70, 70, 70),
    )

    panels = [
        ("Raw image", fit_on_white(raw)),
        ("Repo CLIP input view", clip),
        ("SparseVLM-Original", overlays["sparse"]),
        ("Ours", overlays["ours"]),
        ("Threshold-Fixed", overlays["threshold"]),
    ]
    panel_y = header_height + label_height
    for idx, (label, image) in enumerate(panels):
        x = margin + idx * (panel_size + gap)
        draw.text((x, header_height), label, fill=(0, 0, 0), font=font())
        if label in {"SparseVLM-Original", "Ours", "Threshold-Fixed"}:
            method = {
                "SparseVLM-Original": "sparse",
                "Ours": "ours",
                "Threshold-Fixed": "threshold",
            }[label]
            trace = traces[method]
            draw.text(
                (x, header_height + 18),
                f"raw={len(trace.indices)}, unique={len(trace.unique_indices)}",
                fill=(55, 55, 55),
                font=font(),
            )
        canvas.paste(image, (x, panel_y))

    footer = (
        "Expanded candidate search: selected because Original-64 is wrong and "
        "Ours-64 is correct. Visual evidence must be manually verified."
    )
    draw_wrapped_text(
        draw,
        (margin, panel_y + panel_size + 12),
        footer,
        160,
        fill=(70, 70, 70),
    )
    return canvas


def render_figures(
    candidates: list[dict],
    metric_rows: list[dict],
    traces_by_case: dict[str, dict[str, SelectionTrace]],
    max_visualizations: int,
) -> dict[str, str]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_by_case = {row["expanded_case_id"]: row for row in metric_rows}
    if max_visualizations and max_visualizations > 0:
        selected = candidates[:max_visualizations]
    else:
        selected = candidates
    paths: dict[str, str] = {}
    for candidate in selected:
        case_id = candidate["expanded_case_id"]
        figure = make_comparison_figure(
            candidate,
            traces_by_case[case_id],
            metrics_by_case[case_id],
        )
        path = (
            FIGURE_ROOT
            / f"{int(candidate['rank']):04d}_{file_safe_case_id(case_id)}_comparison.png"
        )
        figure.save(path)
        paths[case_id] = path.relative_to(REPO_ROOT).as_posix()
    return paths


def build_review_rows(candidates: list[dict], figure_paths: dict[str, str]) -> list[dict]:
    review_path = OUTPUT_ROOT / "expanded_visualization_review.csv"
    existing = read_existing_review(review_path)
    rows: list[dict] = []
    for candidate in candidates:
        case_id = candidate["expanded_case_id"]
        if case_id not in figure_paths:
            continue
        row = {
            "rank": candidate["rank"],
            "expanded_case_id": case_id,
            "dataset": candidate["dataset"],
            "question_id": candidate["question_id"],
            "case_group": candidate["case_group"],
            "comparison_figure": figure_paths[case_id],
            "prompt": candidate["prompt"],
            "ground_truth": candidate["ground_truth"],
            "dense_prediction": candidate["dense_prediction"],
            "sparse_prediction": candidate["sparse_prediction"],
            "ours_prediction": candidate["ours_prediction"],
            "threshold_prediction": candidate["threshold_prediction"],
        }
        for field in MANUAL_REVIEW_FIELDS:
            row[field] = existing.get(case_id, {}).get(field, "")
        rows.append(row)
    return rows


def write_summary(candidates: list[dict], figure_paths: dict[str, str]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for candidate in candidates:
        key = (candidate["dataset"], candidate["case_group"])
        counts[key] = counts.get(key, 0) + 1

    lines = [
        "# Expanded Stage 7 Candidate Search",
        "",
        (
            "This output searches non-failure-mining benchmark cases where "
            "SparseVLM-Original-64 is wrong and Ours-64 is correct. These are "
            "manual visualization candidates only; selecting examples from this "
            "pool must be reported as qualitative case selection, not as an "
            "unbiased prevalence estimate."
        ),
        "",
        f"- Total candidates: {len(candidates)}",
        f"- Rendered comparison figures: {len(figure_paths)}",
        "",
        "| Dataset | Group | Cases |",
        "| --- | --- | ---: |",
    ]
    for dataset in sorted({candidate["dataset"] for candidate in candidates}):
        for group in ["ours_only_recovery", "ours_and_threshold_recovery"]:
            lines.append(f"| {dataset} | {group} | {counts.get((dataset, group), 0)} |")
    lines.extend(
        [
            "",
            "Primary files:",
            "",
            "- `expanded_recovery_candidates.csv`: all mined candidates.",
            "- `expanded_recovery_metrics.csv`: selected-token counts and unique-set Jaccard.",
            "- `expanded_visualization_review.csv`: rendered candidates with blank manual fields.",
            "- `figures/`: comparison images for manual inspection.",
            "",
            "Use caution: a correct answer plus different token map is not causal proof. "
            "Keep only cases whose visual overlay actually supports the hypothesized "
            "evidence-preservation explanation.",
            "",
        ]
    )
    (OUTPUT_ROOT / "expanded_candidate_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    fm_keys = failure_mining_keys()

    all_candidates: list[dict] = []
    all_traces: dict[str, dict[str, SelectionTrace]] = {}
    for dataset in args.datasets:
        candidates, traces = build_candidates_for_dataset(
            dataset,
            expected_final_layer=args.expected_final_layer,
            fm_keys=fm_keys,
        )
        all_candidates.extend(candidates)
        all_traces.update(traces)

    metric_rows = build_metric_rows(all_candidates, all_traces)
    candidates, metric_rows = sort_candidates(all_candidates, metric_rows)
    write_csv(OUTPUT_ROOT / "expanded_recovery_candidates.csv", candidates, CANDIDATE_FIELDS)
    write_csv(OUTPUT_ROOT / "expanded_recovery_metrics.csv", metric_rows, METRIC_FIELDS)

    figure_paths: dict[str, str] = {}
    if not args.skip_figures:
        figure_paths = render_figures(
            candidates,
            metric_rows,
            all_traces,
            max_visualizations=args.max_visualizations,
        )
        review_rows = build_review_rows(candidates, figure_paths)
        write_csv(
            OUTPUT_ROOT / "expanded_visualization_review.csv",
            review_rows,
            REVIEW_FIELDS,
        )

    write_summary(candidates, figure_paths)
    print(f"Wrote candidates: {OUTPUT_ROOT / 'expanded_recovery_candidates.csv'}")
    print(f"Wrote metrics: {OUTPUT_ROOT / 'expanded_recovery_metrics.csv'}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Rendered figures: {len(figure_paths)}")
    if figure_paths:
        print(f"Wrote review: {OUTPUT_ROOT / 'expanded_visualization_review.csv'}")
    print(f"Wrote summary: {OUTPUT_ROOT / 'expanded_candidate_summary.md'}")


if __name__ == "__main__":
    main()
