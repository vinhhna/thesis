from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS_ROOT = REPO_ROOT / "outputs" / "raw_results"
STAGE8_ROOT = REPO_ROOT / "outputs" / "stage8"

SELECTED_TOKEN_METRICS_PATH = STAGE8_ROOT / "stage8_selected_token_metrics.csv"
PAIRWISE_SUMMARY_PATH = STAGE8_ROOT / "stage8_pairwise_similarity_summary.csv"
SPATIAL_SUMMARY_PATH = STAGE8_ROOT / "stage8_spatial_coverage_summary.csv"
OVERLAP_SUMMARY_PATH = STAGE8_ROOT / "stage8_overlap_jaccard_summary.csv"
FAILURE_RECOVERY_PATH = STAGE8_ROOT / "stage8_failure_recovery_summary.csv"

GRID_SIZE = 24
PATCH_COUNT = GRID_SIZE * GRID_SIZE
PAIRWISE_FIELDS = [
    "mean_pairwise_similarity",
    "median_pairwise_similarity",
    "max_pairwise_similarity",
    "p90_pairwise_similarity",
    "similarity_above_0.80_ratio",
    "similarity_above_0.85_ratio",
    "similarity_above_0.90_ratio",
]

TOKEN_METRIC_FIELDS = [
    "dataset",
    "run_id",
    "method",
    "selection_method",
    "token_setting",
    "threshold_tau",
    "candidate_pool_factor",
    "lambda_relevance",
    "case_id",
    "question_id",
    "layer_idx",
    "is_final_layer",
    "current_visual_token_count",
    "per_layer_budget",
    "selected_count",
    "retained_token_count",
    "raw_selected_patch_index_count",
    "unique_patch_count",
    "duplicate_patch_index_count",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
    "row_span",
    "col_span",
    "bbox_area_ratio",
    "grid_occupancy_ratio",
    "centroid_row",
    "centroid_col",
    "mean_distance_to_centroid",
    "quadrant_coverage_count",
    "pairwise_available",
    "pairwise_similarity_token_count",
    *PAIRWISE_FIELDS,
]


def format_float(value: object, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def normalize_dataset(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"failure_mining", "failure_mining_set", "failure"}:
        return "failure_mining"
    if text in {"gqa", "gqa_val_balanced", "gqa_val"}:
        return "gqa"
    if text in {"pope"}:
        return "pope"
    return text


def manifest_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("runs") or payload.get("records") or payload.get("results") or [payload]
    else:
        records = []
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("run_id") and record.get("prediction_file")
    ]


def resolve_prediction_path(manifest_path: Path, record: dict) -> Path:
    prediction_file = str(record.get("prediction_file", ""))
    prediction_name = Path(prediction_file).name
    if not prediction_name:
        raise RuntimeError(f"{manifest_path}: run {record.get('run_id')} has no prediction_file")
    matches = sorted(manifest_path.parent.rglob(prediction_name))
    if not matches:
        matches = sorted(RAW_RESULTS_ROOT.rglob(prediction_name))
    if len(matches) != 1:
        raise RuntimeError(
            f"{manifest_path}: could not uniquely resolve {prediction_name}; "
            f"found {len(matches)} matches"
        )
    return matches[0]


def discover_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for manifest_path in sorted(RAW_RESULTS_ROOT.rglob("*manifest*.json")):
        for record in manifest_records(manifest_path):
            if str(record.get("status", "ok")).lower() != "ok":
                continue
            run_id = str(record["run_id"])
            if run_id in runs:
                raise RuntimeError(f"Duplicate run_id in manifests: {run_id}")
            prediction_path = resolve_prediction_path(manifest_path, record)
            runs[run_id] = {
                "run_id": run_id,
                "manifest": record,
                "manifest_path": manifest_path,
                "prediction_path": prediction_path,
                "dataset": normalize_dataset(record.get("dataset")),
                "method": str(record.get("method", "")),
                "selection_method": str(record.get("selection_method", "")),
                "token_setting": str(record.get("retained_tokens", "")),
                "threshold_tau": record.get("threshold_tau", ""),
            }
    if not runs:
        raise RuntimeError(f"No completed run manifests found under {RAW_RESULTS_ROOT}")
    return runs


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not rows:
        raise RuntimeError(f"{path}: no prediction rows")
    return rows


def case_key(row: dict) -> str:
    value = row.get("case_id") or row.get("question_id") or row.get("pope_question_id")
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Prediction row has no case_id/question_id: {row.keys()}")
    return str(value)


def normalize_answer(text: object) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = []
    for token in text.split():
        if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def pope_label(text: object) -> str:
    words = normalize_answer(text).split()
    return "no" if "no" in words or "not" in words else "yes"


def exact_or_phrase_match(prediction: object, target: object) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if not pred or not gold:
        return False
    if gold in {"yes", "no"}:
        return pope_label(pred) == gold
    return re.search(rf"(?:^| ){re.escape(gold)}(?: |$)", pred) is not None


def row_is_correct(row: dict, dataset: str) -> bool:
    if isinstance(row.get("is_correct"), bool):
        return bool(row["is_correct"])
    if row.get("is_correct") is not None and str(row["is_correct"]).strip() != "":
        return str(row["is_correct"]).strip().lower() in {"1", "true", "yes"}
    if dataset == "pope":
        return pope_label(row.get("text", "")) == normalize_answer(row.get("ground_truth", ""))
    return exact_or_phrase_match(row.get("text", ""), row.get("ground_truth", ""))


def validate_patch_indices(indices: object, run_id: str, case_id: str, layer_idx: object) -> list[int]:
    if not isinstance(indices, list):
        raise RuntimeError(
            f"{run_id}/{case_id}/layer={layer_idx}: selected_original_token_indices is missing or not a list"
        )
    normalized = []
    for value in indices:
        if not isinstance(value, int):
            raise RuntimeError(
                f"{run_id}/{case_id}/layer={layer_idx}: patch index {value!r} is not an integer"
            )
        if value < 0 or value >= PATCH_COUNT:
            raise RuntimeError(
                f"{run_id}/{case_id}/layer={layer_idx}: patch index {value} outside 0-{PATCH_COUNT - 1}"
            )
        normalized.append(value)
    return normalized


def spatial_stats(indices: Iterable[int]) -> dict[str, str]:
    unique = sorted(set(indices))
    if not unique:
        return {
            "row_min": "",
            "row_max": "",
            "col_min": "",
            "col_max": "",
            "row_span": "",
            "col_span": "",
            "bbox_area_ratio": "",
            "grid_occupancy_ratio": "0.000000",
            "centroid_row": "",
            "centroid_col": "",
            "mean_distance_to_centroid": "",
            "quadrant_coverage_count": "0",
        }
    rows = [idx // GRID_SIZE for idx in unique]
    cols = [idx % GRID_SIZE for idx in unique]
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    row_span = row_max - row_min + 1
    col_span = col_max - col_min + 1
    centroid_row = sum(rows) / len(rows)
    centroid_col = sum(cols) / len(cols)
    mean_distance = mean(
        math.sqrt((row - centroid_row) ** 2 + (col - centroid_col) ** 2)
        for row, col in zip(rows, cols)
    )
    quadrants = set()
    for row, col in zip(rows, cols):
        quadrants.add((0 if row < GRID_SIZE / 2 else 1, 0 if col < GRID_SIZE / 2 else 1))
    return {
        "row_min": str(row_min),
        "row_max": str(row_max),
        "col_min": str(col_min),
        "col_max": str(col_max),
        "row_span": str(row_span),
        "col_span": str(col_span),
        "bbox_area_ratio": format_float((row_span * col_span) / PATCH_COUNT),
        "grid_occupancy_ratio": format_float(len(unique) / PATCH_COUNT),
        "centroid_row": format_float(centroid_row),
        "centroid_col": format_float(centroid_col),
        "mean_distance_to_centroid": format_float(mean_distance),
        "quadrant_coverage_count": str(len(quadrants)),
    }


def pairwise_available(layer: dict) -> bool:
    if layer.get("pairwise_similarity_available") is True:
        return True
    return all(layer.get(field) is not None for field in PAIRWISE_FIELDS)


def build_token_metric_rows(runs: dict[str, dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    metric_rows = []
    prediction_rows_by_run: dict[str, list[dict]] = {}
    for run_id, run in sorted(runs.items()):
        selection_method = run["selection_method"]
        predictions = load_jsonl(run["prediction_path"])
        prediction_rows_by_run[run_id] = predictions
        if selection_method == "dense":
            continue
        for prediction in predictions:
            metadata = prediction.get("metadata") or {}
            layer_stats = metadata.get("layer_token_stats")
            if not isinstance(layer_stats, list) or not layer_stats:
                raise RuntimeError(f"{run_id}/{case_key(prediction)}: missing layer_token_stats")
            final_layer_idx = layer_stats[-1].get("layer_idx")
            for layer in layer_stats:
                cid = case_key(prediction)
                layer_idx = layer.get("layer_idx", "")
                selected = validate_patch_indices(
                    layer.get("selected_original_token_indices"), run_id, cid, layer_idx
                )
                unique = sorted(set(selected))
                row = {
                    "dataset": run["dataset"],
                    "run_id": run_id,
                    "method": run["method"],
                    "selection_method": selection_method,
                    "token_setting": run["token_setting"],
                    "threshold_tau": "" if run.get("threshold_tau") is None else str(run.get("threshold_tau", "")),
                    "candidate_pool_factor": str(metadata.get("candidate_pool_factor", "")),
                    "lambda_relevance": str(metadata.get("lambda_relevance", layer.get("lambda_relevance", "0.8" if selection_method == "mmr" else ""))),
                    "case_id": cid,
                    "question_id": str(prediction.get("question_id", "")),
                    "layer_idx": str(layer_idx),
                    "is_final_layer": str(layer_idx == final_layer_idx),
                    "current_visual_token_count": str(layer.get("current_visual_token_count", "")),
                    "per_layer_budget": str(layer.get("per_layer_budget", "")),
                    "selected_count": str(layer.get("selected_count", "")),
                    "retained_token_count": str(layer.get("retained_token_count", metadata.get("retained_token_count", ""))),
                    "raw_selected_patch_index_count": str(len(selected)),
                    "unique_patch_count": str(len(unique)),
                    "duplicate_patch_index_count": str(len(selected) - len(unique)),
                    "pairwise_available": str(pairwise_available(layer)).lower(),
                    "pairwise_similarity_token_count": str(layer.get("pairwise_similarity_token_count", "")),
                }
                row.update(spatial_stats(unique))
                for field in PAIRWISE_FIELDS:
                    row[field] = format_float(layer.get(field))
                metric_rows.append(row)
    metric_rows.sort(
        key=lambda row: (
            row["dataset"],
            int(row["token_setting"]) if str(row["token_setting"]).isdigit() else 0,
            row["selection_method"],
            row["run_id"],
            row["case_id"],
            int(row["layer_idx"]) if str(row["layer_idx"]).isdigit() else 0,
        )
    )
    return metric_rows, prediction_rows_by_run


def numeric_values(rows: Iterable[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def mean_field(rows: list[dict], field: str) -> str:
    values = numeric_values(rows, field)
    return format_float(mean(values)) if values else ""


def grouped(rows: Iterable[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    result: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def build_spatial_summary(metric_rows: list[dict]) -> list[dict]:
    rows = []
    keys = ["dataset", "token_setting", "method", "selection_method", "run_id", "layer_idx"]
    for key, items in sorted(grouped(metric_rows, keys).items()):
        dataset, token, method, selection_method, run_id, layer_idx = key
        rows.append({
            "dataset": dataset,
            "token_setting": token,
            "method": method,
            "selection_method": selection_method,
            "run_id": run_id,
            "layer_idx": layer_idx,
            "sample_count": str(len(items)),
            "mean_unique_patch_count": mean_field(items, "unique_patch_count"),
            "mean_duplicate_patch_index_count": mean_field(items, "duplicate_patch_index_count"),
            "mean_bbox_area_ratio": mean_field(items, "bbox_area_ratio"),
            "mean_grid_occupancy_ratio": mean_field(items, "grid_occupancy_ratio"),
            "mean_distance_to_centroid": mean_field(items, "mean_distance_to_centroid"),
            "mean_quadrant_coverage_count": mean_field(items, "quadrant_coverage_count"),
        })
    return rows


def build_pairwise_summary(metric_rows: list[dict]) -> list[dict]:
    rows = []
    keys = ["dataset", "token_setting", "method", "selection_method", "run_id", "layer_idx"]
    for key, items in sorted(grouped(metric_rows, keys).items()):
        dataset, token, method, selection_method, run_id, layer_idx = key
        available = [row for row in items if row.get("pairwise_available") == "true"]
        row = {
            "dataset": dataset,
            "token_setting": token,
            "method": method,
            "selection_method": selection_method,
            "run_id": run_id,
            "layer_idx": layer_idx,
            "sample_count": str(len(items)),
            "pairwise_available_count": str(len(available)),
            "pairwise_available": str(bool(available)).lower(),
            "mean_pairwise_similarity_token_count": mean_field(available, "pairwise_similarity_token_count"),
        }
        for field in PAIRWISE_FIELDS:
            row[f"mean_{field}"] = mean_field(available, field)
        rows.append(row)
    return rows


def method_pair_key(row: dict) -> str:
    method = row["selection_method"]
    if method == "topk":
        return "sparse"
    if method == "mmr":
        return "ours"
    if method == "threshold_fixed":
        return "threshold_fixed"
    return method


def is_official_fixed_run(row: dict) -> bool:
    if "STAGE8" in row["run_id"]:
        return False
    return row["selection_method"] in {"topk", "mmr", "threshold_fixed"} and row["token_setting"] in {"64", "128"}


def overlap_and_jaccard(a: set[int], b: set[int]) -> tuple[int, float, float, float]:
    overlap = len(a & b)
    union = len(a | b)
    ratio_a = overlap / len(a) if a else 0.0
    ratio_b = overlap / len(b) if b else 0.0
    jaccard = overlap / union if union else 0.0
    return overlap, ratio_a, ratio_b, jaccard


def build_overlap_summary(metric_rows: list[dict]) -> list[dict]:
    rows = []
    final_or_layer_rows = [row for row in metric_rows if is_official_fixed_run(row)]
    group_keys = ["dataset", "token_setting", "layer_idx"]
    for key, items in sorted(grouped(final_or_layer_rows, group_keys).items()):
        dataset, token, layer_idx = key
        by_method: dict[str, list[dict]] = defaultdict(list)
        for row in items:
            by_method[method_pair_key(row)].append(row)
        for a_name, b_name in [
            ("ours", "sparse"),
            ("threshold_fixed", "sparse"),
            ("ours", "threshold_fixed"),
        ]:
            if a_name not in by_method or b_name not in by_method:
                continue
            if len({row["run_id"] for row in by_method[a_name]}) != 1:
                raise RuntimeError(f"Ambiguous {a_name} run for {dataset}/{token}/layer {layer_idx}")
            if len({row["run_id"] for row in by_method[b_name]}) != 1:
                raise RuntimeError(f"Ambiguous {b_name} run for {dataset}/{token}/layer {layer_idx}")
            a_map = {row["case_id"]: row for row in by_method[a_name]}
            b_map = {row["case_id"]: row for row in by_method[b_name]}
            if set(a_map) != set(b_map):
                raise RuntimeError(
                    f"Cannot compare {a_name} and {b_name} for {dataset}/{token}/layer {layer_idx}: "
                    "case coverage differs"
                )
            overlaps = []
            ratios_a = []
            ratios_b = []
            jaccards = []
            for case in sorted(a_map):
                a_set = patches_from_metric_row(a_map[case])
                b_set = patches_from_metric_row(b_map[case])
                overlap, ratio_a, ratio_b, jaccard = overlap_and_jaccard(a_set, b_set)
                overlaps.append(overlap)
                ratios_a.append(ratio_a)
                ratios_b.append(ratio_b)
                jaccards.append(jaccard)
            rows.append({
                "dataset": dataset,
                "token_setting": token,
                "layer_idx": layer_idx,
                "pair": f"{a_name}_vs_{b_name}",
                "run_id_a": by_method[a_name][0]["run_id"],
                "run_id_b": by_method[b_name][0]["run_id"],
                "sample_count": str(len(a_map)),
                "mean_overlap_count": format_float(mean(overlaps)),
                "mean_overlap_ratio_a": format_float(mean(ratios_a)),
                "mean_overlap_ratio_b": format_float(mean(ratios_b)),
                "mean_jaccard_overlap": format_float(mean(jaccards)),
            })
    return rows


def patches_from_metric_row(row: dict) -> set[int]:
    # Reconstructing from only spatial summary is impossible, so the detailed row
    # stores the patch set out-of-band through this process-level cache.
    return PATCH_SET_CACHE[(row["run_id"], row["case_id"], row["layer_idx"])]


PATCH_SET_CACHE: dict[tuple[str, str, str], set[int]] = {}


def populate_patch_set_cache(metric_rows: list[dict], runs: dict[str, dict]) -> None:
    PATCH_SET_CACHE.clear()
    for run_id, run in runs.items():
        if run["selection_method"] == "dense":
            continue
        for prediction in load_jsonl(run["prediction_path"]):
            cid = case_key(prediction)
            for layer in prediction.get("metadata", {}).get("layer_token_stats", []):
                layer_idx = str(layer.get("layer_idx", ""))
                selected = validate_patch_indices(
                    layer.get("selected_original_token_indices"), run_id, cid, layer_idx
                )
                PATCH_SET_CACHE[(run_id, cid, layer_idx)] = set(selected)


def prediction_map(rows: list[dict]) -> dict[str, dict]:
    return {case_key(row): row for row in rows}


def official_sparse_run_id(dataset: str, token: str) -> str:
    prefix = {"gqa": "GQA", "pope": "POPE", "failure_mining": "FM"}[dataset]
    return f"{prefix}-SPARSE-ORIG-{token}"


def build_failure_recovery_summary(
    runs: dict[str, dict],
    prediction_rows_by_run: dict[str, list[dict]],
) -> list[dict]:
    rows = []
    predictions = {
        run_id: prediction_map(items)
        for run_id, items in prediction_rows_by_run.items()
    }
    official_group_counts: dict[tuple[str, str], dict[str, int]] = {}

    for dataset in ["gqa", "pope", "failure_mining"]:
        for token in ["64", "128"]:
            sparse_id = official_sparse_run_id(dataset, token)
            ours_id = sparse_id.replace("SPARSE-ORIG", "OURS")
            threshold_id = sparse_id.replace("SPARSE-ORIG", "THRESHOLD-FIXED")
            if sparse_id not in predictions or ours_id not in predictions or threshold_id not in predictions:
                continue
            sparse = predictions[sparse_id]
            ours = predictions[ours_id]
            threshold = predictions[threshold_id]
            if set(sparse) != set(ours) or set(sparse) != set(threshold):
                raise RuntimeError(f"{dataset}/{token}: official sparse/ours/threshold case coverage differs")
            sparse_wrong = {
                case for case, row in sparse.items() if not row_is_correct(row, dataset)
            }
            ours_right = {case for case in sparse_wrong if row_is_correct(ours[case], dataset)}
            threshold_right = {case for case in sparse_wrong if row_is_correct(threshold[case], dataset)}
            official_group_counts[(dataset, token)] = {
                "both_recovered": len(ours_right & threshold_right),
                "ours_only_recovered": len(ours_right - threshold_right),
                "threshold_only_recovered": len(threshold_right - ours_right),
                "unresolved_by_ours_and_threshold": len(sparse_wrong - (ours_right | threshold_right)),
            }

    for run_id, run in sorted(runs.items()):
        if "STAGE8" in run_id:
            # Stage 8 subset runs use 500-sample GQA/POPE subsets and optional
            # targeted failure-mining probes. Their recovery/regression numbers
            # are calculated in build_stage8_ablation_results.py against the
            # matching Stage 8 subset baseline, not in this full-run summary.
            continue
        dataset = run["dataset"]
        selection_method = run["selection_method"]
        token = run["token_setting"]
        if dataset not in {"gqa", "pope", "failure_mining"}:
            continue
        if selection_method in {"dense", "topk"}:
            continue
        if selection_method == "threshold_adaptive":
            token = "64"
        if token not in {"64", "128"}:
            continue
        sparse_id = official_sparse_run_id(dataset, token)
        if sparse_id not in predictions or run_id not in predictions:
            continue
        sparse = predictions[sparse_id]
        current = predictions[run_id]
        if set(sparse) != set(current):
            raise RuntimeError(f"{run_id}: case coverage differs from {sparse_id}")
        sparse_wrong = {case for case, row in sparse.items() if not row_is_correct(row, dataset)}
        sparse_correct = set(sparse) - sparse_wrong
        recovered = {case for case in sparse_wrong if row_is_correct(current[case], dataset)}
        regressions = {case for case in sparse_correct if not row_is_correct(current[case], dataset)}
        group_counts = official_group_counts.get((dataset, token), {})
        rows.append({
            "dataset": dataset,
            "token_setting": token,
            "comparison_method": run["method"],
            "selection_method": selection_method,
            "run_id": run_id,
            "baseline_run_id": sparse_id,
            "sample_count": str(len(sparse)),
            "sparse_wrong_count": str(len(sparse_wrong)),
            "recovered_baseline_failures": str(len(recovered)),
            "recovery_rate": format_float(len(recovered) / len(sparse_wrong) if sparse_wrong else 0.0),
            "regressions_vs_baseline": str(len(regressions)),
            "regression_rate": format_float(len(regressions) / len(sparse_correct) if sparse_correct else 0.0),
            "net_gain": str(len(recovered) - len(regressions)),
            "both_recovered": str(group_counts.get("both_recovered", "")),
            "ours_only_recovered": str(group_counts.get("ours_only_recovered", "")),
            "threshold_only_recovered": str(group_counts.get("threshold_only_recovered", "")),
            "unresolved_by_ours_and_threshold": str(group_counts.get("unresolved_by_ours_and_threshold", "")),
            "comparison_note": (
                "variable-token trade-off baseline"
                if selection_method == "threshold_adaptive"
                else "fixed-budget comparison"
            ),
        })
    rows.sort(key=lambda row: (row["dataset"], int(row["token_setting"]), row["selection_method"], row["run_id"]))
    validate_known_stage6_counts(rows)
    return rows


def validate_known_stage6_counts(rows: list[dict]) -> None:
    fm64 = [
        row for row in rows
        if row["dataset"] == "failure_mining"
        and row["token_setting"] == "64"
        and row["run_id"] in {"FM-OURS-64", "FM-THRESHOLD-FIXED-64"}
    ]
    if not fm64:
        return
    expected_by_run = {
        "FM-OURS-64": {"recovered_baseline_failures": "10"},
        "FM-THRESHOLD-FIXED-64": {"recovered_baseline_failures": "7"},
    }
    for row in fm64:
        expected = {
            "sparse_wrong_count": "30",
            "both_recovered": "7",
            "ours_only_recovered": "3",
            "threshold_only_recovered": "0",
            "unresolved_by_ours_and_threshold": "20",
            **expected_by_run[row["run_id"]],
        }
        for field, value in expected.items():
            if row[field] != value:
                raise RuntimeError(
                    f"Stage 6 validation mismatch for {row['run_id']} field {field}: "
                    f"got {row[field]}, expected {value}"
                )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 8 selected-token metrics from saved predictions.")
    parser.add_argument("--require-pairwise", action="store_true", help="Fail unless instrumented pairwise metrics exist.")
    args = parser.parse_args()

    runs = discover_runs()
    metric_rows, prediction_rows_by_run = build_token_metric_rows(runs)
    populate_patch_set_cache(metric_rows, runs)
    spatial_summary = build_spatial_summary(metric_rows)
    pairwise_summary = build_pairwise_summary(metric_rows)
    overlap_summary = build_overlap_summary(metric_rows)
    failure_recovery = build_failure_recovery_summary(runs, prediction_rows_by_run)

    if args.require_pairwise and not any(row["pairwise_available"] == "true" for row in pairwise_summary):
        raise RuntimeError("Pairwise similarity was required, but no instrumented pairwise metrics were found.")

    write_csv(SELECTED_TOKEN_METRICS_PATH, metric_rows, TOKEN_METRIC_FIELDS)
    write_csv(SPATIAL_SUMMARY_PATH, spatial_summary, list(spatial_summary[0].keys()) if spatial_summary else [])
    write_csv(PAIRWISE_SUMMARY_PATH, pairwise_summary, list(pairwise_summary[0].keys()) if pairwise_summary else [])
    write_csv(OVERLAP_SUMMARY_PATH, overlap_summary, list(overlap_summary[0].keys()) if overlap_summary else [])
    write_csv(FAILURE_RECOVERY_PATH, failure_recovery, list(failure_recovery[0].keys()) if failure_recovery else [])

    print(f"Wrote {len(metric_rows)} detailed token metric rows to {SELECTED_TOKEN_METRICS_PATH}")
    print(f"Wrote {len(spatial_summary)} spatial summary rows")
    print(f"Wrote {len(pairwise_summary)} pairwise summary rows")
    print(f"Wrote {len(overlap_summary)} overlap summary rows")
    print(f"Wrote {len(failure_recovery)} failure recovery rows")


if __name__ == "__main__":
    main()
