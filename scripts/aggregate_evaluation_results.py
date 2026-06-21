from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
RAW_RESULTS_ROOT = (
    OUTPUTS_ROOT / "raw_results"
    if (OUTPUTS_ROOT / "raw_results").is_dir()
    else OUTPUTS_ROOT
)
SUMMARY_ROOT = OUTPUTS_ROOT / "summary"

EXPECTED_RUN_IDS = {
    "GQA-DENSE-576",
    "GQA-SPARSE-ORIG-128",
    "GQA-SPARSE-ORIG-64",
    "GQA-OURS-128",
    "GQA-OURS-64",
    "GQA-THRESHOLD-FIXED-128",
    "GQA-THRESHOLD-FIXED-64",
    "GQA-THRESHOLD-ADAPT-080",
    "GQA-THRESHOLD-ADAPT-085",
    "GQA-THRESHOLD-ADAPT-090",
    "POPE-DENSE-576",
    "POPE-SPARSE-ORIG-128",
    "POPE-SPARSE-ORIG-64",
    "POPE-OURS-128",
    "POPE-OURS-64",
    "POPE-THRESHOLD-FIXED-128",
    "POPE-THRESHOLD-FIXED-64",
    "POPE-THRESHOLD-ADAPT-080",
    "POPE-THRESHOLD-ADAPT-085",
    "POPE-THRESHOLD-ADAPT-090",
    "FM-DENSE-576",
    "FM-SPARSE-ORIG-128",
    "FM-SPARSE-ORIG-64",
    "FM-OURS-128",
    "FM-OURS-64",
    "FM-THRESHOLD-FIXED-128",
    "FM-THRESHOLD-FIXED-64",
    "FM-THRESHOLD-ADAPT-080",
    "FM-THRESHOLD-ADAPT-085",
    "FM-THRESHOLD-ADAPT-090",
}

METHOD_ORDER = {
    "Dense / Vanilla": 0,
    "SparseVLM-Original": 1,
    "Ours": 2,
    "Threshold-Fixed-k": 3,
    "Threshold-Adaptive": 4,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_local_file(run_root: Path, manifest_path: str, suffix: str) -> Path:
    requested_name = Path(manifest_path).name
    matches = sorted(run_root.rglob(requested_name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        fallback = sorted(run_root.rglob(f"*{suffix}"))
        if len(fallback) == 1:
            return fallback[0]
    raise RuntimeError(
        f"Could not resolve {manifest_path!r} under {run_root}; "
        f"matches={matches}"
    )


def load_registry() -> dict[str, dict]:
    registry: dict[str, dict] = {}
    manifest_paths = sorted(RAW_RESULTS_ROOT.rglob("*manifest*.json"))
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            run_id = record.get("run_id")
            if run_id not in EXPECTED_RUN_IDS:
                continue
            if run_id in registry:
                raise RuntimeError(f"Duplicate manifest for {run_id}")
            if record.get("status") != "ok":
                raise RuntimeError(f"{run_id} is not complete: {record.get('status')}")
            prediction_path = find_local_file(
                manifest_path.parent,
                record["prediction_file"],
                ".jsonl",
            )
            predictions = read_jsonl(prediction_path)
            if len(predictions) != int(record["sample_count"]):
                raise RuntimeError(
                    f"{run_id}: manifest sample_count={record['sample_count']} "
                    f"but predictions={len(predictions)}"
                )
            registry[run_id] = {
                "manifest": record,
                "manifest_path": manifest_path,
                "run_root": manifest_path.parent,
                "prediction_path": prediction_path,
                "predictions": predictions,
            }

    missing = sorted(EXPECTED_RUN_IDS - set(registry))
    extra = sorted(set(registry) - EXPECTED_RUN_IDS)
    if missing or extra:
        raise RuntimeError(f"Run registry mismatch: missing={missing}, extra={extra}")
    return registry


def retained_stats(predictions: list[dict]) -> dict:
    counts = [
        int(row["metadata"]["retained_token_count"])
        for row in predictions
        if row.get("metadata", {}).get("retained_token_count") is not None
    ]
    if not counts:
        return {
            "average_retained_tokens": "",
            "min_retained_tokens": "",
            "max_retained_tokens": "",
        }
    return {
        "average_retained_tokens": sum(counts) / len(counts),
        "min_retained_tokens": min(counts),
        "max_retained_tokens": max(counts),
    }


def normalize_pope_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pope_label(text: str) -> str:
    words = normalize_pope_answer(text).split()
    return "no" if "no" in words or "not" in words else "yes"


def compute_pope_metrics(predictions: list[dict]) -> dict:
    y_true = [
        1 if normalize_pope_answer(row["ground_truth"]) == "yes" else 0
        for row in predictions
    ]
    y_pred = [1 if pope_label(row["text"]) == "yes" else 0 for row in predictions]
    tp = sum(p == 1 and g == 1 for p, g in zip(y_pred, y_true))
    tn = sum(p == 0 and g == 0 for p, g in zip(y_pred, y_true))
    fp = sum(p == 1 and g == 0 for p, g in zip(y_pred, y_true))
    fn = sum(p == 0 and g == 1 for p, g in zip(y_pred, y_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "sample_count": len(predictions),
        "accuracy": (tp + tn) / len(predictions) if predictions else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "yes_ratio": sum(y_pred) / len(y_pred) if y_pred else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def method_sort_key(row: dict) -> tuple:
    token_setting = row.get("token_setting")
    token_order = -int(token_setting) if str(token_setting).isdigit() else 0
    return (
        METHOD_ORDER.get(row["method"], 99),
        token_order,
        str(row.get("tau", "")),
    )


def build_gqa_rows(registry: dict[str, dict]) -> list[dict]:
    rows = []
    for run_id, item in registry.items():
        if not run_id.startswith("GQA-"):
            continue
        manifest = item["manifest"]
        if manifest["selection_method"] == "threshold_adaptive":
            continue
        predictions = item["predictions"]
        correct = sum(bool(row["is_correct"]) for row in predictions)
        rows.append({
            "dataset": "GQA",
            "run_id": run_id,
            "method": manifest["method"],
            "selection_method": manifest["selection_method"],
            "token_setting": manifest["retained_tokens"],
            "sample_count": len(predictions),
            "official_gqa_accuracy": correct / len(predictions),
            "correct": correct,
            "total": len(predictions),
            "subset_seed": manifest.get("subset_seed", ""),
            **retained_stats(predictions),
            "prediction_file": item["prediction_path"].relative_to(
                REPO_ROOT
            ).as_posix(),
        })
    return sorted(rows, key=method_sort_key)


def pope_comparison_ids(registry: dict[str, dict]) -> set[str]:
    comparison_runs = [
        run_id
        for run_id in EXPECTED_RUN_IDS
        if run_id.startswith("POPE-") and run_id != "POPE-DENSE-576"
    ]
    id_sets = [
        {str(row["question_id"]) for row in registry[run_id]["predictions"]}
        for run_id in comparison_runs
    ]
    first = id_sets[0]
    if any(ids != first for ids in id_sets[1:]):
        raise RuntimeError("POPE sparse and adaptive runs do not share one sample set")
    return first


def pope_predictions_for_comparison(
    registry: dict[str, dict],
    run_id: str,
    comparison_ids: set[str],
) -> list[dict]:
    predictions = registry[run_id]["predictions"]
    if run_id == "POPE-DENSE-576":
        predictions = [
            row for row in predictions
            if str(row["question_id"]) in comparison_ids
        ]
    ids = {str(row["question_id"]) for row in predictions}
    if ids != comparison_ids:
        raise RuntimeError(f"{run_id} does not match the POPE comparison sample set")
    return predictions


def build_pope_rows(registry: dict[str, dict]) -> list[dict]:
    comparison_ids = pope_comparison_ids(registry)
    rows = []
    for run_id, item in registry.items():
        if not run_id.startswith("POPE-"):
            continue
        manifest = item["manifest"]
        if manifest["selection_method"] == "threshold_adaptive":
            continue
        predictions = pope_predictions_for_comparison(
            registry,
            run_id,
            comparison_ids,
        )
        metrics = compute_pope_metrics(predictions)
        rows.append({
            "dataset": "POPE",
            "run_id": run_id,
            "method": manifest["method"],
            "selection_method": manifest["selection_method"],
            "token_setting": manifest["retained_tokens"],
            "sample_count": metrics["sample_count"],
            "source_sample_count": len(item["predictions"]),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "yes_ratio": metrics["yes_ratio"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            **retained_stats(predictions),
            "prediction_file": item["prediction_path"].relative_to(
                REPO_ROOT
            ).as_posix(),
        })
    return sorted(rows, key=method_sort_key)


def failure_maps(registry: dict[str, dict]) -> dict[str, dict[str, dict]]:
    maps = {}
    reference_ids = None
    for run_id, item in registry.items():
        if not run_id.startswith("FM-"):
            continue
        by_id = {row["case_id"]: row for row in item["predictions"]}
        if len(by_id) != len(item["predictions"]):
            raise RuntimeError(f"{run_id} contains duplicate case IDs")
        if reference_ids is None:
            reference_ids = set(by_id)
        elif set(by_id) != reference_ids:
            raise RuntimeError(f"{run_id} has a different failure-mining case set")
        maps[run_id] = by_id
    return maps


def failure_comparison_stats(
    maps: dict[str, dict[str, dict]],
    budget: int,
) -> dict:
    sparse = maps[f"FM-SPARSE-ORIG-{budget}"]
    ours = maps[f"FM-OURS-{budget}"]
    threshold = maps[f"FM-THRESHOLD-FIXED-{budget}"]
    sparse_wrong = {
        case_id for case_id, row in sparse.items() if not row["is_correct"]
    }
    ours_right = {
        case_id for case_id in sparse_wrong if ours[case_id]["is_correct"]
    }
    threshold_right = {
        case_id for case_id in sparse_wrong if threshold[case_id]["is_correct"]
    }
    unresolved = sparse_wrong - (ours_right | threshold_right)
    return {
        "sparse_wrong_count": len(sparse_wrong),
        "sparse_wrong_ours_right": len(ours_right),
        "sparse_wrong_threshold_right": len(threshold_right),
        "sparse_wrong_both_not_fixed": len(unresolved),
        "ours_recovery_rate": (
            len(ours_right) / len(sparse_wrong) if sparse_wrong else 0.0
        ),
        "threshold_recovery_rate": (
            len(threshold_right) / len(sparse_wrong) if sparse_wrong else 0.0
        ),
    }


def build_failure_rows(registry: dict[str, dict]) -> list[dict]:
    maps = failure_maps(registry)
    comparisons = {
        64: failure_comparison_stats(maps, 64),
        128: failure_comparison_stats(maps, 128),
    }
    rows = []
    for run_id, item in registry.items():
        if not run_id.startswith("FM-"):
            continue
        manifest = item["manifest"]
        predictions = item["predictions"]
        correct = sum(bool(row["is_correct"]) for row in predictions)
        failure_counts = Counter(row["failure_label"] for row in predictions)
        budget = (
            64
            if manifest["selection_method"] == "threshold_adaptive"
            else int(manifest["retained_tokens"])
        )
        comparison = comparisons.get(budget, {})
        run_recovered = ""
        run_recovery_rate = ""
        run_regressions = ""
        baseline_run_id = ""
        if budget in {64, 128} and manifest["selection_method"] != "dense":
            baseline_run_id = f"FM-SPARSE-ORIG-{budget}"
            baseline = maps[baseline_run_id]
            sparse_wrong = {
                case_id for case_id, row in baseline.items()
                if not row["is_correct"]
            }
            current = maps[run_id]
            run_recovered = sum(
                bool(current[case_id]["is_correct"]) for case_id in sparse_wrong
            )
            run_recovery_rate = (
                run_recovered / len(sparse_wrong) if sparse_wrong else 0.0
            )
            run_regressions = sum(
                bool(row["is_correct"]) and not current[case_id]["is_correct"]
                for case_id, row in baseline.items()
            )

        rows.append({
            "dataset": "Failure-mining",
            "run_id": run_id,
            "method": manifest["method"],
            "selection_method": manifest["selection_method"],
            "token_setting": manifest["retained_tokens"],
            "tau": (
                manifest["threshold_tau"]
                if manifest["selection_method"] == "threshold_adaptive"
                else ""
            ),
            "sample_count": len(predictions),
            "correctness": correct / len(predictions),
            "correct": correct,
            "incorrect": len(predictions) - correct,
            "failure_type_binary_mismatch": failure_counts.get(
                "binary_mismatch", 0
            ),
            "failure_type_open_answer_mismatch": failure_counts.get(
                "open_answer_mismatch", 0
            ),
            "failure_type_empty_answer": failure_counts.get("empty_answer", 0),
            "baseline_run_id": baseline_run_id,
            "recovered_baseline_failures": run_recovered,
            "recovery_rate": run_recovery_rate,
            "regressions_vs_baseline": run_regressions,
            "sparse_wrong_count": comparison.get("sparse_wrong_count", ""),
            "sparse_wrong_ours_right": comparison.get(
                "sparse_wrong_ours_right", ""
            ),
            "sparse_wrong_threshold_right": comparison.get(
                "sparse_wrong_threshold_right", ""
            ),
            "sparse_wrong_both_not_fixed": comparison.get(
                "sparse_wrong_both_not_fixed", ""
            ),
            **retained_stats(predictions),
            "prediction_file": item["prediction_path"].relative_to(
                REPO_ROOT
            ).as_posix(),
        })
    return sorted(rows, key=method_sort_key)


def build_adaptive_rows(
    registry: dict[str, dict],
    failure_rows: list[dict],
) -> list[dict]:
    comparison_ids = pope_comparison_ids(registry)
    failure_by_run = {row["run_id"]: row for row in failure_rows}
    rows = []
    for dataset_prefix, dataset_name in [
        ("GQA", "GQA"),
        ("POPE", "POPE"),
        ("FM", "Failure-mining"),
    ]:
        selected_run_ids = [
            f"{dataset_prefix}-OURS-128",
            f"{dataset_prefix}-OURS-64",
            f"{dataset_prefix}-THRESHOLD-ADAPT-080",
            f"{dataset_prefix}-THRESHOLD-ADAPT-085",
            f"{dataset_prefix}-THRESHOLD-ADAPT-090",
        ]
        for run_id in selected_run_ids:
            item = registry[run_id]
            manifest = item["manifest"]
            predictions = item["predictions"]
            f1 = ""
            correct = ""
            total = len(predictions)
            recovery_rate = ""
            if dataset_prefix == "POPE":
                predictions = pope_predictions_for_comparison(
                    registry,
                    run_id,
                    comparison_ids,
                )
                metrics = compute_pope_metrics(predictions)
                score = metrics["accuracy"]
                f1 = metrics["f1"]
            elif dataset_prefix == "FM":
                failure_row = failure_by_run[run_id]
                score = failure_row["correctness"]
                correct = failure_row["correct"]
                recovery_rate = failure_row["recovery_rate"]
            else:
                correct = sum(bool(row["is_correct"]) for row in predictions)
                score = correct / len(predictions)

            rows.append({
                "dataset": dataset_name,
                "run_id": run_id,
                "method": manifest["method"],
                "tau": (
                    manifest["threshold_tau"]
                    if manifest["selection_method"] == "threshold_adaptive"
                    else ""
                ),
                "token_setting": manifest["retained_tokens"],
                "sample_count": len(predictions),
                "accuracy_or_score": score,
                "f1": f1,
                "correct": correct,
                "total": total,
                "recovery_rate": recovery_rate,
                **retained_stats(predictions),
                "prediction_file": item["prediction_path"].relative_to(
                    REPO_ROOT
                ).as_posix(),
            })
    return rows


def build_final_rows(
    registry: dict[str, dict],
    gqa_rows: list[dict],
    pope_rows: list[dict],
    failure_rows: list[dict],
    adaptive_rows: list[dict],
) -> list[dict]:
    rows = []
    adaptive_by_run = {row["run_id"]: row for row in adaptive_rows}
    for row in gqa_rows:
        rows.append({
            "evaluation_group": "main_fixed_budget",
            "dataset": "GQA",
            "run_id": row["run_id"],
            "method": row["method"],
            "selection_method": row["selection_method"],
            "token_setting": row["token_setting"],
            "tau": "",
            "sample_count": row["sample_count"],
            "source_sample_count": row["sample_count"],
            "accuracy_or_score": row["official_gqa_accuracy"],
            "f1": "",
            "correct": row["correct"],
            "total": row["total"],
            "recovery_rate": "",
            "failure_type_binary_mismatch": "",
            "failure_type_open_answer_mismatch": "",
            "failure_type_empty_answer": "",
            "sparse_wrong_ours_right": "",
            "sparse_wrong_threshold_right": "",
            "sparse_wrong_both_not_fixed": "",
            "average_retained_tokens": row["average_retained_tokens"],
            "min_retained_tokens": row["min_retained_tokens"],
            "max_retained_tokens": row["max_retained_tokens"],
            "prediction_file": row["prediction_file"],
        })
    for row in pope_rows:
        rows.append({
            "evaluation_group": "main_fixed_budget",
            "dataset": "POPE",
            "run_id": row["run_id"],
            "method": row["method"],
            "selection_method": row["selection_method"],
            "token_setting": row["token_setting"],
            "tau": "",
            "sample_count": row["sample_count"],
            "source_sample_count": row["source_sample_count"],
            "accuracy_or_score": row["accuracy"],
            "f1": row["f1"],
            "correct": row["tp"] + row["tn"],
            "total": row["sample_count"],
            "recovery_rate": "",
            "failure_type_binary_mismatch": "",
            "failure_type_open_answer_mismatch": "",
            "failure_type_empty_answer": "",
            "sparse_wrong_ours_right": "",
            "sparse_wrong_threshold_right": "",
            "sparse_wrong_both_not_fixed": "",
            "average_retained_tokens": row["average_retained_tokens"],
            "min_retained_tokens": row["min_retained_tokens"],
            "max_retained_tokens": row["max_retained_tokens"],
            "prediction_file": row["prediction_file"],
        })
    for row in failure_rows:
        rows.append({
            "evaluation_group": (
                "adaptive_threshold"
                if row["selection_method"] == "threshold_adaptive"
                else "main_fixed_budget"
            ),
            "dataset": "Failure-mining",
            "run_id": row["run_id"],
            "method": row["method"],
            "selection_method": row["selection_method"],
            "token_setting": row["token_setting"],
            "tau": row["tau"],
            "sample_count": row["sample_count"],
            "source_sample_count": row["sample_count"],
            "accuracy_or_score": row["correctness"],
            "f1": "",
            "correct": row["correct"],
            "total": row["sample_count"],
            "recovery_rate": row["recovery_rate"],
            "failure_type_binary_mismatch": row[
                "failure_type_binary_mismatch"
            ],
            "failure_type_open_answer_mismatch": row[
                "failure_type_open_answer_mismatch"
            ],
            "failure_type_empty_answer": row["failure_type_empty_answer"],
            "sparse_wrong_ours_right": row["sparse_wrong_ours_right"],
            "sparse_wrong_threshold_right": row[
                "sparse_wrong_threshold_right"
            ],
            "sparse_wrong_both_not_fixed": row[
                "sparse_wrong_both_not_fixed"
            ],
            "average_retained_tokens": row["average_retained_tokens"],
            "min_retained_tokens": row["min_retained_tokens"],
            "max_retained_tokens": row["max_retained_tokens"],
            "prediction_file": row["prediction_file"],
        })

    fixed_run_ids = {row["run_id"] for row in rows}
    for run_id, row in adaptive_by_run.items():
        if run_id in fixed_run_ids:
            continue
        manifest = registry[run_id]["manifest"]
        rows.append({
            "evaluation_group": "adaptive_threshold",
            "dataset": row["dataset"],
            "run_id": run_id,
            "method": row["method"],
            "selection_method": manifest["selection_method"],
            "token_setting": row["token_setting"],
            "tau": row["tau"],
            "sample_count": row["sample_count"],
            "source_sample_count": len(registry[run_id]["predictions"]),
            "accuracy_or_score": row["accuracy_or_score"],
            "f1": row["f1"],
            "correct": row["correct"],
            "total": row["total"],
            "recovery_rate": row["recovery_rate"],
            "failure_type_binary_mismatch": "",
            "failure_type_open_answer_mismatch": "",
            "failure_type_empty_answer": "",
            "sparse_wrong_ours_right": "",
            "sparse_wrong_threshold_right": "",
            "sparse_wrong_both_not_fixed": "",
            "average_retained_tokens": row["average_retained_tokens"],
            "min_retained_tokens": row["min_retained_tokens"],
            "max_retained_tokens": row["max_retained_tokens"],
            "prediction_file": row["prediction_file"],
        })

    if {row["run_id"] for row in rows} != EXPECTED_RUN_IDS:
        missing = sorted(EXPECTED_RUN_IDS - {row["run_id"] for row in rows})
        extra = sorted({row["run_id"] for row in rows} - EXPECTED_RUN_IDS)
        raise RuntimeError(f"Final table mismatch: missing={missing}, extra={extra}")

    dataset_order = {"GQA": 0, "POPE": 1, "Failure-mining": 2}
    return sorted(
        rows,
        key=lambda row: (
            dataset_order[row["dataset"]],
            0 if row["evaluation_group"] == "main_fixed_budget" else 1,
            METHOD_ORDER.get(row["method"], 99),
            -int(row["token_setting"]),
            str(row["tau"]),
        ),
    )


def main() -> None:
    registry = load_registry()
    gqa_rows = build_gqa_rows(registry)
    pope_rows = build_pope_rows(registry)
    failure_rows = build_failure_rows(registry)
    adaptive_rows = build_adaptive_rows(registry, failure_rows)
    final_rows = build_final_rows(
        registry,
        gqa_rows,
        pope_rows,
        failure_rows,
        adaptive_rows,
    )

    write_csv(
        SUMMARY_ROOT / "gqa_summary.csv",
        gqa_rows,
        [
            "dataset",
            "run_id",
            "method",
            "selection_method",
            "token_setting",
            "sample_count",
            "official_gqa_accuracy",
            "correct",
            "total",
            "subset_seed",
            "average_retained_tokens",
            "min_retained_tokens",
            "max_retained_tokens",
            "prediction_file",
        ],
    )
    write_csv(
        SUMMARY_ROOT / "pope_summary.csv",
        pope_rows,
        [
            "dataset",
            "run_id",
            "method",
            "selection_method",
            "token_setting",
            "sample_count",
            "source_sample_count",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "yes_ratio",
            "tp",
            "tn",
            "fp",
            "fn",
            "average_retained_tokens",
            "min_retained_tokens",
            "max_retained_tokens",
            "prediction_file",
        ],
    )
    write_csv(
        SUMMARY_ROOT / "failure_mining_summary.csv",
        failure_rows,
        [
            "dataset",
            "run_id",
            "method",
            "selection_method",
            "token_setting",
            "tau",
            "sample_count",
            "correctness",
            "correct",
            "incorrect",
            "failure_type_binary_mismatch",
            "failure_type_open_answer_mismatch",
            "failure_type_empty_answer",
            "baseline_run_id",
            "recovered_baseline_failures",
            "recovery_rate",
            "regressions_vs_baseline",
            "sparse_wrong_count",
            "sparse_wrong_ours_right",
            "sparse_wrong_threshold_right",
            "sparse_wrong_both_not_fixed",
            "average_retained_tokens",
            "min_retained_tokens",
            "max_retained_tokens",
            "prediction_file",
        ],
    )
    write_csv(
        SUMMARY_ROOT / "adaptive_threshold_summary.csv",
        adaptive_rows,
        [
            "dataset",
            "run_id",
            "method",
            "tau",
            "token_setting",
            "sample_count",
            "accuracy_or_score",
            "f1",
            "correct",
            "total",
            "recovery_rate",
            "average_retained_tokens",
            "min_retained_tokens",
            "max_retained_tokens",
            "prediction_file",
        ],
    )
    write_csv(
        SUMMARY_ROOT / "final_evaluation_table.csv",
        final_rows,
        [
            "evaluation_group",
            "dataset",
            "run_id",
            "method",
            "selection_method",
            "token_setting",
            "tau",
            "sample_count",
            "source_sample_count",
            "accuracy_or_score",
            "f1",
            "correct",
            "total",
            "recovery_rate",
            "failure_type_binary_mismatch",
            "failure_type_open_answer_mismatch",
            "failure_type_empty_answer",
            "sparse_wrong_ours_right",
            "sparse_wrong_threshold_right",
            "sparse_wrong_both_not_fixed",
            "average_retained_tokens",
            "min_retained_tokens",
            "max_retained_tokens",
            "prediction_file",
        ],
    )

    print("Raw results:", RAW_RESULTS_ROOT)
    print("Summary output:", SUMMARY_ROOT)
    print("GQA rows:", len(gqa_rows))
    print("POPE rows:", len(pope_rows))
    print("Failure-mining rows:", len(failure_rows))
    print("Adaptive rows:", len(adaptive_rows))
    print("Final rows:", len(final_rows))


if __name__ == "__main__":
    main()
