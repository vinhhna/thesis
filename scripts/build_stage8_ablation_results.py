from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_stage8_token_metrics import (
    REPO_ROOT,
    STAGE8_ROOT,
    build_failure_recovery_summary,
    discover_runs,
    format_float,
    load_jsonl,
    prediction_map,
    row_is_correct,
)


ABLATION_RESULTS_PATH = STAGE8_ROOT / "stage8_ablation_results.csv"
PAIRWISE_SUMMARY_PATH = STAGE8_ROOT / "stage8_pairwise_similarity_summary.csv"
SPATIAL_SUMMARY_PATH = STAGE8_ROOT / "stage8_spatial_coverage_summary.csv"

DATASETS = {
    "gqa": "GQA",
    "pope": "POPE",
    "failure_mining": "FM",
}
POOLS = [2, 3]
LAMBDAS = [0.5, 0.7]
OFFICIAL_BASELINE_LAMBDA = 0.8

FIELDS = [
    "dataset",
    "run_id",
    "run_available",
    "ablation_role",
    "method",
    "selection_method",
    "token_setting",
    "candidate_pool_factor",
    "lambda_relevance",
    "threshold_tau",
    "sample_count",
    "accuracy",
    "f1",
    "correct",
    "recovered_baseline_failures",
    "recovery_rate",
    "regressions_vs_baseline",
    "net_gain",
    "pairwise_available_count",
    "mean_pairwise_similarity",
    "p90_pairwise_similarity",
    "similarity_above_0.85_ratio",
    "mean_bbox_area_ratio",
    "mean_grid_occupancy_ratio",
    "mean_distance_to_centroid",
    "notes",
]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def planned_run_id(dataset: str, pool: int, lambda_value: float) -> str:
    prefix = DATASETS[dataset]
    lambda_code = str(lambda_value).replace(".", "")
    return f"{prefix}-STAGE8-OURS-64-P{pool}-L{lambda_code}"


def official_run_id(dataset: str) -> str:
    prefix = DATASETS[dataset]
    return f"{prefix}-OURS-64"


def pope_label(text: object) -> str:
    from build_stage8_token_metrics import pope_label as _pope_label

    return _pope_label(text)


def performance_for_predictions(dataset: str, predictions: list[dict]) -> dict[str, str]:
    if not predictions:
        return {"sample_count": "", "accuracy": "", "f1": "", "correct": ""}
    correct_values = [row_is_correct(row, dataset) for row in predictions]
    accuracy = sum(correct_values) / len(correct_values)
    result = {
        "sample_count": str(len(predictions)),
        "accuracy": format_float(accuracy),
        "correct": str(sum(correct_values)),
        "f1": "",
    }
    if dataset == "pope":
        y_true = [1 if str(row.get("ground_truth", "")).strip().lower() == "yes" else 0 for row in predictions]
        y_pred = [1 if pope_label(row.get("text", "")) == "yes" else 0 for row in predictions]
        tp = sum(p == 1 and g == 1 for p, g in zip(y_pred, y_true))
        fp = sum(p == 1 and g == 0 for p, g in zip(y_pred, y_true))
        fn = sum(p == 0 and g == 1 for p, g in zip(y_pred, y_true))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["f1"] = format_float(f1)
    return result


def recovery_by_run_id(runs: dict[str, dict], prediction_rows_by_run: dict[str, list[dict]]) -> dict[str, dict]:
    recovery = build_failure_recovery_summary(runs, prediction_rows_by_run)
    return {row["run_id"]: row for row in recovery}


def final_layer_summary_by_run(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    result = {}
    for row in rows:
        if str(row.get("layer_idx", "")) == "15":
            result[row["run_id"]] = row
    return result


def build_row(
    dataset: str,
    run_id: str,
    role: str,
    pool: int,
    lambda_value: float,
    runs: dict[str, dict],
    prediction_rows_by_run: dict[str, list[dict]],
    recovery_rows: dict[str, dict],
    pairwise_by_run: dict[str, dict],
    spatial_by_run: dict[str, dict],
) -> dict:
    run = runs.get(run_id)
    available = run is not None
    predictions = prediction_rows_by_run.get(run_id, [])
    performance = performance_for_predictions(dataset, predictions) if available else {}
    recovery = recovery_rows.get(run_id, {})
    pairwise = pairwise_by_run.get(run_id, {})
    spatial = spatial_by_run.get(run_id, {})
    return {
        "dataset": dataset,
        "run_id": run_id,
        "run_available": str(available).lower(),
        "ablation_role": role,
        "method": run.get("method", "Ours") if run else "Ours",
        "selection_method": run.get("selection_method", "mmr") if run else "mmr",
        "token_setting": "64",
        "candidate_pool_factor": str(pool),
        "lambda_relevance": str(lambda_value),
        "threshold_tau": str(run.get("threshold_tau", "")) if run else "0.85",
        "sample_count": performance.get("sample_count", ""),
        "accuracy": performance.get("accuracy", ""),
        "f1": performance.get("f1", ""),
        "correct": performance.get("correct", ""),
        "recovered_baseline_failures": recovery.get("recovered_baseline_failures", ""),
        "recovery_rate": recovery.get("recovery_rate", ""),
        "regressions_vs_baseline": recovery.get("regressions_vs_baseline", ""),
        "net_gain": recovery.get("net_gain", ""),
        "pairwise_available_count": pairwise.get("pairwise_available_count", ""),
        "mean_pairwise_similarity": pairwise.get("mean_mean_pairwise_similarity", ""),
        "p90_pairwise_similarity": pairwise.get("mean_p90_pairwise_similarity", ""),
        "similarity_above_0.85_ratio": pairwise.get("mean_similarity_above_0.85_ratio", ""),
        "mean_bbox_area_ratio": spatial.get("mean_bbox_area_ratio", ""),
        "mean_grid_occupancy_ratio": spatial.get("mean_grid_occupancy_ratio", ""),
        "mean_distance_to_centroid": spatial.get("mean_distance_to_centroid", ""),
        "notes": (
            "official Ours-64 baseline"
            if role == "official_baseline"
            else "planned Stage 8 ablation run; metrics populate after Kaggle output is imported"
            if not available
            else "Stage 8 ablation run"
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 8 Ours ablation summary table.")
    parser.add_argument("--require-ablation", action="store_true", help="Fail unless all planned ablation runs are present.")
    args = parser.parse_args()

    runs = discover_runs()
    prediction_rows_by_run = {
        run_id: load_jsonl(run["prediction_path"])
        for run_id, run in runs.items()
    }
    recovery_rows = recovery_by_run_id(runs, prediction_rows_by_run)
    pairwise_by_run = final_layer_summary_by_run(PAIRWISE_SUMMARY_PATH)
    spatial_by_run = final_layer_summary_by_run(SPATIAL_SUMMARY_PATH)

    rows = []
    missing = []
    for dataset in DATASETS:
        baseline_id = official_run_id(dataset)
        rows.append(
            build_row(
                dataset,
                baseline_id,
                "official_baseline",
                2,
                OFFICIAL_BASELINE_LAMBDA,
                runs,
                prediction_rows_by_run,
                recovery_rows,
                pairwise_by_run,
                spatial_by_run,
            )
        )
        for pool in POOLS:
            for lambda_value in LAMBDAS:
                run_id = planned_run_id(dataset, pool, lambda_value)
                if run_id not in runs:
                    missing.append(run_id)
                rows.append(
                    build_row(
                        dataset,
                        run_id,
                        "ablation_variant",
                        pool,
                        lambda_value,
                        runs,
                        prediction_rows_by_run,
                        recovery_rows,
                        pairwise_by_run,
                        spatial_by_run,
                    )
                )

    write_csv(ABLATION_RESULTS_PATH, rows)
    if args.require_ablation and missing:
        raise RuntimeError(f"Missing required Stage 8 ablation runs: {missing}")
    print(f"Wrote {len(rows)} Stage 8 ablation rows to {ABLATION_RESULTS_PATH}")
    if missing:
        print(f"Pending ablation runs: {len(missing)}")


if __name__ == "__main__":
    main()
