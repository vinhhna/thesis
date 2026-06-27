from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from build_stage8_token_metrics import (
    STAGE8_ROOT,
    case_key,
    discover_runs,
    format_float,
    load_jsonl,
    pope_label,
    row_is_correct,
)


ABLATION_RESULTS_PATH = STAGE8_ROOT / "stage8_ablation_results.csv"
PAIRWISE_SUMMARY_PATH = STAGE8_ROOT / "stage8_pairwise_similarity_summary.csv"
SPATIAL_SUMMARY_PATH = STAGE8_ROOT / "stage8_spatial_coverage_summary.csv"

STAGE8_SUBSET_SEED = 20260610
STAGE8_PRIMARY_SUBSET_SIZE = 500
POOLS = [2, 3]
LAMBDAS = [0.5, 0.7]
OFFICIAL_BASELINE_LAMBDA = 0.8


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    run_id: str
    ablation_role: str
    row_role: str
    method: str
    selection_method: str
    retained_tokens: int = 64
    candidate_pool_factor: int = 2
    lambda_relevance: float = OFFICIAL_BASELINE_LAMBDA
    threshold_tau: float = 0.85
    required: bool = True


FIELDS = [
    "dataset",
    "run_id",
    "run_available",
    "ablation_role",
    "row_role",
    "method",
    "selection_method",
    "token_setting",
    "candidate_pool_factor",
    "lambda_relevance",
    "threshold_tau",
    "record_selection_similarity",
    "subset_seed",
    "subset_size",
    "sample_count",
    "accuracy",
    "f1",
    "correct",
    "stage8_sparse_baseline_run_id",
    "sparse_wrong_count",
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


def make_primary_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for dataset, prefix, role in [
        ("gqa", "GQA", "general_ablation_gqa"),
        ("pope", "POPE", "general_ablation_pope"),
    ]:
        specs.extend(
            [
                RunSpec(
                    dataset=dataset,
                    run_id=f"{prefix}-STAGE8-SPARSE-ORIG-64",
                    ablation_role=role,
                    row_role="required_general_comparison",
                    method="SparseVLM-Original",
                    selection_method="topk",
                ),
                RunSpec(
                    dataset=dataset,
                    run_id=f"{prefix}-STAGE8-OURS-64-P2-L08",
                    ablation_role=role,
                    row_role="required_general_baseline",
                    method="Ours",
                    selection_method="mmr",
                    candidate_pool_factor=2,
                    lambda_relevance=0.8,
                ),
                RunSpec(
                    dataset=dataset,
                    run_id=f"{prefix}-STAGE8-THRESHOLD-FIXED-64",
                    ablation_role=role,
                    row_role="required_general_comparison",
                    method="Threshold-Fixed-k",
                    selection_method="threshold_fixed",
                ),
            ]
        )
        for pool in POOLS:
            for lambda_value in LAMBDAS:
                lambda_code = str(lambda_value).replace(".", "")
                specs.append(
                    RunSpec(
                        dataset=dataset,
                        run_id=f"{prefix}-STAGE8-OURS-64-P{pool}-L{lambda_code}",
                        ablation_role=role,
                        row_role="required_general_ablation",
                        method="Ours",
                        selection_method="mmr",
                        candidate_pool_factor=pool,
                        lambda_relevance=lambda_value,
                    )
                )
    return specs


def make_optional_failure_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for pool in POOLS:
        for lambda_value in LAMBDAS:
            lambda_code = str(lambda_value).replace(".", "")
            specs.append(
                RunSpec(
                    dataset="failure_mining",
                    run_id=f"FM-STAGE8-OURS-64-P{pool}-L{lambda_code}",
                    ablation_role="targeted_failure_recovery",
                    row_role="optional_failure_ablation",
                    method="Ours",
                    selection_method="mmr",
                    candidate_pool_factor=pool,
                    lambda_relevance=lambda_value,
                    required=False,
                )
            )
    return specs


REQUIRED_SPECS = make_primary_specs()
OPTIONAL_SPECS = make_optional_failure_specs()
ALL_SPECS = REQUIRED_SPECS + OPTIONAL_SPECS


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
        tp = sum(pred == 1 and gold == 1 for pred, gold in zip(y_pred, y_true))
        fp = sum(pred == 1 and gold == 0 for pred, gold in zip(y_pred, y_true))
        fn = sum(pred == 0 and gold == 1 for pred, gold in zip(y_pred, y_true))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["f1"] = format_float(f1)
    return result


def final_layer_summary_by_run(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_csv(path):
        if str(row.get("layer_idx", "")) == "15":
            result[row["run_id"]] = row
    return result


def manifest_value(run: dict | None, key: str, default: object = "") -> object:
    if not run:
        return default
    return run.get("manifest", {}).get(key, default)


def as_bool_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value in {"true", "false"}:
        return str(value)
    if value == "":
        return ""
    return str(value).strip().lower()


def validate_available_run(spec: RunSpec, run: dict, predictions: list[dict]) -> None:
    manifest = run["manifest"]
    problems = []

    def require_equal(field: str, expected: object, actual: object) -> None:
        if str(actual) != str(expected):
            problems.append(f"{field}={actual!r}, expected {expected!r}")

    require_equal("dataset", spec.dataset, run.get("dataset"))
    require_equal("selection_method", spec.selection_method, run.get("selection_method"))
    require_equal("retained_tokens", spec.retained_tokens, run.get("token_setting"))
    require_equal("candidate_pool_factor", spec.candidate_pool_factor, manifest.get("candidate_pool_factor"))
    require_equal("record_selection_similarity", True, manifest.get("record_selection_similarity"))

    actual_lambda = manifest.get("lambda_relevance")
    if actual_lambda is None:
        problems.append("lambda_relevance missing")
    elif abs(float(actual_lambda) - float(spec.lambda_relevance)) > 1e-9:
        problems.append(f"lambda_relevance={actual_lambda!r}, expected {spec.lambda_relevance!r}")

    if spec.dataset in {"gqa", "pope"}:
        require_equal("subset_seed", STAGE8_SUBSET_SEED, manifest.get("subset_seed"))
        subset_size = manifest.get("subset_size", manifest.get("sample_count"))
        require_equal("subset_size", STAGE8_PRIMARY_SUBSET_SIZE, subset_size)
        if len(predictions) != STAGE8_PRIMARY_SUBSET_SIZE:
            problems.append(f"prediction row count={len(predictions)}, expected {STAGE8_PRIMARY_SUBSET_SIZE}")

    if problems:
        raise RuntimeError(f"{spec.run_id}: invalid Stage 8 run metadata: {problems}")


def sparse_baseline_run_id(dataset: str, runs: dict[str, dict]) -> str:
    if dataset == "gqa":
        return "GQA-STAGE8-SPARSE-ORIG-64"
    if dataset == "pope":
        return "POPE-STAGE8-SPARSE-ORIG-64"
    if "FM-STAGE8-SPARSE-ORIG-64" in runs:
        return "FM-STAGE8-SPARSE-ORIG-64"
    return "FM-SPARSE-ORIG-64"


def recovery_against_sparse(
    spec: RunSpec,
    runs: dict[str, dict],
    prediction_rows_by_run: dict[str, list[dict]],
) -> dict[str, str]:
    if spec.selection_method == "topk":
        return {"stage8_sparse_baseline_run_id": ""}

    baseline_id = sparse_baseline_run_id(spec.dataset, runs)
    current_rows = prediction_rows_by_run.get(spec.run_id)
    baseline_rows = prediction_rows_by_run.get(baseline_id)
    if not current_rows or not baseline_rows:
        return {"stage8_sparse_baseline_run_id": baseline_id}

    current_map = {case_key(row): row for row in current_rows}
    baseline_map = {case_key(row): row for row in baseline_rows}
    current_keys = set(current_map)
    baseline_keys = set(baseline_map)
    common_keys = sorted(current_keys & baseline_keys)
    if not common_keys:
        return {"stage8_sparse_baseline_run_id": baseline_id}

    if spec.dataset in {"gqa", "pope"} and current_keys != baseline_keys:
        raise RuntimeError(
            f"{spec.run_id} and {baseline_id} do not use the same Stage 8 subset "
            f"({len(current_keys)} vs {len(baseline_keys)} case IDs)."
        )

    sparse_wrong = 0
    recovered = 0
    regressions = 0
    baseline_correct_total = 0
    current_correct_total = 0
    for key in common_keys:
        baseline_correct = row_is_correct(baseline_map[key], spec.dataset)
        current_correct = row_is_correct(current_map[key], spec.dataset)
        baseline_correct_total += int(baseline_correct)
        current_correct_total += int(current_correct)
        if not baseline_correct:
            sparse_wrong += 1
            if current_correct:
                recovered += 1
        elif not current_correct:
            regressions += 1

    return {
        "stage8_sparse_baseline_run_id": baseline_id,
        "sparse_wrong_count": str(sparse_wrong),
        "recovered_baseline_failures": str(recovered),
        "recovery_rate": format_float(recovered / sparse_wrong) if sparse_wrong else "",
        "regressions_vs_baseline": str(regressions),
        "net_gain": str(current_correct_total - baseline_correct_total),
    }


def build_row(
    spec: RunSpec,
    runs: dict[str, dict],
    prediction_rows_by_run: dict[str, list[dict]],
    pairwise_by_run: dict[str, dict],
    spatial_by_run: dict[str, dict],
) -> dict:
    run = runs.get(spec.run_id)
    available = run is not None
    predictions = prediction_rows_by_run.get(spec.run_id, [])
    performance = performance_for_predictions(spec.dataset, predictions) if available else {}
    recovery = recovery_against_sparse(spec, runs, prediction_rows_by_run)
    pairwise = pairwise_by_run.get(spec.run_id, {})
    spatial = spatial_by_run.get(spec.run_id, {})

    manifest = run.get("manifest", {}) if run else {}
    expected_subset_seed = STAGE8_SUBSET_SEED if spec.dataset in {"gqa", "pope"} else ""
    expected_subset_size = STAGE8_PRIMARY_SUBSET_SIZE if spec.dataset in {"gqa", "pope"} else ""
    subset_size = manifest.get("subset_size", manifest.get("sample_count", expected_subset_size))
    if not available:
        notes = (
            "required Stage 8 GQA/POPE subset run pending"
            if spec.required
            else "optional failure-mining stress-test run pending"
        )
    elif spec.row_role == "required_general_baseline":
        notes = "official Ours-64 baseline setting on the Stage 8 500-sample subset"
    elif spec.ablation_role.startswith("general_ablation"):
        notes = "Stage 8 GQA/POPE 500-sample subset row; auxiliary, not an official Stage 5 benchmark score"
    else:
        notes = "optional failure-mining targeted recovery/stress-test row; not representative"

    return {
        "dataset": spec.dataset,
        "run_id": spec.run_id,
        "run_available": str(available).lower(),
        "ablation_role": spec.ablation_role,
        "row_role": spec.row_role,
        "method": manifest.get("method", spec.method),
        "selection_method": manifest.get("selection_method", spec.selection_method),
        "token_setting": str(manifest.get("retained_tokens", spec.retained_tokens)),
        "candidate_pool_factor": str(manifest.get("candidate_pool_factor", spec.candidate_pool_factor)),
        "lambda_relevance": str(manifest.get("lambda_relevance", spec.lambda_relevance)),
        "threshold_tau": str(manifest.get("threshold_tau", spec.threshold_tau)),
        "record_selection_similarity": as_bool_text(manifest.get("record_selection_similarity", True)),
        "subset_seed": str(manifest.get("subset_seed", expected_subset_seed)),
        "subset_size": str(subset_size),
        "sample_count": performance.get("sample_count", ""),
        "accuracy": performance.get("accuracy", ""),
        "f1": performance.get("f1", ""),
        "correct": performance.get("correct", ""),
        "stage8_sparse_baseline_run_id": recovery.get("stage8_sparse_baseline_run_id", ""),
        "sparse_wrong_count": recovery.get("sparse_wrong_count", ""),
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
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 8 GQA/POPE subset ablation summary table.")
    parser.add_argument(
        "--require-ablation",
        action="store_true",
        help="Fail unless all required GQA/POPE Stage 8 subset runs are present.",
    )
    args = parser.parse_args()

    runs = discover_runs()
    prediction_rows_by_run = {
        run_id: load_jsonl(run["prediction_path"])
        for run_id, run in runs.items()
    }
    pairwise_by_run = final_layer_summary_by_run(PAIRWISE_SUMMARY_PATH)
    spatial_by_run = final_layer_summary_by_run(SPATIAL_SUMMARY_PATH)

    missing_required = []
    for spec in REQUIRED_SPECS:
        run = runs.get(spec.run_id)
        if run is None:
            missing_required.append(spec.run_id)
            continue
        validate_available_run(spec, run, prediction_rows_by_run[spec.run_id])

    for spec in OPTIONAL_SPECS:
        run = runs.get(spec.run_id)
        if run is not None:
            validate_available_run(spec, run, prediction_rows_by_run[spec.run_id])

    rows = [
        build_row(spec, runs, prediction_rows_by_run, pairwise_by_run, spatial_by_run)
        for spec in ALL_SPECS
    ]
    write_csv(ABLATION_RESULTS_PATH, rows)

    if args.require_ablation and missing_required:
        raise RuntimeError(f"Missing required Stage 8 GQA/POPE subset runs: {missing_required}")

    print(f"Wrote {len(rows)} Stage 8 ablation rows to {ABLATION_RESULTS_PATH}")
    if missing_required:
        print(f"Pending required GQA/POPE Stage 8 runs: {len(missing_required)}")
    optional_missing = [spec.run_id for spec in OPTIONAL_SPECS if spec.run_id not in runs]
    if optional_missing:
        print(f"Pending optional failure-mining Stage 8 runs: {len(optional_missing)}")


if __name__ == "__main__":
    main()
