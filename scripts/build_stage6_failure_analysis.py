from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS_ROOT = REPO_ROOT / "outputs" / "raw_results"
STAGE6_ROOT = REPO_ROOT / "outputs" / "stage6"
REVIEW_PATH = STAGE6_ROOT / "failure_pattern_review.csv"
CASES_PATH = STAGE6_ROOT / "failure_pattern_cases.csv"
SUMMARY_PATH = STAGE6_ROOT / "failure_pattern_summary.csv"
VISUALIZATION_PATH = STAGE6_ROOT / "visualization_cases.txt"
ANALYSIS_PATH = STAGE6_ROOT / "stage6_failure_analysis.md"
ANNOTATION_SEED_PATH = REPO_ROOT / "scripts" / "stage6_failure_annotations.csv"

RUN_SPECS = {
    "dense": {
        "run_id": "FM-DENSE-576",
        "selection_method": "dense",
        "retained_tokens": 576,
    },
    "sparse": {
        "run_id": "FM-SPARSE-ORIG-64",
        "selection_method": "topk",
        "retained_tokens": 64,
    },
    "ours": {
        "run_id": "FM-OURS-64",
        "selection_method": "mmr",
        "retained_tokens": 64,
    },
    "threshold": {
        "run_id": "FM-THRESHOLD-FIXED-64",
        "selection_method": "threshold_fixed",
        "retained_tokens": 64,
    },
}

GROUP_ORDER = {
    "ours_only_recovery": 0,
    "ours_and_threshold_recovery": 1,
    "threshold_only_recovery": 2,
    "unresolved_by_both": 3,
}

REVIEW_FAILURE_TYPES = {
    "",
    "missed_relevant_visual_evidence",
    "redundant_or_dominant_region_selection",
    "unclear_before_visualization",
}
CURATED_FAILURE_TYPES = {
    "missed_relevant_visual_evidence",
    "redundant_or_dominant_region_selection",
}
CLASSIFICATION_STATUSES = {"", "provisional", "confirmed_after_visualization"}
VISUALIZATION_PRIORITIES = {"", "high", "medium", "low"}
CASE_ROLES = {"", "recovery", "limitation"}
BOOLEAN_TRUE = {"1", "true", "yes", "y"}

MANUAL_FIELDS = [
    "include_in_curated",
    "curation_order",
    "case_role",
    "main_failure_type",
    "classification_status",
    "why_sparse_failed",
    "why_ours_helped",
    "evidence_needed",
    "visualization_priority",
    "verification_evidence",
    "notes",
]

REVIEW_FIELDS = [
    "case_id",
    "dataset",
    "question_type",
    "image_path",
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
    "sparse_selected_original_token_indices",
    "ours_selected_original_token_indices",
    "threshold_selected_original_token_indices",
    *MANUAL_FIELDS,
]

CURATED_FIELDS = [
    "case_id",
    "dataset",
    "question_type",
    "image_path",
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
    "case_role",
    "main_failure_type",
    "classification_status",
    "why_sparse_failed",
    "why_ours_helped",
    "evidence_needed",
    "visualization_priority",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Stage 6 failure-case analysis deliverables."
    )
    parser.add_argument(
        "--strict-locked-data",
        action="store_true",
        help=(
            "Assert counts and recovery relationships for the locked 100-case "
            "failure-mining dataset."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                raise RuntimeError(f"{path}:{line_number}: missing case_id")
            if case_id in seen:
                raise RuntimeError(f"{path}: duplicate case_id {case_id}")
            seen.add(case_id)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"{path}: no prediction rows")
    return rows


def read_csv_by_case_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "case_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"{path}: missing case_id column")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            if case_id in rows:
                raise RuntimeError(f"{path}: duplicate case_id {case_id}")
            rows[case_id] = {key: str(value or "") for key, value in row.items()}
    return rows


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def resolve_prediction_path(manifest_path: Path, manifest: dict) -> Path:
    prediction_name = Path(str(manifest.get("prediction_file", ""))).name
    if not prediction_name:
        raise RuntimeError(
            f"{manifest_path}: run {manifest.get('run_id')} has no prediction_file"
        )
    matches = sorted(manifest_path.parent.rglob(prediction_name))
    if len(matches) != 1:
        raise RuntimeError(
            f"{manifest_path}: could not uniquely resolve {prediction_name}; "
            f"matches={matches}"
        )
    return matches[0]


def load_required_runs() -> dict[str, dict]:
    required_ids = {spec["run_id"] for spec in RUN_SPECS.values()}
    manifests: dict[str, tuple[Path, dict]] = {}
    for manifest_path in sorted(RAW_RESULTS_ROOT.rglob("*manifest*.json")):
        for record in manifest_records(manifest_path):
            run_id = record.get("run_id")
            if run_id not in required_ids:
                continue
            if run_id in manifests:
                raise RuntimeError(f"Duplicate manifest for required run {run_id}")
            if record.get("status") != "ok":
                raise RuntimeError(f"Required run {run_id} is not complete")
            manifests[run_id] = (manifest_path, record)

    missing = sorted(required_ids - set(manifests))
    if missing:
        raise RuntimeError(f"Missing required run manifests: {missing}")

    runs: dict[str, dict] = {}
    for role, spec in RUN_SPECS.items():
        manifest_path, manifest = manifests[spec["run_id"]]
        if manifest.get("selection_method") != spec["selection_method"]:
            raise RuntimeError(
                f"{spec['run_id']}: expected selection_method="
                f"{spec['selection_method']}, got {manifest.get('selection_method')}"
            )
        if int(manifest.get("retained_tokens", -1)) != spec["retained_tokens"]:
            raise RuntimeError(
                f"{spec['run_id']}: expected retained_tokens="
                f"{spec['retained_tokens']}, got {manifest.get('retained_tokens')}"
            )
        prediction_path = resolve_prediction_path(manifest_path, manifest)
        predictions = read_jsonl(prediction_path)
        expected_count = int(manifest.get("sample_count", -1))
        if len(predictions) != expected_count:
            raise RuntimeError(
                f"{spec['run_id']}: manifest sample_count={expected_count}, "
                f"predictions={len(predictions)}"
            )
        runs[role] = {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "prediction_path": prediction_path,
            "predictions": {row["case_id"]: row for row in predictions},
        }
    return runs


def validate_case_coverage(runs: dict[str, dict]) -> list[str]:
    case_sets = {
        role: set(item["predictions"])
        for role, item in runs.items()
    }
    reference = case_sets["sparse"]
    for role, case_ids in case_sets.items():
        if case_ids != reference:
            raise RuntimeError(
                f"Case coverage mismatch for {role}: "
                f"missing={sorted(reference - case_ids)}, "
                f"extra={sorted(case_ids - reference)}"
            )

    for case_id in sorted(reference):
        sparse_row = runs["sparse"]["predictions"][case_id]
        expected_image = str(sparse_row.get("image_path", "")).strip()
        if not expected_image:
            raise RuntimeError(f"{case_id}: missing image_path")
        image_path = Path(expected_image)
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        if not image_path.is_file():
            raise RuntimeError(f"{case_id}: image does not exist: {image_path}")

        for role, item in runs.items():
            row = item["predictions"][case_id]
            for field in ("dataset", "image_path", "ground_truth"):
                if str(row.get(field, "")) != str(sparse_row.get(field, "")):
                    raise RuntimeError(
                        f"{case_id}: {field} mismatch between sparse and {role}"
                    )
    return sorted(reference)


def bool_text(value: object) -> str:
    return "True" if bool(value) else "False"


def is_truthy(value: object) -> bool:
    return str(value or "").strip().lower() in BOOLEAN_TRUE


def selected_original_indices(row: dict) -> str:
    indices = row.get("metadata", {}).get("selected_original_token_indices", [])
    if not isinstance(indices, list):
        raise RuntimeError(
            f"{row.get('case_id')}: selected_original_token_indices is not a list"
        )
    return json.dumps(indices, separators=(",", ":"))


def derive_group(ours_correct: bool, threshold_correct: bool) -> str:
    if ours_correct and threshold_correct:
        return "ours_and_threshold_recovery"
    if ours_correct:
        return "ours_only_recovery"
    if threshold_correct:
        return "threshold_only_recovery"
    return "unresolved_by_both"


def load_manual_annotations() -> dict[str, dict[str, str]]:
    seed = read_csv_by_case_id(ANNOTATION_SEED_PATH)
    existing = read_csv_by_case_id(REVIEW_PATH)
    if not existing:
        return seed

    merged = dict(seed)
    for case_id, row in existing.items():
        merged[case_id] = {
            field: str(row.get(field, ""))
            for field in MANUAL_FIELDS
        }
    return merged


def build_review_rows(
    runs: dict[str, dict],
    case_ids: list[str],
    manual: dict[str, dict[str, str]],
) -> list[dict]:
    rows: list[dict] = []
    for case_id in case_ids:
        dense = runs["dense"]["predictions"][case_id]
        sparse = runs["sparse"]["predictions"][case_id]
        ours = runs["ours"]["predictions"][case_id]
        threshold = runs["threshold"]["predictions"][case_id]
        if bool(sparse.get("is_correct")):
            continue

        ours_correct = bool(ours.get("is_correct"))
        threshold_correct = bool(threshold.get("is_correct"))
        row = {
            "case_id": case_id,
            "dataset": sparse.get("dataset", ""),
            "question_type": sparse.get("question_type", ""),
            "image_path": sparse.get("image_path", ""),
            "prompt": sparse.get("raw_question") or sparse.get("prompt", ""),
            "ground_truth": sparse.get("ground_truth", ""),
            "dense_prediction": dense.get("text", ""),
            "sparse_prediction": sparse.get("text", ""),
            "ours_prediction": ours.get("text", ""),
            "threshold_prediction": threshold.get("text", ""),
            "dense_is_correct": bool_text(dense.get("is_correct")),
            "is_sparse_wrong": "True",
            "is_ours_correct": bool_text(ours_correct),
            "is_threshold_correct": bool_text(threshold_correct),
            "case_group": derive_group(ours_correct, threshold_correct),
            "sparse_selected_original_token_indices": selected_original_indices(sparse),
            "ours_selected_original_token_indices": selected_original_indices(ours),
            "threshold_selected_original_token_indices": selected_original_indices(
                threshold
            ),
        }
        annotations = manual.get(case_id, {})
        for field in MANUAL_FIELDS:
            row[field] = str(annotations.get(field, ""))
        rows.append(row)

    def sort_key(row: dict) -> tuple:
        order_text = str(row.get("curation_order", "")).strip()
        order = int(order_text) if order_text.isdigit() else 9999
        return (GROUP_ORDER[row["case_group"]], order, row["case_id"])

    return sorted(rows, key=sort_key)


def validate_review_values(rows: list[dict]) -> None:
    for row in rows:
        case_id = row["case_id"]
        failure_type = row.get("main_failure_type", "").strip()
        status = row.get("classification_status", "").strip()
        priority = row.get("visualization_priority", "").strip()
        role = row.get("case_role", "").strip()
        if failure_type not in REVIEW_FAILURE_TYPES:
            raise RuntimeError(
                f"{case_id}: unsupported main_failure_type={failure_type!r}"
            )
        if status not in CLASSIFICATION_STATUSES:
            raise RuntimeError(
                f"{case_id}: unsupported classification_status={status!r}"
            )
        if priority not in VISUALIZATION_PRIORITIES:
            raise RuntimeError(
                f"{case_id}: unsupported visualization_priority={priority!r}"
            )
        if role not in CASE_ROLES:
            raise RuntimeError(f"{case_id}: unsupported case_role={role!r}")


def validate_and_build_curated_rows(review_rows: list[dict]) -> list[dict]:
    curated: list[dict] = []
    for row in review_rows:
        if not is_truthy(row.get("include_in_curated")):
            continue
        case_id = row["case_id"]
        required = [
            "case_role",
            "main_failure_type",
            "classification_status",
            "why_sparse_failed",
            "why_ours_helped",
            "evidence_needed",
            "visualization_priority",
        ]
        missing = [field for field in required if not str(row.get(field, "")).strip()]
        if missing:
            raise RuntimeError(
                f"{case_id}: curated case is missing required fields {missing}"
            )
        if row["main_failure_type"] not in CURATED_FAILURE_TYPES:
            raise RuntimeError(
                f"{case_id}: curated case has unsupported failure type "
                f"{row['main_failure_type']!r}"
            )
        if row["classification_status"] not in {
            "provisional",
            "confirmed_after_visualization",
        }:
            raise RuntimeError(
                f"{case_id}: curated case must have a valid classification status"
            )
        if row["case_role"] == "recovery":
            if row["case_group"] == "unresolved_by_both":
                raise RuntimeError(
                    f"{case_id}: recovery case is unresolved by both methods"
                )
        elif row["case_role"] == "limitation":
            if row["case_group"] != "unresolved_by_both":
                raise RuntimeError(
                    f"{case_id}: limitation case is not unresolved by both methods"
                )
        else:
            raise RuntimeError(f"{case_id}: invalid case_role={row['case_role']!r}")

        if (
            row["main_failure_type"]
            == "redundant_or_dominant_region_selection"
            and row["classification_status"] != "provisional"
        ):
            evidence = row.get("verification_evidence", "").lower()
            if not evidence or not any(
                term in evidence for term in ("visualization", "similarity")
            ):
                raise RuntimeError(
                    f"{case_id}: redundancy classification cannot be confirmed "
                    "without recorded visualization or similarity evidence"
                )

        curated.append({field: row.get(field, "") for field in CURATED_FIELDS})

    def sort_key(row: dict) -> tuple:
        review = next(item for item in review_rows if item["case_id"] == row["case_id"])
        order_text = str(review.get("curation_order", "")).strip()
        order = int(order_text) if order_text.isdigit() else 9999
        return (order, row["case_id"])

    return sorted(curated, key=sort_key)


def recovery_sets(review_rows: list[dict]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {
        group: set() for group in GROUP_ORDER
    }
    for row in review_rows:
        groups[row["case_group"]].add(row["case_id"])
    return groups


def validate_strict_locked_data(
    runs: dict[str, dict],
    review_rows: list[dict],
) -> None:
    if len(runs["sparse"]["predictions"]) != 100:
        raise RuntimeError("Strict mode: expected exactly 100 failure-mining cases")
    groups = recovery_sets(review_rows)
    expected_counts = {
        "ours_and_threshold_recovery": 7,
        "ours_only_recovery": 3,
        "threshold_only_recovery": 0,
        "unresolved_by_both": 20,
    }
    actual_counts = {group: len(case_ids) for group, case_ids in groups.items()}
    if len(review_rows) != 30:
        raise RuntimeError(
            f"Strict mode: expected 30 Original-64 failures, got {len(review_rows)}"
        )
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Strict mode: recovery-group counts changed: {actual_counts}"
        )
    ours_recoveries = (
        groups["ours_and_threshold_recovery"] | groups["ours_only_recovery"]
    )
    threshold_recoveries = (
        groups["ours_and_threshold_recovery"] | groups["threshold_only_recovery"]
    )
    if not threshold_recoveries <= ours_recoveries:
        raise RuntimeError(
            "Strict mode: Threshold recoveries are no longer a subset of Ours"
        )


def build_summary_rows(
    review_rows: list[dict],
    curated_rows: list[dict],
) -> list[dict]:
    group_counts = Counter(row["case_group"] for row in review_rows)
    type_counts = Counter(row["main_failure_type"] for row in curated_rows)
    role_counts = Counter(row["case_role"] for row in curated_rows)
    high_priority = sum(
        row["visualization_priority"] == "high" for row in curated_rows
    )
    provisional_redundancy = sum(
        row["main_failure_type"] == "redundant_or_dominant_region_selection"
        and row["classification_status"] == "provisional"
        for row in curated_rows
    )

    rows = [
        {
            "scope": "full_set",
            "category": "original_64_failures",
            "count": len(review_rows),
            "notes": "All cases where SparseVLM-Original-64 is incorrect.",
        }
    ]
    group_notes = {
        "ours_and_threshold_recovery": "Both Ours-64 and Threshold-Fixed-64 recover the case.",
        "ours_only_recovery": "Only Ours-64 recovers the case.",
        "threshold_only_recovery": "Only Threshold-Fixed-64 recovers the case.",
        "unresolved_by_both": "Neither Ours-64 nor Threshold-Fixed-64 recovers the case.",
    }
    for group in GROUP_ORDER:
        rows.append(
            {
                "scope": "full_set",
                "category": group,
                "count": group_counts[group],
                "notes": group_notes[group],
            }
        )

    rows.extend(
        [
            {
                "scope": "curated",
                "category": "selected_cases",
                "count": len(curated_rows),
                "notes": "Manually interpreted cases in failure_pattern_cases.csv.",
            },
            {
                "scope": "curated",
                "category": "missed_relevant_visual_evidence",
                "count": type_counts["missed_relevant_visual_evidence"],
                "notes": "Selected cases provisionally attributed to missing required evidence.",
            },
            {
                "scope": "curated",
                "category": "redundant_or_dominant_region_selection",
                "count": type_counts["redundant_or_dominant_region_selection"],
                "notes": "This label remains provisional until independently verified.",
            },
            {
                "scope": "curated",
                "category": "provisional_redundancy_cases_awaiting_verification",
                "count": provisional_redundancy,
                "notes": "Requires Stage 7 visualization or similarity analysis.",
            },
            {
                "scope": "curated",
                "category": "recovery_cases",
                "count": role_counts["recovery"],
                "notes": "Curated cases recovered by Ours or Threshold-Fixed.",
            },
            {
                "scope": "curated",
                "category": "limitation_cases",
                "count": role_counts["limitation"],
                "notes": "Curated cases unresolved by both sparse alternatives.",
            },
            {
                "scope": "visualization",
                "category": "high_priority_cases",
                "count": high_priority,
                "notes": "Cases written to visualization_cases.txt.",
            },
        ]
    )
    return rows


def markdown_escape(text: object) -> str:
    return str(text or "").replace("|", r"\|").replace("\n", " ")


def write_visualization_list(curated_rows: list[dict]) -> None:
    selected = [
        row for row in curated_rows if row["visualization_priority"] == "high"
    ]
    lines = [
        "# Stage 7 visualization candidates",
        "# case_id\tfailure_type\tevidence_needed",
    ]
    for row in selected:
        lines.append(
            "\t".join(
                [
                    row["case_id"],
                    row["main_failure_type"],
                    str(row["evidence_needed"]).replace("\t", " "),
                ]
            )
        )
    VISUALIZATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_markdown(
    review_rows: list[dict],
    curated_rows: list[dict],
) -> None:
    groups = Counter(row["case_group"] for row in review_rows)
    type_counts = Counter(row["main_failure_type"] for row in curated_rows)
    high_priority = [
        row["case_id"]
        for row in curated_rows
        if row["visualization_priority"] == "high"
    ]
    ours_recoveries = (
        groups["ours_only_recovery"] + groups["ours_and_threshold_recovery"]
    )
    threshold_recoveries = (
        groups["threshold_only_recovery"] + groups["ours_and_threshold_recovery"]
    )

    lines = [
        "# Stage 6 — Failure-Case Analysis",
        "",
        "## Objective and protocol",
        "",
        (
            "This stage examines cases where SparseVLM-Original fails under the "
            "aggressive 64-token preset. The analysis is intentionally targeted: "
            "it asks whether Ours recovers failures that are plausibly connected "
            "to discarded visual evidence or redundant token allocation. It does "
            "not attempt to construct a complete taxonomy of VLM errors."
        ),
        "",
        (
            "Dense-576, SparseVLM-Original-64, Ours-64, and "
            "Threshold-Fixed-64 predictions were joined by `case_id`. The source "
            "images and final selected original-patch indices were inspected for "
            "candidate cases. Answer changes and patch indices are treated as "
            "diagnostic evidence, not as causal proof."
        ),
        "",
        "## Full-set recovery results",
        "",
        "| Group | Cases |",
        "| --- | ---: |",
        f"| Original-64 failures | {len(review_rows)} |",
        f"| Recovered by Ours-64 | {ours_recoveries} |",
        f"| Recovered by Threshold-Fixed-64 | {threshold_recoveries} |",
        (
            "| Recovered by both Ours and Threshold-Fixed | "
            f"{groups['ours_and_threshold_recovery']} |"
        ),
        f"| Recovered only by Ours | {groups['ours_only_recovery']} |",
        f"| Recovered only by Threshold-Fixed | {groups['threshold_only_recovery']} |",
        f"| Unresolved by both | {groups['unresolved_by_both']} |",
        "",
        (
            "These groups are computed directly from the current prediction files. "
            "The general builder does not assume that one method's recoveries must "
            "be a subset of another method's recoveries."
        ),
        "",
        "## Curated failure patterns",
        "",
        (
            f"The curated table contains {len(curated_rows)} cases. "
            f"{type_counts['missed_relevant_visual_evidence']} are provisionally "
            "classified as `missed_relevant_visual_evidence`, and "
            f"{type_counts['redundant_or_dominant_region_selection']} as "
            "`redundant_or_dominant_region_selection`."
        ),
        "",
    ]

    if type_counts["redundant_or_dominant_region_selection"] == 0:
        lines.extend(
            [
                (
                    "No case was assigned the redundancy/dominant-region label in "
                    "the main table. Manual image inspection alone was not strong "
                    "enough to support that mechanism. Stage 7 token overlays or "
                    "feature-similarity analysis are required before using the label."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "| Case | Group | Ground truth | Original | Ours | Evidence needed |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in curated_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(row["case_id"]),
                    markdown_escape(row["case_group"]),
                    markdown_escape(row["ground_truth"]),
                    markdown_escape(row["sparse_prediction"]),
                    markdown_escape(row["ours_prediction"]),
                    markdown_escape(row["evidence_needed"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Across the selected recoveries, the recurring pattern is that the "
                "answer depends on evidence that is small, overlapping, peripheral, "
                "relational, or distributed across the scene. Examples include a "
                "small camera, a dark tie, a partially visible chair, two overlapping "
                "elephants, and the spatial relation between a dog and nearby tents. "
                "SparseVLM-Original gives an incorrect answer in these cases, while "
                "Ours produces the correct answer with a different retained patch "
                "set. Stage 7 visualization is needed to verify whether the changed "
                "token selection covers the hypothesized evidence region."
            ),
            "",
            (
                "This pattern is consistent with improved preservation of relevant "
                "visual evidence under aggressive pruning. It does not establish "
                "that any particular retained patch caused the answer, and it does "
                "not yet establish reduced feature redundancy."
            ),
            "",
            (
                "Since no defensible redundancy/dominant-region case was found at "
                "this stage, the Stage 6 evidence mainly supports the missed-evidence "
                "aspect of the thesis claim. The redundancy aspect will be evaluated "
                "through visualization and selected-token metrics in later stages."
            ),
            "",
            "## Stage 7 handoff",
            "",
            (
                "The high-priority visualization cases are: "
                + (", ".join(f"`{case_id}`" for case_id in high_priority) or "none")
                + "."
            ),
            "",
            (
                "For each case, Stage 7 should overlay the final original-patch "
                "selections for Original, Ours, and Threshold-Fixed on the same "
                "CLIP 24×24 image grid. The visualization should test whether Ours "
                "actually covers the hypothesized evidence region. Any future "
                "redundancy claim should additionally use pairwise feature "
                "similarity or another explicit redundancy measurement."
            ),
            "",
            "## Limitations",
            "",
            (
                "- The failure-mining set is targeted and small, so curated-case "
                "counts are not prevalence estimates."
            ),
            (
                "- Exact-match recovery can reflect generation or language-model "
                "effects in addition to visual-token selection."
            ),
            (
                "- Dense is not correct on every recovery, so it is a reference "
                "rather than an infallible upper bound."
            ),
            (
                "- Ours regressions are intentionally deferred and must be discussed "
                "later for a balanced account."
            ),
            (
                "- Redundancy/dominant-region explanations remain provisional until "
                "visualization or similarity analysis supports them."
            ),
            "",
        ]
    )
    ANALYSIS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = load_required_runs()
    case_ids = validate_case_coverage(runs)
    manual = load_manual_annotations()
    review_rows = build_review_rows(runs, case_ids, manual)
    validate_review_values(review_rows)
    curated_rows = validate_and_build_curated_rows(review_rows)
    if args.strict_locked_data:
        validate_strict_locked_data(runs, review_rows)

    write_csv(REVIEW_PATH, review_rows, REVIEW_FIELDS)
    write_csv(CASES_PATH, curated_rows, CURATED_FIELDS)
    write_csv(
        SUMMARY_PATH,
        build_summary_rows(review_rows, curated_rows),
        ["scope", "category", "count", "notes"],
    )
    write_visualization_list(curated_rows)
    write_analysis_markdown(review_rows, curated_rows)

    groups = Counter(row["case_group"] for row in review_rows)
    print(f"Wrote {len(review_rows)} review cases to {REVIEW_PATH}")
    print(f"Wrote {len(curated_rows)} curated cases to {CASES_PATH}")
    print(f"Recovery groups: {dict(sorted(groups.items()))}")
    print(f"Wrote summary and Stage 7 handoff under {STAGE6_ROOT}")


if __name__ == "__main__":
    main()
