from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS_ROOT = REPO_ROOT / "outputs" / "raw_results"
STAGE6_ROOT = REPO_ROOT / "outputs" / "stage6"
STAGE7_ROOT = REPO_ROOT / "outputs" / "stage7"
FIGURE_ROOT = STAGE7_ROOT / "figures"

STAGE6_REVIEW_PATH = STAGE6_ROOT / "failure_pattern_review.csv"
STAGE6_CASES_PATH = STAGE6_ROOT / "failure_pattern_cases.csv"
STAGE6_VIS_CASES_PATH = STAGE6_ROOT / "visualization_cases.txt"

METRICS_PATH = STAGE7_ROOT / "stage7_token_metrics.csv"
REVIEW_PATH = STAGE7_ROOT / "stage7_visualization_review.csv"
ANALYSIS_PATH = STAGE7_ROOT / "stage7_visualization_analysis.md"

GRID_SIZE = 24
MAX_PATCH_INDEX = GRID_SIZE * GRID_SIZE - 1
DEFAULT_CLIP_INPUT_SIZE = 336
DEFAULT_FINAL_PRUNING_LAYER = 15

# LLaVA-1.5 uses OpenAI CLIP-L/336. These constants mirror the image processor
# used by SparseVLMs/llava/mm_utils.py::process_images without requiring a model
# load or a GPU rerun.
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_MEAN_RGB = tuple(int(x * 255) for x in CLIP_IMAGE_MEAN)

try:
    BICUBIC = Image.Resampling.BICUBIC
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - old Pillow compatibility.
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS


RUN_SPECS = {
    "sparse": {
        "run_id": "FM-SPARSE-ORIG-64",
        "label": "SparseVLM-Original-64",
        "color": (220, 40, 40),
    },
    "ours": {
        "run_id": "FM-OURS-64",
        "label": "Ours-64",
        "color": (30, 145, 70),
    },
    "threshold": {
        "run_id": "FM-THRESHOLD-FIXED-64",
        "label": "Threshold-Fixed-64",
        "color": (40, 95, 220),
    },
}

MANUAL_FIELDS = [
    "stage7_verified",
    "visual_evidence_region",
    "sparse_covers_evidence",
    "ours_covers_evidence",
    "threshold_covers_evidence",
    "visual_interpretation",
    "redundancy_signal",
    "redundancy_notes",
    "recommended_failure_type",
    "recommended_classification_status",
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
    "case_group",
    "stage6_failure_type",
    "stage6_classification_status",
    "evidence_needed",
    "comparison_figure",
    "sparse_overlay_figure",
    "ours_overlay_figure",
    "threshold_overlay_figure",
    *MANUAL_FIELDS,
]

METRIC_FIELDS = [
    "case_id",
    "selected_index_semantics",
    "final_pruning_layer",
    "metrics_note",
    "sparse_final_visual_token_count",
    "sparse_raw_selected_patch_index_count",
    "sparse_unique_patch_count",
    "sparse_duplicate_patch_index_count",
    "sparse_bbox",
    "sparse_centroid_row",
    "sparse_centroid_col",
    "sparse_spatial_spread_mean_grid_distance",
    "ours_final_visual_token_count",
    "ours_raw_selected_patch_index_count",
    "ours_unique_patch_count",
    "ours_duplicate_patch_index_count",
    "ours_bbox",
    "ours_centroid_row",
    "ours_centroid_col",
    "ours_spatial_spread_mean_grid_distance",
    "threshold_final_visual_token_count",
    "threshold_raw_selected_patch_index_count",
    "threshold_unique_patch_count",
    "threshold_duplicate_patch_index_count",
    "threshold_bbox",
    "threshold_centroid_row",
    "threshold_centroid_col",
    "threshold_spatial_spread_mean_grid_distance",
    "sparse_ours_unique_overlap_count",
    "sparse_ours_unique_jaccard",
    "sparse_threshold_unique_overlap_count",
    "sparse_threshold_unique_jaccard",
    "ours_threshold_unique_overlap_count",
    "ours_threshold_unique_jaccard",
]

STAGE7_VERIFIED_VALUES = {"", "yes", "no", "unclear"}
COVER_VALUES = {"", "yes", "partial", "no", "unclear"}
REDUNDANCY_SIGNAL_VALUES = {"", "yes", "no", "unclear"}
RECOMMENDED_FAILURE_TYPES = {
    "",
    "missed_relevant_visual_evidence",
    "redundant_or_dominant_region_selection",
    "unclear_before_visualization",
}
RECOMMENDED_STATUSES = {
    "",
    "provisional",
    "confirmed_after_visualization",
    "unclear",
}


@dataclass(frozen=True)
class SelectionTrace:
    indices: list[int]
    unique_indices: set[int]
    final_pruning_layer: int
    final_visual_token_count: int | None


@dataclass(frozen=True)
class FigurePaths:
    comparison: Path
    sparse_overlay: Path
    ours_overlay: Path
    threshold_overlay: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage 7 token-selection visualizations and metrics."
    )
    parser.add_argument(
        "--require-complete-review",
        action="store_true",
        help=(
            "Require completed manual Stage 7 review fields and generate the final "
            "analysis markdown."
        ),
    )
    parser.add_argument(
        "--image-aspect-ratio",
        default=None,
        choices=["pad", "anyres", "square"],
        help=(
            "Override the image_aspect_ratio branch used by the repo preprocessing "
            "mirror. By default the value is inferred from run manifests; the current "
            "failure-mining manifests leave it unset."
        ),
    )
    parser.add_argument(
        "--expected-final-layer",
        type=int,
        default=DEFAULT_FINAL_PRUNING_LAYER,
        help="Expected final sparse pruning layer for selected_original_token_indices.",
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


def read_csv_by_case_id(path: Path, required: bool = True) -> dict[str, dict[str, str]]:
    if not path.is_file():
        if required:
            raise RuntimeError(f"Missing required file: {path}")
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "case_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"{path}: missing case_id column")
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
                raise RuntimeError(
                    f"{manifest_path}: required run {run_id} status is not ok"
                )
            manifests[run_id] = (manifest_path, record)

    missing = sorted(required_ids - set(manifests))
    if missing:
        raise RuntimeError(f"Missing required run manifests: {missing}")

    runs: dict[str, dict] = {}
    for method, spec in RUN_SPECS.items():
        manifest_path, manifest = manifests[spec["run_id"]]
        prediction_path = resolve_prediction_path(manifest_path, manifest)
        rows = read_jsonl(prediction_path)
        runs[method] = {
            "manifest_path": manifest_path,
            "manifest": manifest,
            "prediction_path": prediction_path,
            "rows": {row["case_id"]: row for row in rows},
        }
    return runs


def infer_image_aspect_ratio(
    runs: dict[str, dict], override: str | None
) -> str | None:
    if override == "square":
        return None
    if override:
        return override
    values = {
        str(run["manifest"].get("image_aspect_ratio", "") or "").strip()
        for run in runs.values()
    }
    values.discard("")
    if len(values) > 1:
        raise RuntimeError(f"Inconsistent image_aspect_ratio values: {sorted(values)}")
    if not values:
        return None
    return next(iter(values))


def read_priority_cases(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise RuntimeError(
                    f"{path}:{line_number}: expected case_id, failure_type, evidence"
                )
            case_id, failure_type, evidence_needed = parts[0], parts[1], "\t".join(parts[2:])
            if case_id in seen:
                raise RuntimeError(f"{path}: duplicate priority case {case_id}")
            seen.add(case_id)
            cases.append(
                {
                    "case_id": case_id,
                    "failure_type": failure_type,
                    "evidence_needed": evidence_needed,
                }
            )
    if not cases:
        raise RuntimeError(f"{path}: no visualization cases")
    return cases


def validate_stage6_sources(
    priority_cases: list[dict[str, str]],
    review_rows: dict[str, dict[str, str]],
    curated_rows: dict[str, dict[str, str]],
) -> None:
    for case in priority_cases:
        case_id = case["case_id"]
        if case_id not in review_rows:
            raise RuntimeError(f"{case_id}: missing from Stage 6 review inventory")
        if case_id not in curated_rows:
            raise RuntimeError(f"{case_id}: missing from Stage 6 curated cases")
        row = review_rows[case_id]
        for field in [
            "image_path",
            "prompt",
            "ground_truth",
            "dense_prediction",
            "sparse_prediction",
            "ours_prediction",
            "threshold_prediction",
        ]:
            if not str(row.get(field, "")).strip():
                raise RuntimeError(f"{case_id}: Stage 6 review missing {field}")
        image_path = REPO_ROOT / row["image_path"]
        if not image_path.is_file():
            raise RuntimeError(f"{case_id}: image path does not exist: {image_path}")


def validate_patch_indices(case_id: str, method: str, indices: object) -> list[int]:
    if not isinstance(indices, list):
        raise RuntimeError(f"{case_id}/{method}: selected patch indices are not a list")
    validated: list[int] = []
    for position, value in enumerate(indices):
        if not isinstance(value, int):
            raise RuntimeError(
                f"{case_id}/{method}: selected patch index at position {position} "
                f"is not an integer: {value!r}"
            )
        if value < 0 or value > MAX_PATCH_INDEX:
            raise RuntimeError(
                f"{case_id}/{method}: selected patch index outside 0-{MAX_PATCH_INDEX}: "
                f"{value}"
            )
        validated.append(value)
    if not validated:
        raise RuntimeError(f"{case_id}/{method}: no selected patch indices")
    return validated


def extract_selection_trace(
    case_id: str,
    method: str,
    prediction_row: dict,
    expected_final_layer: int,
) -> SelectionTrace:
    metadata = prediction_row.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{case_id}/{method}: missing prediction metadata")

    top_level_indices = validate_patch_indices(
        case_id,
        method,
        metadata.get("selected_original_token_indices"),
    )

    layer_stats = metadata.get("layer_token_stats")
    if not isinstance(layer_stats, list) or not layer_stats:
        raise RuntimeError(f"{case_id}/{method}: missing layer_token_stats")
    final_layer = layer_stats[-1]
    if not isinstance(final_layer, dict):
        raise RuntimeError(f"{case_id}/{method}: final layer stats are malformed")

    final_layer_idx = final_layer.get("layer_idx")
    if final_layer_idx != expected_final_layer:
        raise RuntimeError(
            f"{case_id}/{method}: expected final pruning layer "
            f"{expected_final_layer}, found {final_layer_idx}"
        )

    final_layer_indices = validate_patch_indices(
        case_id,
        method,
        final_layer.get("selected_original_token_indices"),
    )
    if top_level_indices != final_layer_indices:
        raise RuntimeError(
            f"{case_id}/{method}: metadata.selected_original_token_indices does not "
            "match the final layer selected_original_token_indices"
        )

    retained_token_count = metadata.get("retained_token_count")
    if retained_token_count is not None and not isinstance(retained_token_count, int):
        raise RuntimeError(
            f"{case_id}/{method}: retained_token_count is not an integer"
        )

    return SelectionTrace(
        indices=top_level_indices,
        unique_indices=set(top_level_indices),
        final_pruning_layer=int(final_layer_idx),
        final_visual_token_count=retained_token_count,
    )


def load_case_traces(
    priority_cases: list[dict[str, str]],
    runs: dict[str, dict],
    expected_final_layer: int,
) -> dict[str, dict[str, SelectionTrace]]:
    traces: dict[str, dict[str, SelectionTrace]] = {}
    for case in priority_cases:
        case_id = case["case_id"]
        traces[case_id] = {}
        for method, run in runs.items():
            row = run["rows"].get(case_id)
            if row is None:
                raise RuntimeError(
                    f"{case_id}: missing prediction row for {RUN_SPECS[method]['run_id']}"
                )
            traces[case_id][method] = extract_selection_trace(
                case_id,
                method,
                row,
                expected_final_layer,
            )
    return traces


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img.copy()
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def resize_shortest_edge(image: Image.Image, shortest_edge: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid image size: {image.size}")
    if width < height:
        new_width = shortest_edge
        new_height = int(round(height * shortest_edge / width))
    elif height < width:
        new_height = shortest_edge
        new_width = int(round(width * shortest_edge / height))
    else:
        new_width = new_height = shortest_edge
    return image.resize((new_width, new_height), BICUBIC)


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    if width < crop_size or height < crop_size:
        padded = Image.new(image.mode, (max(width, crop_size), max(height, crop_size)))
        padded.paste(image, ((padded.width - width) // 2, (padded.height - height) // 2))
        image = padded
        width, height = image.size
    left = int(round((width - crop_size) / 2.0))
    top = int(round((height - crop_size) / 2.0))
    return image.crop((left, top, left + crop_size, top + crop_size))


def repo_clip_input_view(
    image: Image.Image,
    image_aspect_ratio: str | None,
    size: int = DEFAULT_CLIP_INPUT_SIZE,
) -> Image.Image:
    """Return the display image aligned with llava.mm_utils.process_images.

    The evaluation code calls process_images([image], image_processor, model.config).
    This mirrors the same branch structure, but returns the pre-normalization PIL
    view so patch overlays are interpretable.
    """
    if image_aspect_ratio == "anyres":
        raise RuntimeError(
            "Stage 7 visualization does not support image_aspect_ratio='anyres' "
            "because the selected patch grid is not a single 24x24 CLIP view."
        )
    image = image.convert("RGB")
    if image_aspect_ratio == "pad":
        image = expand2square(image, CLIP_MEAN_RGB)
    elif image_aspect_ratio not in (None, "", "square"):
        raise RuntimeError(f"Unsupported image_aspect_ratio: {image_aspect_ratio}")

    # Default CLIPImageProcessor behavior for OpenAI CLIP-L/336: resize shortest
    # edge to 336 using bicubic resampling, then center-crop to 336x336.
    return center_crop(resize_shortest_edge(image, size), size)


def patch_box(index: int, image_size: int = DEFAULT_CLIP_INPUT_SIZE) -> tuple[int, int, int, int]:
    row, col = divmod(index, GRID_SIZE)
    left = round(col * image_size / GRID_SIZE)
    top = round(row * image_size / GRID_SIZE)
    right = round((col + 1) * image_size / GRID_SIZE)
    bottom = round((row + 1) * image_size / GRID_SIZE)
    return left, top, right, bottom


def draw_overlay(
    clip_view: Image.Image,
    unique_indices: set[int],
    color: tuple[int, int, int],
    fade: float = 0.68,
) -> Image.Image:
    base = clip_view.convert("RGB")
    white = Image.new("RGB", base.size, "white")
    canvas = Image.blend(base, white, fade).convert("RGBA")
    source = base.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for row in range(GRID_SIZE + 1):
        coord = round(row * base.height / GRID_SIZE)
        draw.line((0, coord, base.width, coord), fill=(80, 80, 80, 36), width=1)
    for col in range(GRID_SIZE + 1):
        coord = round(col * base.width / GRID_SIZE)
        draw.line((coord, 0, coord, base.height), fill=(80, 80, 80, 36), width=1)

    fill = (*color, 58)
    outline = (*color, 235)
    for index in sorted(unique_indices):
        box = patch_box(index, base.width)
        canvas.paste(source.crop(box), box)
        draw.rectangle(box, fill=fill, outline=outline, width=2)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def fit_on_white(image: Image.Image, size: int = DEFAULT_CLIP_INPUT_SIZE) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width_chars: int,
    fill: tuple[int, int, int] = (0, 0, 0),
    line_spacing: int = 4,
) -> int:
    x, y = xy
    current_y = y
    for paragraph in str(text).splitlines() or [""]:
        lines = textwrap.wrap(paragraph, width=width_chars) or [""]
        for line in lines:
            draw.text((x, current_y), line, fill=fill, font=font())
            bbox = draw.textbbox((x, current_y), line, font=font())
            current_y += (bbox[3] - bbox[1]) + line_spacing
    return current_y


def make_comparison_figure(
    case_id: str,
    stage6_row: dict[str, str],
    evidence_needed: str,
    raw_image: Image.Image,
    clip_view: Image.Image,
    overlays: dict[str, Image.Image],
    traces: dict[str, SelectionTrace],
) -> Image.Image:
    panel_size = DEFAULT_CLIP_INPUT_SIZE
    margin = 18
    gap = 14
    label_height = 46
    header_height = 154
    footer_height = 54
    panel_count = 5
    width = margin * 2 + panel_count * panel_size + (panel_count - 1) * gap
    height = header_height + label_height + panel_size + footer_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"{case_id} - Stage 7 selected-token comparison"
    draw.text((margin, 14), title, fill=(0, 0, 0), font=font())

    prediction_line = (
        f"GT: {stage6_row['ground_truth']} | Dense: {stage6_row['dense_prediction']} | "
        f"Original: {stage6_row['sparse_prediction']} | Ours: {stage6_row['ours_prediction']} | "
        f"Threshold: {stage6_row['threshold_prediction']}"
    )
    y = draw_wrapped_text(draw, (margin, 34), f"Question: {stage6_row['prompt']}", 160)
    y = draw_wrapped_text(draw, (margin, y + 3), prediction_line, 160)
    draw_wrapped_text(draw, (margin, y + 3), f"Hypothesized evidence: {evidence_needed}", 160)

    panels = [
        ("Raw image", fit_on_white(raw_image)),
        ("Repo CLIP input view", clip_view),
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
            count_line = (
                f"raw={len(trace.indices)}, unique={len(trace.unique_indices)}, "
                f"final_tokens={trace.final_visual_token_count}"
            )
            draw.text((x, header_height + 18), count_line, fill=(55, 55, 55), font=font())
        canvas.paste(image, (x, panel_y))

    footer = (
        "Selected indices are final-layer original CLIP patch IDs on a 24x24 grid; "
        "merged tokens are not shown as individual original patches."
    )
    draw_wrapped_text(draw, (margin, panel_y + panel_size + 12), footer, 160, fill=(70, 70, 70))
    return canvas


def create_figures(
    priority_cases: list[dict[str, str]],
    review_rows: dict[str, dict[str, str]],
    traces_by_case: dict[str, dict[str, SelectionTrace]],
    image_aspect_ratio: str | None,
) -> dict[str, FigurePaths]:
    figure_paths: dict[str, FigurePaths] = {}
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for case in priority_cases:
        case_id = case["case_id"]
        stage6_row = review_rows[case_id]
        raw_image = Image.open(REPO_ROOT / stage6_row["image_path"]).convert("RGB")
        clip_view = repo_clip_input_view(raw_image, image_aspect_ratio)
        traces = traces_by_case[case_id]

        overlay_images: dict[str, Image.Image] = {}
        overlay_paths: dict[str, Path] = {}
        for method, trace in traces.items():
            overlay = draw_overlay(
                clip_view,
                trace.unique_indices,
                RUN_SPECS[method]["color"],
            )
            overlay_path = FIGURE_ROOT / f"{case_id}_{method}_overlay.png"
            overlay.save(overlay_path)
            overlay_images[method] = overlay
            overlay_paths[method] = overlay_path

        comparison = make_comparison_figure(
            case_id,
            stage6_row,
            case["evidence_needed"],
            raw_image,
            clip_view,
            overlay_images,
            traces,
        )
        comparison_path = FIGURE_ROOT / f"{case_id}_comparison.png"
        comparison.save(comparison_path)
        figure_paths[case_id] = FigurePaths(
            comparison=comparison_path,
            sparse_overlay=overlay_paths["sparse"],
            ours_overlay=overlay_paths["ours"],
            threshold_overlay=overlay_paths["threshold"],
        )
    return figure_paths


def coordinate_stats(indices: set[int]) -> dict[str, str]:
    if not indices:
        return {
            "bbox": "",
            "centroid_row": "",
            "centroid_col": "",
            "spread": "",
        }
    rows = [idx // GRID_SIZE for idx in sorted(indices)]
    cols = [idx % GRID_SIZE for idx in sorted(indices)]
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    centroid_row = sum(rows) / len(rows)
    centroid_col = sum(cols) / len(cols)
    spread = sum(
        math.sqrt((row - centroid_row) ** 2 + (col - centroid_col) ** 2)
        for row, col in zip(rows, cols)
    ) / len(rows)
    return {
        "bbox": f"rows={row_min}-{row_max};cols={col_min}-{col_max}",
        "centroid_row": f"{centroid_row:.3f}",
        "centroid_col": f"{centroid_col:.3f}",
        "spread": f"{spread:.3f}",
    }


def unique_overlap_and_jaccard(a: set[int], b: set[int]) -> tuple[int, str]:
    union = a | b
    intersection = a & b
    if not union:
        return 0, "0.000000"
    return len(intersection), f"{len(intersection) / len(union):.6f}"


def build_metrics_rows(
    priority_cases: list[dict[str, str]],
    traces_by_case: dict[str, dict[str, SelectionTrace]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    semantics = (
        "metadata.selected_original_token_indices from the final sparse pruning "
        "layer; original CLIP patch IDs on the 24x24 grid before merge tokens."
    )
    note = (
        "Counts and Jaccard values are diagnostic only. Jaccard is computed from "
        "unique patch sets, not raw selected-index lists."
    )
    for case in priority_cases:
        case_id = case["case_id"]
        traces = traces_by_case[case_id]
        final_layers = {trace.final_pruning_layer for trace in traces.values()}
        if len(final_layers) != 1:
            raise RuntimeError(f"{case_id}: inconsistent final pruning layers")
        row: dict[str, str] = {
            "case_id": case_id,
            "selected_index_semantics": semantics,
            "final_pruning_layer": str(next(iter(final_layers))),
            "metrics_note": note,
        }
        for method, trace in traces.items():
            stats = coordinate_stats(trace.unique_indices)
            row[f"{method}_final_visual_token_count"] = (
                "" if trace.final_visual_token_count is None else str(trace.final_visual_token_count)
            )
            row[f"{method}_raw_selected_patch_index_count"] = str(len(trace.indices))
            row[f"{method}_unique_patch_count"] = str(len(trace.unique_indices))
            row[f"{method}_duplicate_patch_index_count"] = str(
                len(trace.indices) - len(trace.unique_indices)
            )
            row[f"{method}_bbox"] = stats["bbox"]
            row[f"{method}_centroid_row"] = stats["centroid_row"]
            row[f"{method}_centroid_col"] = stats["centroid_col"]
            row[f"{method}_spatial_spread_mean_grid_distance"] = stats["spread"]

        pairs = [
            ("sparse", "ours", "sparse_ours"),
            ("sparse", "threshold", "sparse_threshold"),
            ("ours", "threshold", "ours_threshold"),
        ]
        for left, right, prefix in pairs:
            overlap, jaccard = unique_overlap_and_jaccard(
                traces[left].unique_indices,
                traces[right].unique_indices,
            )
            row[f"{prefix}_unique_overlap_count"] = str(overlap)
            row[f"{prefix}_unique_jaccard"] = jaccard
        rows.append(row)

    for row in rows:
        for field in METRIC_FIELDS:
            if field not in row:
                raise RuntimeError(f"{row['case_id']}: missing metric field {field}")
    return rows


def relative_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def validate_manual_row(row: dict[str, str], require_complete: bool) -> None:
    case_id = row.get("case_id", "<unknown>")
    if row.get("stage7_verified", "") not in STAGE7_VERIFIED_VALUES:
        raise RuntimeError(f"{case_id}: unsupported stage7_verified value")
    for field in [
        "sparse_covers_evidence",
        "ours_covers_evidence",
        "threshold_covers_evidence",
    ]:
        if row.get(field, "") not in COVER_VALUES:
            raise RuntimeError(f"{case_id}: unsupported {field} value")
    if row.get("redundancy_signal", "") not in REDUNDANCY_SIGNAL_VALUES:
        raise RuntimeError(f"{case_id}: unsupported redundancy_signal value")
    if row.get("recommended_failure_type", "") not in RECOMMENDED_FAILURE_TYPES:
        raise RuntimeError(f"{case_id}: unsupported recommended_failure_type value")
    if row.get("recommended_classification_status", "") not in RECOMMENDED_STATUSES:
        raise RuntimeError(
            f"{case_id}: unsupported recommended_classification_status value"
        )

    if row.get("stage7_verified") == "yes" and not row.get(
        "visual_interpretation", ""
    ).strip():
        raise RuntimeError(
            f"{case_id}: stage7_verified=yes requires visual_interpretation"
        )
    if row.get("redundancy_signal") == "yes" and not row.get(
        "redundancy_notes", ""
    ).strip():
        raise RuntimeError(f"{case_id}: redundancy_signal=yes requires notes")
    if (
        row.get("recommended_failure_type") == "redundant_or_dominant_region_selection"
        and row.get("recommended_classification_status")
        == "confirmed_after_visualization"
        and not (
            row.get("redundancy_signal") == "yes"
            and row.get("redundancy_notes", "").strip()
        )
    ):
        raise RuntimeError(
            f"{case_id}: confirmed redundancy requires recorded visual or metric support"
        )

    if require_complete:
        required = [
            "stage7_verified",
            "visual_evidence_region",
            "sparse_covers_evidence",
            "ours_covers_evidence",
            "threshold_covers_evidence",
            "visual_interpretation",
            "redundancy_signal",
            "recommended_failure_type",
            "recommended_classification_status",
        ]
        missing = [field for field in required if not row.get(field, "").strip()]
        if missing:
            raise RuntimeError(
                f"{case_id}: --require-complete-review missing fields: {missing}"
            )


def build_review_rows(
    priority_cases: list[dict[str, str]],
    stage6_rows: dict[str, dict[str, str]],
    figure_paths: dict[str, FigurePaths],
    require_complete: bool,
) -> list[dict[str, str]]:
    existing_rows = read_csv_by_case_id(REVIEW_PATH, required=False)
    rows: list[dict[str, str]] = []
    for case in priority_cases:
        case_id = case["case_id"]
        stage6 = stage6_rows[case_id]
        paths = figure_paths[case_id]
        existing = existing_rows.get(case_id, {})
        row = {
            "case_id": case_id,
            "dataset": stage6.get("dataset", ""),
            "question_type": stage6.get("question_type", ""),
            "image_path": stage6.get("image_path", ""),
            "prompt": stage6.get("prompt", ""),
            "ground_truth": stage6.get("ground_truth", ""),
            "dense_prediction": stage6.get("dense_prediction", ""),
            "sparse_prediction": stage6.get("sparse_prediction", ""),
            "ours_prediction": stage6.get("ours_prediction", ""),
            "threshold_prediction": stage6.get("threshold_prediction", ""),
            "case_group": stage6.get("case_group", ""),
            "stage6_failure_type": stage6.get("main_failure_type", case["failure_type"]),
            "stage6_classification_status": stage6.get("classification_status", ""),
            "evidence_needed": case["evidence_needed"],
            "comparison_figure": relative_path(paths.comparison),
            "sparse_overlay_figure": relative_path(paths.sparse_overlay),
            "ours_overlay_figure": relative_path(paths.ours_overlay),
            "threshold_overlay_figure": relative_path(paths.threshold_overlay),
        }
        for field in MANUAL_FIELDS:
            row[field] = existing.get(field, "")
        validate_manual_row(row, require_complete=require_complete)
        rows.append(row)
    return rows


def markdown_link(path: str) -> str:
    return path.replace("\\", "/")


def generate_analysis_markdown(
    review_rows: list[dict[str, str]],
    metrics_rows: list[dict[str, str]],
) -> str:
    metrics_by_case = {row["case_id"]: row for row in metrics_rows}
    verification_counts: dict[str, int] = {"yes": 0, "no": 0, "unclear": 0}
    redundancy_counts: dict[str, int] = {"yes": 0, "no": 0, "unclear": 0}
    for row in review_rows:
        verification_counts[row["stage7_verified"]] += 1
        redundancy_counts[row["redundancy_signal"]] += 1

    lines = [
        "# Stage 7 — Selected-Token Visualization",
        "",
        "## Objective and protocol",
        "",
        (
            "Stage 7 checks whether the Stage 6 failure-case explanations are "
            "supported by selected-token overlays. The figures use saved "
            "64-token failure-mining metadata; no model inference is rerun."
        ),
        "",
        (
            "The displayed CLIP input view mirrors the repository preprocessing "
            "path in `SparseVLMs/llava/mm_utils.py::process_images`. Selected "
            "patches are final-layer original CLIP patch IDs on the 24×24 grid. "
            "Merged tokens are not visualized as individual original patches."
        ),
        "",
        "## Verification summary",
        "",
        "| Result | Cases |",
        "| --- | ---: |",
        f"| Verified | {verification_counts['yes']} |",
        f"| Not supported | {verification_counts['no']} |",
        f"| Unclear | {verification_counts['unclear']} |",
        "",
        "| Redundancy signal | Cases |",
        "| --- | ---: |",
        f"| Yes | {redundancy_counts['yes']} |",
        f"| No | {redundancy_counts['no']} |",
        f"| Unclear | {redundancy_counts['unclear']} |",
        "",
        "## Case-level visual findings",
        "",
    ]

    for row in review_rows:
        metric = metrics_by_case[row["case_id"]]
        lines.extend(
            [
                f"### {row['case_id']}",
                "",
                f"![{row['case_id']} comparison]({markdown_link(row['comparison_figure'])})",
                "",
                f"- Question: {row['prompt']}",
                f"- Ground truth: {row['ground_truth']}",
                (
                    f"- Predictions: Dense={row['dense_prediction']}; "
                    f"Original={row['sparse_prediction']}; Ours={row['ours_prediction']}; "
                    f"Threshold={row['threshold_prediction']}"
                ),
                f"- Evidence region: {row['visual_evidence_region']}",
                (
                    f"- Coverage: Original={row['sparse_covers_evidence']}; "
                    f"Ours={row['ours_covers_evidence']}; "
                    f"Threshold={row['threshold_covers_evidence']}"
                ),
                f"- Verdict: {row['stage7_verified']}",
                f"- Interpretation: {row['visual_interpretation']}",
                (
                    f"- Unique-set Jaccard: Original/Ours="
                    f"{metric['sparse_ours_unique_jaccard']}; "
                    f"Original/Threshold={metric['sparse_threshold_unique_jaccard']}; "
                    f"Ours/Threshold={metric['ours_threshold_unique_jaccard']}"
                ),
                f"- Redundancy signal: {row['redundancy_signal']}",
            ]
        )
        if row["redundancy_notes"].strip():
            lines.append(f"- Redundancy notes: {row['redundancy_notes']}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "These visualizations can strengthen the missed-evidence explanation "
                "only for cases where SparseVLM-Original visibly fails to cover the "
                "hypothesized evidence region and Ours covers it better. Cases marked "
                "`no` or `unclear` should remain cautious and may involve downstream "
                "generation or language-model effects."
            ),
            "",
            (
                "The token metrics are diagnostic support for the overlays. Raw "
                "selected-index counts, unique patch counts, and unique-set Jaccard "
                "similarities should not be treated as causal proof."
            ),
            "",
            "## Limitations",
            "",
            "- The visualized cases are selected qualitative examples, not prevalence estimates.",
            "- Selected patch coverage does not prove that a patch caused the generated answer.",
            "- Merged visual tokens are not directly localizable to one original patch in these overlays.",
            "- Redundancy claims require visual support and should ideally be paired with explicit similarity or coverage metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    priority_cases = read_priority_cases(STAGE6_VIS_CASES_PATH)
    stage6_review = read_csv_by_case_id(STAGE6_REVIEW_PATH)
    stage6_curated = read_csv_by_case_id(STAGE6_CASES_PATH)
    validate_stage6_sources(priority_cases, stage6_review, stage6_curated)

    runs = load_required_runs()
    image_aspect_ratio = infer_image_aspect_ratio(runs, args.image_aspect_ratio)
    traces_by_case = load_case_traces(
        priority_cases,
        runs,
        expected_final_layer=args.expected_final_layer,
    )

    figure_paths = create_figures(
        priority_cases,
        stage6_review,
        traces_by_case,
        image_aspect_ratio=image_aspect_ratio,
    )

    metrics_rows = build_metrics_rows(priority_cases, traces_by_case)
    write_csv(METRICS_PATH, metrics_rows, METRIC_FIELDS)

    review_rows = build_review_rows(
        priority_cases,
        stage6_review,
        figure_paths,
        require_complete=args.require_complete_review,
    )
    write_csv(REVIEW_PATH, review_rows, REVIEW_FIELDS)

    if args.require_complete_review:
        ANALYSIS_PATH.write_text(
            generate_analysis_markdown(review_rows, metrics_rows),
            encoding="utf-8",
        )

    print(f"Wrote {len(priority_cases)} Stage 7 case visualizations to {FIGURE_ROOT}")
    print(f"Wrote metrics: {METRICS_PATH}")
    print(f"Wrote review table: {REVIEW_PATH}")
    if args.require_complete_review:
        print(f"Wrote analysis: {ANALYSIS_PATH}")
    else:
        print(
            "Skipped final analysis markdown because --require-complete-review "
            "was not set."
        )


if __name__ == "__main__":
    main()
