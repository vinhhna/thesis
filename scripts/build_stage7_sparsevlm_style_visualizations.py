from __future__ import annotations

import argparse
import csv
import textwrap
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from build_stage7_expanded_visual_candidates import (
    DATASET_SPECS,
    display_prediction,
    file_safe_case_id,
    local_image_path,
    read_jsonl_by_question_id,
)
from build_stage7_visualizations import GRID_SIZE, MAX_PATCH_INDEX


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_CSV = (
    REPO_ROOT
    / "outputs"
    / "stage7"
    / "stage7_selected_visual_support_cases.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "stage7"
DEFAULT_FIGURE_ROOT = DEFAULT_OUTPUT_ROOT / "figures"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "stage7_visualization_manifest.csv"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "stage7_visualization_summary.md"

PRUNING_LAYERS = [2, 6, 15]
METHODS = {
    "sparse": {
        "label": "SparseVLM-Original-64",
        "prediction_field": "sparse_prediction",
    },
    "ours": {
        "label": "Ours-64",
        "prediction_field": "ours_prediction",
    },
    "threshold": {
        "label": "Threshold-Fixed-64",
        "prediction_field": "threshold_prediction",
    },
}

MANIFEST_FIELDS = [
    "rank",
    "expanded_case_id",
    "dataset",
    "question_id",
    "method",
    "method_label",
    "figure_path",
    "image_name",
    "image_path",
    "prompt",
    "ground_truth",
    "prediction",
    "layer2_raw_selected_patch_count",
    "layer2_unique_patch_count",
    "layer6_raw_selected_patch_count",
    "layer6_unique_patch_count",
    "layer15_raw_selected_patch_count",
    "layer15_unique_patch_count",
    "visualization_note",
]

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - old Pillow compatibility.
    LANCZOS = Image.LANCZOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render SparseVLM-paper-style layer-wise retained-patch figures for "
            "the manually selected expanded Stage 7 visual-support cases."
        )
    )
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=DEFAULT_SELECTED_CSV,
        help="Selected-case CSV produced from expanded Stage 7 inspection.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for SparseVLM-style figures and manifest.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit the number of selected cases rendered; 0 renders all.",
    )
    parser.add_argument(
        "--fade",
        type=float,
        default=0.74,
        help="White fade strength for non-retained patches, matching paper style.",
    )
    parser.add_argument(
        "--panel-max-width",
        type=int,
        default=260,
        help="Maximum width for each image/pruning panel.",
    )
    parser.add_argument(
        "--panel-max-height",
        type=int,
        default=220,
        help="Maximum height for each image/pruning panel.",
    )
    return parser.parse_args()


def font(size: int = 16, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_selected_rows(path: Path, max_cases: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing selected-case CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "rank",
            "expanded_case_id",
            "dataset",
            "question_id",
            "prompt",
            "ground_truth",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path}: missing required columns: {sorted(missing)}")
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    rows = sorted(rows, key=lambda row: int(row["rank"]))
    if max_cases and max_cases > 0:
        rows = rows[:max_cases]
    if not rows:
        raise RuntimeError(f"{path}: no selected rows")
    return rows


def load_runs_for_datasets(datasets: Iterable[str]) -> dict[str, dict[str, dict[str, dict]]]:
    loaded: dict[str, dict[str, dict[str, dict]]] = {}
    for dataset in sorted(set(datasets)):
        if dataset not in DATASET_SPECS:
            raise RuntimeError(f"Unsupported dataset in selected CSV: {dataset}")
        loaded[dataset] = {
            method: read_jsonl_by_question_id(path)
            for method, path in DATASET_SPECS[dataset].items()
            if method in METHODS
        }
    return loaded


def validate_indices(
    case_id: str,
    method: str,
    layer_idx: int,
    indices: object,
) -> list[int]:
    if not isinstance(indices, list):
        raise RuntimeError(
            f"{case_id}/{method}/layer{layer_idx}: selected_original_token_indices "
            "is not a list"
        )
    validated: list[int] = []
    for position, value in enumerate(indices):
        if not isinstance(value, int):
            raise RuntimeError(
                f"{case_id}/{method}/layer{layer_idx}: patch index at position "
                f"{position} is not an integer: {value!r}"
            )
        if value < 0 or value > MAX_PATCH_INDEX:
            raise RuntimeError(
                f"{case_id}/{method}/layer{layer_idx}: patch index outside "
                f"0-{MAX_PATCH_INDEX}: {value}"
            )
        validated.append(value)
    if not validated:
        raise RuntimeError(
            f"{case_id}/{method}/layer{layer_idx}: no selected original patches"
        )
    return validated


def layer_indices_by_pruning_layer(
    case_id: str,
    method: str,
    prediction_row: dict,
) -> dict[int, list[int]]:
    metadata = prediction_row.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{case_id}/{method}: missing prediction metadata")
    layer_stats = metadata.get("layer_token_stats")
    if not isinstance(layer_stats, list) or not layer_stats:
        raise RuntimeError(f"{case_id}/{method}: missing layer_token_stats")

    by_layer: dict[int, list[int]] = {}
    for layer in layer_stats:
        if not isinstance(layer, dict):
            continue
        layer_idx = layer.get("layer_idx")
        if layer_idx in PRUNING_LAYERS:
            by_layer[int(layer_idx)] = validate_indices(
                case_id,
                method,
                int(layer_idx),
                layer.get("selected_original_token_indices"),
            )
    missing = [layer for layer in PRUNING_LAYERS if layer not in by_layer]
    if missing:
        raise RuntimeError(f"{case_id}/{method}: missing pruning layers {missing}")
    return by_layer


def display_image(raw: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = raw.convert("RGB").copy()
    image.thumbnail((max_width, max_height), LANCZOS)
    if image.width <= 0 or image.height <= 0:
        raise RuntimeError(f"Invalid display image size: {image.size}")
    return image


def patch_box(index: int, width: int, height: int) -> tuple[int, int, int, int]:
    row, col = divmod(index, GRID_SIZE)
    left = round(col * width / GRID_SIZE)
    top = round(row * height / GRID_SIZE)
    right = round((col + 1) * width / GRID_SIZE)
    bottom = round((row + 1) * height / GRID_SIZE)
    return left, top, right, bottom


def sparsevlm_style_pruned_view(
    base: Image.Image,
    selected_indices: Iterable[int],
    fade: float,
) -> Image.Image:
    base = base.convert("RGB")
    fade = min(max(float(fade), 0.0), 1.0)
    faded = Image.blend(base, Image.new("RGB", base.size, "white"), fade)

    for index in sorted(set(selected_indices)):
        box = patch_box(index, base.width, base.height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        faded.paste(base.crop(box), box)
    return faded


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: tuple[int, int, int] = (150, 150, 150),
    dash: int = 8,
    gap: int = 5,
    width: int = 1,
) -> None:
    left, top, right, bottom = xy
    x = left
    while x < right:
        draw.line((x, top, min(x + dash, right), top), fill=fill, width=width)
        draw.line((x, bottom, min(x + dash, right), bottom), fill=fill, width=width)
        x += dash + gap
    y = top
    while y < bottom:
        draw.line((left, y, left, min(y + dash, bottom)), fill=fill, width=width)
        draw.line((right, y, right, min(y + dash, bottom)), fill=fill, width=width)
        y += dash + gap


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_chars: int,
    text_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (35, 35, 35),
    line_gap: int = 4,
) -> int:
    x, y = xy
    current_y = y
    paragraphs = str(text or "").splitlines() or [""]
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=max_chars) or [""]:
            draw.text((x, current_y), line, font=text_font, fill=fill)
            bbox = draw.textbbox((x, current_y), line, font=text_font)
            current_y += bbox[3] - bbox[1] + line_gap
    return current_y


def make_sparsevlm_style_figure(
    selected_row: dict[str, str],
    method: str,
    prediction_row: dict,
    layer_indices: dict[int, list[int]],
    raw_image: Image.Image,
    image_path: Path,
    args: argparse.Namespace,
) -> Image.Image:
    panel = display_image(raw_image, args.panel_max_width, args.panel_max_height)
    pruning_panels = [
        sparsevlm_style_pruned_view(panel, layer_indices[layer], args.fade)
        for layer in PRUNING_LAYERS
    ]

    panel_gap = 6
    margin = 14
    label_height = 24
    metadata_width = 370
    metadata_gap = 28
    row_height = max(panel.height + label_height + margin * 2, 330)
    panels_width = panel.width * 4 + panel_gap * 3
    width = margin * 2 + panels_width + metadata_gap + metadata_width
    height = row_height

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_dashed_rectangle(draw, (6, 6, width - 7, height - 7))

    title_font = font(16, bold=True)
    small_font = font(13)
    meta_font = font(16)
    meta_bold = font(17, bold=True)

    labels = ["Original", "Layer 2", "Layer 6", "Layer 15 final"]
    images = [panel, *pruning_panels]
    panel_top = margin + label_height
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = margin + idx * (panel.width + panel_gap)
        draw.text((x, margin), label, font=small_font, fill=(35, 35, 35))
        canvas.paste(image, (x, panel_top))

    metadata_x = margin + panels_width + metadata_gap
    y = margin + 4
    draw.text(
        (metadata_x, y),
        METHODS[method]["label"],
        font=meta_bold,
        fill=(20, 80, 40),
    )
    y += 28
    draw.text(
        (metadata_x, y),
        f"Rank {selected_row['rank']} | {selected_row['expanded_case_id']}",
        font=small_font,
        fill=(70, 70, 70),
    )
    y += 25
    y = draw_wrapped(
        draw,
        (metadata_x, y),
        f"Image: {image_path.name}",
        max_chars=42,
        text_font=small_font,
        fill=(70, 70, 70),
    )
    y += 8
    draw.text((metadata_x, y), "Question", font=title_font, fill=(35, 35, 35))
    y += 21
    y = draw_wrapped(
        draw,
        (metadata_x, y),
        selected_row["prompt"],
        max_chars=34,
        text_font=meta_font,
        fill=(35, 35, 35),
    )
    y += 8
    y = draw_wrapped(
        draw,
        (metadata_x, y),
        f"Ground truth: {selected_row['ground_truth']}",
        max_chars=36,
        text_font=meta_bold,
        fill=(0, 110, 40),
    )
    prediction = display_prediction(selected_row["dataset"], prediction_row)
    y = draw_wrapped(
        draw,
        (metadata_x, y + 3),
        f"Prediction: {prediction}",
        max_chars=36,
        text_font=meta_bold,
        fill=(0, 110, 40),
    )
    y += 8
    for layer in PRUNING_LAYERS:
        indices = layer_indices[layer]
        draw.text(
            (metadata_x, y),
            f"Layer {layer}: raw={len(indices)}, unique={len(set(indices))}",
            font=small_font,
            fill=(70, 70, 70),
        )
        y += 18
    y += 4
    draw_wrapped(
        draw,
        (metadata_x, y),
        "Paper-style mask: faded cells are not explicitly retained. Merged/recycled tokens are not visualized.",
        max_chars=42,
        text_font=small_font,
        fill=(95, 95, 95),
    )
    return canvas


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    figure_root = output_root / "figures"
    manifest_path = output_root / DEFAULT_MANIFEST_PATH.name
    summary_path = output_root / DEFAULT_SUMMARY_PATH.name

    selected_rows = read_selected_rows(args.selected_csv, args.max_cases)
    runs = load_runs_for_datasets(row["dataset"] for row in selected_rows)

    figure_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    for selected_row in selected_rows:
        dataset = selected_row["dataset"]
        question_id = selected_row["question_id"]
        case_id = selected_row["expanded_case_id"]

        sparse_row = runs[dataset]["sparse"].get(question_id)
        if sparse_row is None:
            raise RuntimeError(f"{case_id}: missing SparseVLM prediction row")
        resolved_image_path = local_image_path(str(sparse_row.get("image_path", "")))
        raw_image = Image.open(resolved_image_path).convert("RGB")

        for method in METHODS:
            prediction_row = runs[dataset][method].get(question_id)
            if prediction_row is None:
                raise RuntimeError(f"{case_id}: missing {method} prediction row")
            layer_indices = layer_indices_by_pruning_layer(
                case_id,
                method,
                prediction_row,
            )
            figure = make_sparsevlm_style_figure(
                selected_row,
                method,
                prediction_row,
                layer_indices,
                raw_image,
                resolved_image_path,
                args,
            )
            figure_path = (
                figure_root
                / f"{int(selected_row['rank']):04d}_{file_safe_case_id(case_id)}_{method}_sparsevlm_style.png"
            )
            figure.save(figure_path)

            row = {
                "rank": selected_row["rank"],
                "expanded_case_id": case_id,
                "dataset": dataset,
                "question_id": question_id,
                "method": method,
                "method_label": METHODS[method]["label"],
                "figure_path": figure_path.relative_to(REPO_ROOT).as_posix(),
                "image_name": resolved_image_path.name,
                "image_path": resolved_image_path.relative_to(REPO_ROOT).as_posix()
                if resolved_image_path.is_relative_to(REPO_ROOT)
                else str(resolved_image_path),
                "prompt": selected_row["prompt"],
                "ground_truth": selected_row["ground_truth"],
                "prediction": display_prediction(dataset, prediction_row),
                "visualization_note": (
                    "SparseVLM-paper-style faded mask using explicit retained "
                    "original-patch indices from layer_token_stats; merged/recycled "
                    "tokens are not shown."
                ),
            }
            for layer in PRUNING_LAYERS:
                indices = layer_indices[layer]
                row[f"layer{layer}_raw_selected_patch_count"] = str(len(indices))
                row[f"layer{layer}_unique_patch_count"] = str(len(set(indices)))
            manifest_rows.append(row)

    write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)

    counts_by_method = Counter(row["method"] for row in manifest_rows)
    figure_counts_by_dataset = Counter(row["dataset"] for row in manifest_rows)
    case_counts_by_dataset = Counter(row["dataset"] for row in selected_rows)
    summary_lines = [
        "# Stage 7 SparseVLM-Style Visualizations",
        "",
        (
            "These figures render the selected expanded Stage 7 cases in the same "
            "visual style as SparseVLM's qualitative pruning examples: original "
            "image first, followed by layer-wise faded masks where explicitly "
            "retained original patches remain visible."
        ),
        "",
        f"- Selected cases rendered: {len(selected_rows)}",
        f"- Figures rendered: {len(manifest_rows)}",
        f"- Figure folder: `{figure_root.relative_to(REPO_ROOT).as_posix()}`",
        f"- Manifest: `{manifest_path.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Counts by method",
        "",
    ]
    for method in METHODS:
        summary_lines.append(f"- {METHODS[method]['label']}: {counts_by_method[method]}")
    summary_lines.extend(["", "## Selected cases by dataset", ""])
    for dataset in sorted(case_counts_by_dataset):
        summary_lines.append(f"- {dataset}: {case_counts_by_dataset[dataset]}")
    summary_lines.extend(["", "## Figures by dataset", ""])
    for dataset in sorted(figure_counts_by_dataset):
        summary_lines.append(f"- {dataset}: {figure_counts_by_dataset[dataset]}")
    summary_lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            (
                "The masks show explicit retained original-patch traces at pruning "
                "layers 2, 6, and 15. SparseVLM's merged/recycled tokens do not map "
                "to a single original patch, so they are intentionally not shown. "
                "Use these figures as qualitative evidence, not causal proof."
            ),
            "",
        ]
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Selected cases: {len(selected_rows)}")
    print(f"Rendered figures: {len(manifest_rows)}")
    print(f"Wrote figures: {figure_root}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
