# SparseVLM Kaggle Defense Demo - Single Image

These cells are for the live defense demo. This demo is POPE-only: every demo
image should come from the POPE/COCO validation image set, not GQA or the
failure-mining sample-image folder. The intended workflow is:

1. Run Demo Cells 1 and 2 after a fresh Kaggle clone/runtime setup.
2. Restart the Kaggle kernel after Demo Cell 2.
3. Run Demo Cells 3 and 5 once to initialize helpers and load the model.
4. For each new demo image, edit only Demo Cell 4, then rerun Demo Cells 6 and 7.

The live demo runs one POPE image at a time through:

- `SparseVLM-Original-64`: original top-k selection.
- `Ours-64`: MMR selection.
- `Threshold-Fixed-64`: fixed-budget threshold baseline.

Dense-576 reference answers are included in the presets when available, but the live dense model is not loaded by default because it requires a separate non-sparse model load and extra GPU memory.

Demo Cell 1: Clone repo
```python
%env USE_FLAX=NO
%env USE_JAX=NO
%env USE_TF=NO
%env WANDB_DISABLED=true
%env WANDB_MODE=disabled
%env TOKENIZERS_PARALLELISM=false

%cd /kaggle/working
!rm -rf thesis
!git clone https://github.com/vinhhna/thesis.git

%cd /kaggle/working/thesis/SparseVLMs
!git rev-parse --short HEAD
```

Demo Cell 2: Install runtime packages
```python
import subprocess
import sys


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])


pip("install", "--upgrade", "pip", "setuptools", "wheel")

pip(
    "install",
    "--force-reinstall",
    "--no-deps",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
    "--index-url",
    "https://download.pytorch.org/whl/cu121",
)

pip(
    "uninstall",
    "-y",
    "tensorflow",
    "tensorflow-cpu",
    "tensorflow-io-gcs-filesystem",
    "keras",
    "tf-keras",
    "tensorboard",
    "tensorboard-data-server",
    "wandb",
    "bitsandbytes",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "chex",
    "orbax-checkpoint",
)

pip("install", "--force-reinstall", "numpy==1.26.4", "protobuf", "sentencepiece", "shortuuid")
pip("install", "transformers==4.37.2", "tokenizers==0.15.1", "accelerate==0.21.0", "peft==0.7.1")
pip("install", "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13", "markdown2[all]")

print("Restart the Kaggle kernel after this cell finishes, then continue from Demo Cell 3.")
```

Demo Cell 3: Demo helpers
```python
%cd /kaggle/working/thesis/SparseVLMs

import csv
import json
import os
import re
import shutil
import sys
import textwrap
import time
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from IPython.display import Markdown, display
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path("/kaggle/working/thesis")
LLAVA_ROOT = REPO_ROOT / "SparseVLMs"
DEMO_OUTPUT_ROOT = Path("/kaggle/working/defense_demo_outputs")

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONPATH"] = str(LLAVA_ROOT)
if str(LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ROOT))


def patch_sparse_private_import():
    path = LLAVA_ROOT / "llava/model/language_model/modelling_sparse_llama.py"
    text = path.read_text(encoding="utf-8")
    anchor = "from .score import *"
    replacement = "from .score import *\nfrom .score import _to_int"
    if replacement not in text:
        if anchor not in text:
            raise RuntimeError(f"Patch anchor not found in {path}")
        path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
        print("Patched SparseVLM private score import:", path)
    else:
        print("SparseVLM private score import already patched.")

    loaded_module = sys.modules.get("llava.model.language_model.modelling_sparse_llama")
    if loaded_module is not None and not hasattr(loaded_module, "_to_int"):
        from llava.model.language_model.score import _to_int as score_to_int
        loaded_module._to_int = score_to_int
        print("Patched already-loaded SparseVLM module with _to_int.")


patch_sparse_private_import()

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init


MODEL_PATH = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "llava_v1"
SPARSE_PRUNING_LOC = [2, 6, 15]
GRID_SIZE = 24
MAX_PATCH_INDEX = GRID_SIZE * GRID_SIZE - 1

METHOD_SPECS = {
    "sparse": {
        "label": "SparseVLM-Original-64",
        "selection_method": "topk",
        "retained_tokens": 64,
        "candidate_pool_factor": 2,
        "lambda_relevance": 0.8,
        "threshold_tau": 0.85,
        "color": (220, 40, 40),
    },
    "ours": {
        "label": "Ours-64",
        "selection_method": "mmr",
        "retained_tokens": 64,
        "candidate_pool_factor": 2,
        "lambda_relevance": 0.8,
        "threshold_tau": 0.85,
        "color": (30, 145, 70),
    },
    "threshold": {
        "label": "Threshold-Fixed-64",
        "selection_method": "threshold_fixed",
        "retained_tokens": 64,
        "candidate_pool_factor": 2,
        "lambda_relevance": 0.8,
        "threshold_tau": 0.85,
        "color": (40, 95, 220),
    },
}


def normalize_answer(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def yes_no_label(text):
    words = normalize_answer(text).split()
    return "no" if "no" in words or "not" in words else "yes"


def rough_correct(prediction, ground_truth):
    gold = normalize_answer(ground_truth)
    pred = normalize_answer(prediction)
    if not gold:
        return ""
    if gold in {"yes", "no"}:
        return yes_no_label(prediction) == gold
    return gold == pred or re.search(rf"(?:^| ){re.escape(gold)}(?: |$)", pred) is not None


def safe_stem(text):
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_")
    return text[:120] or "demo_case"


def find_file_by_name(filename, dataset="", allow_zip_extract=False):
    dataset = str(dataset or "pope").lower()
    if dataset != "pope":
        raise ValueError(
            f"This defense demo is POPE-only. Got dataset={dataset!r}; "
            "use a POPE/COCO val2014 image instead."
        )

    filename = str(filename or "").strip()
    if not filename:
        raise ValueError("image_filename is empty")

    direct_candidates = [
        Path(filename),
        REPO_ROOT / filename,
        Path("/kaggle/working") / filename,
    ]
    for path in direct_candidates:
        if path.is_file():
            return path.resolve()

    likely_roots = [
        REPO_ROOT / "data/kaggle_datasets/POPE/val2014",
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]

    for root in likely_roots:
        if not root.exists():
            continue
        direct = root / filename
        if direct.is_file():
            return direct.resolve()
        matches = sorted(path for path in root.rglob(filename) if path.is_file())
        if matches:
            return matches[0].resolve()

    if allow_zip_extract:
        zip_matches = sorted(Path("/kaggle/input").rglob("*.zip"))
        extract_root = Path("/kaggle/working/demo_zip_extract")
        for zip_path in zip_matches:
            target_dir = extract_root / zip_path.stem
            marker = target_dir / ".extract_complete"
            if not marker.exists():
                print("Extracting zip because DEMO_ALLOW_ZIP_EXTRACT=True:", zip_path)
                target_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as archive:
                    archive.extractall(target_dir)
                marker.write_text(str(zip_path), encoding="utf-8")
            matches = sorted(path for path in target_dir.rglob(filename) if path.is_file())
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find image file {filename!r}. Upload the image as a Kaggle Dataset, "
        "use an absolute path in CUSTOM_DEMO['image_path'], or set DEMO_ALLOW_ZIP_EXTRACT=True."
    )


def validate_pope_image(case):
    image_name = Path(case["image_path"]).name
    if not (image_name.startswith("COCO_val2014_") and image_name.lower().endswith((".jpg", ".jpeg", ".png"))):
        raise ValueError(
            "This defense demo is POPE-only. Expected a COCO val2014 image named "
            f"COCO_val2014_*.jpg, got {image_name!r}."
        )


def build_demo_case():
    if DEMO_PRESET_NAME == "custom":
        case = dict(CUSTOM_DEMO)
    else:
        if DEMO_PRESET_NAME not in DEMO_PRESETS:
            raise ValueError(f"Unknown DEMO_PRESET_NAME={DEMO_PRESET_NAME!r}")
        case = dict(DEMO_PRESETS[DEMO_PRESET_NAME])

    case["dataset"] = str(case.get("dataset", "pope")).lower()
    if case["dataset"] != "pope":
        raise ValueError(
            f"This defense demo is POPE-only. Got dataset={case['dataset']!r}; "
            "change the preset/custom image to a POPE case."
        )

    image_path = str(case.get("image_path", "")).strip()
    if image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        case["image_path"] = str(path.resolve())
    else:
        case["image_path"] = str(
            find_file_by_name(
                case.get("image_filename", ""),
                dataset=case.get("dataset", ""),
                allow_zip_extract=DEMO_ALLOW_ZIP_EXTRACT,
            )
        )

    validate_pope_image(case)
    case.setdefault("case_id", DEMO_PRESET_NAME)
    case.setdefault("dataset", "custom")
    case.setdefault("ground_truth", "")
    case["question"] = str(case["question"]).strip()
    case["ground_truth"] = str(case.get("ground_truth", "")).strip()
    case["prompt_for_model"] = (
        case["question"] + DEMO_SHORT_ANSWER_SUFFIX
        if DEMO_USE_SHORT_ANSWER_SUFFIX
        else case["question"]
    )
    return case


def build_prompt(question):
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prepare_image_tensor(image):
    images_tensor = process_images([image], image_processor, model.config)
    if isinstance(images_tensor, list):
        return [
            image_tensor.to(model.device, dtype=torch.float16)
            for image_tensor in images_tensor
        ]
    return images_tensor.to(model.device, dtype=torch.float16)


def run_demo_method(case, method_key):
    spec = METHOD_SPECS[method_key]
    image = Image.open(case["image_path"]).convert("RGB")
    prompt = build_prompt(case["prompt_for_model"])
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    images_tensor = prepare_image_tensor(image)

    sparse_core = model.get_model()
    original_pruning_loc = list(getattr(sparse_core, "pruning_loc", SPARSE_PRUNING_LOC))
    sparse_core.pruning_loc = SPARSE_PRUNING_LOC
    sparse_core.last_sparse_metadata = {}

    generation_kwargs = {
        "do_sample": DEMO_TEMPERATURE > 0,
        "num_beams": DEMO_NUM_BEAMS,
        "max_new_tokens": DEMO_MAX_NEW_TOKENS,
        "use_cache": True,
    }
    if DEMO_TEMPERATURE > 0:
        generation_kwargs["temperature"] = DEMO_TEMPERATURE

    start = time.time()
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                retained_tokens=spec["retained_tokens"],
                selection_method=spec["selection_method"],
                threshold_tau=spec["threshold_tau"],
                candidate_pool_factor=spec["candidate_pool_factor"],
                lambda_relevance=spec["lambda_relevance"],
                record_selection_similarity=True,
                **generation_kwargs,
            )
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        metadata = deepcopy(getattr(sparse_core, "last_sparse_metadata", {}))
    finally:
        sparse_core.pruning_loc = original_pruning_loc

    elapsed = time.time() - start
    metadata.setdefault("selection_method", spec["selection_method"])
    metadata.setdefault("retained_tokens", spec["retained_tokens"])
    metadata.setdefault("threshold_tau", spec["threshold_tau"])
    metadata.setdefault("candidate_pool_factor", spec["candidate_pool_factor"])
    metadata.setdefault("lambda_relevance", spec["lambda_relevance"])

    torch.cuda.empty_cache()
    return {
        "method_key": method_key,
        "method_label": spec["label"],
        "answer": answer,
        "is_rough_correct": rough_correct(answer, case.get("ground_truth", "")),
        "seconds": elapsed,
        "metadata": metadata,
    }


def run_defense_demo():
    case = build_demo_case()
    output_dir = DEMO_OUTPUT_ROOT / safe_stem(case.get("case_id", DEMO_PRESET_NAME))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Demo case:", case.get("case_id"))
    print("Dataset:", case.get("dataset"))
    print("Image:", case["image_path"])
    print("Question:", case["question"])
    print("Ground truth:", case.get("ground_truth", ""))
    if case.get("defense_use"):
        print("Defense use:", case["defense_use"])
    print()

    results = []
    for method_key in DEMO_METHODS:
        print("Running", METHOD_SPECS[method_key]["label"])
        result = run_demo_method(case, method_key)
        results.append(result)
        print(f"  Answer: {result['answer']}")
        print(f"  Time: {result['seconds']:.2f}s")

    payload = {
        "case": case,
        "methods": DEMO_METHODS,
        "results": results,
        "reference_answers": case.get("reference_answers", {}),
    }
    json_path = output_dir / "demo_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    rows = [
        "| Method | Live answer | Rough correct | Seconds | Reference answer |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    reference = case.get("reference_answers", {})
    for result in results:
        ref = reference.get(result["method_key"], "")
        correctness = result["is_rough_correct"]
        correctness_text = "" if correctness == "" else str(bool(correctness))
        rows.append(
            "| {method} | {answer} | {correct} | {seconds:.2f} | {ref} |".format(
                method=result["method_label"],
                answer=str(result["answer"]).replace("|", "\\|"),
                correct=correctness_text,
                seconds=result["seconds"],
                ref=str(ref).replace("|", "\\|"),
            )
        )
    display(Markdown("\n".join(rows)))
    print("Saved:", json_path)
    return case, results, output_dir


try:
    BICUBIC = Image.Resampling.BICUBIC
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS


def default_font(size=14, bold=False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def expand2square(pil_img, background_color):
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


def tensor_to_pil(pixel_values):
    tensor = pixel_values.detach().float().cpu()
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    tensor = tensor * std + mean
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def clip_input_view(raw_image):
    ratio = getattr(model.config, "image_aspect_ratio", None)
    image = raw_image.convert("RGB")
    if ratio == "anyres":
        raise RuntimeError("This demo visualizer expects the single 24x24 CLIP grid, not anyres.")
    if ratio == "pad":
        image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
    pixel = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return tensor_to_pil(pixel)


def patch_box(index, width, height):
    row, col = divmod(int(index), GRID_SIZE)
    left = round(col * width / GRID_SIZE)
    top = round(row * height / GRID_SIZE)
    right = round((col + 1) * width / GRID_SIZE)
    bottom = round((row + 1) * height / GRID_SIZE)
    return left, top, right, bottom


def draw_grid(image, fill=(80, 80, 80, 52)):
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for row in range(GRID_SIZE + 1):
        y = round(row * canvas.height / GRID_SIZE)
        draw.line((0, y, canvas.width, y), fill=fill, width=1)
    for col in range(GRID_SIZE + 1):
        x = round(col * canvas.width / GRID_SIZE)
        draw.line((x, 0, x, canvas.height), fill=fill, width=1)
    return canvas.convert("RGB")


def sparsevlm_style_pruned_view(base, selected_indices, color, fade=0.74):
    base = base.convert("RGB")
    faded = Image.blend(base, Image.new("RGB", base.size, "white"), max(0.0, min(float(fade), 1.0)))
    draw = ImageDraw.Draw(faded)
    for index in sorted(set(int(i) for i in selected_indices if 0 <= int(i) <= MAX_PATCH_INDEX)):
        box = patch_box(index, base.width, base.height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        faded.paste(base.crop(box), box)
        draw.rectangle(box, outline=color, width=2)
    return faded


def layer_indices(metadata, layer_idx):
    for layer in metadata.get("layer_token_stats", []) or []:
        if int(layer.get("layer_idx", -1)) == int(layer_idx):
            values = layer.get("selected_original_token_indices", [])
            return [int(v) for v in values if isinstance(v, int) and 0 <= int(v) <= MAX_PATCH_INDEX]
    return []


def draw_wrapped(draw, xy, text, width_chars, font, fill=(20, 20, 20), line_gap=4):
    x, y = xy
    for paragraph in str(text or "").splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width_chars) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            bbox = draw.textbbox((x, y), line, font=font)
            y += bbox[3] - bbox[1] + line_gap
    return y


def make_demo_visualization(case, results, output_dir):
    raw = Image.open(case["image_path"]).convert("RGB")
    clip_view = clip_input_view(raw)
    panel_size = DEMO_PANEL_SIZE
    clip_panel = clip_view.resize((panel_size, panel_size), LANCZOS)
    original_panel = draw_grid(clip_panel)

    margin = 18
    text_width = 360
    gap = 12
    label_height = 30
    header_height = 128
    row_height = panel_size + label_height + 28
    width = margin * 2 + text_width + gap + 4 * panel_size + 3 * gap
    height = header_height + row_height * len(results) + margin

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = default_font(18, bold=True)
    bold_font = default_font(14, bold=True)
    small_font = default_font(12)
    body_font = default_font(13)

    y = margin
    draw.text((margin, y), "Single-image SparseVLM pruning demo", font=title_font, fill=(0, 0, 0))
    y += 28
    y = draw_wrapped(draw, (margin, y), f"Case: {case.get('case_id', '')} | Dataset: {case.get('dataset', '')}", 150, body_font)
    y = draw_wrapped(draw, (margin, y), f"Question: {case['question']}", 150, body_font)
    y = draw_wrapped(draw, (margin, y), f"Ground truth: {case.get('ground_truth', '')}", 150, bold_font, fill=(0, 105, 40))

    x0 = margin + text_width + gap
    for idx, label in enumerate(["CLIP input grid", "Layer 2", "Layer 6", "Layer 15 final"]):
        x = x0 + idx * (panel_size + gap)
        draw.text((x, header_height - 24), label, font=bold_font, fill=(0, 0, 0))

    for row_idx, result in enumerate(results):
        spec = METHOD_SPECS[result["method_key"]]
        row_top = header_height + row_idx * row_height
        text_x = margin
        text_y = row_top + 5
        draw.text((text_x, text_y), result["method_label"], font=bold_font, fill=spec["color"])
        text_y += 22
        text_y = draw_wrapped(draw, (text_x, text_y), f"Answer: {result['answer']}", 42, body_font)
        text_y = draw_wrapped(draw, (text_x, text_y + 2), f"Time: {result['seconds']:.2f}s", 42, small_font, fill=(70, 70, 70))
        if case.get("reference_answers", {}).get(result["method_key"]):
            text_y = draw_wrapped(
                draw,
                (text_x, text_y + 2),
                f"Reference: {case['reference_answers'][result['method_key']]}",
                42,
                small_font,
                fill=(70, 70, 70),
            )

        panels = [original_panel]
        for layer_idx in SPARSE_PRUNING_LOC:
            indices = layer_indices(result["metadata"], layer_idx)
            panels.append(sparsevlm_style_pruned_view(clip_panel, indices, spec["color"], fade=DEMO_FADE))
        for panel_idx, panel in enumerate(panels):
            x = x0 + panel_idx * (panel_size + gap)
            y_panel = row_top + label_height
            canvas.paste(panel, (x, y_panel))
            if panel_idx > 0:
                layer_idx = SPARSE_PRUNING_LOC[panel_idx - 1]
                indices = layer_indices(result["metadata"], layer_idx)
                count_text = f"raw={len(indices)}, unique={len(set(indices))}"
                draw.text((x, y_panel + panel_size + 4), count_text, font=small_font, fill=(70, 70, 70))

    note = (
        "Faded patches are not explicitly retained original CLIP patches. "
        "Merged/recycled tokens are not shown as individual patches."
    )
    draw_wrapped(draw, (margin, height - margin - 22), note, 160, small_font, fill=(85, 85, 85))

    output_path = output_dir / "demo_sparsevlm_style_comparison.png"
    canvas.save(output_path)
    display(canvas)
    print("Saved visualization:", output_path)
    return output_path


print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Repository root:", REPO_ROOT)
print("Demo output root:", DEMO_OUTPUT_ROOT)
```

Demo Cell 4: Choose the one image to demo
```python
# Rerun this cell whenever you want to switch to another image.

DEMO_PRESETS = {
    "pope_ours_recovery_potted_plant": {
        "case_id": "POPE_adversarial_21",
        "dataset": "pope",
        "image_filename": "COCO_val2014_000000211674.jpg",
        "question": "Is there a potted plant in the image?",
        "ground_truth": "yes",
        "defense_use": "Ours-only POPE recovery: SparseVLM-Original and Threshold-Fixed answer no, Ours answers yes.",
        "reference_answers": {
            "dense": "Yes",
            "sparse": "No",
            "ours": "Yes",
            "threshold": "No",
        },
    },
    "pope_ours_recovery_toothbrush": {
        "case_id": "POPE_adversarial_177",
        "dataset": "pope",
        "image_filename": "COCO_val2014_000000288639.jpg",
        "question": "Is there a toothbrush in the image?",
        "ground_truth": "yes",
        "defense_use": "POPE recovery where both Ours and Threshold-Fixed recover the SparseVLM-Original failure.",
        "reference_answers": {
            "dense": "Yes",
            "sparse": "No",
            "ours": "Yes",
            "threshold": "Yes",
        },
    },
    "pope_both_correct_snowboard": {
        "case_id": "POPE_adversarial_1",
        "dataset": "pope",
        "image_filename": "COCO_val2014_000000310196.jpg",
        "question": "Is there a snowboard in the image?",
        "ground_truth": "yes",
        "defense_use": "Both SparseVLM-Original and Ours are correct; use this to show stability on POPE.",
        "reference_answers": {
            "dense": "Yes",
            "sparse": "Yes",
            "ours": "Yes",
            "threshold": "Yes",
        },
    },
    "pope_ours_regression_person_cats": {
        "case_id": "POPE_adversarial_128",
        "dataset": "pope",
        "image_filename": "COCO_val2014_000000075591.jpg",
        "question": "Is there a person in the image?",
        "ground_truth": "no",
        "defense_use": "POPE regression: SparseVLM-Original is correct, while Ours and Threshold-Fixed hallucinate a person.",
        "reference_answers": {
            "dense": "No",
            "sparse": "No",
            "ours": "Yes",
            "threshold": "Yes",
        },
    },
    "pope_both_wrong_book_on_bed": {
        "case_id": "POPE_adversarial_131",
        "dataset": "pope",
        "image_filename": "COCO_val2014_000000075591.jpg",
        "question": "Is there a book in the image?",
        "ground_truth": "yes",
        "defense_use": "Both SparseVLM-Original and Ours are wrong, while Dense and Threshold-Fixed answer correctly.",
        "reference_answers": {
            "dense": "Yes",
            "sparse": "No",
            "ours": "No",
            "threshold": "Yes",
        },
    },
}

# Use one of the preset names above, or set this to "custom".
# Every preset is POPE-only. Make sure the POPE/COCO val2014 images are mounted.
DEMO_PRESET_NAME = "pope_ours_recovery_potted_plant"

# For custom demos, set DEMO_PRESET_NAME = "custom" and edit these fields.
CUSTOM_DEMO = {
    "case_id": "custom_demo",
    "dataset": "pope",
    "image_filename": "COCO_val2014_000000211674.jpg",
    "image_path": "",  # optional absolute path; overrides image_filename when set
    "question": "Is there a potted plant in the image?",
    "ground_truth": "yes",
    "defense_use": "Custom POPE single-image demo.",
    "reference_answers": {},
}

# Keep this False for live demos unless your image is inside a zip and you accept
# the extraction time and disk usage.
DEMO_ALLOW_ZIP_EXTRACT = False

# Run only one image, but compare these methods on that image.
DEMO_METHODS = ["sparse", "ours", "threshold"]

# Generation and visualization settings.
DEMO_USE_SHORT_ANSWER_SUFFIX = True
DEMO_SHORT_ANSWER_SUFFIX = "\nAnswer using a single word or short phrase."
DEMO_MAX_NEW_TOKENS = 32
DEMO_TEMPERATURE = 0.0
DEMO_NUM_BEAMS = 1
DEMO_PANEL_SIZE = 224
DEMO_FADE = 0.74

demo_case_preview = build_demo_case()
display(Markdown(
    f"**Selected demo:** `{demo_case_preview['case_id']}`  \n"
    f"**Dataset:** `{demo_case_preview.get('dataset', '')}`  \n"
    f"**Image:** `{demo_case_preview['image_path']}`  \n"
    f"**Question:** {demo_case_preview['question']}  \n"
    f"**Ground truth:** `{demo_case_preview.get('ground_truth', '')}`  \n"
    f"**Purpose:** {demo_case_preview.get('defense_use', '')}"
))
display(Image.open(demo_case_preview["image_path"]).convert("RGB"))
```

Demo Cell 5: Load the sparse model once
```python
disable_torch_init()
torch.backends.cuda.matmul.allow_tf32 = True

model_name = get_model_name_from_path(MODEL_PATH)
load_start = time.time()
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    model_base=None,
    model_name=model_name,
    load_4bit=False,
    load_8bit=False,
    device="cuda",
    dynamic_sparse=True,
)
model.eval()

print("Loaded:", MODEL_PATH)
print("Conversation mode:", CONV_MODE)
print("Context length:", context_len)
print("Methods available:", ", ".join(METHOD_SPECS))
print("Load seconds:", round(time.time() - load_start, 2))
```

Demo Cell 6: Run the selected image through the demo methods
```python
demo_case, demo_results, demo_output_dir = run_defense_demo()
```

Demo Cell 7: Render the SparseVLM-style patch demo
```python
demo_figure_path = make_demo_visualization(demo_case, demo_results, demo_output_dir)
```

Demo Cell 8: Zip current demo output for download
```python
download_zip = Path("/kaggle/working") / f"{safe_stem(demo_case.get('case_id', 'demo'))}_defense_demo.zip"
if download_zip.exists():
    download_zip.unlink()

created_zip = shutil.make_archive(
    base_name=str(download_zip.with_suffix("")),
    format="zip",
    root_dir=str(demo_output_dir),
)

print("Created:", created_zip)
print("Size MB:", round(Path(created_zip).stat().st_size / (1024 * 1024), 2))
```