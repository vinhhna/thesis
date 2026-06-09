# SparseVLM Kaggle POPE Production Run

These cells run POPE experiments from `evaluation_protocol_updated.md`.

Kaggle may not handle many methods in one session, so this notebook runs a configurable subset. By default it runs one non-dense configuration on a balanced POPE subset: 500 adversarial, 500 popular, and 500 random samples. For later Kaggle sessions, change `RUN_IDS_TO_RUN` in Cell 4 to the next run ID.

The output goes to `/kaggle/working/pope_main_results`. The final cell creates `/kaggle/working/pope_main_results_download.zip` so the results can be downloaded as one file from Kaggle.

Dense has already been completed on the full POPE split, so the default order starts from SparseVLM-Original.

Suggested run order:

1. `POPE-SPARSE-ORIG-64`
2. `POPE-OURS-64`
3. `POPE-SPARSE-ORIG-128`
4. `POPE-OURS-128`
5. `POPE-THRESHOLD-FIXED-64`
6. `POPE-THRESHOLD-FIXED-128`
7. `POPE-THRESHOLD-ADAPT-080`
8. `POPE-THRESHOLD-ADAPT-085`
9. `POPE-THRESHOLD-ADAPT-090`

Run Cell 2 only after a fresh clone or package reset, then restart the Kaggle kernel before continuing.

Cell 1: Clone repo
```python
%env USE_FLAX=NO
%env USE_JAX=NO
%env USE_TF=NO
%env WANDB_DISABLED=true
%env WANDB_MODE=disabled

%cd /kaggle/working
!rm -rf thesis
!git clone https://github.com/vinhhna/thesis.git

%cd /kaggle/working/thesis/SparseVLMs
!git rev-parse --short HEAD
```

Cell 2: Install runtime packages
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
    "pandas",
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

print("Restart the Kaggle kernel after this cell finishes, then continue from Cell 3.")
```

Cell 3: Runtime paths and environment
```python
%cd /kaggle/working/thesis/SparseVLMs

import os
import zipfile
import sys
from pathlib import Path

import torch


def find_kaggle_input_dir(required_child):
    input_root = Path("/kaggle/input")
    working_extract_root = Path("/kaggle/working/input_unzipped")

    matches = sorted(path for path in input_root.rglob(required_child) if path.is_dir())
    if matches:
        return matches[0]

    zip_matches = sorted(path for path in input_root.rglob(f"{required_child}.zip") if path.is_file())
    if zip_matches:
        zip_path = zip_matches[0]
        target_root = working_extract_root / zip_path.stem
        marker = target_root / f".{required_child.lower()}_unzip_complete"
        if not marker.exists():
            target_root.mkdir(parents=True, exist_ok=True)
            print("Unzipping Kaggle input archive:", zip_path)
            print("Unzip target:", target_root)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(target_root)
            marker.write_text(str(zip_path), encoding="utf-8")

        extracted_matches = sorted(
            path for path in target_root.rglob(required_child)
            if path.is_dir()
        )
        if extracted_matches:
            return extracted_matches[0]

    available = [str(path) for path in input_root.rglob("*") if path.is_dir() or path.suffix == ".zip"]
    raise FileNotFoundError(
        f"Could not find an extracted {required_child!r} folder or {required_child}.zip. "
        f"Available input paths: {available[:80]}"
    )


REPO_ROOT = "/kaggle/working/thesis"
LLAVA_ROOT = "/kaggle/working/thesis/SparseVLMs"
POPE_ROOT = find_kaggle_input_dir("POPE")
POPE_ANNOTATIONS_DIR = POPE_ROOT / "annotations"
POPE_IMAGE_DIR = POPE_ROOT / "val2014"
OUTPUT_BASE_ROOT = "/kaggle/working/pope_runs"

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONPATH"] = LLAVA_ROOT
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("POPE root:", POPE_ROOT)
print("POPE annotations:", POPE_ANNOTATIONS_DIR)
print("POPE images:", POPE_IMAGE_DIR)
print("Output base root:", OUTPUT_BASE_ROOT)
```

Cell 4: POPE production helpers
```python
import csv
import json
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import shortuuid
import torch
from PIL import Image


def patch_sparse_private_import():
    path = Path(LLAVA_ROOT) / "llava/model/language_model/modelling_sparse_llama.py"
    text = path.read_text(encoding="utf-8")
    anchor = "from .score import *"
    replacement = "from .score import *\nfrom .score import _to_int"
    if replacement not in text:
        if anchor not in text:
            raise RuntimeError(f"Patch anchor not found in {path}")
        text = text.replace(anchor, replacement, 1)
        path.write_text(text, encoding="utf-8")
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
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# ----- Config -----

MODEL_PATH = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "llava_v1"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0
NUM_BEAMS = 1
CANDIDATE_POOL_FACTOR = 2
DEFAULT_THRESHOLD_TAU = 0.85
SPARSE_PRUNING_LOC = [2, 6, 15]
POPE_SAMPLES_PER_CATEGORY = 500

POPE_ROOT = Path(POPE_ROOT)
POPE_ANNOTATIONS_DIR = Path(POPE_ANNOTATIONS_DIR)
POPE_IMAGE_DIR = Path(POPE_IMAGE_DIR)
OUTPUT_BASE_ROOT = Path(OUTPUT_BASE_ROOT)


# ----- Run catalog -----

@dataclass(frozen=True)
class PopeRun:
    run_id: str
    method_label: str
    selection_method: str
    retained_tokens: int
    prediction_relpath: str
    output_relpath: str
    threshold_tau: float = DEFAULT_THRESHOLD_TAU


POPE_RUNS = [
    PopeRun("POPE-DENSE-576", "Dense / Vanilla", "dense", 576, "pope/predictions/pope_dense_576.jsonl", "pope/pope_dense_576.csv"),
    PopeRun("POPE-SPARSE-ORIG-128", "SparseVLM-Original", "topk", 128, "pope/predictions/pope_sparsevlm_original_128.jsonl", "pope/pope_sparsevlm_original_128.csv"),
    PopeRun("POPE-SPARSE-ORIG-64", "SparseVLM-Original", "topk", 64, "pope/predictions/pope_sparsevlm_original_64.jsonl", "pope/pope_sparsevlm_original_64.csv"),
    PopeRun("POPE-OURS-128", "Ours", "mmr", 128, "pope/predictions/pope_ours_128.jsonl", "pope/pope_ours_128.csv"),
    PopeRun("POPE-OURS-64", "Ours", "mmr", 64, "pope/predictions/pope_ours_64.jsonl", "pope/pope_ours_64.csv"),
    PopeRun("POPE-THRESHOLD-FIXED-128", "Threshold-Fixed-k", "threshold_fixed", 128, "pope/predictions/pope_threshold_fixed_128.jsonl", "pope/pope_threshold_fixed_128.csv"),
    PopeRun("POPE-THRESHOLD-FIXED-64", "Threshold-Fixed-k", "threshold_fixed", 64, "pope/predictions/pope_threshold_fixed_64.jsonl", "pope/pope_threshold_fixed_64.csv"),
    PopeRun("POPE-THRESHOLD-ADAPT-080", "Threshold-Adaptive", "threshold_adaptive", 64, "pope/predictions/pope_threshold_adaptive_tau080.jsonl", "pope/pope_threshold_adaptive_tau080.csv", threshold_tau=0.80),
    PopeRun("POPE-THRESHOLD-ADAPT-085", "Threshold-Adaptive", "threshold_adaptive", 64, "pope/predictions/pope_threshold_adaptive_tau085.jsonl", "pope/pope_threshold_adaptive_tau085.csv", threshold_tau=0.85),
    PopeRun("POPE-THRESHOLD-ADAPT-090", "Threshold-Adaptive", "threshold_adaptive", 64, "pope/predictions/pope_threshold_adaptive_tau090.jsonl", "pope/pope_threshold_adaptive_tau090.csv", threshold_tau=0.90),
]

# Change this list for each Kaggle session. Keep it short if Kaggle memory or
# runtime is tight. Dense has already been completed, so the default next run is
# the aggressive SparseVLM baseline.
RUN_IDS_TO_RUN = [
    "POPE-SPARSE-ORIG-64",
]


def selected_pope_runs():
    run_by_id = {run.run_id: run for run in POPE_RUNS}
    missing = [run_id for run_id in RUN_IDS_TO_RUN if run_id not in run_by_id]
    if missing:
        raise ValueError(f"Unknown RUN_IDS_TO_RUN entries: {missing}")
    return [run_by_id[run_id] for run_id in RUN_IDS_TO_RUN]


SELECTED_POPE_RUNS = selected_pope_runs()
if len(SELECTED_POPE_RUNS) != 1:
    raise ValueError(
        "Run exactly one POPE experiment per Kaggle session so each download zip "
        "contains only one run. Set RUN_IDS_TO_RUN to a single run ID."
    )

OUTPUT_ROOT = OUTPUT_BASE_ROOT / SELECTED_POPE_RUNS[0].run_id
DOWNLOAD_ZIP = Path(f"/kaggle/working/{SELECTED_POPE_RUNS[0].run_id}_download.zip")
RESULTS_ROOT = OUTPUT_ROOT / "results"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = RESULTS_ROOT / "summary"
MANIFEST_PATH = OUTPUT_ROOT / "pope_main_manifest.json"

for path in [RESULTS_ROOT, LOG_DIR, SUMMARY_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ----- Metrics -----

def normalize_answer(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pope_label_from_answer(text):
    words = normalize_answer(text).split()
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_pope_metrics(predictions):
    y_true = [1 if normalize_answer(item["ground_truth"]) == "yes" else 0 for item in predictions]
    y_pred = [1 if pope_label_from_answer(item["text"]) == "yes" else 0 for item in predictions]

    tp = sum(1 for pred, gold in zip(y_pred, y_true) if pred == 1 and gold == 1)
    tn = sum(1 for pred, gold in zip(y_pred, y_true) if pred == 0 and gold == 0)
    fp = sum(1 for pred, gold in zip(y_pred, y_true) if pred == 1 and gold == 0)
    fn = sum(1 for pred, gold in zip(y_pred, y_true) if pred == 0 and gold == 1)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, len(y_true))
    yes_ratio = safe_div(sum(y_pred), len(y_pred))
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "yes_ratio": yes_ratio,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ----- Dataset -----

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pope_samples():
    samples = []
    for category in ["adversarial", "popular", "random"]:
        path = POPE_ANNOTATIONS_DIR / f"coco_pope_{category}.json"
        rows = read_jsonl(path)
        if POPE_SAMPLES_PER_CATEGORY is not None:
            rows = rows[:POPE_SAMPLES_PER_CATEGORY]
        for row in rows:
            image_path = POPE_IMAGE_DIR / row["image"]
            if not image_path.exists():
                raise FileNotFoundError(f"Missing POPE image: {image_path}")
            samples.append({
                "question_id": f"{category}_{row['question_id']}",
                "pope_question_id": row["question_id"],
                "pope_category": category,
                "question": row["text"],
                "image": row["image"],
                "image_path": str(image_path),
                "ground_truth": row["label"],
            })
    return samples


POPE_SAMPLES = build_pope_samples()
print("POPE sample count:", len(POPE_SAMPLES))
print("POPE category counts:", {
    category: sum(1 for item in POPE_SAMPLES if item["pope_category"] == category)
    for category in ["adversarial", "popular", "random"]
})
print("POPE samples per category cap:", POPE_SAMPLES_PER_CATEGORY)
print("All POPE configured run count:", len(POPE_RUNS))
print("Selected POPE run count:", len(SELECTED_POPE_RUNS))
print("Selected run IDs:", [run.run_id for run in SELECTED_POPE_RUNS])


# ----- Inference -----

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


def run_generation(row, run):
    image = Image.open(row["image_path"]).convert("RGB")
    prompt = build_prompt(row["question"])
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    images_tensor = prepare_image_tensor(image)

    sparse_core = model.get_model()
    original_pruning_loc = list(getattr(sparse_core, "pruning_loc", SPARSE_PRUNING_LOC))

    if run.selection_method == "dense":
        sparse_core.pruning_loc = []
        sparse_core.last_sparse_metadata = {}
        generate_selection_method = "topk"
    else:
        sparse_core.pruning_loc = SPARSE_PRUNING_LOC
        sparse_core.last_sparse_metadata = {}
        generate_selection_method = run.selection_method

    try:
        generation_kwargs = {
            "do_sample": TEMPERATURE > 0,
            "num_beams": NUM_BEAMS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": True,
        }
        if TEMPERATURE > 0:
            generation_kwargs["temperature"] = TEMPERATURE

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                retained_tokens=run.retained_tokens,
                selection_method=generate_selection_method,
                threshold_tau=run.threshold_tau,
                candidate_pool_factor=CANDIDATE_POOL_FACTOR,
                **generation_kwargs,
            )
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        metadata = deepcopy(getattr(sparse_core, "last_sparse_metadata", {}))
    finally:
        sparse_core.pruning_loc = original_pruning_loc

    if run.selection_method == "dense":
        metadata = {
            "selection_method": "dense",
            "retained_tokens": 576,
            "retained_token_count": 576,
            "layer_token_stats": [],
        }
    else:
        metadata.setdefault("selection_method", run.selection_method)
        metadata.setdefault("retained_tokens", run.retained_tokens)
        metadata.setdefault("threshold_tau", run.threshold_tau)
        metadata.setdefault("candidate_pool_factor", CANDIDATE_POOL_FACTOR)

    return answer, metadata


# ----- Validation and output -----

def validate_metadata(run, metadata, sample_id):
    problems = []
    if metadata.get("selection_method") != run.selection_method:
        problems.append(f"selection_method={metadata.get('selection_method')} expected {run.selection_method}")

    if int(metadata.get("retained_tokens", -1)) != int(run.retained_tokens):
        problems.append(f"retained_tokens={metadata.get('retained_tokens')} expected {run.retained_tokens}")

    if run.selection_method == "dense":
        if metadata.get("retained_token_count") != 576:
            problems.append("dense retained_token_count should be 576")
        return problems

    layer_stats = metadata.get("layer_token_stats")
    if not isinstance(layer_stats, list) or not layer_stats:
        problems.append("missing layer_token_stats")
        return problems

    if metadata.get("retained_token_count") is None:
        problems.append("missing retained_token_count")

    if not isinstance(metadata.get("selected_original_token_indices"), list):
        problems.append("missing selected_original_token_indices")

    for layer in layer_stats:
        if layer.get("selection_method") != run.selection_method:
            problems.append(f"layer {layer.get('layer_idx')} used {layer.get('selection_method')}")
        selected_count = int(layer.get("selected_count", -1))
        per_layer_budget = int(layer.get("per_layer_budget", -1))
        if selected_count < 0 or per_layer_budget < 0:
            problems.append(f"layer {layer.get('layer_idx')} has invalid selected_count or budget")
        if run.selection_method == "threshold_fixed" and selected_count != per_layer_budget:
            problems.append(f"threshold_fixed layer {layer.get('layer_idx')} selected {selected_count}, budget {per_layer_budget}")
        if run.selection_method == "threshold_adaptive" and "threshold_tau" not in layer:
            problems.append(f"threshold_adaptive layer {layer.get('layer_idx')} missing threshold_tau")

    return [f"{sample_id}: {problem}" for problem in problems]


def prediction_record(row, run, answer, metadata, elapsed):
    return {
        "question_id": row["question_id"],
        "pope_question_id": row["pope_question_id"],
        "pope_category": row["pope_category"],
        "prompt": row["question"],
        "text": answer,
        "answer_id": shortuuid.uuid(),
        "model_id": MODEL_PATH,
        "dataset": "pope",
        "image": row["image"],
        "image_path": row["image_path"],
        "ground_truth": row["ground_truth"],
        "run_id": run.run_id,
        "metadata": metadata,
        "inference_seconds": elapsed,
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_rows_for_run(run, predictions):
    rows = []
    for split_name in ["all", "adversarial", "popular", "random"]:
        if split_name == "all":
            split_predictions = predictions
        else:
            split_predictions = [
                item for item in predictions
                if item["pope_category"] == split_name
            ]
        metrics = compute_pope_metrics(split_predictions)
        rows.append({
            "run_id": run.run_id,
            "dataset": "POPE",
            "pope_split": split_name,
            "method": run.method_label,
            "selection_method": run.selection_method,
            "token_setting": run.retained_tokens,
            "threshold_tau": run.threshold_tau if run.selection_method == "threshold_adaptive" else "",
            "sample_count": len(split_predictions),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "yes_ratio": metrics["yes_ratio"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        })
    return rows


def sparse_stats_for_predictions(run, predictions):
    counts = [
        int(item["metadata"]["retained_token_count"])
        for item in predictions
        if item.get("metadata", {}).get("retained_token_count") is not None
    ]
    if not counts:
        return {}
    return {
        "selection_method": run.selection_method,
        "retained_tokens": run.retained_tokens,
        "threshold_tau": run.threshold_tau,
        "sample_count": len(counts),
        "average_retained_tokens": sum(counts) / len(counts),
        "min_retained_tokens": min(counts),
        "max_retained_tokens": max(counts),
    }


def write_run_metric_csv(run, predictions):
    output_path = RESULTS_ROOT / run.output_relpath
    rows = metric_rows_for_run(run, predictions)
    fieldnames = [
        "run_id", "dataset", "pope_split", "method", "selection_method",
        "token_setting", "threshold_tau", "sample_count", "accuracy", "f1", "precision",
        "recall", "yes_ratio", "tp", "tn", "fp", "fn",
    ]
    write_csv(output_path, rows, fieldnames)
    return output_path, rows


def write_pope_summary(manifest):
    summary_rows = []
    for item in manifest:
        stats = {}
        pred_path = Path(item["prediction_file"])
        predictions = [json.loads(line) for line in open(pred_path, "r", encoding="utf-8")]
        if item["selection_method"] != "dense":
            stats = sparse_stats_for_predictions(
                PopeRun(
                    run_id=item["run_id"],
                    method_label=item["method"],
                    selection_method=item["selection_method"],
                    retained_tokens=item["retained_tokens"],
                    threshold_tau=item.get("threshold_tau", DEFAULT_THRESHOLD_TAU),
                    prediction_relpath="",
                    output_relpath="",
                ),
                predictions,
            )

        for metric in item["metrics"]:
            summary_rows.append({
                "run_id": item["run_id"],
                "dataset": "POPE",
                "pope_split": metric["pope_split"],
                "method": item["method"],
                "selection_method": item["selection_method"],
                "token_setting": item["retained_tokens"],
                "sample_count": metric["sample_count"],
                "accuracy": metric["accuracy"],
                "f1": metric["f1"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "yes_ratio": metric["yes_ratio"],
                "average_retained_tokens": stats.get("average_retained_tokens", ""),
                "min_retained_tokens": stats.get("min_retained_tokens", ""),
                "max_retained_tokens": stats.get("max_retained_tokens", ""),
                "threshold_tau": item.get("threshold_tau", ""),
                "prediction_file": item["prediction_file"],
                "metric_file": item["metric_file"],
                "log_file": item["log_file"],
                "status": item["status"],
            })

    fieldnames = [
        "run_id", "dataset", "pope_split", "method", "selection_method",
        "token_setting", "sample_count", "accuracy", "f1", "precision",
        "recall", "yes_ratio", "average_retained_tokens",
        "min_retained_tokens", "max_retained_tokens", "threshold_tau",
        "prediction_file", "metric_file", "log_file", "status",
    ]
    write_csv(SUMMARY_DIR / "pope_summary.csv", summary_rows, fieldnames)
    write_csv(SUMMARY_DIR / "final_evaluation_table.csv", summary_rows, fieldnames)
    return summary_rows


# ----- Runner -----

def run_one_pope_experiment(run):
    predictions = []
    metadata_errors = []
    log_path = LOG_DIR / f"{run.run_id}.jsonl"
    prediction_path = RESULTS_ROOT / run.prediction_relpath

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "event": "run_start",
            "run_id": run.run_id,
            "selection_method": run.selection_method,
            "retained_tokens": run.retained_tokens,
            "threshold_tau": run.threshold_tau,
            "pope_samples_per_category": POPE_SAMPLES_PER_CATEGORY,
            "sample_count": len(POPE_SAMPLES),
            "time": time.time(),
        }) + "\n")

        for sample_idx, row in enumerate(POPE_SAMPLES, start=1):
            start = time.time()
            answer, metadata = run_generation(row, run)
            elapsed = time.time() - start
            record = prediction_record(row, run, answer, metadata, elapsed)
            predictions.append(record)
            metadata_errors.extend(validate_metadata(run, metadata, row["question_id"]))

            if sample_idx % 50 == 0 or sample_idx == len(POPE_SAMPLES):
                print(f"  {run.run_id}: {sample_idx}/{len(POPE_SAMPLES)} samples")

            log_file.write(json.dumps({
                "event": "sample_done",
                "run_id": run.run_id,
                "sample_index": sample_idx,
                "question_id": row["question_id"],
                "pope_category": row["pope_category"],
                "seconds": elapsed,
                "retained_token_count": metadata.get("retained_token_count"),
            }) + "\n")
            log_file.flush()
            if sample_idx % 50 == 0:
                write_jsonl(prediction_path, predictions)
            torch.cuda.empty_cache()

        log_file.write(json.dumps({
            "event": "run_end",
            "run_id": run.run_id,
            "sample_count": len(predictions),
            "metadata_error_count": len(metadata_errors),
            "time": time.time(),
        }) + "\n")

    if metadata_errors:
        raise RuntimeError(f"{run.run_id} metadata validation failed:\n" + "\n".join(metadata_errors[:20]))

    write_jsonl(prediction_path, predictions)
    metric_path, metric_rows = write_run_metric_csv(run, predictions)

    manifest_record = {
        "run_id": run.run_id,
        "dataset": "pope",
        "method": run.method_label,
        "selection_method": run.selection_method,
        "retained_tokens": run.retained_tokens,
        "threshold_tau": run.threshold_tau,
        "sample_count": len(predictions),
        "prediction_file": str(prediction_path),
        "metric_file": str(metric_path),
        "log_file": str(log_path),
        "metrics": metric_rows,
        "status": "ok",
    }
    return manifest_record


def run_pope_main_experiments():
    manifest = []
    start = time.time()
    for index, run in enumerate(SELECTED_POPE_RUNS, start=1):
        print(f"[{index}/{len(SELECTED_POPE_RUNS)}] {run.run_id} ({run.selection_method}, retained={run.retained_tokens})")
        record = run_one_pope_experiment(run)
        manifest.append(record)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("  ok:", record["prediction_file"])

    summary_rows = write_pope_summary(manifest)
    print()
    print("POPE main experiment complete.")
    print("Runs:", len(manifest))
    print("Samples:", len(POPE_SAMPLES))
    print("Output root:", OUTPUT_ROOT)
    print("Manifest:", MANIFEST_PATH)
    print("Summary rows:", len(summary_rows))
    print("Total minutes:", round((time.time() - start) / 60, 2))
    return manifest
```

Cell 5: Load model
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
print("Max new tokens:", MAX_NEW_TOKENS)
print("POPE sample count:", len(POPE_SAMPLES))
print("Selected POPE run count:", len(SELECTED_POPE_RUNS))
print("Selected run IDs:", [run.run_id for run in SELECTED_POPE_RUNS])
print("Load seconds:", round(time.time() - load_start, 2))
```

Cell 6: Run selected POPE main fixed-budget experiment(s)
```python
manifest = run_pope_main_experiments()
```

Cell 7: Zip all outputs for download
```python
import shutil
from pathlib import Path

output_root = Path(OUTPUT_ROOT)
download_zip = Path(DOWNLOAD_ZIP)
if download_zip.exists():
    download_zip.unlink()

archive_base = download_zip.with_suffix("")
created_zip = shutil.make_archive(
    base_name=str(archive_base),
    format="zip",
    root_dir=str(output_root),
)

print("Created:", created_zip)
print("Size MB:", round(Path(created_zip).stat().st_size / (1024 * 1024), 2))
```
