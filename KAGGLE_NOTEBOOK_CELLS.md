# SparseVLM Kaggle GQA Production Run

These cells run GQA experiments from `evaluation_protocol_updated.md`.

Kaggle should run one method per session. Change `RUN_IDS_TO_RUN` in Cell 4 to the next run ID, then run the notebook. Each run writes to a run-specific output directory and creates one zip file for download.

This notebook uses a deterministic 5000-sample GQA validation subset. The subset is selected once per run from the same question file using a fixed seed, so every method sees the same questions.

Important dataset note: the local GQA package currently contains images and `val_choices.json`, but the actual question file is required for inference. The notebook first looks for `val_balanced_questions.json` or `val_all_questions.json` in the Kaggle input. If it cannot find one, it tries to download `questions1.2.zip` from the official GQA release into `/kaggle/working`. If Kaggle internet is disabled, upload the GQA questions file as part of the Kaggle Dataset.

Suggested run order:

1. `GQA-SPARSE-ORIG-64`
2. `GQA-OURS-64`
3. `GQA-THRESHOLD-FIXED-64`
4. `GQA-SPARSE-ORIG-128`
5. `GQA-OURS-128`
6. `GQA-THRESHOLD-FIXED-128`
7. `GQA-DENSE-576`
8. `GQA-THRESHOLD-ADAPT-080`
9. `GQA-THRESHOLD-ADAPT-085`
10. `GQA-THRESHOLD-ADAPT-090`

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

pip("install", "--force-reinstall", "numpy==1.26.4", "protobuf", "sentencepiece", "shortuuid", "ijson")
pip("install", "transformers==4.37.2", "tokenizers==0.15.1", "accelerate==0.21.0", "peft==0.7.1")
pip("install", "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13", "markdown2[all]")

print("Restart the Kaggle kernel after this cell finishes, then continue from Cell 3.")
```

Cell 3: Runtime paths and environment
```python
%cd /kaggle/working/thesis/SparseVLMs

import os
import sys
import zipfile
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
GQA_ROOT = find_kaggle_input_dir("GQA")
GQA_IMAGE_DIR = GQA_ROOT / "images"
GQA_EVAL_DIR = GQA_ROOT / "eval"
OUTPUT_BASE_ROOT = "/kaggle/working/gqa_runs"

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
print("GQA root:", GQA_ROOT)
print("GQA eval dir:", GQA_EVAL_DIR)
print("GQA image dir:", GQA_IMAGE_DIR)
print("Output base root:", OUTPUT_BASE_ROOT)
```

Cell 4: GQA production helpers
```python
import csv
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import ijson
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

GQA_SUBSET_SIZE = 5000
GQA_SUBSET_SEED = 20260610
GQA_SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."
GQA_DOWNLOAD_QUESTIONS_IF_MISSING = True
GQA_QUESTIONS_URLS = [
    "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
    "https://nlp.stanford.edu/data/gqa/questions1.2.zip",
]
GQA_QUESTION_FILE_PREFERENCE = [
    "val_balanced_questions.json",
    "val_all_questions.json",
]

GQA_ROOT = Path(GQA_ROOT)
GQA_IMAGE_DIR = Path(GQA_IMAGE_DIR)
GQA_EVAL_DIR = Path(GQA_EVAL_DIR)
OUTPUT_BASE_ROOT = Path(OUTPUT_BASE_ROOT)


# ----- Run catalog -----

@dataclass(frozen=True)
class GqaRun:
    run_id: str
    method_label: str
    selection_method: str
    retained_tokens: int
    prediction_relpath: str
    output_relpath: str
    threshold_tau: float = DEFAULT_THRESHOLD_TAU


GQA_RUNS = [
    GqaRun("GQA-DENSE-576", "Dense / Vanilla", "dense", 576, "gqa/predictions/gqa_dense_576.jsonl", "gqa/gqa_dense_576.csv"),
    GqaRun("GQA-SPARSE-ORIG-128", "SparseVLM-Original", "topk", 128, "gqa/predictions/gqa_sparsevlm_original_128.jsonl", "gqa/gqa_sparsevlm_original_128.csv"),
    GqaRun("GQA-SPARSE-ORIG-64", "SparseVLM-Original", "topk", 64, "gqa/predictions/gqa_sparsevlm_original_64.jsonl", "gqa/gqa_sparsevlm_original_64.csv"),
    GqaRun("GQA-OURS-128", "Ours", "mmr", 128, "gqa/predictions/gqa_ours_128.jsonl", "gqa/gqa_ours_128.csv"),
    GqaRun("GQA-OURS-64", "Ours", "mmr", 64, "gqa/predictions/gqa_ours_64.jsonl", "gqa/gqa_ours_64.csv"),
    GqaRun("GQA-THRESHOLD-FIXED-128", "Threshold-Fixed-k", "threshold_fixed", 128, "gqa/predictions/gqa_threshold_fixed_128.jsonl", "gqa/gqa_threshold_fixed_128.csv"),
    GqaRun("GQA-THRESHOLD-FIXED-64", "Threshold-Fixed-k", "threshold_fixed", 64, "gqa/predictions/gqa_threshold_fixed_64.jsonl", "gqa/gqa_threshold_fixed_64.csv"),
    GqaRun("GQA-THRESHOLD-ADAPT-080", "Threshold-Adaptive", "threshold_adaptive", 64, "gqa/predictions/gqa_threshold_adaptive_tau080.jsonl", "gqa/gqa_threshold_adaptive_tau080.csv", threshold_tau=0.80),
    GqaRun("GQA-THRESHOLD-ADAPT-085", "Threshold-Adaptive", "threshold_adaptive", 64, "gqa/predictions/gqa_threshold_adaptive_tau085.jsonl", "gqa/gqa_threshold_adaptive_tau085.csv", threshold_tau=0.85),
    GqaRun("GQA-THRESHOLD-ADAPT-090", "Threshold-Adaptive", "threshold_adaptive", 64, "gqa/predictions/gqa_threshold_adaptive_tau090.jsonl", "gqa/gqa_threshold_adaptive_tau090.csv", threshold_tau=0.90),
]

# Change this list for each Kaggle session. Keep exactly one run ID here.
RUN_IDS_TO_RUN = [
    "GQA-SPARSE-ORIG-64",
]


def selected_gqa_runs():
    run_by_id = {run.run_id: run for run in GQA_RUNS}
    missing = [run_id for run_id in RUN_IDS_TO_RUN if run_id not in run_by_id]
    if missing:
        raise ValueError(f"Unknown RUN_IDS_TO_RUN entries: {missing}")
    return [run_by_id[run_id] for run_id in RUN_IDS_TO_RUN]


SELECTED_GQA_RUNS = selected_gqa_runs()
if len(SELECTED_GQA_RUNS) != 1:
    raise ValueError(
        "Run exactly one GQA experiment per Kaggle session so each download zip "
        "contains only one run. Set RUN_IDS_TO_RUN to a single run ID."
    )

OUTPUT_ROOT = OUTPUT_BASE_ROOT / SELECTED_GQA_RUNS[0].run_id
DOWNLOAD_ZIP = Path(f"/kaggle/working/{SELECTED_GQA_RUNS[0].run_id}_download.zip")
RESULTS_ROOT = OUTPUT_ROOT / "results"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = RESULTS_ROOT / "summary"
MANIFEST_PATH = OUTPUT_ROOT / "gqa_main_manifest.json"
SUBSET_PATH = OUTPUT_ROOT / f"gqa_subset_seed{GQA_SUBSET_SEED}_n{GQA_SUBSET_SIZE}.jsonl"

for path in [RESULTS_ROOT, LOG_DIR, SUMMARY_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ----- GQA data discovery -----

def find_first_existing(root, names):
    for name in names:
        matches = sorted(path for path in root.rglob(name) if path.is_file())
        if matches:
            return matches[0]
    return None


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print("Using existing download:", destination)
        return destination
    print("Downloading:", url)
    print("Destination:", destination)
    with urllib.request.urlopen(url, timeout=60) as response, open(destination, "wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        start = time.time()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total and (downloaded // (128 * 1024 * 1024)) != ((downloaded - len(chunk)) // (128 * 1024 * 1024)):
                pct = downloaded * 100 / total
                elapsed = max(time.time() - start, 1e-6)
                speed = downloaded / elapsed
                remaining = (total - downloaded) / speed if speed else 0
                print(f"  {pct:.1f}% downloaded, ETA {remaining / 60:.1f} min")
    return destination


def extract_gqa_question_file_from_zip(zip_path):
    target_root = Path("/kaggle/working/gqa_questions")
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for desired in GQA_QUESTION_FILE_PREFERENCE:
            matches = [name for name in names if name.endswith(desired)]
            if matches:
                member = sorted(matches)[0]
                target_path = target_root / desired
                if not target_path.exists():
                    print("Extracting GQA question file:", member)
                    with archive.open(member) as src, open(target_path, "wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                return target_path
    raise FileNotFoundError(f"No validation question file found inside {zip_path}")


def find_gqa_question_file():
    question_path = find_first_existing(GQA_ROOT, GQA_QUESTION_FILE_PREFERENCE)
    if question_path:
        return question_path

    input_root = Path("/kaggle/input")
    zip_matches = sorted(input_root.rglob("questions1.2.zip")) + sorted(GQA_ROOT.rglob("questions1.2.zip"))
    if zip_matches:
        return extract_gqa_question_file_from_zip(zip_matches[0])

    if GQA_DOWNLOAD_QUESTIONS_IF_MISSING:
        download_dir = Path("/kaggle/working/gqa_downloads")
        last_error = None
        for url in GQA_QUESTIONS_URLS:
            try:
                zip_path = download_file(url, download_dir / "questions1.2.zip")
                return extract_gqa_question_file_from_zip(zip_path)
            except Exception as exc:
                last_error = exc
                print("Question download attempt failed:", repr(exc))
        raise FileNotFoundError(
            "Could not find or download GQA validation questions. "
            "Upload val_balanced_questions.json or questions1.2.zip to the Kaggle Dataset."
        ) from last_error

    raise FileNotFoundError(
        "Missing GQA validation questions. Upload val_balanced_questions.json, "
        "val_all_questions.json, or questions1.2.zip to the Kaggle Dataset."
    )


GQA_QUESTION_PATH = find_gqa_question_file()
GQA_CHOICES_PATH = find_first_existing(GQA_ROOT, ["val_choices.json"])

print("GQA question file:", GQA_QUESTION_PATH)
print("GQA choices file:", GQA_CHOICES_PATH)


# ----- Metrics -----

def normalize_gqa_answer(text):
    text = str(text).strip().lower()
    text = text.rstrip(".")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_gqa_metrics(predictions):
    correct = 0
    for item in predictions:
        if normalize_gqa_answer(item["text"]) == normalize_gqa_answer(item["ground_truth"]):
            correct += 1
    total = len(predictions)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


# ----- Sample selection -----

def iter_gqa_questions(question_path):
    with open(question_path, "rb") as f:
        for question_id, question in ijson.kvitems(f, ""):
            yield str(question_id), question


def build_gqa_samples():
    rng = random.Random(GQA_SUBSET_SEED)
    selected = []
    seen = 0
    skipped_missing_image = 0
    skipped_unbalanced = 0

    for question_id, question in iter_gqa_questions(GQA_QUESTION_PATH):
        if not question.get("isBalanced", True):
            skipped_unbalanced += 1
            continue

        question_text = question.get("question", "")
        answer = question.get("answer", "")
        image_id = str(question.get("imageId", ""))
        image_path = GQA_IMAGE_DIR / f"{image_id}.jpg"
        if not question_text or not answer or not image_id:
            continue
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        types = question.get("types", {}) or {}
        sample = {
            "question_id": question_id,
            "question": question_text,
            "prompt": question_text + GQA_SHORT_ANSWER_SUFFIX,
            "ground_truth": str(answer),
            "image_id": image_id,
            "image_path": str(image_path),
            "is_balanced": bool(question.get("isBalanced", True)),
            "structural_type": types.get("structural", ""),
            "semantic_type": types.get("semantic", ""),
            "detailed_type": types.get("detailed", ""),
        }

        seen += 1
        if len(selected) < GQA_SUBSET_SIZE:
            selected.append(sample)
        else:
            replace_idx = rng.randrange(seen)
            if replace_idx < GQA_SUBSET_SIZE:
                selected[replace_idx] = sample

    if len(selected) < GQA_SUBSET_SIZE:
        raise RuntimeError(
            f"Only found {len(selected)} valid GQA samples, expected {GQA_SUBSET_SIZE}. "
            f"Skipped missing images: {skipped_missing_image}; skipped unbalanced: {skipped_unbalanced}."
        )

    selected.sort(key=lambda item: item["question_id"])
    return selected, {
        "eligible_seen": seen,
        "skipped_missing_image": skipped_missing_image,
        "skipped_unbalanced": skipped_unbalanced,
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


GQA_SAMPLES, GQA_SAMPLE_STATS = build_gqa_samples()
write_jsonl(SUBSET_PATH, GQA_SAMPLES)

print("GQA subset count:", len(GQA_SAMPLES))
print("GQA subset seed:", GQA_SUBSET_SEED)
print("GQA subset path:", SUBSET_PATH)
print("GQA sample stats:", GQA_SAMPLE_STATS)
print("Selected GQA run count:", len(SELECTED_GQA_RUNS))
print("Selected run IDs:", [run.run_id for run in SELECTED_GQA_RUNS])


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
    prompt = build_prompt(row["prompt"])
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
        return [f"{sample_id}: {problem}" for problem in problems]

    layer_stats = metadata.get("layer_token_stats")
    if not isinstance(layer_stats, list) or not layer_stats:
        problems.append("missing layer_token_stats")
        return [f"{sample_id}: {problem}" for problem in problems]

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
        "prompt": row["prompt"],
        "raw_question": row["question"],
        "text": answer,
        "normalized_text": normalize_gqa_answer(answer),
        "answer_id": shortuuid.uuid(),
        "model_id": MODEL_PATH,
        "dataset": "gqa",
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "ground_truth": row["ground_truth"],
        "normalized_ground_truth": normalize_gqa_answer(row["ground_truth"]),
        "is_correct": normalize_gqa_answer(answer) == normalize_gqa_answer(row["ground_truth"]),
        "structural_type": row.get("structural_type", ""),
        "semantic_type": row.get("semantic_type", ""),
        "detailed_type": row.get("detailed_type", ""),
        "run_id": run.run_id,
        "metadata": metadata,
        "inference_seconds": elapsed,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gqa_submission_json(path, predictions):
    rows = [
        {
            "questionId": item["question_id"],
            "prediction": normalize_gqa_answer(item["text"]),
        }
        for item in predictions
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


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
        "candidate_pool_factor": CANDIDATE_POOL_FACTOR,
        "sample_count": len(counts),
        "average_retained_tokens": sum(counts) / len(counts),
        "min_retained_tokens": min(counts),
        "max_retained_tokens": max(counts),
    }


def metric_rows_for_run(run, predictions):
    metrics = compute_gqa_metrics(predictions)
    row = {
        "run_id": run.run_id,
        "dataset": "GQA",
        "method": run.method_label,
        "selection_method": run.selection_method,
        "token_setting": run.retained_tokens,
        "threshold_tau": run.threshold_tau if run.selection_method == "threshold_adaptive" else "",
        "sample_count": len(predictions),
        "subset_seed": GQA_SUBSET_SEED,
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
    }
    return [row]


def write_run_metric_csv(run, predictions):
    output_path = RESULTS_ROOT / run.output_relpath
    rows = metric_rows_for_run(run, predictions)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    write_csv(output_path, rows, fieldnames)
    submission_path = output_path.with_name(output_path.stem + "_submission.json")
    write_gqa_submission_json(submission_path, predictions)
    return output_path, submission_path, rows


def write_gqa_summary(manifest):
    summary_rows = []
    for item in manifest:
        stats = {}
        pred_path = Path(item["prediction_file"])
        predictions = [json.loads(line) for line in open(pred_path, "r", encoding="utf-8")]
        if item["selection_method"] != "dense":
            stats = sparse_stats_for_predictions(
                GqaRun(
                    run_id=item["run_id"],
                    method_label=item["method"],
                    selection_method=item["selection_method"],
                    retained_tokens=item["retained_tokens"],
                    threshold_tau=item["threshold_tau"],
                    prediction_relpath="",
                    output_relpath="",
                ),
                predictions,
            )
        metric = item["metrics"][0]
        summary_rows.append({
            "run_id": item["run_id"],
            "dataset": "GQA",
            "method": item["method"],
            "selection_method": item["selection_method"],
            "token_setting": item["retained_tokens"],
            "threshold_tau": item["threshold_tau"] if item["selection_method"] == "threshold_adaptive" else "",
            "sample_count": item["sample_count"],
            "subset_seed": GQA_SUBSET_SEED,
            "accuracy": metric.get("accuracy", ""),
            "correct": metric.get("correct", ""),
            "total": metric.get("total", ""),
            "average_retained_tokens": stats.get("average_retained_tokens", ""),
            "min_retained_tokens": stats.get("min_retained_tokens", ""),
            "max_retained_tokens": stats.get("max_retained_tokens", ""),
            "prediction_file": item["prediction_file"],
            "metric_file": item["metric_file"],
            "submission_file": item["submission_file"],
            "log_file": item["log_file"],
            "status": item["status"],
        })

    fieldnames = [
        "run_id", "dataset", "method", "selection_method", "token_setting",
        "threshold_tau", "sample_count", "subset_seed", "accuracy", "correct",
        "total", "average_retained_tokens", "min_retained_tokens",
        "max_retained_tokens", "prediction_file", "metric_file",
        "submission_file", "log_file", "status",
    ]
    write_csv(SUMMARY_DIR / "gqa_summary.csv", summary_rows, fieldnames)
    write_csv(SUMMARY_DIR / "final_evaluation_table.csv", summary_rows, fieldnames)
    return summary_rows


# ----- Runner -----

def run_one_gqa_experiment(run):
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
            "sample_count": len(GQA_SAMPLES),
            "subset_seed": GQA_SUBSET_SEED,
            "subset_path": str(SUBSET_PATH),
            "time": time.time(),
        }) + "\n")

        for sample_idx, row in enumerate(GQA_SAMPLES, start=1):
            start = time.time()
            answer, metadata = run_generation(row, run)
            elapsed = time.time() - start
            record = prediction_record(row, run, answer, metadata, elapsed)
            predictions.append(record)
            metadata_errors.extend(validate_metadata(run, metadata, row["question_id"]))

            if sample_idx % 50 == 0 or sample_idx == len(GQA_SAMPLES):
                print(f"  {run.run_id}: {sample_idx}/{len(GQA_SAMPLES)} samples")
                write_jsonl(prediction_path, predictions)

            log_file.write(json.dumps({
                "event": "sample_done",
                "run_id": run.run_id,
                "sample_index": sample_idx,
                "question_id": row["question_id"],
                "seconds": elapsed,
                "retained_token_count": metadata.get("retained_token_count"),
                "is_correct": record["is_correct"],
            }) + "\n")
            log_file.flush()
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
    metric_path, submission_path, metric_rows = write_run_metric_csv(run, predictions)

    manifest_record = {
        "run_id": run.run_id,
        "dataset": "gqa",
        "method": run.method_label,
        "selection_method": run.selection_method,
        "retained_tokens": run.retained_tokens,
        "threshold_tau": run.threshold_tau,
        "sample_count": len(predictions),
        "subset_seed": GQA_SUBSET_SEED,
        "subset_file": str(SUBSET_PATH),
        "prediction_file": str(prediction_path),
        "metric_file": str(metric_path),
        "submission_file": str(submission_path),
        "log_file": str(log_path),
        "metrics": metric_rows,
        "status": "ok",
    }
    return manifest_record


def run_gqa_main_experiments():
    manifest = []
    start = time.time()
    for index, run in enumerate(SELECTED_GQA_RUNS, start=1):
        print(f"[{index}/{len(SELECTED_GQA_RUNS)}] {run.run_id} ({run.selection_method}, retained={run.retained_tokens})")
        record = run_one_gqa_experiment(run)
        manifest.append(record)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("  ok:", record["prediction_file"])

    summary_rows = write_gqa_summary(manifest)
    print()
    print("GQA main experiment complete.")
    print("Runs:", len(manifest))
    print("Samples:", len(GQA_SAMPLES))
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
print("GQA subset count:", len(GQA_SAMPLES))
print("Selected GQA run count:", len(SELECTED_GQA_RUNS))
print("Selected run IDs:", [run.run_id for run in SELECTED_GQA_RUNS])
print("Load seconds:", round(time.time() - load_start, 2))
```

Cell 6: Run selected GQA experiment
```python
manifest = run_gqa_main_experiments()
```

Cell 7: Zip current run outputs for download
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
