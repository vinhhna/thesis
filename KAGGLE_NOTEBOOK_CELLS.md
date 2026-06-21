# SparseVLM Kaggle Failure-Mining Production Run

These cells run the final failure-mining experiments from `evaluation_protocol_updated.md`.

The evaluation set is already stored in the repository:

- `failure_mining_set.csv`
- `data/sample_images/`

It contains 100 cases: 70 GQA, 20 VQAv2, and 10 TextVQA. No additional Kaggle Dataset input is required.

Run one method per Kaggle session. Change only `RUN_IDS_TO_RUN` in Cell 4 for the next experiment. Every run writes to a unique directory and creates one zip file under `/kaggle/working`.

Suggested run order:

1. `FM-DENSE-576`
2. `FM-SPARSE-ORIG-64`
3. `FM-OURS-64`
4. `FM-THRESHOLD-FIXED-64`
5. `FM-SPARSE-ORIG-128`
6. `FM-OURS-128`
7. `FM-THRESHOLD-FIXED-128`
8. `FM-THRESHOLD-ADAPT-080`
9. `FM-THRESHOLD-ADAPT-085`
10. `FM-THRESHOLD-ADAPT-090`

Recovery rates require paired predictions from multiple methods. Each run therefore exports a case-analysis CSV with stable `case_id` values. The final recovery comparison will be calculated after all run zips are downloaded.

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
import sys
from pathlib import Path

import torch


REPO_ROOT = Path("/kaggle/working/thesis")
LLAVA_ROOT = REPO_ROOT / "SparseVLMs"
FAILURE_MINING_CSV = REPO_ROOT / "failure_mining_set.csv"
FAILURE_MINING_IMAGE_ROOT = REPO_ROOT
OUTPUT_BASE_ROOT = Path("/kaggle/working/failure_mining_runs")

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONPATH"] = str(LLAVA_ROOT)
if str(LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ROOT))

if not FAILURE_MINING_CSV.exists():
    raise FileNotFoundError(f"Missing failure-mining CSV: {FAILURE_MINING_CSV}")

print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Repository root:", REPO_ROOT)
print("Failure-mining CSV:", FAILURE_MINING_CSV)
print("Output base root:", OUTPUT_BASE_ROOT)
```

Cell 4: Failure-mining production helpers
```python
import csv
import json
import re
import sys
import time
from collections import Counter
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
SHORT_ANSWER_SUFFIX = "\nAnswer using a single word or short phrase."
CHECKPOINT_EVERY = 10
EXPECTED_SAMPLE_COUNT = 100

REPO_ROOT = Path(REPO_ROOT)
FAILURE_MINING_CSV = Path(FAILURE_MINING_CSV)
FAILURE_MINING_IMAGE_ROOT = Path(FAILURE_MINING_IMAGE_ROOT)
OUTPUT_BASE_ROOT = Path(OUTPUT_BASE_ROOT)


# ----- Run catalog -----

@dataclass(frozen=True)
class FailureMiningRun:
    run_id: str
    method_label: str
    selection_method: str
    retained_tokens: int
    prediction_relpath: str
    output_relpath: str
    threshold_tau: float = DEFAULT_THRESHOLD_TAU


FAILURE_MINING_RUNS = [
    FailureMiningRun(
        "FM-DENSE-576",
        "Dense / Vanilla",
        "dense",
        576,
        "failure_mining/predictions/failure_mining_dense_576.jsonl",
        "failure_mining/failure_mining_dense_576.csv",
    ),
    FailureMiningRun(
        "FM-SPARSE-ORIG-128",
        "SparseVLM-Original",
        "topk",
        128,
        "failure_mining/predictions/failure_mining_sparsevlm_original_128.jsonl",
        "failure_mining/failure_mining_sparsevlm_original_128.csv",
    ),
    FailureMiningRun(
        "FM-SPARSE-ORIG-64",
        "SparseVLM-Original",
        "topk",
        64,
        "failure_mining/predictions/failure_mining_sparsevlm_original_64.jsonl",
        "failure_mining/failure_mining_sparsevlm_original_64.csv",
    ),
    FailureMiningRun(
        "FM-OURS-128",
        "Ours",
        "mmr",
        128,
        "failure_mining/predictions/failure_mining_ours_128.jsonl",
        "failure_mining/failure_mining_ours_128.csv",
    ),
    FailureMiningRun(
        "FM-OURS-64",
        "Ours",
        "mmr",
        64,
        "failure_mining/predictions/failure_mining_ours_64.jsonl",
        "failure_mining/failure_mining_ours_64.csv",
    ),
    FailureMiningRun(
        "FM-THRESHOLD-FIXED-128",
        "Threshold-Fixed-k",
        "threshold_fixed",
        128,
        "failure_mining/predictions/failure_mining_threshold_fixed_128.jsonl",
        "failure_mining/failure_mining_threshold_fixed_128.csv",
    ),
    FailureMiningRun(
        "FM-THRESHOLD-FIXED-64",
        "Threshold-Fixed-k",
        "threshold_fixed",
        64,
        "failure_mining/predictions/failure_mining_threshold_fixed_64.jsonl",
        "failure_mining/failure_mining_threshold_fixed_64.csv",
    ),
    FailureMiningRun(
        "FM-THRESHOLD-ADAPT-080",
        "Threshold-Adaptive",
        "threshold_adaptive",
        64,
        "failure_mining/predictions/failure_mining_threshold_adaptive_tau080.jsonl",
        "failure_mining/failure_mining_threshold_adaptive_tau080.csv",
        threshold_tau=0.80,
    ),
    FailureMiningRun(
        "FM-THRESHOLD-ADAPT-085",
        "Threshold-Adaptive",
        "threshold_adaptive",
        64,
        "failure_mining/predictions/failure_mining_threshold_adaptive_tau085.jsonl",
        "failure_mining/failure_mining_threshold_adaptive_tau085.csv",
        threshold_tau=0.85,
    ),
    FailureMiningRun(
        "FM-THRESHOLD-ADAPT-090",
        "Threshold-Adaptive",
        "threshold_adaptive",
        64,
        "failure_mining/predictions/failure_mining_threshold_adaptive_tau090.jsonl",
        "failure_mining/failure_mining_threshold_adaptive_tau090.csv",
        threshold_tau=0.90,
    ),
]


# Change this single value for each Kaggle session.
RUN_IDS_TO_RUN = [
    "FM-DENSE-576",
]


def selected_failure_mining_runs():
    run_by_id = {run.run_id: run for run in FAILURE_MINING_RUNS}
    missing = [run_id for run_id in RUN_IDS_TO_RUN if run_id not in run_by_id]
    if missing:
        raise ValueError(f"Unknown RUN_IDS_TO_RUN entries: {missing}")
    return [run_by_id[run_id] for run_id in RUN_IDS_TO_RUN]


SELECTED_RUNS = selected_failure_mining_runs()
if len(SELECTED_RUNS) != 1:
    raise ValueError(
        "Run exactly one failure-mining experiment per Kaggle session. "
        "Set RUN_IDS_TO_RUN to one run ID."
    )

CURRENT_RUN = SELECTED_RUNS[0]
OUTPUT_ROOT = OUTPUT_BASE_ROOT / CURRENT_RUN.run_id
DOWNLOAD_ZIP = Path(f"/kaggle/working/{CURRENT_RUN.run_id}_download.zip")
RESULTS_ROOT = OUTPUT_ROOT / "results"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = RESULTS_ROOT / "summary"
MANIFEST_PATH = OUTPUT_ROOT / "failure_mining_manifest.json"
DATASET_SNAPSHOT_PATH = OUTPUT_ROOT / "failure_mining_set_snapshot.csv"

for path in [RESULTS_ROOT, LOG_DIR, SUMMARY_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ----- General file helpers -----

def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----- Dataset loading and validation -----

def load_failure_mining_samples():
    with open(FAILURE_MINING_CSV, "r", encoding="utf-8", newline="") as f:
        source_rows = list(csv.DictReader(f))

    required_columns = {
        "case_id",
        "dataset",
        "image_path",
        "question",
        "ground_truth",
        "question_type",
        "note",
    }
    actual_columns = set(source_rows[0].keys()) if source_rows else set()
    missing_columns = sorted(required_columns - actual_columns)
    if missing_columns:
        raise RuntimeError(f"Failure-mining CSV is missing columns: {missing_columns}")

    if len(source_rows) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLE_COUNT} failure-mining rows, found {len(source_rows)}"
        )

    seen_ids = set()
    samples = []
    for row_number, row in enumerate(source_rows, start=2):
        case_id = row["case_id"].strip()
        if not case_id:
            raise RuntimeError(f"Missing case_id on CSV row {row_number}")
        if case_id in seen_ids:
            raise RuntimeError(f"Duplicate case_id: {case_id}")
        seen_ids.add(case_id)

        question = row["question"].strip()
        ground_truth = row["ground_truth"].strip()
        if not question or not ground_truth:
            raise RuntimeError(f"{case_id}: question or ground_truth is empty")

        relative_image_path = Path(row["image_path"])
        image_path = (FAILURE_MINING_IMAGE_ROOT / relative_image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"{case_id}: missing image {image_path}")

        samples.append({
            "case_id": case_id,
            "dataset": row["dataset"].strip(),
            "image_path": str(image_path),
            "source_image_path": relative_image_path.as_posix(),
            "question": question,
            "prompt": question + SHORT_ANSWER_SUFFIX,
            "ground_truth": ground_truth,
            "question_type": row["question_type"].strip(),
            "note": row.get("note", "").strip(),
        })

    return samples, source_rows


FAILURE_MINING_SAMPLES, FAILURE_MINING_SOURCE_ROWS = load_failure_mining_samples()
write_csv(
    DATASET_SNAPSHOT_PATH,
    FAILURE_MINING_SOURCE_ROWS,
    list(FAILURE_MINING_SOURCE_ROWS[0].keys()),
)

print("Failure-mining sample count:", len(FAILURE_MINING_SAMPLES))
print("Dataset distribution:", dict(Counter(row["dataset"] for row in FAILURE_MINING_SAMPLES)))
print("Question-type distribution:", dict(Counter(row["question_type"] for row in FAILURE_MINING_SAMPLES)))
print("Selected run:", CURRENT_RUN.run_id)
print("Output root:", OUTPUT_ROOT)
print("Download zip:", DOWNLOAD_ZIP)


# ----- Answer scoring -----

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_token(token):
    token = NUMBER_WORDS.get(token, token)
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        token = token[:-1]
    return token


def normalize_answer(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = re.sub(r"\s+", " ", text).strip().split()
    return " ".join(normalize_token(token) for token in tokens)


def binary_label(text):
    words = normalize_answer(text).split()
    if "no" in words or "not" in words or "never" in words:
        return "no"
    if "yes" in words:
        return "yes"
    return ""


def exact_or_phrase_match(prediction, target):
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if not pred or not gold:
        return False
    if gold in {"yes", "no"}:
        return binary_label(pred) == gold
    if pred == gold:
        return True
    return re.search(rf"(?:^| ){re.escape(gold)}(?: |$)", pred) is not None


def classify_failure(prediction, target):
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if exact_or_phrase_match(prediction, target):
        return "correct"
    if not pred:
        return "empty_answer"
    if gold in {"yes", "no"}:
        return "binary_mismatch"
    return "open_answer_mismatch"


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


# ----- Metadata validation -----

def validate_metadata(run, metadata, sample_id):
    problems = []
    if metadata.get("selection_method") != run.selection_method:
        problems.append(
            f"selection_method={metadata.get('selection_method')} expected {run.selection_method}"
        )

    if int(metadata.get("retained_tokens", -1)) != int(run.retained_tokens):
        problems.append(
            f"retained_tokens={metadata.get('retained_tokens')} expected {run.retained_tokens}"
        )

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
            problems.append(
                f"layer {layer.get('layer_idx')} used {layer.get('selection_method')}"
            )
        selected_count = int(layer.get("selected_count", -1))
        per_layer_budget = int(layer.get("per_layer_budget", -1))
        if selected_count < 0 or per_layer_budget < 0:
            problems.append(
                f"layer {layer.get('layer_idx')} has invalid selected_count or budget"
            )
        if run.selection_method == "threshold_fixed" and selected_count != per_layer_budget:
            problems.append(
                f"threshold_fixed layer {layer.get('layer_idx')} selected "
                f"{selected_count}, budget {per_layer_budget}"
            )
        if run.selection_method == "threshold_adaptive" and "threshold_tau" not in layer:
            problems.append(
                f"threshold_adaptive layer {layer.get('layer_idx')} missing threshold_tau"
            )

    return [f"{sample_id}: {problem}" for problem in problems]


# ----- Prediction and metrics -----

def prediction_record(row, run, answer, metadata, elapsed):
    is_correct = exact_or_phrase_match(answer, row["ground_truth"])
    return {
        "case_id": row["case_id"],
        "question_id": row["case_id"],
        "prompt": row["prompt"],
        "raw_question": row["question"],
        "text": answer,
        "normalized_text": normalize_answer(answer),
        "answer_id": shortuuid.uuid(),
        "model_id": MODEL_PATH,
        "dataset": row["dataset"],
        "image_path": row["source_image_path"],
        "ground_truth": row["ground_truth"],
        "normalized_ground_truth": normalize_answer(row["ground_truth"]),
        "question_type": row["question_type"],
        "note": row["note"],
        "is_correct": is_correct,
        "failure_label": classify_failure(answer, row["ground_truth"]),
        "run_id": run.run_id,
        "metadata": metadata,
        "inference_seconds": elapsed,
    }


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


def aggregate_metrics(predictions):
    correct = sum(bool(item["is_correct"]) for item in predictions)
    counts = Counter(item["failure_label"] for item in predictions)
    total = len(predictions)
    return {
        "sample_count": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "incorrect": total - correct,
        "correct_count": counts.get("correct", 0),
        "binary_mismatch_count": counts.get("binary_mismatch", 0),
        "open_answer_mismatch_count": counts.get("open_answer_mismatch", 0),
        "empty_answer_count": counts.get("empty_answer", 0),
    }


def metric_row(run, predictions, scope, scope_value):
    metrics = aggregate_metrics(predictions)
    stats = sparse_stats_for_predictions(run, predictions)
    return {
        "run_id": run.run_id,
        "dataset": "Failure-mining",
        "method": run.method_label,
        "selection_method": run.selection_method,
        "token_setting": run.retained_tokens,
        "threshold_tau": run.threshold_tau if run.selection_method == "threshold_adaptive" else "",
        "scope": scope,
        "scope_value": scope_value,
        **metrics,
        "average_retained_tokens": stats.get("average_retained_tokens", ""),
        "min_retained_tokens": stats.get("min_retained_tokens", ""),
        "max_retained_tokens": stats.get("max_retained_tokens", ""),
    }


def metric_rows_for_run(run, predictions):
    rows = [metric_row(run, predictions, "overall", "all")]

    for dataset_name in sorted({item["dataset"] for item in predictions}):
        group = [item for item in predictions if item["dataset"] == dataset_name]
        rows.append(metric_row(run, group, "dataset", dataset_name))

    for question_type in sorted({item["question_type"] for item in predictions}):
        group = [item for item in predictions if item["question_type"] == question_type]
        rows.append(metric_row(run, group, "question_type", question_type))

    return rows


def case_analysis_rows(predictions):
    return [
        {
            "case_id": item["case_id"],
            "dataset": item["dataset"],
            "question_type": item["question_type"],
            "image_path": item["image_path"],
            "question": item["raw_question"],
            "ground_truth": item["ground_truth"],
            "prediction": item["text"],
            "is_correct": item["is_correct"],
            "failure_label": item["failure_label"],
            "retained_token_count": item["metadata"].get("retained_token_count", ""),
            "inference_seconds": item["inference_seconds"],
            "manual_failure_cause": "",
            "review_notes": "",
        }
        for item in predictions
    ]


def write_run_outputs(run, predictions):
    metric_path = RESULTS_ROOT / run.output_relpath
    metric_rows = metric_rows_for_run(run, predictions)
    metric_fields = list(metric_rows[0].keys())
    write_csv(metric_path, metric_rows, metric_fields)

    analysis_path = metric_path.with_name(metric_path.stem + "_case_analysis.csv")
    analysis_rows = case_analysis_rows(predictions)
    write_csv(analysis_path, analysis_rows, list(analysis_rows[0].keys()))

    stats_path = ""
    if run.selection_method == "threshold_adaptive":
        stats = sparse_stats_for_predictions(run, predictions)
        stats_path = metric_path.with_name(metric_path.stem + "_sparse_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return metric_path, analysis_path, stats_path, metric_rows


def write_failure_mining_summary(manifest):
    summary_rows = []
    for item in manifest:
        overall = next(
            row for row in item["metrics"]
            if row["scope"] == "overall" and row["scope_value"] == "all"
        )
        summary_rows.append({
            "run_id": item["run_id"],
            "dataset": "Failure-mining",
            "method": item["method"],
            "selection_method": item["selection_method"],
            "token_setting": item["retained_tokens"],
            "threshold_tau": (
                item["threshold_tau"]
                if item["selection_method"] == "threshold_adaptive"
                else ""
            ),
            "sample_count": item["sample_count"],
            "accuracy": overall["accuracy"],
            "correct": overall["correct"],
            "incorrect": overall["incorrect"],
            "binary_mismatch_count": overall["binary_mismatch_count"],
            "open_answer_mismatch_count": overall["open_answer_mismatch_count"],
            "empty_answer_count": overall["empty_answer_count"],
            "average_retained_tokens": overall["average_retained_tokens"],
            "min_retained_tokens": overall["min_retained_tokens"],
            "max_retained_tokens": overall["max_retained_tokens"],
            "prediction_file": item["prediction_file"],
            "metric_file": item["metric_file"],
            "case_analysis_file": item["case_analysis_file"],
            "log_file": item["log_file"],
            "status": item["status"],
        })

    fieldnames = list(summary_rows[0].keys())
    write_csv(SUMMARY_DIR / "failure_mining_summary.csv", summary_rows, fieldnames)
    write_csv(SUMMARY_DIR / "final_evaluation_table.csv", summary_rows, fieldnames)
    return summary_rows


# ----- Runner -----

def run_one_failure_mining_experiment(run):
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
            "sample_count": len(FAILURE_MINING_SAMPLES),
            "dataset_snapshot": str(DATASET_SNAPSHOT_PATH),
            "time": time.time(),
        }) + "\n")

        for sample_idx, row in enumerate(FAILURE_MINING_SAMPLES, start=1):
            start = time.time()
            answer, metadata = run_generation(row, run)
            elapsed = time.time() - start
            record = prediction_record(row, run, answer, metadata, elapsed)
            predictions.append(record)
            metadata_errors.extend(validate_metadata(run, metadata, row["case_id"]))

            log_file.write(json.dumps({
                "event": "sample_done",
                "run_id": run.run_id,
                "sample_index": sample_idx,
                "case_id": row["case_id"],
                "seconds": elapsed,
                "retained_token_count": metadata.get("retained_token_count"),
                "is_correct": record["is_correct"],
                "failure_label": record["failure_label"],
            }) + "\n")
            log_file.flush()

            if sample_idx % CHECKPOINT_EVERY == 0 or sample_idx == len(FAILURE_MINING_SAMPLES):
                print(f"  {run.run_id}: {sample_idx}/{len(FAILURE_MINING_SAMPLES)} samples")
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
        raise RuntimeError(
            f"{run.run_id} metadata validation failed:\n"
            + "\n".join(metadata_errors[:20])
        )

    write_jsonl(prediction_path, predictions)
    metric_path, analysis_path, stats_path, metric_rows = write_run_outputs(
        run,
        predictions,
    )

    manifest_record = {
        "run_id": run.run_id,
        "dataset": "failure_mining",
        "method": run.method_label,
        "selection_method": run.selection_method,
        "retained_tokens": run.retained_tokens,
        "threshold_tau": run.threshold_tau,
        "sample_count": len(predictions),
        "dataset_snapshot": str(DATASET_SNAPSHOT_PATH),
        "prediction_file": str(prediction_path),
        "metric_file": str(metric_path),
        "case_analysis_file": str(analysis_path),
        "adaptive_sparse_stats_file": str(stats_path) if stats_path else "",
        "log_file": str(log_path),
        "metrics": metric_rows,
        "status": "ok",
    }
    return manifest_record


def run_failure_mining_experiment():
    manifest = []
    start = time.time()
    for index, run in enumerate(SELECTED_RUNS, start=1):
        print(
            f"[{index}/{len(SELECTED_RUNS)}] {run.run_id} "
            f"({run.selection_method}, retained={run.retained_tokens}, "
            f"tau={run.threshold_tau})"
        )
        record = run_one_failure_mining_experiment(run)
        manifest.append(record)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("  ok:", record["prediction_file"])

    summary_rows = write_failure_mining_summary(manifest)
    print()
    print("Failure-mining experiment complete.")
    print("Runs:", len(manifest))
    print("Samples:", len(FAILURE_MINING_SAMPLES))
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
print("Failure-mining sample count:", len(FAILURE_MINING_SAMPLES))
print("Selected run:", CURRENT_RUN.run_id)
print("Load seconds:", round(time.time() - load_start, 2))
```

Cell 6: Run selected failure-mining experiment
```python
manifest = run_failure_mining_experiment()
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
