# SparseVLM Kaggle Stage 8 Subset Runner - One Run

These cells run one Stage 8 ablation/comparison experiment on a deterministic 500-sample subset.

Current configured run:

```text
POPE-STAGE8-SPARSE-ORIG-64
```

Change only `RUN_ID_TO_RUN` in Cell 4 for the next Kaggle session. Each run writes to a unique output folder and creates one zip file under `/kaggle/working`.

This notebook is currently configured for the POPE Stage 8 subset runs. Use the POPE run list below as the working order, then change `RUN_ID_TO_RUN` to the next POPE run after each Kaggle download.

Important: Cell 1 clones from GitHub. Before running this on Kaggle, push the Stage 8 code changes that add `lambda_relevance` and `record_selection_similarity`.

Stage 8 uses fixed subset seed:

```text
20260610
```

Required GQA runs:

```text
GQA-STAGE8-SPARSE-ORIG-64
GQA-STAGE8-OURS-64-P2-L08
GQA-STAGE8-THRESHOLD-FIXED-64
GQA-STAGE8-OURS-64-P2-L05
GQA-STAGE8-OURS-64-P2-L07
GQA-STAGE8-OURS-64-P3-L05
GQA-STAGE8-OURS-64-P3-L07
```

Required POPE runs:

```text
POPE-STAGE8-SPARSE-ORIG-64
POPE-STAGE8-OURS-64-P2-L08
POPE-STAGE8-THRESHOLD-FIXED-64
POPE-STAGE8-OURS-64-P2-L05
POPE-STAGE8-OURS-64-P2-L07
POPE-STAGE8-OURS-64-P3-L05
POPE-STAGE8-OURS-64-P3-L07
```

Optional failure-mining stress-test runs:

```text
FM-STAGE8-OURS-64-P2-L05
FM-STAGE8-OURS-64-P2-L07
FM-STAGE8-OURS-64-P3-L05
FM-STAGE8-OURS-64-P3-L07
```

These optional runs are not representative ablation evidence; use them only for targeted recovery behavior.

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


REPO_ROOT = Path("/kaggle/working/thesis")
LLAVA_ROOT = REPO_ROOT / "SparseVLMs"
OUTPUT_BASE_ROOT = Path("/kaggle/working/stage8_subset_runs")

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONPATH"] = str(LLAVA_ROOT)
if str(LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ROOT))

print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Repository root:", REPO_ROOT)
print("Output base root:", OUTPUT_BASE_ROOT)
```

Cell 4: Stage 8 subset helpers
```python
import csv
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter
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
DEFAULT_THRESHOLD_TAU = 0.85
DEFAULT_LAMBDA_RELEVANCE = 0.8
RECORD_SELECTION_SIMILARITY = True
SPARSE_PRUNING_LOC = [2, 6, 15]

STAGE8_SUBSET_SEED = 20260610
GQA_STAGE8_SUBSET_N = 500
POPE_STAGE8_SUBSET_N = 500
GQA_SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."
FAILURE_MINING_CSV = REPO_ROOT / "failure_mining_set.csv"
FAILURE_MINING_IMAGE_ROOT = REPO_ROOT
FAILURE_MINING_SHORT_ANSWER_SUFFIX = "\nAnswer using a single word or short phrase."
GQA_QUESTIONS_URLS = [
    "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
    "https://nlp.stanford.edu/data/gqa/questions1.2.zip",
]
GQA_QUESTION_FILE_PREFERENCE = [
    "val_balanced_questions.json",
    "val_all_questions.json",
]
CHECKPOINT_EVERY = 25

REPO_ROOT = Path(REPO_ROOT)
LLAVA_ROOT = Path(LLAVA_ROOT)
OUTPUT_BASE_ROOT = Path(OUTPUT_BASE_ROOT)


# ----- Run catalog -----

@dataclass(frozen=True)
class Stage8Run:
    run_id: str
    dataset: str
    ablation_role: str
    method_label: str
    selection_method: str
    retained_tokens: int = 64
    candidate_pool_factor: int = 2
    lambda_relevance: float = DEFAULT_LAMBDA_RELEVANCE
    threshold_tau: float = DEFAULT_THRESHOLD_TAU
    record_selection_similarity: bool = RECORD_SELECTION_SIMILARITY

    @property
    def safe_name(self):
        return self.run_id.lower().replace("-", "_")

    @property
    def prediction_relpath(self):
        return f"{self.dataset}/predictions/{self.safe_name}.jsonl"

    @property
    def metric_relpath(self):
        return f"{self.dataset}/{self.safe_name}.csv"


STAGE8_RUNS = [
    Stage8Run("GQA-STAGE8-SPARSE-ORIG-64", "gqa", "general_ablation_gqa", "SparseVLM-Original", "topk"),
    Stage8Run("GQA-STAGE8-OURS-64-P2-L08", "gqa", "general_ablation_gqa", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.8),
    Stage8Run("GQA-STAGE8-THRESHOLD-FIXED-64", "gqa", "general_ablation_gqa", "Threshold-Fixed-k", "threshold_fixed"),
    Stage8Run("GQA-STAGE8-OURS-64-P2-L05", "gqa", "general_ablation_gqa", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.5),
    Stage8Run("GQA-STAGE8-OURS-64-P2-L07", "gqa", "general_ablation_gqa", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.7),
    Stage8Run("GQA-STAGE8-OURS-64-P3-L05", "gqa", "general_ablation_gqa", "Ours", "mmr", candidate_pool_factor=3, lambda_relevance=0.5),
    Stage8Run("GQA-STAGE8-OURS-64-P3-L07", "gqa", "general_ablation_gqa", "Ours", "mmr", candidate_pool_factor=3, lambda_relevance=0.7),
    Stage8Run("POPE-STAGE8-SPARSE-ORIG-64", "pope", "general_ablation_pope", "SparseVLM-Original", "topk"),
    Stage8Run("POPE-STAGE8-OURS-64-P2-L08", "pope", "general_ablation_pope", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.8),
    Stage8Run("POPE-STAGE8-THRESHOLD-FIXED-64", "pope", "general_ablation_pope", "Threshold-Fixed-k", "threshold_fixed"),
    Stage8Run("POPE-STAGE8-OURS-64-P2-L05", "pope", "general_ablation_pope", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.5),
    Stage8Run("POPE-STAGE8-OURS-64-P2-L07", "pope", "general_ablation_pope", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.7),
    Stage8Run("POPE-STAGE8-OURS-64-P3-L05", "pope", "general_ablation_pope", "Ours", "mmr", candidate_pool_factor=3, lambda_relevance=0.5),
    Stage8Run("POPE-STAGE8-OURS-64-P3-L07", "pope", "general_ablation_pope", "Ours", "mmr", candidate_pool_factor=3, lambda_relevance=0.7),
    Stage8Run("FM-STAGE8-OURS-64-P2-L05", "failure_mining", "targeted_failure_recovery", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.5),
    Stage8Run("FM-STAGE8-OURS-64-P2-L07", "failure_mining", "targeted_failure_recovery", "Ours", "mmr", candidate_pool_factor=2, lambda_relevance=0.7),
    Stage8Run("FM-STAGE8-OURS-64-P3-L05", "failure_mining", "targeted_failure_recovery", "Ours", "mmr", candidate_pool_factor=3, lambda_relevance=0.5),
    Stage8Run("FM-STAGE8-OURS-64-P3-L07", "failure_mining", "targeted_failure_recovery", "Ours", "mmr", candidate_pool_factor=3, lambda_relevance=0.7),
]

# Change this value for each Kaggle session. Keep exactly one run ID here.
RUN_ID_TO_RUN = "POPE-STAGE8-SPARSE-ORIG-64"

RUN_BY_ID = {run.run_id: run for run in STAGE8_RUNS}
if RUN_ID_TO_RUN not in RUN_BY_ID:
    raise ValueError(f"Unknown RUN_ID_TO_RUN: {RUN_ID_TO_RUN}")

CURRENT_RUN = RUN_BY_ID[RUN_ID_TO_RUN]


def expected_sample_count(dataset):
    if dataset == "gqa":
        return GQA_STAGE8_SUBSET_N
    if dataset == "pope":
        return POPE_STAGE8_SUBSET_N
    if dataset == "failure_mining":
        return 100
    raise ValueError(f"Unsupported Stage 8 dataset: {dataset}")


OUTPUT_ROOT = OUTPUT_BASE_ROOT / CURRENT_RUN.run_id
DOWNLOAD_ZIP = Path(f"/kaggle/working/{CURRENT_RUN.run_id}_download.zip")
RESULTS_ROOT = OUTPUT_ROOT / "results"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = RESULTS_ROOT / "summary"
MANIFEST_PATH = OUTPUT_ROOT / "stage8_subset_manifest.json"
SUBSET_PATH = OUTPUT_ROOT / f"stage8_{CURRENT_RUN.dataset}_subset_seed{STAGE8_SUBSET_SEED}_n{expected_sample_count(CURRENT_RUN.dataset)}.jsonl"

for path in [RESULTS_ROOT, LOG_DIR, SUMMARY_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ----- Common IO and metrics -----

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_answer(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_gqa_answer(text):
    text = str(text or "").strip().lower().rstrip(".")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pope_label_from_answer(text):
    words = normalize_answer(text).split()
    return "no" if "no" in words or "not" in words else "yes"


def exact_or_phrase_match(prediction, target):
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if not pred or not gold:
        return False
    if gold in {"yes", "no"}:
        return pope_label_from_answer(pred) == gold
    return re.search(rf"(?:^| ){re.escape(gold)}(?: |$)", pred) is not None


def compute_metrics(dataset, predictions):
    if dataset == "gqa":
        correct = sum(
            normalize_gqa_answer(item["text"]) == normalize_gqa_answer(item["ground_truth"])
            for item in predictions
        )
        total = len(predictions)
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        }

    if dataset == "failure_mining":
        correct = sum(
            exact_or_phrase_match(item["text"], item["ground_truth"])
            for item in predictions
        )
        total = len(predictions)
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        }

    y_true = [1 if normalize_answer(item["ground_truth"]) == "yes" else 0 for item in predictions]
    y_pred = [1 if pope_label_from_answer(item["text"]) == "yes" else 0 for item in predictions]
    tp = sum(pred == 1 and gold == 1 for pred, gold in zip(y_pred, y_true))
    tn = sum(pred == 0 and gold == 0 for pred, gold in zip(y_pred, y_true))
    fp = sum(pred == 1 and gold == 0 for pred, gold in zip(y_pred, y_true))
    fn = sum(pred == 0 and gold == 1 for pred, gold in zip(y_pred, y_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true) if y_true else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "correct": tp + tn,
        "total": len(y_true),
    }


# ----- GQA data -----

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
    with urllib.request.urlopen(url, timeout=60) as response, open(destination, "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
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


def find_gqa_question_file(gqa_root):
    question_path = find_first_existing(gqa_root, GQA_QUESTION_FILE_PREFERENCE)
    if question_path:
        return question_path

    input_root = Path("/kaggle/input")
    zip_matches = sorted(input_root.rglob("questions1.2.zip")) + sorted(gqa_root.rglob("questions1.2.zip"))
    if zip_matches:
        return extract_gqa_question_file_from_zip(zip_matches[0])

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


def iter_gqa_questions(question_path):
    with open(question_path, "rb") as f:
        for question_id, question in ijson.kvitems(f, ""):
            yield str(question_id), question


def build_gqa_samples():
    gqa_root = find_kaggle_input_dir("GQA")
    gqa_image_dir = gqa_root / "images"
    gqa_question_path = find_gqa_question_file(gqa_root)
    print("GQA root:", gqa_root)
    print("GQA image dir:", gqa_image_dir)
    print("GQA question file:", gqa_question_path)

    samples = []
    skipped_missing_image = 0
    skipped_unbalanced = 0
    for question_id, question in iter_gqa_questions(gqa_question_path):
        if not question.get("isBalanced", True):
            skipped_unbalanced += 1
            continue
        question_text = question.get("question", "")
        answer = question.get("answer", "")
        image_id = str(question.get("imageId", ""))
        image_path = gqa_image_dir / f"{image_id}.jpg"
        if not question_text or not answer or not image_id:
            continue
        if not image_path.exists():
            skipped_missing_image += 1
            continue
        types = question.get("types", {}) or {}
        samples.append({
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
        })

    if len(samples) < GQA_STAGE8_SUBSET_N:
        raise RuntimeError(
            f"Only found {len(samples)} valid GQA samples, expected {GQA_STAGE8_SUBSET_N}. "
            f"Skipped missing images: {skipped_missing_image}; skipped unbalanced: {skipped_unbalanced}."
        )
    rng = random.Random(STAGE8_SUBSET_SEED)
    selected = rng.sample(sorted(samples, key=lambda item: item["question_id"]), GQA_STAGE8_SUBSET_N)
    selected.sort(key=lambda item: item["question_id"])
    write_jsonl(SUBSET_PATH, selected)
    return selected, {
        "eligible_seen": len(samples),
        "skipped_missing_image": skipped_missing_image,
        "skipped_unbalanced": skipped_unbalanced,
        "subset_seed": STAGE8_SUBSET_SEED,
        "subset_size": GQA_STAGE8_SUBSET_N,
        "subset_file": str(SUBSET_PATH),
    }


# ----- POPE data -----

def build_pope_samples():
    pope_root = find_kaggle_input_dir("POPE")
    pope_annotations_dir = pope_root / "annotations"
    pope_image_dir = pope_root / "val2014"
    print("POPE root:", pope_root)
    print("POPE annotations:", pope_annotations_dir)
    print("POPE image dir:", pope_image_dir)

    samples = []
    for category in ["adversarial", "popular", "random"]:
        path = pope_annotations_dir / f"coco_pope_{category}.json"
        rows = read_jsonl(path)
        for row in rows:
            image_path = pope_image_dir / row["image"]
            if not image_path.exists():
                raise FileNotFoundError(f"Missing POPE image: {image_path}")
            samples.append({
                "question_id": f"{category}_{row['question_id']}",
                "pope_question_id": row["question_id"],
                "pope_category": category,
                "question": row["text"],
                "prompt": row["text"],
                "image": row["image"],
                "image_path": str(image_path),
                "ground_truth": row["label"],
            })

    if len(samples) < POPE_STAGE8_SUBSET_N:
        raise RuntimeError(f"Only found {len(samples)} valid POPE samples, expected {POPE_STAGE8_SUBSET_N}.")
    rng = random.Random(STAGE8_SUBSET_SEED)
    selected = rng.sample(sorted(samples, key=lambda item: item["question_id"]), POPE_STAGE8_SUBSET_N)
    selected.sort(key=lambda item: item["question_id"])
    write_jsonl(SUBSET_PATH, selected)
    return selected, {
        "eligible_seen": len(samples),
        "subset_seed": STAGE8_SUBSET_SEED,
        "subset_size": POPE_STAGE8_SUBSET_N,
        "subset_file": str(SUBSET_PATH),
        "category_counts": {
            category: sum(1 for item in selected if item["pope_category"] == category)
            for category in ["adversarial", "popular", "random"]
        },
    }


# ----- Failure-mining data, optional targeted stress test -----

def build_failure_mining_samples():
    if not FAILURE_MINING_CSV.exists():
        raise FileNotFoundError(f"Missing failure-mining CSV: {FAILURE_MINING_CSV}")

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
    if len(source_rows) != 100:
        raise RuntimeError(f"Expected 100 failure-mining rows, found {len(source_rows)}")

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
            "question_id": case_id,
            "dataset_source": row["dataset"].strip(),
            "image_path": str(image_path),
            "source_image_path": relative_image_path.as_posix(),
            "question": question,
            "prompt": question + FAILURE_MINING_SHORT_ANSWER_SUFFIX,
            "ground_truth": ground_truth,
            "question_type": row["question_type"].strip(),
            "note": row.get("note", "").strip(),
        })

    write_jsonl(SUBSET_PATH, samples)
    return samples, {
        "eligible_seen": len(samples),
        "subset_seed": STAGE8_SUBSET_SEED,
        "subset_size": len(samples),
        "subset_file": str(SUBSET_PATH),
        "source_csv": str(FAILURE_MINING_CSV),
        "dataset_counts": dict(Counter(row["dataset_source"] for row in samples)),
        "question_type_counts": dict(Counter(row["question_type"] for row in samples)),
    }


def build_stage8_samples():
    if CURRENT_RUN.dataset == "gqa":
        samples, stats = build_gqa_samples()
    elif CURRENT_RUN.dataset == "pope":
        samples, stats = build_pope_samples()
    elif CURRENT_RUN.dataset == "failure_mining":
        samples, stats = build_failure_mining_samples()
    else:
        raise ValueError(f"Unsupported Stage 8 subset dataset: {CURRENT_RUN.dataset}")

    expected_size = expected_sample_count(CURRENT_RUN.dataset)
    if len(samples) != expected_size:
        raise RuntimeError(f"{CURRENT_RUN.dataset} subset has {len(samples)} samples; expected {expected_size}")
    return samples, stats


STAGE8_SAMPLES, STAGE8_SAMPLE_STATS = build_stage8_samples()

print("Current run:", CURRENT_RUN.run_id)
print("Dataset:", CURRENT_RUN.dataset)
print("Subset count:", len(STAGE8_SAMPLES))
print("Subset stats:", STAGE8_SAMPLE_STATS)
print("Subset path:", SUBSET_PATH)
print("Output root:", OUTPUT_ROOT)
print("Download zip:", DOWNLOAD_ZIP)


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

    sparse_core.pruning_loc = SPARSE_PRUNING_LOC
    sparse_core.last_sparse_metadata = {}

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
                selection_method=run.selection_method,
                threshold_tau=run.threshold_tau,
                candidate_pool_factor=run.candidate_pool_factor,
                lambda_relevance=run.lambda_relevance,
                record_selection_similarity=run.record_selection_similarity,
                **generation_kwargs,
            )
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        metadata = deepcopy(getattr(sparse_core, "last_sparse_metadata", {}))
    finally:
        sparse_core.pruning_loc = original_pruning_loc

    metadata.setdefault("selection_method", run.selection_method)
    metadata.setdefault("retained_tokens", run.retained_tokens)
    metadata.setdefault("threshold_tau", run.threshold_tau)
    metadata.setdefault("candidate_pool_factor", run.candidate_pool_factor)
    metadata.setdefault("lambda_relevance", run.lambda_relevance)
    metadata.setdefault("record_selection_similarity", run.record_selection_similarity)

    return answer, metadata


# ----- Validation and output -----

def validate_metadata(run, metadata, sample_id):
    problems = []
    if metadata.get("selection_method") != run.selection_method:
        problems.append(f"selection_method={metadata.get('selection_method')} expected {run.selection_method}")

    if int(metadata.get("retained_tokens", -1)) != int(run.retained_tokens):
        problems.append(f"retained_tokens={metadata.get('retained_tokens')} expected {run.retained_tokens}")

    if int(metadata.get("candidate_pool_factor", -1)) != int(run.candidate_pool_factor):
        problems.append(f"candidate_pool_factor={metadata.get('candidate_pool_factor')} expected {run.candidate_pool_factor}")

    if run.selection_method == "mmr":
        actual_lambda = float(metadata.get("lambda_relevance", -1))
        if abs(actual_lambda - run.lambda_relevance) > 1e-9:
            problems.append(f"lambda_relevance={actual_lambda} expected {run.lambda_relevance}")

    layer_stats = metadata.get("layer_token_stats")
    if not isinstance(layer_stats, list) or not layer_stats:
        problems.append("missing layer_token_stats")
        return [f"{sample_id}: {problem}" for problem in problems]

    if metadata.get("retained_token_count") is None:
        problems.append("missing retained_token_count")

    selected_original = metadata.get("selected_original_token_indices")
    if not isinstance(selected_original, list):
        problems.append("missing selected_original_token_indices")
    else:
        bad_indices = [idx for idx in selected_original if not isinstance(idx, int) or idx < 0 or idx > 575]
        if bad_indices:
            problems.append(f"selected_original_token_indices outside 0-575: {bad_indices[:5]}")

    for layer in layer_stats:
        if layer.get("selection_method") != run.selection_method:
            problems.append(f"layer {layer.get('layer_idx')} used {layer.get('selection_method')}")
        selected_count = int(layer.get("selected_count", -1))
        per_layer_budget = int(layer.get("per_layer_budget", -1))
        if selected_count < 0 or per_layer_budget < 0:
            problems.append(f"layer {layer.get('layer_idx')} has invalid selected_count or budget")
        if run.selection_method == "threshold_fixed" and selected_count != per_layer_budget:
            problems.append(f"threshold_fixed layer {layer.get('layer_idx')} selected {selected_count}, budget {per_layer_budget}")
        if run.record_selection_similarity and layer.get("pairwise_similarity_available") is not True:
            problems.append(f"layer {layer.get('layer_idx')} missing pairwise similarity stats")

    return [f"{sample_id}: {problem}" for problem in problems]


def prediction_record(row, run, answer, metadata, elapsed):
    base = {
        "question_id": row["question_id"],
        "prompt": row["prompt"],
        "text": answer,
        "answer_id": shortuuid.uuid(),
        "model_id": MODEL_PATH,
        "dataset": run.dataset,
        "image_path": row["image_path"],
        "ground_truth": row["ground_truth"],
        "run_id": run.run_id,
        "subset_seed": STAGE8_SUBSET_SEED,
        "subset_size": len(STAGE8_SAMPLES),
        "metadata": metadata,
        "inference_seconds": elapsed,
    }
    if run.dataset == "gqa":
        base.update({
            "raw_question": row["question"],
            "image_id": row["image_id"],
            "normalized_text": normalize_gqa_answer(answer),
            "normalized_ground_truth": normalize_gqa_answer(row["ground_truth"]),
            "is_correct": normalize_gqa_answer(answer) == normalize_gqa_answer(row["ground_truth"]),
            "structural_type": row.get("structural_type", ""),
            "semantic_type": row.get("semantic_type", ""),
            "detailed_type": row.get("detailed_type", ""),
        })
    elif run.dataset == "pope":
        base.update({
            "pope_question_id": row["pope_question_id"],
            "pope_category": row["pope_category"],
            "image": row["image"],
        })
    elif run.dataset == "failure_mining":
        base.update({
            "case_id": row["case_id"],
            "source_dataset": row["dataset_source"],
            "source_image_path": row["source_image_path"],
            "question_type": row["question_type"],
            "note": row.get("note", ""),
            "is_correct": exact_or_phrase_match(answer, row["ground_truth"]),
        })
    return base


def sparse_stats_for_predictions(run, predictions):
    counts = [
        int(item["metadata"]["retained_token_count"])
        for item in predictions
        if item.get("metadata", {}).get("retained_token_count") is not None
    ]
    if not counts:
        return {}
    return {
        "average_retained_tokens": sum(counts) / len(counts),
        "min_retained_tokens": min(counts),
        "max_retained_tokens": max(counts),
    }


def metric_rows_for_run(run, predictions):
    metrics = compute_metrics(run.dataset, predictions)
    row = {
        "run_id": run.run_id,
        "dataset": run.dataset.upper(),
        "ablation_role": run.ablation_role,
        "method": run.method_label,
        "selection_method": run.selection_method,
        "token_setting": run.retained_tokens,
        "threshold_tau": run.threshold_tau if run.selection_method.startswith("threshold") else "",
        "candidate_pool_factor": run.candidate_pool_factor,
        "lambda_relevance": run.lambda_relevance,
        "record_selection_similarity": run.record_selection_similarity,
        "sample_count": len(predictions),
        "subset_seed": STAGE8_SUBSET_SEED,
        "subset_size": len(STAGE8_SAMPLES),
        "accuracy": metrics.get("accuracy", ""),
        "correct": metrics.get("correct", ""),
        "total": metrics.get("total", ""),
    }
    if run.dataset == "pope":
        row.update({
            "f1": metrics.get("f1", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "tp": metrics.get("tp", ""),
            "tn": metrics.get("tn", ""),
            "fp": metrics.get("fp", ""),
            "fn": metrics.get("fn", ""),
        })
    row.update(sparse_stats_for_predictions(run, predictions))
    return [row]


def write_run_metric_csv(run, predictions):
    output_path = RESULTS_ROOT / run.metric_relpath
    rows = metric_rows_for_run(run, predictions)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    write_csv(output_path, rows, fieldnames)
    return output_path, rows


def write_stage8_summary(manifest):
    summary_rows = []
    for item in manifest:
        metric = item["metrics"][0] if item.get("metrics") else {}
        summary_rows.append({
            "run_id": item["run_id"],
            "dataset": item["dataset"],
            "ablation_role": item["ablation_role"],
            "method": item["method"],
            "selection_method": item["selection_method"],
            "token_setting": item["retained_tokens"],
            "threshold_tau": item["threshold_tau"] if item["selection_method"].startswith("threshold") else "",
            "candidate_pool_factor": item["candidate_pool_factor"],
            "lambda_relevance": item["lambda_relevance"],
            "record_selection_similarity": item["record_selection_similarity"],
            "sample_count": item["sample_count"],
            "subset_seed": item["subset_seed"],
            "subset_size": item["subset_size"],
            "accuracy": metric.get("accuracy", ""),
            "f1": metric.get("f1", ""),
            "correct": metric.get("correct", ""),
            "total": metric.get("total", ""),
            "average_retained_tokens": metric.get("average_retained_tokens", ""),
            "min_retained_tokens": metric.get("min_retained_tokens", ""),
            "max_retained_tokens": metric.get("max_retained_tokens", ""),
            "prediction_file": item["prediction_file"],
            "metric_file": item["metric_file"],
            "subset_file": item["subset_file"],
            "log_file": item["log_file"],
            "status": item["status"],
        })

    fieldnames = [
        "run_id", "dataset", "ablation_role", "method", "selection_method",
        "token_setting", "threshold_tau", "candidate_pool_factor",
        "lambda_relevance", "record_selection_similarity", "sample_count",
        "subset_seed", "subset_size", "accuracy", "f1", "correct", "total",
        "average_retained_tokens", "min_retained_tokens", "max_retained_tokens",
        "prediction_file", "metric_file", "subset_file", "log_file", "status",
    ]
    write_csv(SUMMARY_DIR / "stage8_subset_summary.csv", summary_rows, fieldnames)
    return summary_rows


def run_one_stage8_experiment(run):
    predictions = []
    metadata_errors = []
    log_path = LOG_DIR / f"{run.run_id}.jsonl"
    prediction_path = RESULTS_ROOT / run.prediction_relpath

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "event": "run_start",
            "run_id": run.run_id,
            "dataset": run.dataset,
            "ablation_role": run.ablation_role,
            "selection_method": run.selection_method,
            "retained_tokens": run.retained_tokens,
            "threshold_tau": run.threshold_tau,
            "candidate_pool_factor": run.candidate_pool_factor,
            "lambda_relevance": run.lambda_relevance,
            "record_selection_similarity": run.record_selection_similarity,
            "sample_count": len(STAGE8_SAMPLES),
            "subset_seed": STAGE8_SUBSET_SEED,
            "subset_path": str(SUBSET_PATH),
            "time": time.time(),
        }) + "\n")

        for sample_idx, row in enumerate(STAGE8_SAMPLES, start=1):
            start = time.time()
            answer, metadata = run_generation(row, run)
            elapsed = time.time() - start
            record = prediction_record(row, run, answer, metadata, elapsed)
            predictions.append(record)
            metadata_errors.extend(validate_metadata(run, metadata, row["question_id"]))

            if sample_idx % CHECKPOINT_EVERY == 0 or sample_idx == len(STAGE8_SAMPLES):
                print(f"  {run.run_id}: {sample_idx}/{len(STAGE8_SAMPLES)} samples")
                write_jsonl(prediction_path, predictions)

            log_file.write(json.dumps({
                "event": "sample_done",
                "run_id": run.run_id,
                "sample_index": sample_idx,
                "question_id": row["question_id"],
                "seconds": elapsed,
                "retained_token_count": metadata.get("retained_token_count"),
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
        raise RuntimeError(
            f"{run.run_id} metadata validation failed:\n"
            + "\n".join(metadata_errors[:30])
        )

    write_jsonl(prediction_path, predictions)
    metric_path, metric_rows = write_run_metric_csv(run, predictions)

    manifest_record = {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "ablation_role": run.ablation_role,
        "method": run.method_label,
        "selection_method": run.selection_method,
        "retained_tokens": run.retained_tokens,
        "threshold_tau": run.threshold_tau,
        "candidate_pool_factor": run.candidate_pool_factor,
        "lambda_relevance": run.lambda_relevance,
        "record_selection_similarity": run.record_selection_similarity,
        "sample_count": len(predictions),
        "subset_seed": STAGE8_SUBSET_SEED,
        "subset_size": len(STAGE8_SAMPLES),
        "subset_file": str(SUBSET_PATH),
        "prediction_file": str(prediction_path),
        "metric_file": str(metric_path),
        "log_file": str(log_path),
        "metrics": metric_rows,
        "status": "ok",
    }
    return manifest_record


def run_stage8_subset_experiment():
    start = time.time()
    print(
        f"Running {CURRENT_RUN.run_id} "
        f"({CURRENT_RUN.dataset}, {CURRENT_RUN.selection_method}, "
        f"retained={CURRENT_RUN.retained_tokens}, pool={CURRENT_RUN.candidate_pool_factor}, "
        f"lambda={CURRENT_RUN.lambda_relevance})"
    )
    record = run_one_stage8_experiment(CURRENT_RUN)
    manifest = [record]
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    summary_rows = write_stage8_summary(manifest)

    print()
    print("Stage 8 subset experiment complete.")
    print("Run:", CURRENT_RUN.run_id)
    print("Dataset:", CURRENT_RUN.dataset)
    print("Samples:", len(STAGE8_SAMPLES))
    print("Subset seed:", STAGE8_SUBSET_SEED)
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
print("Current run:", CURRENT_RUN.run_id)
print("Dataset:", CURRENT_RUN.dataset)
print("Subset count:", len(STAGE8_SAMPLES))
print("Candidate pool factor:", CURRENT_RUN.candidate_pool_factor)
print("Lambda relevance:", CURRENT_RUN.lambda_relevance)
print("Record pairwise similarity:", CURRENT_RUN.record_selection_similarity)
print("Load seconds:", round(time.time() - load_start, 2))
```

Cell 6: Run selected Stage 8 subset experiment
```python
manifest = run_stage8_subset_experiment()
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
