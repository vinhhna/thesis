# SparseVLM Kaggle Smoke Test

These cells run a small 10-sample end-to-end smoke test for every algorithm and token setting in `evaluation_protocol_updated.md`. The output goes to `/kaggle/working/smoke_protocol_results`.

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
import torch

REPO_ROOT = "/kaggle/working/thesis"
LLAVA_ROOT = "/kaggle/working/thesis/SparseVLMs"
CSV_PATH = f"{REPO_ROOT}/failure_mining_set.csv"
OUTPUT_ROOT = "/kaggle/working/smoke_protocol_results"

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
print("Output root:", OUTPUT_ROOT)
```

Cell 4: Selector unit tests
```python
# Fast selector unit tests. This catches dispatch, fixed-k backfill, and adaptive
# threshold regressions before loading the 7B model.
!python tests/test_sparse_selection.py
```

Cell 5: Smoke-test helpers
```python
import csv
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import shortuuid
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# ----- Config -----

MODEL_PATH = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "llava_v1"
SMOKE_SAMPLE_COUNT = 10
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0
NUM_BEAMS = 1
CANDIDATE_POOL_FACTOR = 2
DEFAULT_THRESHOLD_TAU = 0.85
SPARSE_PRUNING_LOC = [2, 6, 15]

OUTPUT_ROOT = Path(OUTPUT_ROOT)
RESULTS_ROOT = OUTPUT_ROOT / "results"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = RESULTS_ROOT / "summary"
MANIFEST_PATH = OUTPUT_ROOT / "smoke_manifest.json"

for path in [RESULTS_ROOT, LOG_DIR, SUMMARY_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ----- Run catalog -----

@dataclass(frozen=True)
class SmokeRun:
    run_id: str
    dataset_key: str
    method_label: str
    selection_method: str
    retained_tokens: int
    threshold_tau: float
    prediction_relpath: str
    output_relpath: str


def make_run(
    dataset_key,
    run_id,
    method_label,
    selection_method,
    retained_tokens,
    stem,
    threshold_tau=DEFAULT_THRESHOLD_TAU,
):
    return SmokeRun(
        run_id=run_id,
        dataset_key=dataset_key,
        method_label=method_label,
        selection_method=selection_method,
        retained_tokens=retained_tokens,
        threshold_tau=float(threshold_tau),
        prediction_relpath=f"{dataset_key}/predictions/{stem}.jsonl",
        output_relpath=f"{dataset_key}/{stem}.csv",
    )


def build_protocol_smoke_runs():
    runs = []
    dataset_specs = [("gqa", "GQA"), ("pope", "POPE"), ("failure_mining", "FM")]

    for dataset_key, run_prefix in dataset_specs:
        runs.append(make_run(
            dataset_key,
            f"{run_prefix}-DENSE-576",
            "Dense / Vanilla",
            "dense",
            576,
            f"{dataset_key}_dense_576",
        ))

        for budget in [128, 64]:
            for run_name, method_label, selection_method, stem in [
                ("SPARSE-ORIG", "SparseVLM-Original", "topk", "sparsevlm_original"),
                ("OURS", "Ours", "mmr", "ours"),
                ("THRESHOLD-FIXED", "Threshold-Fixed-k", "threshold_fixed", "threshold_fixed"),
            ]:
                runs.append(make_run(
                    dataset_key,
                    f"{run_prefix}-{run_name}-{budget}",
                    method_label,
                    selection_method,
                    budget,
                    f"{dataset_key}_{stem}_{budget}",
                ))

        for tau in [0.80, 0.85, 0.90]:
            tau_suffix = f"tau{int(round(tau * 100)):03d}"
            runs.append(make_run(
                dataset_key,
                f"{run_prefix}-THRESHOLD-ADAPT-{int(round(tau * 100)):03d}",
                "Threshold-Adaptive",
                "threshold_adaptive",
                64,
                f"{dataset_key}_threshold_adaptive_{tau_suffix}",
                threshold_tau=tau,
            ))

    return runs


PROTOCOL_SMOKE_RUNS = build_protocol_smoke_runs()


# ----- Metrics -----

def normalize_answer(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction, target):
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if not pred or not gold:
        return False
    return pred == gold or gold in pred.split()


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


def compute_exact_metrics(predictions):
    correct = sum(1 for item in predictions if exact_match(item["text"], item["ground_truth"]))
    total = len(predictions)
    return {"accuracy": safe_div(correct, total), "correct": correct, "total": total}


def classify_failure(prediction, target):
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if exact_match(pred, gold):
        return "correct"
    if gold in {"yes", "no"}:
        return "binary_mismatch"
    if not pred:
        return "empty_answer"
    return "open_answer_mismatch"


# ----- Sample selection -----

def read_failure_mining_rows():
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows, start=1):
        row["question_id"] = row.get("case_id") or f"sample_{idx:04d}"
    return rows


def build_sample_sets():
    rows = read_failure_mining_rows()
    gqa_rows = [row for row in rows if row.get("dataset", "").lower().startswith("gqa")]
    if len(gqa_rows) < SMOKE_SAMPLE_COUNT:
        gqa_rows = rows

    yes_no_rows = [
        row for row in rows
        if normalize_answer(row.get("ground_truth", "")) in {"yes", "no"}
    ]
    if len(yes_no_rows) < SMOKE_SAMPLE_COUNT:
        raise RuntimeError("Need at least SMOKE_SAMPLE_COUNT yes/no rows to exercise POPE metrics.")

    return {
        "gqa": gqa_rows[:SMOKE_SAMPLE_COUNT],
        "failure_mining": rows[:SMOKE_SAMPLE_COUNT],
        "pope": yes_no_rows[:SMOKE_SAMPLE_COUNT],
    }


SAMPLE_SETS = build_sample_sets()
print("Smoke sample counts:", {key: len(value) for key, value in SAMPLE_SETS.items()})
print("Protocol run count:", len(PROTOCOL_SMOKE_RUNS))


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
    image_path = Path(REPO_ROOT) / row["image_path"]
    image = Image.open(image_path).convert("RGB")
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
        "prompt": row["question"],
        "text": answer,
        "answer_id": shortuuid.uuid(),
        "model_id": MODEL_PATH,
        "dataset": row.get("dataset", run.dataset_key),
        "image_path": row["image_path"],
        "ground_truth": row.get("ground_truth", ""),
        "question_type": row.get("question_type", ""),
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
    if run.dataset_key == "pope":
        metrics = compute_pope_metrics(predictions)
        row = {
            "run_id": run.run_id,
            "dataset": "POPE-smoke",
            "method": run.method_label,
            "token_setting": run.retained_tokens,
            "threshold_tau": run.threshold_tau if run.selection_method == "threshold_adaptive" else "",
            "sample_count": len(predictions),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "yes_ratio": metrics["yes_ratio"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        return [row]

    metrics = compute_exact_metrics(predictions)
    rows = [{
        "run_id": run.run_id,
        "dataset": f"{run.dataset_key}-smoke",
        "method": run.method_label,
        "token_setting": run.retained_tokens,
        "threshold_tau": run.threshold_tau if run.selection_method == "threshold_adaptive" else "",
        "sample_count": len(predictions),
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
    }]

    if run.dataset_key == "failure_mining":
        counts = {}
        for item in predictions:
            label = classify_failure(item["text"], item["ground_truth"])
            counts[label] = counts.get(label, 0) + 1
        rows[0].update({
            "correct_count": counts.get("correct", 0),
            "binary_mismatch_count": counts.get("binary_mismatch", 0),
            "open_answer_mismatch_count": counts.get("open_answer_mismatch", 0),
            "empty_answer_count": counts.get("empty_answer", 0),
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
        "candidate_pool_factor": CANDIDATE_POOL_FACTOR,
        "sample_count": len(counts),
        "retained_token_count": counts,
        "average_retained_tokens": sum(counts) / len(counts),
        "min_retained_tokens": min(counts),
        "max_retained_tokens": max(counts),
    }


def write_run_metric_csv(run, predictions):
    output_path = RESULTS_ROOT / run.output_relpath
    rows = metric_rows_for_run(run, predictions)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    write_csv(output_path, rows, fieldnames)
    return output_path, rows[0]


def write_adaptive_sparse_stats(run, predictions):
    if run.selection_method != "threshold_adaptive":
        return None
    stats = sparse_stats_for_predictions(run, predictions)
    if not stats:
        raise RuntimeError(f"{run.run_id}: no adaptive retained-token stats were captured")
    stats_path = (RESULTS_ROOT / run.prediction_relpath).with_name(
        Path(run.prediction_relpath).stem + "_sparse_stats.json"
    )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_path


# ----- Smoke runner -----

def run_one_protocol_smoke(run):
    predictions = []
    metadata_errors = []
    log_path = LOG_DIR / f"{run.run_id}.jsonl"
    prediction_path = RESULTS_ROOT / run.prediction_relpath
    samples = SAMPLE_SETS[run.dataset_key]

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "event": "run_start",
            "run_id": run.run_id,
            "selection_method": run.selection_method,
            "retained_tokens": run.retained_tokens,
            "threshold_tau": run.threshold_tau,
            "sample_count": len(samples),
            "time": time.time(),
        }) + "\n")

        for sample_idx, row in enumerate(samples, start=1):
            start = time.time()
            answer, metadata = run_generation(row, run)
            elapsed = time.time() - start
            record = prediction_record(row, run, answer, metadata, elapsed)
            predictions.append(record)
            metadata_errors.extend(validate_metadata(run, metadata, row["question_id"]))

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
        raise RuntimeError(f"{run.run_id} metadata validation failed:\n" + "\n".join(metadata_errors[:20]))

    write_jsonl(prediction_path, predictions)
    metric_path, metric_row = write_run_metric_csv(run, predictions)
    adaptive_stats_path = write_adaptive_sparse_stats(run, predictions)

    manifest_record = {
        "run_id": run.run_id,
        "dataset": run.dataset_key,
        "method": run.method_label,
        "selection_method": run.selection_method,
        "retained_tokens": run.retained_tokens,
        "threshold_tau": run.threshold_tau,
        "sample_count": len(predictions),
        "prediction_file": str(prediction_path),
        "metric_file": str(metric_path),
        "log_file": str(log_path),
        "adaptive_sparse_stats_file": str(adaptive_stats_path) if adaptive_stats_path else "",
        "metric": metric_row,
        "status": "ok",
    }
    return manifest_record


def write_summary_tables(manifest):
    summary_rows = []
    for item in manifest:
        stats = {}
        pred_path = Path(item["prediction_file"])
        predictions = [json.loads(line) for line in open(pred_path, "r", encoding="utf-8")]
        if item["selection_method"] != "dense":
            stats = sparse_stats_for_predictions(
                SmokeRun(
                    run_id=item["run_id"],
                    dataset_key=item["dataset"],
                    method_label=item["method"],
                    selection_method=item["selection_method"],
                    retained_tokens=item["retained_tokens"],
                    threshold_tau=item["threshold_tau"],
                    prediction_relpath="",
                    output_relpath="",
                ),
                predictions,
            )
        metric = item["metric"]
        summary_rows.append({
            "run_id": item["run_id"],
            "dataset": item["dataset"],
            "method": item["method"],
            "selection_method": item["selection_method"],
            "token_setting": item["retained_tokens"],
            "threshold_tau": item["threshold_tau"] if item["selection_method"] == "threshold_adaptive" else "",
            "sample_count": item["sample_count"],
            "accuracy": metric.get("accuracy", ""),
            "f1": metric.get("f1", ""),
            "average_retained_tokens": stats.get("average_retained_tokens", ""),
            "min_retained_tokens": stats.get("min_retained_tokens", ""),
            "max_retained_tokens": stats.get("max_retained_tokens", ""),
            "prediction_file": item["prediction_file"],
            "metric_file": item["metric_file"],
            "log_file": item["log_file"],
            "status": item["status"],
        })

    fieldnames = [
        "run_id", "dataset", "method", "selection_method", "token_setting",
        "threshold_tau", "sample_count", "accuracy", "f1",
        "average_retained_tokens", "min_retained_tokens", "max_retained_tokens",
        "prediction_file", "metric_file", "log_file", "status",
    ]
    write_csv(SUMMARY_DIR / "final_evaluation_table.csv", summary_rows, fieldnames)

    write_csv(
        SUMMARY_DIR / "gqa_summary.csv",
        [row for row in summary_rows if row["dataset"] == "gqa" and row["selection_method"] != "threshold_adaptive"],
        fieldnames,
    )
    write_csv(
        SUMMARY_DIR / "pope_summary.csv",
        [row for row in summary_rows if row["dataset"] == "pope" and row["selection_method"] != "threshold_adaptive"],
        fieldnames,
    )
    write_csv(
        SUMMARY_DIR / "failure_mining_summary.csv",
        [row for row in summary_rows if row["dataset"] == "failure_mining" and row["selection_method"] != "threshold_adaptive"],
        fieldnames,
    )
    write_csv(
        SUMMARY_DIR / "adaptive_threshold_summary.csv",
        [
            row for row in summary_rows
            if row["selection_method"] == "threshold_adaptive"
            or row["run_id"].endswith("OURS-64")
            or row["run_id"].endswith("OURS-128")
        ],
        fieldnames,
    )

    return summary_rows


def run_protocol_smoke_test():
    manifest = []
    start = time.time()
    for index, run in enumerate(PROTOCOL_SMOKE_RUNS, start=1):
        print(f"[{index}/{len(PROTOCOL_SMOKE_RUNS)}] {run.run_id} ({run.selection_method}, retained={run.retained_tokens}, tau={run.threshold_tau})")
        record = run_one_protocol_smoke(run)
        manifest.append(record)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("  ok:", record["prediction_file"])

    summary_rows = write_summary_tables(manifest)
    print()
    print("Smoke protocol complete.")
    print("Runs:", len(manifest))
    print("Samples per dataset:", {key: len(value) for key, value in SAMPLE_SETS.items()})
    print("Output root:", OUTPUT_ROOT)
    print("Manifest:", MANIFEST_PATH)
    print("Summary rows:", len(summary_rows))
    print("Total minutes:", round((time.time() - start) / 60, 2))
    return manifest
```

Cell 6: Load model
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
print("Smoke sample count:", SMOKE_SAMPLE_COUNT)
print("Load seconds:", round(time.time() - load_start, 2))
```

Cell 7: Run smoke test
```python
manifest = run_protocol_smoke_test()
```
