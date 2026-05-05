Cell 1:
```python
%env USE_FLAX=NO
%env USE_JAX=NO
%env USE_TF=NO

%cd /kaggle/working
!rm -rf thesis
!git clone https://github.com/vinhhna/thesis.git

%cd /kaggle/working/thesis/SparseVLMs
!git rev-parse --short HEAD

from collections import Counter
from pathlib import Path
import csv

repo_root = Path("/kaggle/working/thesis")
csv_path = repo_root / "failure_mining_set.csv"

rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
missing = [row["image_path"] for row in rows if not (repo_root / row["image_path"]).exists()]

print("Samples:", len(rows))
print("Datasets:", dict(Counter(row["dataset"] for row in rows)))
print("Question types:", dict(Counter(row["question_type"] for row in rows)))
print("Missing images:", len(missing))
if missing:
    print("First missing image paths:", missing[:10])
    raise RuntimeError("Some image paths from failure_mining_set.csv are missing.")
```

Cell 2:
```python
!python -m pip install --upgrade pip setuptools wheel

!python -m pip install --force-reinstall --no-deps torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

!python -m pip install "numpy<2" protobuf sentencepiece shortuuid
!python -m pip install transformers==4.37.2 tokenizers==0.15.1 accelerate==0.21.0 peft==0.7.1
!python -m pip install bitsandbytes==0.45.5 einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 "markdown2[all]"
!python -m pip uninstall -y jax jaxlib flax optax chex orbax-checkpoint

print("Restart the Kaggle kernel after this cell finishes, then continue from Cell 3.")
```

Cell 3:
```python
import os
import sys
import torch
from pathlib import Path

REPO_ROOT = "/kaggle/working/thesis"
SPARSEVLM_ROOT = "/kaggle/working/thesis/SparseVLMs"
CSV_PATH = f"{REPO_ROOT}/failure_mining_set.csv"
OUTPUT_JSONL = "/kaggle/working/failure_mining_budget_sweep_outputs.jsonl"

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"
os.environ["PYTHONPATH"] = SPARSEVLM_ROOT
if SPARSEVLM_ROOT not in sys.path:
    sys.path.insert(0, SPARSEVLM_ROOT)

print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU accelerator in Kaggle before running SparseVLM inference.")

gpu_name = torch.cuda.get_device_name(0)
gpu_capability = torch.cuda.get_device_capability(0)
torch_arches = torch.cuda.get_arch_list()

print("GPU:", gpu_name)
print("GPU capability:", gpu_capability)
print("Torch CUDA architectures:", torch_arches)

if gpu_capability[0] == 6 and "sm_60" not in torch_arches:
    raise RuntimeError(
        "Kaggle assigned a Tesla P100 GPU, but the current PyTorch build does not support sm_60. "
        "Run Cell 2 from a fresh kernel so it installs the P100-compatible PyTorch build, "
        "then continue from Cell 3."
    )

# SparseVLM's score.py supports 192/128/64 by default. Add an interpolated
# 96-token budget for the sweep without requiring a repo commit.
score_path = Path(SPARSEVLM_ROOT) / "llava" / "model" / "language_model" / "score.py"
score_text = score_path.read_text(encoding="utf-8")
if "sparse_token_list_96" not in score_text:
    score_text = score_text.replace(
        "sparse_token_list_64 = [66,30,17]          \n",
        "sparse_token_list_64 = [66,30,17]          \n"
        "sparse_token_list_96 = [184,70,26]         # midpoint between 128 and 64 budgets\n",
    )
    score_text = score_text.replace(
        "    128: sparse_token_list_128,\n    64 : sparse_token_list_64\n}",
        "    128: sparse_token_list_128,\n    96 : sparse_token_list_96,\n    64 : sparse_token_list_64\n}",
    )
    score_path.write_text(score_text, encoding="utf-8")
    print("Patched score.py to support retained_tokens=96.")
else:
    print("score.py already supports retained_tokens=96.")
```

Cell 4:
```python
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"

for module_name in list(sys.modules):
    if (
        module_name == "transformers"
        or module_name.startswith("transformers.")
        or module_name == "jax"
        or module_name.startswith("jax.")
        or module_name == "flax"
        or module_name.startswith("flax.")
    ):
        del sys.modules[module_name]

import torch
from PIL import Image as PILImage
from IPython.display import Image as DisplayImage
from IPython.display import FileLink, Markdown, display

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


MODEL_PATH = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "llava_v1"
MAX_NEW_TOKENS = 64
SWEEP_MODES = [
    {"mode": "dense", "retained_tokens": None, "dynamic_sparse": False},
    {"mode": "sparse_192", "retained_tokens": 192, "dynamic_sparse": True},
    {"mode": "sparse_128", "retained_tokens": 128, "dynamic_sparse": True},
    {"mode": "sparse_96", "retained_tokens": 96, "dynamic_sparse": True},
    {"mode": "sparse_64", "retained_tokens": 64, "dynamic_sparse": True},
]


def build_prompt(question):
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_model_bundle(dynamic_sparse):
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
        dynamic_sparse=dynamic_sparse,
    )
    model.eval()
    print(
        "Loaded:",
        MODEL_PATH,
        "| dynamic_sparse:",
        dynamic_sparse,
        "| context length:",
        context_len,
        "| seconds:",
        round(time.time() - load_start, 2),
    )
    return tokenizer, model, image_processor


def unload_model_bundle(tokenizer=None, model=None, image_processor=None):
    del tokenizer, model, image_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_answer(tokenizer, model, image_processor, image_path, question, retained_tokens=None):
    prompt = build_prompt(question)
    image = PILImage.open(image_path).convert("RGB")
    image_sizes = [image.size]

    images_tensor = process_images([image], image_processor, model.config)
    if isinstance(images_tensor, list):
        images_tensor = [
            image_tensor.to(model.device, dtype=torch.float16)
            for image_tensor in images_tensor
        ]
    else:
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    generate_kwargs = {
        "inputs": input_ids,
        "images": images_tensor,
        "image_sizes": image_sizes,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
    }
    if retained_tokens is not None:
        generate_kwargs["retained_tokens"] = retained_tokens

    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def load_rows(start=0, limit=None):
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = rows[start:] if limit is None else rows[start:start + limit]
    return rows, selected


def load_completed_keys(output_jsonl):
    output_path = Path(output_jsonl)
    completed = set()
    if not output_path.exists():
        return completed
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        completed.add((record.get("case_id"), record.get("mode")))
    return completed


def run_mode(mode_cfg, selected, total, start_offset, output_jsonl, image_width=420, resume=True):
    mode = mode_cfg["mode"]
    retained_tokens = mode_cfg["retained_tokens"]
    dynamic_sparse = mode_cfg["dynamic_sparse"]
    completed = load_completed_keys(output_jsonl) if resume else set()
    tokenizer, model, image_processor = load_model_bundle(dynamic_sparse=dynamic_sparse)

    try:
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a" if resume else "w", encoding="utf-8") as f:
            for local_idx, row in enumerate(selected, start=1):
                offset = start_offset + local_idx - 1
                case_id = row["case_id"]
                if (case_id, mode) in completed:
                    print(f"[skip] {mode} {case_id}")
                    continue

                image_path = Path(REPO_ROOT) / row["image_path"]
                dataset = row["dataset"]
                question_type = row.get("question_type", "")
                question = row["question"]

                display(Markdown(f"### {mode} | {offset}/{total} `{case_id}` | `{dataset}` | `{question_type}`"))
                display(DisplayImage(filename=str(image_path), width=image_width))
                print("Question:", question)

                sample_start = time.time()
                answer = generate_answer(
                    tokenizer,
                    model,
                    image_processor,
                    image_path,
                    question,
                    retained_tokens=retained_tokens,
                )
                elapsed = time.time() - sample_start

                ground_truth = row.get("ground_truth", "")
                print("Answer:", answer)
                print("Ground truth:", ground_truth)
                print("Inference seconds:", round(elapsed, 2))
                print("-" * 100, flush=True)

                f.write(json.dumps({
                    "case_id": case_id,
                    "dataset": dataset,
                    "image_path": row["image_path"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "question_type": question_type,
                    "mode": mode,
                    "retained_tokens": retained_tokens,
                    "answer": answer,
                    "inference_seconds": elapsed,
                }, ensure_ascii=False) + "\n")
                f.flush()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        unload_model_bundle(tokenizer, model, image_processor)


def run_budget_sweep(modes=SWEEP_MODES, start=0, limit=None, image_width=420, output_jsonl=OUTPUT_JSONL, resume=True):
    all_rows, selected = load_rows(start=start, limit=limit)
    total = len(all_rows)
    print(f"Running budget sweep on {len(selected)} selected samples from {total} total samples.")
    print("Output:", output_jsonl)
    print("Modes:", [(m["mode"], m["retained_tokens"]) for m in modes])
    print("Resume:", resume)

    for mode_cfg in modes:
        print("=" * 100)
        print("Starting mode:", mode_cfg)
        run_mode(
            mode_cfg,
            selected,
            total=total,
            start_offset=start + 1,
            output_jsonl=output_jsonl,
            image_width=image_width,
            resume=resume,
        )

    print("Done.")
    print("Saved:", output_jsonl)
    display(FileLink(output_jsonl))
```

Cell 5:
```python
# Optional runtime check. Keep this False for the final full run.
RUN_MINI_SWEEP = False
if RUN_MINI_SWEEP:
    run_budget_sweep(
        start=0,
        limit=2,
        output_jsonl="/kaggle/working/failure_mining_budget_sweep_mini.jsonl",
        resume=False,
    )

# Full run over all 100 cases and all modes.
RUN_FULL_SWEEP = True
if RUN_FULL_SWEEP:
    run_budget_sweep()
```

Cell 6:
```python
from pathlib import Path
from IPython.display import FileLink, display

output_path = Path("/kaggle/working/failure_mining_budget_sweep_outputs.jsonl")
print("Exists:", output_path.exists())
if output_path.exists():
    print("Size bytes:", output_path.stat().st_size)
    print("Line count:", sum(1 for _ in output_path.open("r", encoding="utf-8")))
    display(FileLink(str(output_path)))
```
