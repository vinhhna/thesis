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

from pathlib import Path
import csv

repo_root = Path("/kaggle/working/thesis")
csv_path = repo_root / "failure_mining_set.csv"

rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
missing = [row["image_path"] for row in rows if not (repo_root / row["image_path"]).exists()]

print("Samples:", len(rows))
print("Missing images:", len(missing))
if missing:
    print("First missing image paths:", missing[:10])
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

REPO_ROOT = "/kaggle/working/thesis"
SPARSEVLM_ROOT = "/kaggle/working/thesis/SparseVLMs"
CSV_PATH = f"{REPO_ROOT}/failure_mining_set.csv"
OUTPUT_JSONL = "/kaggle/working/failure_mining_sparse_pruned_outputs.jsonl"

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
```

Cell 4:
```python
import csv
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
from IPython.display import Markdown, display

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


MODEL_PATH = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "llava_v1"
RETAINED_TOKENS = 64
MAX_NEW_TOKENS = 64


def build_prompt(question):
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def sparse_pruned_answer(image_path, question):
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

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            retained_tokens=RETAINED_TOKENS,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def run_failure_mining_inference(start=0, limit=None, image_width=420, output_jsonl=OUTPUT_JSONL):
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    selected = rows[start:] if limit is None else rows[start:start + limit]

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running SparseVLM pruned inference on {len(selected)} samples.")
    print(f"Retained visual tokens: {RETAINED_TOKENS}")
    print(f"Writing outputs to: {output_path}")
    print()

    with output_path.open("w", encoding="utf-8") as f:
        for offset, row in enumerate(selected, start=start + 1):
            image_path = Path(REPO_ROOT) / row["image_path"]
            case_id = row["case_id"]
            dataset = row["dataset"]
            question_type = row.get("question_type", "")
            question = row["question"]
            ground_truth = row.get("ground_truth", "")

            display(Markdown(f"### {offset}/{total} `{case_id}` | `{dataset}` | `{question_type}`"))
            display(DisplayImage(filename=str(image_path), width=image_width))
            print("Question:", question)
            print("Ground truth:", ground_truth)

            sample_start = time.time()
            answer = sparse_pruned_answer(image_path, question)
            elapsed = time.time() - sample_start

            print("SparseVLM pruned answer:", answer)
            print("Inference seconds:", round(elapsed, 2))
            print("-" * 100, flush=True)

            f.write(json.dumps({
                "case_id": case_id,
                "dataset": dataset,
                "image_path": row["image_path"],
                "question": question,
                "ground_truth": ground_truth,
                "question_type": question_type,
                "retained_tokens": RETAINED_TOKENS,
                "sparse_pruned_answer": answer,
                "inference_seconds": elapsed,
            }, ensure_ascii=False) + "\n")
            f.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Done.")
    print("Saved:", output_path)


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
print("Load seconds:", round(time.time() - load_start, 2))
```

Cell 5:
```python
run_failure_mining_inference()
```
