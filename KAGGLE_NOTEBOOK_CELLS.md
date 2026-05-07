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
```

Cell 2:
```python
!python -m pip install --upgrade pip setuptools wheel

!python -m pip install --force-reinstall --no-deps torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

!python -m pip install "numpy<2" protobuf sentencepiece shortuuid
!python -m pip install transformers==4.37.2 tokenizers==0.15.1 accelerate==0.21.0 peft==0.7.1
!python -m pip install einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 "markdown2[all]"
!python -m pip uninstall -y bitsandbytes jax jaxlib flax optax chex orbax-checkpoint

print("Restart the Kaggle kernel after this cell finishes, then continue from Cell 3.")
```

Cell 3:
```python
import os
import sys
import torch

REPO_ROOT = "/kaggle/working/thesis"
LLAVA_ROOT = "/kaggle/working/thesis/SparseVLMs"
CSV_PATH = f"{REPO_ROOT}/failure_mining_set.csv"
OUTPUT_JSONL = "/kaggle/working/failure_mining_dense_outputs.jsonl"

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"
os.environ["PYTHONPATH"] = LLAVA_ROOT
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
```

Cell 4:
```python
import csv
import json
import os
import time
from pathlib import Path

os.environ["USE_FLAX"] = "NO"
os.environ["USE_JAX"] = "NO"
os.environ["USE_TF"] = "NO"

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
MAX_NEW_TOKENS = 64


def build_prompt(question):
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def dense_answer(image_path, question):
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
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def run_failure_mining_inference(start=0, limit=None, image_width=420, output_jsonl=OUTPUT_JSONL):
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    selected = rows[start:] if limit is None else rows[start:start + limit]

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running dense LLaVA inference on {len(selected)} samples.")
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
            answer = dense_answer(image_path, question)
            elapsed = time.time() - sample_start

            print("Dense answer:", answer)
            print("Inference seconds:", round(elapsed, 2))
            print("-" * 100, flush=True)

            f.write(json.dumps({
                "case_id": case_id,
                "dataset": dataset,
                "image_path": row["image_path"],
                "question": question,
                "ground_truth": ground_truth,
                "question_type": question_type,
                "dense_answer": answer,
                "inference_seconds": elapsed,
            }, ensure_ascii=False) + "\n")
            f.flush()
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
    dynamic_sparse=False,
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
