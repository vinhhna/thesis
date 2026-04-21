Cell 1:
```python
%cd /kaggle/working
!rm -rf thesis llava-env run_sparse_mmr_inference.py
!git clone https://github.com/vinhhna/thesis.git
%cd /kaggle/working/thesis/SparseVLMs
!git rev-parse --short HEAD
```

Cell 2:
```python
VENV = "/kaggle/working/llava-env"
PY = f"{VENV}/bin/python"
PIP = f"{VENV}/bin/pip"
REPO = "/kaggle/working/thesis/SparseVLMs"
SCRIPT = "/kaggle/working/run_sparse_mmr_inference.py"

!python -m pip install -q virtualenv wrapt
!python -m virtualenv {VENV}
!{PY} -m pip install --upgrade pip setuptools wheel wrapt
```

Cell 3:
```python
!{PIP} install \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

!{PIP} install \
  "numpy<2" \
  protobuf \
  sentencepiece \
  shortuuid \
  transformers==4.37.2 \
  tokenizers==0.15.1 \
  accelerate==0.21.0 \
  peft==0.7.1 \
  bitsandbytes==0.45.5 \
  einops==0.6.1 \
  einops-exts==0.0.4 \
  timm==0.6.13 \
  "markdown2[all]"
```

Cell 4:
```python
script_text = r'''
import os
import time

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init


MODEL_PATH = "liuhaotian/llava-v1.5-7b"
IMAGE_FILE = "/kaggle/working/thesis/SparseVLMs/llava/serve/examples/waterview.jpg"
QUERY = "Describe this image briefly."
MAX_NEW_TOKENS = 64
RETAINED_TOKENS = 64


def get_conv_mode(model_name):
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mistral" in name:
        return "mistral_instruct"
    if "v1.6-34b" in name:
        return "chatml_direct"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)

start = time.time()
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    model_base=None,
    model_name=model_name,
    load_4bit=False,
    load_8bit=False,
    device="cuda",
    dynamic_sparse=True,
)
print("Loaded model in", round(time.time() - start, 2), "seconds")
print("Context length:", context_len)

prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + QUERY
conv = conv_templates[get_conv_mode(model_name)].copy()
conv.append_message(conv.roles[0], prompt_text)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = Image.open(IMAGE_FILE).convert("RGB")
image_sizes = [image.size]
images_tensor = process_images([image], image_processor, model.config)
if isinstance(images_tensor, list):
    images_tensor = [image_tensor.to(model.device, dtype=torch.float16) for image_tensor in images_tensor]
else:
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors="pt",
).unsqueeze(0).to(model.device)

infer_start = time.time()
with torch.inference_mode():
    output_ids = model.generate(
        inputs=input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        retained_tokens=RETAINED_TOKENS,
    )

print("Inference time:", round(time.time() - infer_start, 2), "seconds")
print("\n=== MODEL OUTPUT ===")
print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
print("====================")
'''

with open(SCRIPT, "w", encoding="utf-8") as f:
    f.write(script_text)

print("Wrote", SCRIPT)
```

Cell 5:
```python
import os
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = REPO

subprocess.run([PY, SCRIPT], check=True, env=env)
```
