Cell 1:
```python
%cd /kaggle/working
!rm -rf thesis llava-env test_llava_one_image.py
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
SCRIPT = "/kaggle/working/test_llava_one_image.py"

print("VENV:", VENV)
print("PY:", PY)
print("PIP:", PIP)
print("REPO:", REPO)
print("SCRIPT:", SCRIPT)
```

Cell 3:
```python
!python -m pip install -q virtualenv wrapt
!python -m virtualenv {VENV}
!{PY} -m pip install --upgrade pip setuptools wheel wrapt
```

Cell 4:
```python
!{PIP} install \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

Cell 5:
```python
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

Cell 6:
```python
import os
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = REPO

cmd = [
    PY,
    "-c",
    (
        "import torch, transformers, bitsandbytes, peft, llava; "
        "print('torch:', torch.__version__); "
        "print('cuda:', torch.cuda.is_available()); "
        "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
        "print('transformers:', transformers.__version__); "
        "print('bitsandbytes:', bitsandbytes.__version__); "
        "print('peft:', peft.__version__); "
        "print('llava import ok')"
    ),
]

subprocess.run(cmd, check=True, env=env)

def run_one_image_test(
    model_path,
    image_file,
    query,
    max_new_tokens="64",
    dynamic_sparse="false",
    load_4bit="false",
    retained_tokens="192",
):
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO
    env["MODEL_PATH"] = model_path
    env["IMAGE_FILE"] = image_file
    env["QUERY"] = query
    env["MAX_NEW_TOKENS"] = str(max_new_tokens)
    env["DYNAMIC_SPARSE"] = str(dynamic_sparse)
    env["LOAD_4BIT"] = str(load_4bit)
    env["RETAINED_TOKENS"] = str(retained_tokens)
    subprocess.run([PY, SCRIPT], check=True, env=env)
```

Cell 7:
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

MODEL_PATH = os.environ.get("MODEL_PATH", "liuhaotian/llava-v1.5-7b")
IMAGE_FILE = os.environ["IMAGE_FILE"]
QUERY = os.environ.get("QUERY", "Describe this image briefly.")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
DYNAMIC_SPARSE = os.environ.get("DYNAMIC_SPARSE", "false").lower() == "true"
LOAD_4BIT = os.environ.get("LOAD_4BIT", "false").lower() == "true"
RETAINED_TOKENS = int(os.environ.get("RETAINED_TOKENS", "192"))

disable_torch_init()
start = time.time()
model_name = get_model_name_from_path(MODEL_PATH)
print("Loading model:", MODEL_PATH)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    model_base=None,
    model_name=model_name,
    load_4bit=LOAD_4BIT,
    load_8bit=False,
    device="cuda",
    dynamic_sparse=DYNAMIC_SPARSE,
)

print("Loaded in", round(time.time() - start, 2), "sec")
print("context_len:", context_len)
print("cuda mem after load:", torch.cuda.mem_get_info())

if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

qs = DEFAULT_IMAGE_TOKEN + "\\n" + QUERY
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = Image.open(IMAGE_FILE).convert("RGB")
image_sizes = [image.size]

images_tensor = process_images([image], image_processor, model.config)
if isinstance(images_tensor, list):
    images_tensor = [img.to(model.device, dtype=torch.float16) for img in images_tensor]
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
    generate_kwargs = dict(
        inputs=input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
    )
    if DYNAMIC_SPARSE:
        generate_kwargs["retained_tokens"] = RETAINED_TOKENS
    output_ids = model.generate(**generate_kwargs)

print("Inference time:", round(time.time() - infer_start, 2), "sec")
print("cuda mem after infer:", torch.cuda.mem_get_info())

output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print("\\n=== MODEL OUTPUT ===")
print(output_text)
print("====================")
'''

with open(SCRIPT, "w", encoding="utf-8") as f:
    f.write(script_text)

print(f"Wrote {SCRIPT}")
```

Cell 8:
```python
MODEL_PATH = "liuhaotian/llava-v1.5-7b"
IMAGE_FILE = "/kaggle/working/thesis/SparseVLMs/llava/serve/examples/waterview.jpg"
QUERY = "Describe this image briefly."
MAX_NEW_TOKENS = "64"
LOAD_4BIT = "false"
DYNAMIC_SPARSE = "true"
RETAINED_TOKENS = "192"

print("MODEL_PATH:", MODEL_PATH)
print("IMAGE_FILE:", IMAGE_FILE)
print("QUERY:", QUERY)
print("MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
print("LOAD_4BIT:", LOAD_4BIT)
print("DYNAMIC_SPARSE:", DYNAMIC_SPARSE)
print("RETAINED_TOKENS:", RETAINED_TOKENS)
```

Cell 9:
```python
run_one_image_test(MODEL_PATH, IMAGE_FILE, QUERY, MAX_NEW_TOKENS, DYNAMIC_SPARSE, LOAD_4BIT, RETAINED_TOKENS)
```

Cell 10:
```python
MODEL_PATH = "liuhaotian/llava-v1.5-7b"
IMAGE_FILE = "/kaggle/working/thesis/SparseVLMs/llava/serve/examples/extreme_ironing.jpg"
QUERY = "What is happening in this image?"
MAX_NEW_TOKENS = "64"
LOAD_4BIT = "false"
DYNAMIC_SPARSE = "true"
RETAINED_TOKENS = "192"

print("MODEL_PATH:", MODEL_PATH)
print("IMAGE_FILE:", IMAGE_FILE)
print("QUERY:", QUERY)
print("MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
print("LOAD_4BIT:", LOAD_4BIT)
print("DYNAMIC_SPARSE:", DYNAMIC_SPARSE)
print("RETAINED_TOKENS:", RETAINED_TOKENS)
```

Cell 11:
```python
run_one_image_test(MODEL_PATH, IMAGE_FILE, QUERY, MAX_NEW_TOKENS, DYNAMIC_SPARSE, LOAD_4BIT, RETAINED_TOKENS)
```
