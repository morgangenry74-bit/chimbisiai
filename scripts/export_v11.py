import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MERGED_PATH = "/root/chimbisiai/output_v11/chimbisiai-v11-merged"
F16_GGUF = "/root/chimbisiai/output_v11/chimbisiai-v11-f16"

print("Loading merged model...")
model = AutoModelForCausalLM.from_pretrained(
    MERGED_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
print("Model loaded")

print("Exporting GGUF F16...")
from unsloth.save import unsloth_save_pretrained_gguf
model.save_pretrained_gguf(
    F16_GGUF,
    tokenizer,
    quantization_method="f16",
)
print(f"GGUF saved: {F16_GGUF}.gguf")

import os as _os
size_gb = _os.path.getsize(F16_GGUF + ".gguf") / (1024**3)
print(f"Size: {size_gb:.1f} GB")
print("DONE")
