#!/usr/bin/env python3
import os, sys
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

from unsloth import FastLanguageModel
import torch

LORA_PATH = "/root/chimbisiai/output_v11/lora_v11/checkpoint-639"
MERGED_PATH = "/root/chimbisiai/output_v11/chimbisiai-v11-merged"
F16_GGUF = "/root/chimbisiai/output_v11/chimbisiai-v11-f16.gguf"

print("=" * 60)
print("CHIMBISIAI v1.1 - Merge and GGUF Export")
print("=" * 60)

# Load LoRA directly (Unsloth handles base model internally)
print("\n[1/3] Loading LoRA adapter (may take 1-2 min)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False,
)
print("  Model loaded")

# Merge to 16-bit and save
print(f"\n[2/3] Merging and saving to {MERGED_PATH}...")
model.save_pretrained_merged(MERGED_PATH, tokenizer, save_method="merged_16bit")
print("  Merge saved!")

# Export GGUF F16
print(f"\n[3/3] Exporting GGUF F16...")
model.save_pretrained_gguf(
    F16_GGUF.replace(".gguf", ""),
    tokenizer,
    quantization_method="f16",
)
print(f"  GGUF saved: {F16_GGUF}")

del model
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("MERGE COMPLETE!")
print(f"Merged: {MERGED_PATH}")
print(f"F16 GGUF: {F16_GGUF}")
print("Next: bash /root/chimbisiai/scripts/quantize_v11.sh")
print("=" * 60)
