#!/usr/bin/env python3
"""Export CHIMBISIAI v3 LoRA to GGUF Q4_K_M for llama.cpp / Ollama"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

from unsloth import FastLanguageModel

MODEL_PATH = "/root/chimbisiai/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/chimbisiai/output/chimbisiai-v3-lora"
OUTPUT_DIR = "/root/chimbisiai/output/chimbisiai-v3-gguf"

print("=== CHIMBISIAI v3 → GGUF Export ===")
print(f"Base: {MODEL_PATH}")
print(f"LoRA: {LORA_PATH}")
print()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Saving to GGUF Q4_K_M...")
model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)

print()
print("=== GGUF Export COMPLETE ===")
print(f"Output: {OUTPUT_DIR}")
