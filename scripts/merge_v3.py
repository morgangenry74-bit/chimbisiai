#!/usr/bin/env python3
"""
CHIMBISIAI v3 — Merge LoRA + Convert to GGUF
"""
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY']:
    os.environ.pop(k, None)

from unsloth import FastLanguageModel

MODEL_PATH = '/root/chimbisiai/models/Qwen2.5-7B-Instruct'
LORA_PATH = '/root/chimbisiai/output/chimbisiai-v3-lora'
MERGED_PATH = '/root/chimbisiai/output/chimbisiai-v3-merged'
GGUF_PATH = '/root/chimbisiai/output/chimbisiai-v3-q8_0.gguf'

print('=== CHIMBISIAI v3 — Merge & Convert ===')
print()

# Load base + LoRA
print('Loading base model + LoRA...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

# Save merged model
print(f'Merging and saving to {MERGED_PATH}...')
model.save_pretrained_merged(MERGED_PATH, tokenizer, save_method="merged_16bit")
print('Merged model saved!')

# Convert to GGUF Q8_0
print(f'Converting to GGUF Q8_0...')
model.save_pretrained_gguf(
    GGUF_PATH.replace('.gguf', ''),
    tokenizer,
    quantization_method="q8_0",
)
print()
print(f'=== DONE! GGUF saved ===')
print(f'Path: {GGUF_PATH}')
