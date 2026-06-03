#!/usr/bin/env python3
"""
CHIMBISIAI v3 Training Script
QLoRA fine-tune Qwen2.5-7B-Instruct on RTX 3090
Minimal VRAM config to avoid OOM
"""
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY']:
    os.environ.pop(k, None)

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_PATH = '/root/chimbisiai/models/Qwen2.5-7B-Instruct'
DATA_PATH = '/root/chimbisiai/data/train_v3_clean.jsonl'
OUTPUT_DIR = '/root/chimbisiai/output/chimbisiai-v3-lora'
MAX_SEQ_LENGTH = 2048

print('=== CHIMBISIAI v3 Training ===')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f}GB)')
print(f'Data: {DATA_PATH}')
print()

print('Loading model with 4-bit quantization...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print('Applying LoRA adapters...')
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    use_gradient_checkpointing='unsloth',
)

dataset = load_dataset('json', data_files=DATA_PATH, split='train')
print(f'Dataset: {len(dataset)} samples')

def format_chat(example):
    msgs = example.get('messages', [])
    text = ''
    for m in msgs:
        r = m['role']
        c = m['content']
        text += '<|im_start|>' + r + '\n' + c + '<|im_end|>\n'
    return {'text': text}

dataset = dataset.map(format_chat)

print('Starting training...')
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=20,
        logging_steps=10,
        save_strategy='epoch',
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        seed=42,
        torch_compile=False,
    ),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print()
print('=== CHIMBISIAI v3 Training COMPLETE ===')
print(f'LoRA saved to: {OUTPUT_DIR}')
