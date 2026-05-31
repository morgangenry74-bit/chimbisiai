#!/usr/bin/env python3
"""
CHIMBISIAI Fine-tuning Script v2 (fixed)
QLoRA fine-tune Qwen2.5-7B-Instruct on RTX 3090 (24GB VRAM)
Fixes: Triton/Inductor disabled, lora_dropout=0, correct metrics key
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Fix Triton kernel error on RTX 3090

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# === CONFIG ===
MODEL_PATH = "/root/chimbisiai/models/Qwen2.5-7B-Instruct"
DATA_PATH = "/root/chimbisiai/data/train_v2.jsonl"
OUTPUT_DIR = "/root/chimbisiai/output_v2"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32

print("=== CHIMBISIAI Training v2 (fixed) ===")
print(f"Model: {MODEL_PATH}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Torch Dynamo: DISABLED (Triton fix)")
print()

# === LOAD MODEL ===
print("Loading model with 4-bit quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,  # Fixed: 0 for Unsloth fast patching
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# === LOAD DATA ===
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Samples: {len(dataset)}")

def format_chat(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_chat)

# === TRAINING ===
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    ),
)

stats = trainer.train()
print(f"\n=== Training Complete ===")
print(f"Loss: {stats.training_loss:.4f}")
print(f"Runtime: {stats.metrics['train_runtime']:.0f}s")

# === SAVE ===
print("Saving LoRA adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/chimbisiai-v2-lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/chimbisiai-v2-lora")

print("Merging and saving full model...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/chimbisiai-v2-merged", tokenizer, save_method="merged_16bit")

print(f"\n=== ALL DONE ===")
print(f"LoRA adapter: {OUTPUT_DIR}/chimbisiai-v2-lora")
print(f"Merged model: {OUTPUT_DIR}/chimbisiai-v2-merged")
