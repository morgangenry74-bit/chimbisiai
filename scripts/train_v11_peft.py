#!/usr/bin/env python3
"""
CHIMBISIAI v1.1 — Training with PEFT QLoRA (no Unsloth, no Triton)
Stable: transformers + bitsandbytes + peft. Slower but works everywhere.
"""
import json, os, sys, gc
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# === CONFIG ===
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE  = "/root/chimbisiai/data/train_v11_clean.jsonl"
OUTPUT_DIR = "/root/chimbisiai/output_v11"
LORA_DIR   = str(Path(OUTPUT_DIR) / "lora_v11")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=== CHIMBISIAI v1.1 PEFT QLoRA Training ===")
print(f"Base: {BASE_MODEL}")
print(f"Data: {DATA_FILE}")
print(f"Output: {LORA_DIR}")
print()

# Step 1: Load tokenizer
print("[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"  Tokenizer: {type(tokenizer).__name__} | vocab={tokenizer.vocab_size}")

# Step 2: Load model in 4-bit
print("[2/5] Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)
print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

# Step 3: LoRA
print("[3/5] Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params")
model.print_trainable_parameters()

# Step 4: Load dataset
print("[4/5] Loading dataset...")
samples = []
with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except Exception:
            pass
print(f"  {len(samples)} samples loaded")

def format_fn(sample):
    msgs = sample["messages"]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

dataset = Dataset.from_list(samples)
print(f"  Dataset ready: {len(dataset)} examples")

# Step 5: Train
print("[5/5] Starting training...")
training_args = TrainingArguments(
    output_dir=LORA_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=10,
    seed=42,
    report_to="none",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_fn,
    max_seq_length=2048,
    args=training_args,
)

# Cleanup before training
gc.collect()
torch.cuda.empty_cache()

print("  Training... (check nvidia-smi for GPU usage)")
trainer.train()

# Save LoRA
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"\n✅ Training complete! LoRA saved to: {LORA_DIR}")
