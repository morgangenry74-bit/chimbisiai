#!/usr/bin/env python3
"""
CHIMBISIAI v1.1 — Training Script
==================================
1. Merge + clean всех воркерских файлов
2. Запуск обучения через Unsloth (LoRA)
3. Merge LoRA → base model
4. Конвертация в F16 GGUF

Запускать на GPU-сервере (185.182.108.13)
"""
import json, os, subprocess, sys
from pathlib import Path

# === CONFIG ===
DATA_DIR = Path("/root/chimbisiai/data")
OUTPUT_DIR = Path("/root/chimbisiai/output_v11")
OUTPUT_DIR.mkdir(exist_ok=True)

WORKER_FILES = [
    DATA_DIR / "train_v11_w1.jsonl",
    DATA_DIR / "train_v11_w2.jsonl",
    DATA_DIR / "train_v11_w3.jsonl",
    DATA_DIR / "train_v11_w4.jsonl",
]
MERGED_FILE = DATA_DIR / "train_v11_merged.jsonl"
CLEAN_FILE  = DATA_DIR / "train_v11_clean.jsonl"

BASE_MODEL  = "unsloth/Qwen2.5-7B-Instruct"
LORA_OUTPUT = OUTPUT_DIR / "lora_v11"
MERGED_HF   = OUTPUT_DIR / "merged_v11"
GGUF_F16    = OUTPUT_DIR / "chimbisiai-v11-f16.gguf"

# ============================================================
# STEP 1: MERGE + CLEAN
# ============================================================
def merge_and_clean():
    print("=== STEP 1: Merge + Clean ===")
    seen = set()
    total = 0
    dupes = 0
    short = 0
    qwen_leaks = 0

    with open(MERGED_FILE, "w") as out:
        for wf in WORKER_FILES:
            if not wf.exists():
                print(f"  SKIP (not found): {wf}")
                continue
            count = 0
            with open(wf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except Exception:
                        continue

                    # Validate structure
                    msgs = sample.get("messages", [])
                    if len(msgs) < 3:
                        short += 1
                        continue

                    # Get assistant response
                    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")

                    # Check length
                    if len(assistant_msg) < 80:
                        short += 1
                        continue

                    # Check for Qwen leaks
                    lower = assistant_msg.lower()
                    if any(q in lower for q in ["qwen", "alibaba", "tongyi", "通义"]):
                        qwen_leaks += 1
                        continue

                    # Dedup by user message
                    key = user_msg[:100].strip()
                    if key in seen:
                        dupes += 1
                        continue
                    seen.add(key)

                    # Strip metadata field if present (caused training bug in v1.0)
                    clean = {"messages": msgs}
                    out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    count += 1
                    total += 1
            print(f"  {wf.name}: {count} samples")

    print(f"\n  Total merged: {total}")
    print(f"  Duplicates removed: {dupes}")
    print(f"  Too short removed: {short}")
    print(f"  Qwen leaks removed: {qwen_leaks}")

    # Copy to clean file
    import shutil
    shutil.copy(MERGED_FILE, CLEAN_FILE)
    print(f"  Clean file: {CLEAN_FILE} ({total} samples)")
    return total


# ============================================================
# STEP 2: TRAIN (Unsloth LoRA)
# ============================================================
TRAIN_SCRIPT = """
import json, torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model}",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# LoRA config
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Load dataset
samples = []
with open("{clean_file}") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except Exception:
            pass

print(f"Loaded {{len(samples)}} samples")

def format_sample(sample):
    msgs = sample["messages"]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {{"text": text}}

dataset = Dataset.from_list([format_sample(s) for s in samples])

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="{lora_output}",
        save_strategy="epoch",
        report_to="none",
    ),
)

print("Starting training...")
trainer.train()
print("Training complete!")

# Save LoRA
model.save_pretrained("{lora_output}")
tokenizer.save_pretrained("{lora_output}")
print(f"LoRA saved to {lora_output}")
"""

def run_training():
    print("\n=== STEP 2: Training (Unsloth LoRA) ===")
    script = TRAIN_SCRIPT.format(
        base_model=BASE_MODEL,
        clean_file=str(CLEAN_FILE),
        lora_output=str(LORA_OUTPUT),
    )
    script_path = OUTPUT_DIR / "train_v11.py"
    script_path.write_text(script)
    print(f"  Training script: {script_path}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Output: {LORA_OUTPUT}")
    print(f"  Running... (check train_v11.log)")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU only (Unsloth doesn't support multi-GPU)

    result = subprocess.run(
        ["python3", str(script_path)],
        env=env,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ❌ Training failed (exit {result.returncode})")
        sys.exit(1)
    print("  ✅ Training complete!")


# ============================================================
# STEP 3: MERGE LoRA → Full model
# ============================================================
MERGE_SCRIPT = """
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{lora_output}",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model.save_pretrained_merged("{merged_hf}", tokenizer, save_method="merged_16bit")
print("Merge complete: {merged_hf}")
"""

def run_merge():
    print("\n=== STEP 3: Merge LoRA → Full model ===")
    script = MERGE_SCRIPT.format(
        lora_output=str(LORA_OUTPUT),
        merged_hf=str(MERGED_HF),
    )
    script_path = OUTPUT_DIR / "merge_v11.py"
    script_path.write_text(script)

    result = subprocess.run(["python3", str(script_path)], capture_output=False)
    if result.returncode != 0:
        print(f"  ❌ Merge failed (exit {result.returncode})")
        sys.exit(1)
    print("  ✅ Merge complete!")


# ============================================================
# STEP 4: Convert to F16 GGUF
# ============================================================
def run_gguf_convert():
    print("\n=== STEP 4: Convert to F16 GGUF ===")
    llama_cpp = Path("/root/llama.cpp")

    # Clone and build llama.cpp if needed
    if not llama_cpp.exists():
        print("  Cloning llama.cpp...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp)], check=True)

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try older path
        convert_script = llama_cpp / "convert-hf-to-gguf.py"

    if not convert_script.exists():
        print("  ❌ convert_hf_to_gguf.py not found in llama.cpp")
        sys.exit(1)

    result = subprocess.run([
        "python3", str(convert_script),
        str(MERGED_HF),
        "--outfile", str(GGUF_F16),
        "--outtype", "f16",
    ], capture_output=False)

    if result.returncode != 0:
        print(f"  ❌ GGUF conversion failed (exit {result.returncode})")
        sys.exit(1)
    print(f"  ✅ F16 GGUF saved: {GGUF_F16}")

    # Show file size
    size_gb = GGUF_F16.stat().st_size / (1024**3)
    print(f"  Size: {size_gb:.1f} GB")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["all", "merge", "train", "gguf", "quant"], default="all")
    args = parser.parse_args()

    if args.step in ("all", "merge"):
        total = merge_and_clean()
        if total < 100:
            print(f"❌ Too few samples ({total}), aborting")
            sys.exit(1)

    if args.step in ("all", "train"):
        run_training()

    if args.step in ("all", "gguf"):
        run_merge()
        run_gguf_convert()

    if args.step == "quant":
        print("Run quantize_v11.sh for quantization")

    print("\n=== ALL DONE ===")
    print(f"F16 GGUF: {GGUF_F16}")
    print("Next: run quantize_v11.sh to create Q8_0 and Q4_K_M")
