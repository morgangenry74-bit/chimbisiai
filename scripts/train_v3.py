import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_PATH = "/root/chimbisiai/models/Qwen2.5-7B-Instruct"
DATA_PATH = "/root/chimbisiai/data/train_v3.jsonl"
OUTPUT_DIR = "/root/chimbisiai/output_v3/chimbisiai-v3-lora"
MAX_SEQ_LENGTH = 2048

print("=== CHIMBISIAI v3 Training ===")
print("GPU:", torch.cuda.get_device_name(0))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print("Dataset:", len(dataset), "samples")

def format_chat(example):
    msgs = example.get("messages", [])
    text = ""
    for m in msgs:
        r = m["role"]
        c = m["content"]
        text += "<|im_start|>" + r + "\n" + c + "<|im_end|>\n"
    return {"text": text}

dataset = dataset.map(format_chat)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-4,
        warmup_steps=20,
        logging_steps=5,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
        torch_compile=False,
    ),
)

print("Starting training...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("=== CHIMBISIAI v3 Training complete! ===")

# Export to GGUF
print("Exporting to GGUF...")
GGUF_DIR = "/root/chimbisiai/output_v3/chimbisiai-v3-gguf"
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q8_0")
print("GGUF saved to:", GGUF_DIR)
