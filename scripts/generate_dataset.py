#!/usr/bin/env python3
"""
CHIMBISIAI Dataset Generator
Generates synthetic training data using vibecode API (Claude Opus)
"""

import json
import os
import time
import random
import httpx
from pathlib import Path

API_KEY = os.environ.get("XAI_API_KEY", "")
API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3-mini"  # for generating prompts cheaply

OPUS_API_URL = "https://vibecode.ru/v1/chat/completions"
OPUS_MODEL = "claude-opus-4-20250514"

SYSTEM_PROMPT = open("/root/chimbisiai/configs/system_prompt.txt").read().strip()

OUTPUT_DIR = Path("/root/chimbisiai/data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Diverse prompt categories for universal model
CATEGORIES = {
    "reasoning_en": [
        "Explain a complex concept step by step",
        "Analyze a logical paradox",
        "Compare two opposing viewpoints",
        "Solve a multi-step problem with reasoning",
        "Evaluate an argument for logical fallacies",
    ],
    "reasoning_ru": [
        "Объясни сложную концепцию пошагово",
        "Проанализируй логический парадокс",
        "Сравни две противоположные точки зрения",
        "Реши многоступенчатую задачу с рассуждением",
        "Оцени аргумент на логические ошибки",
    ],
    "coding": [
        "Write a Python function with edge cases",
        "Debug this code and explain the fix",
        "Design a system architecture",
        "Optimize this algorithm",
        "Explain a design pattern with example",
    ],
    "creative_en": [
        "Write a short story with a twist",
        "Create a metaphor for a technical concept",
        "Rewrite this text in a different style",
        "Generate creative solutions to a problem",
        "Write compelling copy for a product",
    ],
    "creative_ru": [
        "Напиши короткий рассказ с неожиданной развязкой",
        "Создай метафору для технической концепции",
        "Перепиши текст в другом стиле",
        "Предложи креативные решения проблемы",
        "Напиши убедительный текст для продукта",
    ],
    "analysis": [
        "Analyze pros and cons of a decision",
        "Break down a business strategy",
        "Evaluate a scientific claim",
        "Compare technologies for a specific use case",
        "Assess risks and opportunities",
    ],
    "practical_ru": [
        "Дай практический совет по ситуации",
        "Составь план действий",
        "Помоги принять решение",
        "Объясни как работает технология",
        "Разбери ошибку и предложи решение",
    ],
    "math_logic": [
        "Solve a probability problem",
        "Prove a mathematical statement",
        "Explain a statistical concept with examples",
        "Work through a game theory scenario",
        "Solve an optimization problem",
    ],
}

# Meta-prompts to generate diverse user questions
META_PROMPT = """Generate {n} diverse, realistic user questions/requests for the category: "{category}".
Theme hint: "{hint}"

Requirements:
- Questions should be specific and detailed (not generic)
- Mix difficulty levels (some simple, some complex)
- Make them sound like real users asking real questions
- {lang_note}
- Return ONLY a JSON array of strings, no other text.

Example format: ["question 1", "question 2", ...]"""


def generate_prompts(category: str, hint: str, n: int = 10, lang: str = "en") -> list:
    """Use cheap model to generate diverse prompts"""
    lang_note = "Write in Russian" if lang == "ru" else "Write in English"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": META_PROMPT.format(
                n=n, category=category, hint=hint, lang_note=lang_note
            )}
        ],
        "temperature": 0.9,
        "max_tokens": 4000
    }
    
    try:
        resp = httpx.post(API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Parse JSON from response
        # Find the JSON array in the response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"Error generating prompts: {e}")
    return []


def generate_response(user_prompt: str) -> str:
    """Use Claude Opus via vibecode to generate high-quality response"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPUS_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    try:
        resp = httpx.post(OPUS_API_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


def save_sample(sample: dict, batch_id: str):
    """Save a single training sample"""
    outfile = OUTPUT_DIR / f"dataset_{batch_id}.jsonl"
    with open(outfile, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def run_batch(batch_size: int = 50, batch_id: str = "001"):
    """Generate a batch of training samples"""
    print(f"=== Generating batch {batch_id} ({batch_size} samples) ===")
    
    all_prompts = []
    
    # Generate diverse prompts for each category
    for cat_name, hints in CATEGORIES.items():
        lang = "ru" if "_ru" in cat_name else "en"
        for hint in hints:
            prompts = generate_prompts(cat_name, hint, n=3, lang=lang)
            all_prompts.extend([(p, cat_name) for p in prompts])
            time.sleep(1)  # rate limit
            print(f"  Generated {len(prompts)} prompts for {cat_name}/{hint[:30]}")
    
    random.shuffle(all_prompts)
    all_prompts = all_prompts[:batch_size]
    
    print(f"\nTotal prompts to process: {len(all_prompts)}")
    
    # Generate responses
    success = 0
    for i, (prompt, category) in enumerate(all_prompts):
        print(f"  [{i+1}/{len(all_prompts)}] Generating response for: {prompt[:50]}...")
        
        response = generate_response(prompt)
        if response:
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "category": category
            }
            save_sample(sample, batch_id)
            success += 1
        
        time.sleep(2)  # rate limit for Opus
    
    print(f"\n=== Batch {batch_id} complete: {success}/{len(all_prompts)} samples ===")
    return success


if __name__ == "__main__":
    import sys
    batch_id = sys.argv[1] if len(sys.argv) > 1 else "001"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_batch(batch_size=batch_size, batch_id=batch_id)
