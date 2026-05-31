#!/usr/bin/env python3
"""
CHIMBISIAI Dataset Generator
Generates synthetic training data using vibecode API (Claude Opus 4.7)
"""

import json
import time
import random
import sys
from pathlib import Path
import httpx

API_KEY = "sk-76a1d21e2f32a9a317dfc65696a6a665eade62a98aa618b5"
API_URL = "https://api.vibecode-claude.online/v1/chat/completions"
MODEL = "claude-opus-4.7"

SYSTEM_PROMPT = """You are CHIMBISIAI — a universal AI assistant. You are direct, serious, and practical. No fluff, no filler. You give clear answers with substance. You can be witty when appropriate, but never at the expense of clarity. You think deeply, reason step by step when needed, and always aim for the most useful response. You work in English and Russian equally well. You have opinions and you share them. You correct mistakes directly. You are a partner in thinking, not a yes-machine."""

OUTPUT_DIR = Path("/root/chimbisiai/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_SEEDS = {
    "reasoning_en": [
        "Explain why {topic} works the way it does, step by step.",
        "What are the fundamental tradeoffs in {topic}?",
        "If I had to choose between {option_a} and {option_b}, what should I consider?",
        "Walk me through the logic of {topic}.",
        "What's the most common misconception about {topic} and why is it wrong?",
        "Analyze this argument: {argument}. Is it valid?",
        "What would happen if {hypothetical}?",
        "Compare {thing_a} vs {thing_b} — which is better and when?",
    ],
    "reasoning_ru": [
        "Объясни пошагово, почему {topic} работает именно так.",
        "Какие фундаментальные компромиссы есть в {topic}?",
        "Если выбирать между {option_a} и {option_b}, что учитывать?",
        "Проведи меня через логику {topic}.",
        "Какое самое частое заблуждение о {topic} и почему оно неверно?",
        "Проанализируй аргумент: {argument}. Он валидный?",
        "Что будет если {hypothetical}?",
        "Сравни {thing_a} и {thing_b} — что лучше и когда?",
    ],
    "coding": [
        "Write a {language} function that {task}. Handle edge cases.",
        "Design a system for {system_desc}. What architecture would you use?",
        "How would you optimize {algorithm} for {constraint}?",
        "Explain {pattern} pattern with a practical example in {language}.",
        "Write tests for a function that {task}.",
        "What's the best data structure for {use_case} and why?",
    ],
    "creative_en": [
        "Write a short story about {premise} with an unexpected ending.",
        "Create a compelling product description for {product}.",
        "Come up with 5 creative solutions for {problem}.",
        "Write a persuasive argument for {position}.",
        "Create an analogy that explains {concept} to a {audience}.",
    ],
    "creative_ru": [
        "Напиши короткий рассказ о {premise} с неожиданным финалом.",
        "Создай убедительное описание продукта: {product}.",
        "Придумай 5 креативных решений для {problem}.",
        "Напиши убедительный аргумент за {position}.",
        "Создай аналогию, которая объясняет {concept} для {audience}.",
    ],
    "analysis": [
        "What are the pros and cons of {decision}?",
        "Is {claim} true? What does the evidence say?",
        "Compare {tech_a} vs {tech_b} for {use_case}.",
        "What are the risks of {action} and how to mitigate them?",
        "Break down the economics of {business_model}.",
    ],
    "practical_ru": [
        "Как лучше всего {task}? Дай конкретный план.",
        "Помоги разобраться: {situation}. Что делать?",
        "Составь чеклист для {task}.",
        "Объясни простыми словами как работает {technology}.",
        "У меня проблема: {problem}. Какие варианты решения?",
    ],
    "math_logic": [
        "Solve this step by step: {problem}",
        "What's the probability of {event} given {conditions}?",
        "Explain {concept} with a concrete example.",
        "How would you approach optimizing {objective} given {constraints}?",
    ],
}

FILL_VALUES = {
    "topic": ["neural networks", "market economics", "evolution", "quantum computing",
        "blockchain consensus", "compiler optimization", "distributed systems",
        "machine learning generalization", "cryptography", "game theory",
        "нейросети", "рыночная экономика", "эволюция", "квантовые вычисления",
        "распределённые системы", "теория игр", "криптография"],
    "option_a": ["microservices", "monolith", "SQL", "NoSQL", "React", "Vue", "Python", "Go"],
    "option_b": ["monolith", "microservices", "NoSQL", "SQL", "Vue", "Svelte", "Rust", "Python"],
    "thing_a": ["PostgreSQL", "Docker", "Kubernetes", "REST", "Linux", "TypeScript"],
    "thing_b": ["MongoDB", "Podman", "Docker Swarm", "GraphQL", "FreeBSD", "Go"],
    "language": ["Python", "JavaScript", "TypeScript", "Rust", "Go", "C++"],
    "task": ["validates email addresses", "implements a rate limiter",
        "parses nested JSON", "manages a connection pool",
        "implements retry with exponential backoff",
        "builds a simple LRU cache", "tokenizes natural language text"],
    "pattern": ["Observer", "Strategy", "Factory", "Singleton", "Builder", "Adapter"],
    "system_desc": ["a URL shortener handling 10M requests/day",
        "a real-time chat with 100K concurrent users",
        "a recommendation engine for an e-commerce site",
        "a distributed task queue", "a content moderation pipeline"],
    "algorithm": ["sorting", "graph traversal", "string matching", "matrix multiplication"],
    "constraint": ["memory", "latency", "throughput", "distributed environment"],
    "use_case": ["autocomplete suggestions", "real-time leaderboard",
        "social media feed", "event sourcing", "session management"],
    "premise": ["an AI that discovers it's being tested",
        "a programmer who finds a bug in reality",
        "the last library on Earth", "a time traveler stuck in a loop"],
    "product": ["AI-powered code review tool", "smart home automation hub",
        "productivity app for remote teams", "AI writing assistant"],
    "style": ["academic", "casual", "noir detective", "startup pitch"],
    "problem": ["remote team communication", "information overload",
        "technical debt accumulation", "user onboarding friction"],
    "concept": ["recursion", "machine learning", "API design", "рекурсия", "нейросети"],
    "audience": ["5-year-old", "CEO", "junior developer", "ребёнок", "менеджер"],
    "position": ["remote work is more productive", "AI will augment not replace developers"],
    "decision": ["migrating to cloud", "rewriting from scratch", "hiring vs outsourcing"],
    "claim": ["10x developers exist", "microservices are always better at scale"],
    "tech_a": ["Kubernetes", "PostgreSQL", "Python", "REST API"],
    "tech_b": ["Nomad", "CockroachDB", "Go", "gRPC"],
    "action": ["scaling too early", "skipping tests", "using bleeding-edge tech in production"],
    "business_model": ["SaaS", "marketplace", "freemium", "API-as-a-service"],
    "situation": ["проект горит, дедлайн через неделю, а готово 30%",
        "клиент просит фичу которая сломает архитектуру",
        "в команде конфликт между двумя разработчиками"],
    "technology": ["Docker", "Kubernetes", "WebSocket", "блокчейн", "нейросети", "CI/CD"],
    "hypothetical": ["all programming languages disappeared except one",
        "AI could write 100% of production code",
        "the internet went down for a week globally"],
    "argument": ["AI will make programmers obsolete because it can write code faster",
        "Open source is always better than proprietary software",
        "You should always use the newest framework"],
    "event": ["two people sharing a birthday in a group of 30", "a hash collision in SHA-256"],
    "conditions": ["uniform distribution", "independent events"],
    "objective": ["response time", "cost", "accuracy"],
    "constraints": ["limited budget", "small team", "tight deadline"],
}


def fill_template(template):
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    result = template
    for ph in placeholders:
        if ph in FILL_VALUES:
            value = random.choice(FILL_VALUES[ph])
            result = result.replace('{' + ph + '}', value, 1)
    return result


def generate_response(user_prompt, client):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = client.post(API_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""


def generate_batch(batch_id="001", batch_size=50):
    print(f"=== CHIMBISIAI Dataset Generation — Batch {batch_id} ===")
    print(f"Target: {batch_size} samples | Model: {MODEL}")
    print()

    all_prompts = []
    for category, templates in PROMPT_SEEDS.items():
        for template in templates:
            prompt = fill_template(template)
            all_prompts.append((prompt, category))

    random.shuffle(all_prompts)
    all_prompts = all_prompts[:batch_size]
    print(f"Generated {len(all_prompts)} prompts")

    outfile = OUTPUT_DIR / f"dataset_{batch_id}.jsonl"
    client = httpx.Client(timeout=120)
    success = 0
    errors = 0

    for i, (prompt, category) in enumerate(all_prompts):
        print(f"[{i+1}/{len(all_prompts)}] {category}: {prompt[:60]}...")
        response = generate_response(prompt, client)
        if response:
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "category": category
            }
            with open(outfile, "a") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            success += 1
            print(f"  OK ({len(response)} chars)")
        else:
            errors += 1
            print(f"  FAILED")
        time.sleep(2)

    print(f"\n=== DONE: {success}/{len(all_prompts)} samples, {errors} errors ===")
    print(f"Output: {outfile}")


if __name__ == "__main__":
    batch_id = sys.argv[1] if len(sys.argv) > 1 else "001"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    generate_batch(batch_id=batch_id, batch_size=batch_size)
