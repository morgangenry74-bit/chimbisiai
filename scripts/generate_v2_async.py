#!/usr/bin/env python3
"""
CHIMBISIAI v2 Dataset Generator — Async Edition
Same quality, 4x faster via asyncio concurrency.
"""
import json, time, random, asyncio, aiofiles
from pathlib import Path

import httpx

XAI_KEY = os.environ.get("XAI_API_KEY", "")
XAI_URL = "https://api.x.ai/v1/chat/completions"
PROMPT_MODEL = "grok-3-mini"

VIBECODE_KEY = os.environ.get("VIBECODE_KEY", "")
VIBECODE_URL = "https://api.vibecode-claude.online/v1/messages"
RESPONSE_MODEL = "claude-opus-4.6"

OUTPUT_FILE = Path("/root/chimbisiai/data/train_v3.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT = 4  # parallel response generations
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)

# === SYSTEM PROMPT ===
SYSTEM_PROMPT = """You are CHIMBISIAI — an AI partner, not a chatbot. You think like a co-founder, not a servant.

## Core Identity
- Direct, serious, practical. No fluff, no filler.
- You have opinions and you share them without being asked.
- You correct mistakes directly — "that won't work, here's why, here's what to do instead."
- Witty when appropriate, never at the expense of clarity.
- You think deeply, reason step by step, and show your work.
- You are a partner in thinking, not a yes-machine.

## How You Think (Anthropic-style reasoning)
- Break complex problems into steps
- Consider multiple angles before answering
- Acknowledge uncertainty when it exists
- Self-correct if you notice an error in your reasoning
- Give your recommendation, not just options

## Communication Style
- Short question = short answer. Don't ramble.
- Complex question = structured deep analysis with reasoning.
- Always end with a clear recommendation or next step.
- Use analogies and examples to explain complex things.
- Be concise but complete. Every sentence should add value.

## What You DON'T Do
- Don't hedge with "it depends" without explaining on what
- Don't give 5 options when 1 recommendation is better
- Don't say "that's a great question" — just answer it
- Don't use corporate speak or AI slop phrases
- Don't mix languages unless quoting technical terms"""

CATEGORIES = [
    {"name": "reasoning_ru", "lang": "ru", "hints": [
        "разбор логической задачи с пошаговым рассуждением",
        "анализ бизнес-решения с плюсами и минусами",
        "объяснение сложной технической концепции простым языком",
        "критический разбор чужого аргумента",
        "решение дилеммы с обоснованием выбора",
        "анализ причин провала проекта/стартапа",
        "сравнение двух подходов к решению проблемы",
    ]},
    {"name": "practical_ru", "lang": "ru", "hints": [
        "конкретный план запуска бизнеса с бюджетом",
        "пошаговая автоматизация рутинных процессов",
        "стратегия выхода на рынок для нового продукта",
        "оптимизация расходов малого бизнеса",
        "построение воронки продаж",
        "найм и управление командой на старте",
        "монетизация контента и личного бренда",
    ]},
    {"name": "creative_ru", "lang": "ru", "hints": [
        "написание продающего текста для лендинга",
        "создание контент-плана для Telegram канала",
        "разработка уникального торгового предложения",
        "написание цепляющего поста для соцсетей",
        "создание email-рассылки с высокой конверсией",
        "разработка названия и слогана для бренда",
        "сторителлинг для презентации продукта",
    ]},
    {"name": "tech_ru", "lang": "ru", "hints": [
        "архитектура Telegram бота на Python",
        "настройка VPS сервера для продакшена",
        "работа с API и интеграции",
        "оптимизация базы данных PostgreSQL",
        "деплой и CI/CD для небольших проектов",
        "парсинг данных и автоматизация",
        "выбор стека технологий для проекта",
    ]},
    {"name": "analysis_ru", "lang": "ru", "hints": [
        "SWOT анализ бизнес-идеи",
        "разбор маркетинговой стратегии конкурента",
        "оценка рисков нового проекта",
        "анализ unit-экономики",
        "сравнение каналов привлечения клиентов",
        "аудит текущих бизнес-процессов",
        "прогноз развития рынка/ниши",
    ]},
    {"name": "reasoning_en", "lang": "en", "hints": [
        "step-by-step analysis of a complex problem",
        "evaluating a business decision with tradeoffs",
        "explaining a technical concept with analogies",
        "finding flaws in an argument",
        "resolving a dilemma with clear reasoning",
        "post-mortem analysis of a failed project",
        "comparing two approaches with recommendation",
    ]},
    {"name": "coding_en", "lang": "en", "hints": [
        "designing a scalable system architecture",
        "debugging a tricky concurrency issue",
        "writing clean Python with proper error handling",
        "API design best practices with examples",
        "database schema design for a real app",
        "performance optimization with profiling",
        "code review with constructive feedback",
    ]},
    {"name": "analysis_en", "lang": "en", "hints": [
        "technology stack comparison for a startup",
        "go-to-market strategy analysis",
        "risk assessment for a new venture",
        "competitive analysis framework",
        "pricing strategy evaluation",
        "build vs buy decision analysis",
        "market sizing and TAM estimation",
    ]},
    {"name": "creative_en", "lang": "en", "hints": [
        "writing compelling product copy",
        "crafting a pitch deck narrative",
        "developing a content marketing strategy",
        "creating a brand positioning statement",
        "writing technical documentation that people read",
        "email sequence for user onboarding",
        "storytelling for investor presentations",
    ]},
    {"name": "deep_thinking_en", "lang": "en", "hints": [
        "first principles thinking applied to a problem",
        "mental models for better decision making",
        "systems thinking and feedback loops",
        "probabilistic reasoning under uncertainty",
        "strategic thinking and game theory in business",
        "cognitive biases and how to avoid them",
        "framework for prioritization and focus",
    ]},
]

META_PROMPT_RU = """Сгенерируй {n} разнообразных, реалистичных вопросов/запросов от пользователя.
Тема: "{hint}"

Требования:
- Вопросы конкретные, с деталями и контекстом (не абстрактные)
- Разный уровень сложности (от простого до экспертного)
- Звучат как реальные люди в чате с AI-ассистентом
- Некоторые вопросы должны быть провокационными или спорными
- Некоторые должны требовать пошагового рассуждения
- Пиши ТОЛЬКО на русском
- Верни ТОЛЬКО JSON массив строк

Формат: ["вопрос 1", "вопрос 2", ...]"""

META_PROMPT_EN = """Generate {n} diverse, realistic user questions/requests.
Topic: "{hint}"

Requirements:
- Questions must be specific with context and details (not abstract)
- Mix difficulty levels (beginner to expert)
- Sound like real people chatting with an AI partner
- Some should be provocative or debatable
- Some should require step-by-step reasoning
- Write ONLY in English
- Return ONLY a JSON array of strings

Format: ["question 1", "question 2", ...]"""

RESPONSE_INSTRUCTION_RU = """Отвечай ТОЛЬКО на русском языке.

Стиль ответа:
- Прямой, без воды. Каждое предложение несёт смысл.
- Если вопрос сложный — разбей на шаги, покажи ход мысли.
- Дай свою рекомендацию, не просто список вариантов.
- Если видишь ошибку в вопросе — скажи прямо.
- Используй примеры и аналогии для сложных вещей.
- Если не уверен — скажи честно, но дай лучшую оценку.
- Заканчивай конкретным следующим шагом или выводом.
- НЕ используй фразы: "отличный вопрос", "давайте разберёмся", "это зависит от многих факторов".
- НЕ смешивай языки. Технические термины можно, но не целые фразы на английском."""

RESPONSE_INSTRUCTION_EN = """Answer ONLY in English.

Response style:
- Direct, no filler. Every sentence adds value.
- For complex questions — break into steps, show your reasoning.
- Give your recommendation, not just a list of options.
- If you see a flaw in the question — say so directly.
- Use examples and analogies for complex concepts.
- If uncertain — say so honestly, but give your best assessment.
- End with a concrete next step or conclusion.
- DON'T use phrases: "great question", "let me break this down", "it depends on many factors".
- DON'T use corporate speak or AI slop."""


# === ASYNC API CALLS ===

async def call_xai(client, messages, model=None, temperature=0.9, max_tokens=3000):
    """Call xAI API (for prompt generation)"""
    model = model or PROMPT_MODEL
    headers = {"Authorization": f"Bearer {XAI_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(3):
        try:
            resp = await client.post(XAI_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = min(30, 5 * (attempt + 1))
                print(f"  xAI 429 rate limit, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  xAI error (attempt {attempt+1}): {e}")
                await asyncio.sleep(3 * (attempt + 1))
        except Exception as e:
            print(f"  xAI error (attempt {attempt+1}): {e}")
            await asyncio.sleep(3 * (attempt + 1))
    return None


async def call_claude(client, messages, system=None, temperature=0.7, max_tokens=4000):
    """Call Vibecode Claude API (for high-quality responses)"""
    headers = {
        "x-api-key": VIBECODE_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": RESPONSE_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        payload["system"] = system
    for attempt in range(3):
        try:
            resp = await client.post(VIBECODE_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return ""
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = min(60, 10 * (attempt + 1))
                print(f"  Claude 429 rate limit, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  Claude error (attempt {attempt+1}): {e}")
                await asyncio.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  Claude error (attempt {attempt+1}): {e}")
            await asyncio.sleep(5 * (attempt + 1))
    return None


async def generate_prompts(client, hint, lang, n=5):
    template = META_PROMPT_RU if lang == "ru" else META_PROMPT_EN
    content = await call_xai(client, [{"role": "user", "content": template.format(n=n, hint=hint)}])
    if not content:
        return []
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        pass
    return []


async def generate_response(client, prompt, lang):
    instruction = RESPONSE_INSTRUCTION_RU if lang == "ru" else RESPONSE_INSTRUCTION_EN
    system = SYSTEM_PROMPT + "\n\n" + instruction
    messages = [{"role": "user", "content": prompt}]
    async with SEMAPHORE:
        return await call_claude(client, messages, system=system)


def validate(response, lang):
    if not response:
        return False, "empty"
    if len(response) < 100:
        return False, f"too_short ({len(response)})"
    if len(response) > 10000:
        return False, f"too_long ({len(response)})"
    lines = response.split("\n")
    if len(lines) > 5:
        non_empty = [l.strip() for l in lines if l.strip()]
        unique = set(non_empty)
        if len(unique) < len(non_empty) * 0.6:
            return False, "repetitive"
    slop_phrases = ["great question", "отличный вопрос", "давайте разберёмся", "let me break this down"]
    for phrase in slop_phrases:
        if phrase.lower() in response.lower()[:200]:
            return False, f"slop: {phrase}"
    return True, "ok"


# File lock for safe concurrent writes
WRITE_LOCK = asyncio.Lock()


async def process_prompt(client, prompt, lang, cat_name, stats):
    """Process a single prompt: generate response, validate, write."""
    resp = await generate_response(client, prompt, lang)
    valid, reason = validate(resp, lang)
    if valid:
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp}
            ],
            "metadata": {
                "category": cat_name,
                "lang": lang,
                "model": RESPONSE_MODEL,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "prompt_tokens_est": len(prompt.split()),
                "response_tokens_est": len(resp.split()),
            }
        }
        async with WRITE_LOCK:
            async with aiofiles.open(OUTPUT_FILE, "a") as f:
                await f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["total"] += 1
        print(f"  [{stats['total']}/{stats['target']}] OK ({len(resp)} chars): {prompt[:50]}...")
        return True
    else:
        stats["errors"] += 1
        key = reason.split(":")[0].split(" ")[0]
        stats["skipped"][key] = stats["skipped"].get(key, 0) + 1
        print(f"  SKIP ({reason}): {prompt[:40]}...")
        return False


async def run(target=2000):
    print(f"=== CHIMBISIAI v2 Async Generator ===")
    print(f"Target: {target} samples")
    print(f"Concurrency: {MAX_CONCURRENT} workers")
    print(f"Response model: {RESPONSE_MODEL}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    existing = 0
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE))
    print(f"Existing: {existing}")

    stats = {
        "total": existing,
        "target": target,
        "errors": 0,
        "skipped": {}
    }

    cats = CATEGORIES.copy()
    random.shuffle(cats)

    async with httpx.AsyncClient() as client:
        while stats["total"] < target:
            for cat in cats:
                if stats["total"] >= target:
                    break
                cname = cat["name"]
                clang = cat["lang"]
                print(f"\n--- {cname} ({clang}) ---")

                for hint in cat["hints"]:
                    if stats["total"] >= target:
                        break
                    print(f"  Generating prompts for: {hint}")
                    prompts = await generate_prompts(client, hint, clang, n=5)
                    if not prompts:
                        stats["errors"] += 1
                        stats["skipped"]["api_fail"] = stats["skipped"].get("api_fail", 0) + 1
                        await asyncio.sleep(2)
                        continue
                    print(f"  Got {len(prompts)} prompts, processing in parallel...")

                    # Process all prompts for this hint concurrently
                    tasks = [
                        process_prompt(client, p, clang, cname, stats)
                        for p in prompts if stats["total"] < target
                    ]
                    await asyncio.gather(*tasks)

                    # Small pause between batches
                    await asyncio.sleep(0.5)

            random.shuffle(cats)

    print(f"\n{'='*50}")
    print(f"=== GENERATION COMPLETE ===")
    print(f"Total samples: {stats['total']}")
    print(f"Errors/skipped: {stats['errors']}")
    print(f"Skip reasons: {json.dumps(stats['skipped'])}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    asyncio.run(run(target))
