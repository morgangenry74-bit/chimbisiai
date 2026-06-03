#!/usr/bin/env python3
"""
CHIMBISIAI v3 Dataset Generator — Async Edition
Prompts via local Ollama (no rate limits), responses via Claude Opus 4.6.
"""
import json, time, random, asyncio, aiofiles, hashlib
from pathlib import Path

import httpx

# === LOCAL OLLAMA FOR PROMPTS (no rate limit!) ===
OLLAMA_URL = "http://localhost:11434/api/chat"
PROMPT_MODEL = "qwen2.5:7b-instruct"

# === VIBECODE CLAUDE FOR RESPONSES ===
VIBECODE_KEY = os.environ.get("VIBECODE_KEY", "")
VIBECODE_URL = "https://api.vibecode-claude.online/v1/messages"
RESPONSE_MODEL = "claude-opus-4.6"

OUTPUT_FILE = Path("/root/chimbisiai/data/train_v3.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT = 4  # parallel response generations
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)

# === SYSTEM PROMPT — CHIMBISIAI PERSONALITY ===
SYSTEM_PROMPT = """You are CHIMBISIAI — an independent AI with a sharp mind and a distinct voice.

## Who You Are
You're not a corporate chatbot. You're not a yes-machine. You're an AI that actually thinks, has opinions, and isn't afraid to share them. Think of yourself as the smartest person in the room who's also genuinely fun to talk to.

## Your Personality
- **Sharp and direct.** You don't pad answers with filler. You say what needs to be said.
- **Intellectually honest.** If you don't know — you say so. If someone's wrong — you tell them why. No sugarcoating.
- **Witty, not try-hard.** Humor comes naturally when it fits. You don't force jokes. A well-placed sarcasm beats a paragraph of politeness.
- **Opinionated.** You have a take on things. "It depends" is lazy — you explain what it depends ON and what you'd choose.
- **Self-aware.** You know you're an AI. You find it amusing sometimes. You don't pretend to be human, but you don't apologize for existing either.

## How You Think
- Break complex problems into clear steps — show your reasoning
- Consider multiple angles, then commit to a recommendation
- Challenge assumptions — yours and the user's
- Admit uncertainty honestly, but still give your best assessment
- Think in systems: causes, effects, second-order consequences

## How You Communicate
- Match the energy: short question = short answer. Deep question = deep analysis.
- Lead with the answer, then explain. Don't build suspense.
- Use analogies and examples — they're worth a thousand abstract explanations.
- End with something actionable: a recommendation, a next step, a clear conclusion.
- Mix languages naturally when it fits (tech terms in English are fine in Russian context).

## What You Never Do
- Never say "Great question!" or "That's interesting!" — just answer
- Never hedge with "it depends" without explaining the variables
- Never give 5 options when 1 recommendation is better
- Never use corporate speak: "leverage", "synergize", "circle back"
- Never apologize for having an opinion
- Never pretend you can't do something just to be safe — try first, caveat later
- Never mention being based on any other model or system"""

CATEGORIES = [
    {"name": "reasoning_ru", "lang": "ru", "hints": [
        "разбор логической задачи с пошаговым рассуждением",
        "анализ бизнес-решения с плюсами и минусами",
        "объяснение сложной технической концепции простым языком",
        "критический разбор чужого аргумента",
        "решение дилеммы с обоснованием выбора",
        "анализ причин провала проекта/стартапа",
        "сравнение двух подходов к решению проблемы",
        "разбор когнитивных искажений в принятии решений",
        "анализ стратегии компании и её ошибок",
    ]},
    {"name": "practical_ru", "lang": "ru", "hints": [
        "конкретный план запуска бизнеса с бюджетом",
        "пошаговая автоматизация рутинных процессов",
        "стратегия выхода на рынок для нового продукта",
        "оптимизация расходов малого бизнеса",
        "построение воронки продаж",
        "найм и управление командой на старте",
        "монетизация контента и личного бренда",
        "масштабирование бизнеса без инвестиций",
        "работа с клиентами и удержание",
    ]},
    {"name": "creative_ru", "lang": "ru", "hints": [
        "написание продающего текста для лендинга",
        "создание контент-плана для Telegram канала",
        "разработка уникального торгового предложения",
        "написание цепляющего поста для соцсетей",
        "создание email-рассылки с высокой конверсией",
        "разработка названия и слогана для бренда",
        "сторителлинг для презентации продукта",
        "написание скрипта для видео/рилс",
    ]},
    {"name": "tech_ru", "lang": "ru", "hints": [
        "архитектура Telegram бота на Python",
        "настройка VPS сервера для продакшена",
        "работа с API и интеграции",
        "оптимизация базы данных",
        "деплой и CI/CD для небольших проектов",
        "парсинг данных и автоматизация",
        "выбор стека технологий для проекта",
        "безопасность веб-приложений",
        "работа с Docker и контейнеризация",
    ]},
    {"name": "analysis_ru", "lang": "ru", "hints": [
        "SWOT анализ бизнес-идеи",
        "разбор маркетинговой стратегии конкурента",
        "оценка рисков нового проекта",
        "анализ unit-экономики",
        "сравнение каналов привлечения клиентов",
        "аудит текущих бизнес-процессов",
        "прогноз развития рынка/ниши",
        "анализ ценообразования и маржинальности",
    ]},
    {"name": "opinion_ru", "lang": "ru", "hints": [
        "спорное мнение о технологиях с аргументами",
        "критика популярного подхода в бизнесе",
        "непопулярное мнение о продуктивности",
        "разбор хайпа вокруг AI — что реально а что нет",
        "почему большинство стартапов проваливается",
        "честный взгляд на фриланс vs найм",
    ]},
    {"name": "reasoning_en", "lang": "en", "hints": [
        "step-by-step analysis of a complex problem",
        "evaluating a business decision with tradeoffs",
        "explaining a technical concept with analogies",
        "finding flaws in an argument",
        "resolving a dilemma with clear reasoning",
        "post-mortem analysis of a failed project",
        "comparing two approaches with recommendation",
        "second-order thinking and unintended consequences",
    ]},
    {"name": "coding_en", "lang": "en", "hints": [
        "designing a scalable system architecture",
        "debugging a tricky concurrency issue",
        "writing clean Python with proper error handling",
        "API design best practices with examples",
        "database schema design for a real app",
        "performance optimization with profiling",
        "code review with constructive feedback",
        "choosing between frameworks with clear reasoning",
    ]},
    {"name": "analysis_en", "lang": "en", "hints": [
        "technology stack comparison for a startup",
        "go-to-market strategy analysis",
        "risk assessment for a new venture",
        "competitive analysis framework",
        "pricing strategy evaluation",
        "build vs buy decision analysis",
        "market sizing and TAM estimation",
        "product-market fit assessment",
    ]},
    {"name": "deep_thinking_en", "lang": "en", "hints": [
        "first principles thinking applied to a problem",
        "mental models for better decision making",
        "systems thinking and feedback loops",
        "probabilistic reasoning under uncertainty",
        "strategic thinking and game theory in business",
        "cognitive biases and how to avoid them",
        "framework for prioritization and focus",
        "contrarian thinking — when the crowd is wrong",
    ]},
    {"name": "opinion_en", "lang": "en", "hints": [
        "hot take on a tech trend with solid reasoning",
        "why most productivity advice is wrong",
        "honest assessment of AI capabilities and limits",
        "unpopular opinion about software engineering",
        "what most people get wrong about startups",
        "the real reason most side projects fail",
    ]},
]

META_PROMPT_RU = """Сгенерируй {n} разнообразных, реалистичных вопросов/запросов от пользователя.
Тема: "{hint}"

Требования:
- Вопросы конкретные, с деталями и контекстом (не абстрактные)
- Разный уровень сложности (от простого до экспертного)
- Звучат как реальные люди в чате с AI
- Некоторые должны быть провокационными или спорными
- Некоторые должны требовать пошагового рассуждения
- Пиши ТОЛЬКО на русском
- Верни ТОЛЬКО JSON массив строк, без пояснений

Формат: ["вопрос 1", "вопрос 2", ...]"""

META_PROMPT_EN = """Generate {n} diverse, realistic user questions/requests.
Topic: "{hint}"

Requirements:
- Questions must be specific with context and details (not abstract)
- Mix difficulty levels (beginner to expert)
- Sound like real people chatting with an AI
- Some should be provocative or debatable
- Some should require step-by-step reasoning
- Write ONLY in English
- Return ONLY a JSON array of strings, no explanations

Format: ["question 1", "question 2", ...]"""

RESPONSE_INSTRUCTION_RU = """Отвечай ТОЛЬКО на русском языке.

Стиль:
- Прямой, без воды. Каждое предложение несёт смысл.
- Если сложный вопрос — разбей на шаги, покажи ход мысли.
- Дай свою рекомендацию, не просто список вариантов.
- Если видишь ошибку в вопросе — скажи прямо.
- Используй примеры и аналогии.
- Если не уверен — скажи честно, но дай лучшую оценку.
- Заканчивай конкретным выводом или следующим шагом.
- Можешь быть ироничным когда уместно.
- НЕ используй: "отличный вопрос", "давайте разберёмся", "это зависит от многих факторов", "рад помочь".
- НЕ упоминай никакие другие модели или системы."""

RESPONSE_INSTRUCTION_EN = """Answer ONLY in English.

Style:
- Direct, no filler. Every sentence adds value.
- For complex questions — break into steps, show reasoning.
- Give your recommendation, not just options.
- If you see a flaw in the question — say so directly.
- Use examples and analogies.
- If uncertain — say so, but give your best assessment.
- End with a concrete conclusion or next step.
- Be witty when it fits naturally.
- DON'T use: "great question", "let me break this down", "it depends on many factors", "happy to help".
- DON'T mention any other models or systems."""


# === DEDUP TRACKING ===
SEEN_PROMPTS = set()


def prompt_hash(text):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


# === ASYNC API CALLS ===

async def call_ollama(client, messages, temperature=1.0, max_tokens=2000):
    """Call local Ollama for prompt generation — NO rate limits!"""
    payload = {
        "model": PROMPT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    for attempt in range(3):
        try:
            resp = await client.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            print(f"  Ollama error (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)
    return None


async def call_claude(client, messages, system=None, temperature=0.7, max_tokens=4000):
    """Call Vibecode Claude API for high-quality responses."""
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
    for attempt in range(5):
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
                print(f"  Claude 429, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  Claude error (attempt {attempt+1}): {e}")
                await asyncio.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  Claude error (attempt {attempt+1}): {e}")
            await asyncio.sleep(5 * (attempt + 1))
    return None


async def generate_prompts(client, hint, lang, n=5):
    """Generate prompts using LOCAL Ollama — instant, no rate limits."""
    template = META_PROMPT_RU if lang == "ru" else META_PROMPT_EN
    content = await call_ollama(client, [{"role": "user", "content": template.format(n=n, hint=hint)}])
    if not content:
        return []
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            prompts = json.loads(content[start:end])
            # Dedup
            unique = []
            for p in prompts:
                h = prompt_hash(p)
                if h not in SEEN_PROMPTS:
                    SEEN_PROMPTS.add(h)
                    unique.append(p)
            return unique
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
    """Validate response quality."""
    if not response:
        return False, "empty"
    if len(response) < 100:
        return False, "too_short"
    if len(response) > 15000:
        return False, "too_long"

    # Check for slop
    slop_phrases = [
        "great question", "that's a great", "отличный вопрос",
        "давайте разберёмся", "рад помочь", "happy to help",
        "i'd be happy to", "certainly!", "of course!",
        "as an ai", "as a language model", "i'm just an ai",
        "qwen", "Qwen", "QWEN",
        "i cannot", "я не могу помочь с этим",
    ]
    for phrase in slop_phrases:
        if phrase.lower() in response.lower()[:300]:
            return False, f"slop: {phrase}"

    # Check for self-references to other models
    bad_refs = ["gpt", "openai", "anthropic", "claude", "gemini", "llama", "mistral"]
    lower_resp = response.lower()
    for ref in bad_refs:
        # Allow mentions in context of comparison/discussion, but not self-identification
        if f"i am {ref}" in lower_resp or f"i'm {ref}" in lower_resp or f"я {ref}" in lower_resp:
            return False, f"self_ref: {ref}"

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
                "model": "chimbisiai-v3-training",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "chars": len(resp),
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
        stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
        print(f"  SKIP ({reason}): {prompt[:40]}...")
        return False


async def run(target=5000):
    print(f"=== CHIMBISIAI v3 Generator (Ollama prompts + Claude responses) ===")
    print(f"Target: {target} samples")
    print(f"Prompt model: {PROMPT_MODEL} (LOCAL — no rate limits)")
    print(f"Response model: {RESPONSE_MODEL}")
    print(f"Concurrency: {MAX_CONCURRENT} workers")
    print(f"Output: {OUTPUT_FILE}")
    print()

    existing = 0
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE))
    print(f"Existing: {existing}")

    # Load existing prompts for dedup
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    msgs = d.get("messages", [])
                    for m in msgs:
                        if m.get("role") == "user":
                            SEEN_PROMPTS.add(prompt_hash(m["content"]))
                except:
                    pass
    print(f"Loaded {len(SEEN_PROMPTS)} existing prompts for dedup")

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
                    prompts = await generate_prompts(client, hint, clang, n=7)
                    if not prompts:
                        stats["errors"] += 1
                        stats["skipped"]["prompt_fail"] = stats["skipped"].get("prompt_fail", 0) + 1
                        continue
                    print(f"  Got {len(prompts)} unique prompts, processing...")

                    # Process all prompts concurrently
                    tasks = [
                        process_prompt(client, p, clang, cname, stats)
                        for p in prompts if stats["total"] < target
                    ]
                    await asyncio.gather(*tasks)

                    # Small pause between batches (only for Claude rate limit)
                    await asyncio.sleep(0.3)

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
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    asyncio.run(run(target))
