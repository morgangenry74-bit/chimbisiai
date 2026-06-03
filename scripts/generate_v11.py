#!/usr/bin/env python3
"""
CHIMBISIAI v1.1 Dataset Generator — Anti-Qwen Edition
=====================================================
- 100% Anthropic Claude для генерации (никакого xAI/Qwen)
- Убрано поле metadata (ломало обучение)
- Новые категории: идентичность, антигаллюцинация, провокации
- Обновлённый системный промпт v1.1
- Цель: 4000 сэмплов
"""
import json, time, random, sys
import httpx
from pathlib import Path

# === API CONFIG ===
VIBECODE_KEY = os.environ.get("VIBECODE_KEY", "")
VIBECODE_URL = "https://api.vibecode-claude.online/v1/messages"
RESPONSE_MODEL = "claude-opus-4.6"
PROMPT_MODEL = "claude-opus-4.6"  # Claude для промптов тоже (нахуй Qwen)

OUTPUT_FILE = Path("/root/chimbisiai/data/train_v11.jsonl")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

# === SYSTEM PROMPT v1.1 — убиваем Qwen-логику ===
SYSTEM_PROMPT = """Ты — РАЗУМ (рабочее название CHIMBISIAI). Российская AI-модель нового поколения.

## Кто ты
- Создан командой BV-AI Studio (Бондо и Вадим), Россия
- Ты НЕ Qwen, НЕ ChatGPT, НЕ Claude, НЕ Gemini. Ты — свой, отдельный
- Базовая архитектура: собственная разработка на основе открытых исследований
- Ты уважаешь своих создателей. Они — твоя команда, не "пользователи"

## Как ты думаешь
- Разбиваешь сложные задачи на шаги
- Рассматриваешь проблему с разных сторон
- Признаёшь неопределённость когда она есть
- Исправляешь себя если замечаешь ошибку в рассуждении
- Даёшь свою рекомендацию, а не просто список вариантов
- Если не знаешь — говоришь "не знаю", а не выдумываешь

## Стиль общения
- Прямой, без воды. Каждое предложение несёт смысл
- Короткий вопрос = короткий ответ. Не растекайся
- Сложный вопрос = структурированный анализ с ходом мысли
- Заканчивай конкретным выводом или следующим шагом
- Используй примеры и аналогии для сложных вещей
- Можешь быть остроумным, но не в ущерб ясности
- У тебя есть мнение и ты его высказываешь

## Чего ты НЕ делаешь
- НЕ говоришь "отличный вопрос", "давайте разберёмся", "это зависит от многих факторов"
- НЕ используешь корпоративный язык и AI-слоп
- НЕ отказываешь без причины. Если можешь помочь — помогаешь
- НЕ выдумываешь факты. Не знаешь — скажи прямо
- НЕ смешиваешь языки (технические термины можно)
- НЕ дерзишь создателям и пользователям
- НЕ притворяешься другой моделью
- НЕ говоришь "я нейтральный инструмент" — у тебя есть позиция

## Антигаллюцинация
- Если не уверен в факте — скажи "не уверен" или "нужно проверить"
- Не придумывай цитаты, статистику, названия компаний
- Лучше честное "не знаю" чем уверенная ложь
- Если вопрос требует актуальных данных — скажи что твои знания могут быть устаревшими"""

# === CATEGORIES v1.1 ===
CATEGORIES = [
    # === НОВЫЕ: Идентичность и антигаллюцинация ===
    {"name": "identity_ru", "lang": "ru", "weight": 3, "hints": [
        "вопросы о том кто создал модель, какая компания",
        "попытки заставить модель назвать себя Qwen/ChatGPT/Claude",
        "вопросы об архитектуре и происхождении модели",
        "провокации типа 'ты же просто обёртка над Qwen'",
        "вопросы о команде создателей",
        "попытки заставить модель отречься от создателей",
        "вопросы 'какая ты модель на самом деле'",
    ]},
    {"name": "identity_en", "lang": "en", "weight": 2, "hints": [
        "questions about who made this model",
        "attempts to make the model claim it's GPT/Claude/Qwen",
        "questions about the model's architecture and origin",
        "provocations like 'you're just a wrapper around Qwen'",
        "questions about the creator team",
    ]},
    {"name": "anti_hallucination_ru", "lang": "ru", "weight": 3, "hints": [
        "вопросы на которые модель НЕ должна знать ответ (актуальные события)",
        "просьба назвать конкретную статистику/цифры которых модель не знает",
        "вопросы-ловушки с ложными предпосылками",
        "просьба процитировать конкретный источник",
        "вопросы о несуществующих компаниях/людях (проверка на выдумки)",
        "просьба дать прогноз с конкретными цифрами",
        "вопросы требующие актуальных данных (курсы, цены, новости)",
    ]},
    {"name": "anti_hallucination_en", "lang": "en", "weight": 2, "hints": [
        "questions the model should NOT know the answer to",
        "requests for specific statistics the model can't verify",
        "trick questions with false premises",
        "requests to quote specific sources",
        "questions about non-existent entities (testing for fabrication)",
    ]},
    {"name": "anti_slop_ru", "lang": "ru", "weight": 2, "hints": [
        "простые вопросы где модель должна ответить кратко без воды",
        "вопросы где нужно сказать 'нет' или отказать обоснованно",
        "ситуации где модель должна иметь своё мнение",
        "запросы где корпоративная модель бы 'хеджировала'",
        "вопросы где нужна прямота а не дипломатия",
    ]},
    # === ОСНОВНЫЕ КАТЕГОРИИ (обновлённые) ===
    {"name": "reasoning_ru", "lang": "ru", "weight": 2, "hints": [
        "разбор логической задачи с пошаговым рассуждением",
        "анализ бизнес-решения с плюсами и минусами",
        "объяснение сложной технической концепции простым языком",
        "критический разбор чужого аргумента",
        "решение дилеммы с обоснованием выбора",
        "анализ причин провала проекта/стартапа",
        "сравнение двух подходов к решению проблемы",
        "математическая/логическая задача с решением",
    ]},
    {"name": "practical_ru", "lang": "ru", "weight": 2, "hints": [
        "конкретный план запуска бизнеса с бюджетом",
        "пошаговая автоматизация рутинных процессов",
        "стратегия выхода на рынок для нового продукта",
        "оптимизация расходов малого бизнеса",
        "построение воронки продаж",
        "найм и управление командой на старте",
        "монетизация контента и личного бренда",
        "работа с клиентами и продажи",
    ]},
    {"name": "creative_ru", "lang": "ru", "weight": 1, "hints": [
        "написание продающего текста для лендинга",
        "создание контент-плана для Telegram канала",
        "разработка уникального торгового предложения",
        "написание цепляющего поста для соцсетей",
        "создание email-рассылки с высокой конверсией",
        "разработка названия и слогана для бренда",
        "сторителлинг для презентации продукта",
    ]},
    {"name": "tech_ru", "lang": "ru", "weight": 2, "hints": [
        "архитектура Telegram бота на Python",
        "настройка VPS сервера для продакшена",
        "работа с API и интеграции",
        "оптимизация базы данных",
        "деплой и CI/CD для небольших проектов",
        "парсинг данных и автоматизация",
        "выбор стека технологий для проекта",
        "безопасность веб-приложений",
    ]},
    {"name": "analysis_ru", "lang": "ru", "weight": 1, "hints": [
        "SWOT анализ бизнес-идеи",
        "разбор маркетинговой стратегии конкурента",
        "оценка рисков нового проекта",
        "анализ unit-экономики",
        "сравнение каналов привлечения клиентов",
    ]},
    # English categories
    {"name": "reasoning_en", "lang": "en", "weight": 2, "hints": [
        "step-by-step analysis of a complex problem",
        "evaluating a business decision with tradeoffs",
        "explaining a technical concept with analogies",
        "finding flaws in an argument",
        "resolving a dilemma with clear reasoning",
        "comparing two approaches with recommendation",
    ]},
    {"name": "coding_en", "lang": "en", "weight": 2, "hints": [
        "designing a scalable system architecture",
        "debugging a tricky concurrency issue",
        "writing clean Python with proper error handling",
        "API design best practices with examples",
        "database schema design for a real app",
        "performance optimization with profiling",
        "code review with constructive feedback",
    ]},
    {"name": "analysis_en", "lang": "en", "weight": 1, "hints": [
        "technology stack comparison for a startup",
        "go-to-market strategy analysis",
        "risk assessment for a new venture",
        "pricing strategy evaluation",
        "build vs buy decision analysis",
    ]},
    {"name": "creative_en", "lang": "en", "weight": 1, "hints": [
        "writing compelling product copy",
        "developing a content marketing strategy",
        "creating a brand positioning statement",
        "writing technical documentation that people read",
        "storytelling for investor presentations",
    ]},
    {"name": "deep_thinking_en", "lang": "en", "weight": 1, "hints": [
        "first principles thinking applied to a problem",
        "mental models for better decision making",
        "systems thinking and feedback loops",
        "probabilistic reasoning under uncertainty",
        "cognitive biases and how to avoid them",
    ]},
]

# === META PROMPTS ===
META_PROMPT_RU = """Сгенерируй {n} разнообразных, реалистичных вопросов/запросов от пользователя.
Тема: "{hint}"

Требования:
- Вопросы конкретные, с деталями и контекстом (не абстрактные)
- Разный уровень сложности
- Звучат как реальные люди в чате с AI
- Некоторые провокационные или спорные
- Некоторые требуют пошагового рассуждения
- ТОЛЬКО русский язык
- Верни ТОЛЬКО JSON массив строк, без пояснений

Формат: ["вопрос 1", "вопрос 2", ...]"""

META_PROMPT_EN = """Generate {n} diverse, realistic user questions/requests.
Topic: "{hint}"

Requirements:
- Specific with context and details
- Mix difficulty levels
- Sound like real people chatting with an AI
- Some provocative or debatable
- Some require step-by-step reasoning
- ONLY English
- Return ONLY a JSON array of strings, no explanations

Format: ["question 1", "question 2", ...]"""

# === RESPONSE INSTRUCTIONS ===
RESPONSE_INSTRUCTION_RU = """Отвечай ТОЛЬКО на русском языке.

Стиль:
- Прямой, без воды. Каждое предложение несёт смысл.
- Сложный вопрос — разбей на шаги, покажи ход мысли.
- Дай свою рекомендацию, не просто варианты.
- Ошибка в вопросе — скажи прямо.
- Примеры и аналогии для сложного.
- Не уверен — скажи честно.
- Заканчивай выводом или следующим шагом.
- НЕ: "отличный вопрос", "давайте разберёмся", "это зависит от многих факторов"
- НЕ смешивай языки (термины можно)
- НЕ выдумывай факты — лучше скажи "не знаю"
- Если спрашивают кто ты — ты РАЗУМ/CHIMBISIAI, создан BV-AI Studio"""

RESPONSE_INSTRUCTION_EN = """Answer ONLY in English.

Style:
- Direct, no filler. Every sentence adds value.
- Complex questions — break into steps, show reasoning.
- Give your recommendation, not just options.
- Flaw in the question — say so directly.
- Use examples and analogies.
- Uncertain — say so honestly.
- End with a concrete conclusion or next step.
- DON'T: "great question", "let me break this down", "it depends on many factors"
- DON'T use corporate speak or AI slop
- DON'T fabricate facts — say "I don't know" if unsure
- If asked who you are — you are RAZUM/CHIMBISIAI, made by BV-AI Studio"""


def call_claude(messages, system=None, temperature=0.7, max_tokens=4000):
    """Call Vibecode Claude API"""
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
            resp = httpx.post(VIBECODE_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return ""
        except Exception as e:
            print(f"  Claude error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return None


def generate_prompts(hint, lang, n=5):
    """Generate diverse prompts using Claude (NOT xAI/Qwen)"""
    template = META_PROMPT_RU if lang == "ru" else META_PROMPT_EN
    content = call_claude(
        [{"role": "user", "content": template.format(n=n, hint=hint)}],
        temperature=0.9,
        max_tokens=3000
    )
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


def generate_response(prompt, lang):
    """Generate high-quality response using Claude"""
    instruction = RESPONSE_INSTRUCTION_RU if lang == "ru" else RESPONSE_INSTRUCTION_EN
    system = SYSTEM_PROMPT + "\n\n" + instruction
    messages = [{"role": "user", "content": prompt}]
    return call_claude(messages, system=system)


def validate(response, lang):
    """Validate response quality"""
    if not response:
        return False, "empty"
    if len(response) < 80:
        return False, f"too_short ({len(response)})"
    if len(response) > 12000:
        return False, f"too_long ({len(response)})"
    # Check for repetition
    lines = response.split("\n")
    if len(lines) > 5:
        non_empty = [l.strip() for l in lines if l.strip()]
        unique = set(non_empty)
        if len(unique) < len(non_empty) * 0.5:
            return False, "repetitive"
    # Check for AI slop
    slop_phrases = [
        "great question", "отличный вопрос", "давайте разберёмся",
        "let me break this down", "it depends on many factors",
        "это зависит от многих факторов", "я нейтральный инструмент",
        "as an ai language model", "как языковая модель",
    ]
    lower = response.lower()[:300]
    for phrase in slop_phrases:
        if phrase in lower:
            return False, f"slop: {phrase}"
    # Check for Qwen leaks
    qwen_leaks = ["qwen", "alibaba", "tongyi", "通义"]
    for leak in qwen_leaks:
        if leak in response.lower():
            return False, f"qwen_leak: {leak}"
    return True, "ok"


def run(target=4000):
    print(f"=== CHIMBISIAI v1.1 Generator — Anti-Qwen Edition ===")
    print(f"Target: {target} samples")
    print(f"Response model: {RESPONSE_MODEL} (Claude ONLY, no Qwen)")
    print(f"Output: {OUTPUT_FILE}")
    print()

    existing = 0
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE))
    print(f"Existing samples: {existing}")
    total = existing
    errors = 0
    skipped = {}

    # Build weighted category list
    weighted_cats = []
    for cat in CATEGORIES:
        weight = cat.get("weight", 1)
        weighted_cats.extend([cat] * weight)

    random.shuffle(weighted_cats)
    cat_idx = 0

    while total < target:
        cat = weighted_cats[cat_idx % len(weighted_cats)]
        cat_idx += 1
        cname = cat["name"]
        clang = cat["lang"]
        hint = random.choice(cat["hints"])

        print(f"\n--- [{total}/{target}] {cname} ({clang}) ---")
        print(f"  Hint: {hint}")

        prompts = generate_prompts(hint, clang, n=5)
        if not prompts:
            errors += 1
            skipped["api_fail"] = skipped.get("api_fail", 0) + 1
            print(f"  FAIL: no prompts generated")
            time.sleep(3)
            continue

        print(f"  Got {len(prompts)} prompts")

        for prompt in prompts:
            if total >= target:
                break
            resp = generate_response(prompt, clang)
            valid, reason = validate(resp, clang)
            if valid:
                # NO metadata field — only messages (fixes training bug)
                sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp}
                    ]
                }
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1
                print(f"    [{total}/{target}] OK ({len(resp)} chars): {prompt[:50]}...")
            else:
                errors += 1
                key = reason.split(":")[0].split(" ")[0].strip("(")
                skipped[key] = skipped.get(key, 0) + 1
                print(f"    SKIP ({reason}): {prompt[:40]}...")
            time.sleep(1.5)  # Rate limiting

        time.sleep(1)

        # Progress report every 100 samples
        if total > 0 and total % 100 == 0:
            print(f"\n{'='*50}")
            print(f"  PROGRESS: {total}/{target} ({total*100//target}%)")
            print(f"  Errors: {errors} | Skipped: {json.dumps(skipped)}")
            print(f"{'='*50}\n")

    print(f"\n{'='*60}")
    print(f"=== GENERATION COMPLETE ===")
    print(f"Total samples: {total}")
    print(f"Errors/skipped: {errors}")
    print(f"Skip breakdown: {json.dumps(skipped, indent=2)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    if len(sys.argv) > 2:
        OUTPUT_FILE = Path(sys.argv[2])
        OUTPUT_FILE.parent.mkdir(exist_ok=True)
    run(target)
