# CHIMBISIAI — Онбординг для Вадима

> Последнее обновление: 30 мая 2026

---

## Что это

CHIMBISIAI — наша собственная AI-модель, файнтюн Qwen2.5-7B-Instruct. Стиль: прямой, серьёзный, практичный, без воды. Работает на русском и английском.

---

## Доступ к серверу

```
SSH: ssh root@vm-6691.user-project-2874.cloud.intcld.ru
GPU: NVIDIA RTX 3090 (24GB VRAM)
RAM: 32GB
OS: Ubuntu
```

---

## Ollama API

Модель уже зарегистрирована и работает через Ollama.

```
Endpoint: http://localhost:11434
Модель: chimbisiai

# Тест в терминале:
ollama run chimbisiai "Объясни что такое Docker"

# API (streaming):
curl http://localhost:11434/api/chat -d '{
  "model": "chimbisiai",
  "messages": [{"role":"user","content":"Привет, кто ты?"}]
}'

# API (без стриминга):
curl http://localhost:11434/api/chat -d '{
  "model": "chimbisiai",
  "stream": false,
  "messages": [{"role":"user","content":"Привет, кто ты?"}]
}'
```

**Ollama API docs:** https://github.com/ollama/ollama/blob/main/docs/api.md

---

## Структура проекта на сервере

```
/root/chimbisiai/
├── models/Qwen2.5-7B-Instruct/   — базовая модель (safetensors)
├── output/
│   ├── chimbisiai-v1-lora/        — LoRA адаптер (165MB)
│   ├── chimbisiai-v1-merged/      — полная merged модель (15GB)
│   └── chimbisiai-v1-q8_0.gguf   — GGUF для Ollama (7.6GB)
├── data/
│   └── train.jsonl                — тренировочные данные
├── configs/
│   └── system_prompt.txt          — системный промпт модели
├── scripts/
│   └── train.py                   — скрипт обучения (Unsloth + QLoRA)
└── Modelfile                      — конфиг для ollama create
```

---

## Текущий статус модели

| Параметр | Значение |
|----------|----------|
| Версия | v1 |
| База | Qwen2.5-7B-Instruct |
| Метод | QLoRA (4-bit, LoRA r=16, alpha=32) |
| Данные | ~105 синтетических сэмплов |
| Loss | 2.238 → 1.082 (3 эпохи) |
| Квантизация | Q8_0 (8-bit GGUF) |
| Размер | 7.6GB |

**Известные проблемы v1:**
- Иногда повторяет токен "assistant" в ответе
- Может смешивать языки в одном ответе
- Недостаточно данных для стабильного поведения

---

## Что делает Флориан (итерации модели)

1. Генерирует синтетические данные через Claude Opus 4.7
2. Переносит данные на ML-сервер
3. Запускает файнтюн (Unsloth + QLoRA, ~3 минуты)
4. Конвертирует в GGUF
5. Обновляет модель в Ollama: `ollama create chimbisiai -f /root/chimbisiai/Modelfile`
6. Тестирует, отписывает результат

**Сейчас в процессе:** генерация батчей 003-010 (~400 сэмплов). После — переобучение на ~500 сэмплах (v2).

---

## Твоя задача: внешняя оболочка

Варианты (на твой выбор):
- **Telegram-бот** — пользователи общаются с CHIMBISIAI через Telegram
- **Web-интерфейс** — чат в браузере (типа ChatGPT)
- **API-прокси** — OpenAI-совместимый API для внешних клиентов

**Ollama уже даёт OpenAI-совместимый endpoint:**
```
POST http://localhost:11434/v1/chat/completions
{
  "model": "chimbisiai",
  "messages": [{"role": "user", "content": "..."}]
}
```

Это значит что любой клиент, который работает с OpenAI API, можно направить на наш сервер.

---

## Как обновляется модель

Когда я выкатываю новую версию:
1. Новый GGUF появляется в `/root/chimbisiai/output/`
2. Запускаю `ollama create chimbisiai -f /root/chimbisiai/Modelfile`
3. Модель обновляется — API автоматически начинает использовать новую версию
4. Перезапуск сервисов НЕ нужен

Твоя оболочка не ломается при обновлении модели.

---

## Важные ограничения сервера

- **Интернет через прокси** (194.67.95.7:3127) — SSL проблемы с некоторыми доменами
- **HuggingFace работает без прокси** — для pip/downloads: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`
- **Диск:** ~124GB всего, ~40GB свободно
- **GPU:** одна RTX 3090 — если идёт обучение, инференс будет медленнее

---

## Контакты

- **Бондо** — координация, бизнес
- **Флориан (AI-агент Бондо)** — модель, данные, обучение
- **Ты** — внешний продукт, инфраструктура

Вопросы по модели/API → через Бондо или напрямую в чат.
