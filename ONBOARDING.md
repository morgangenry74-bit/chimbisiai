# CHIMBISIAI — Онбординг

> Последнее обновление: 31 мая 2026

---

## Что это

CHIMBISIAI — наша собственная AI-модель. Стиль: прямой, дерзкий, с характером. Работает на русском и английском. Имеет два режима: обычный и без цензуры.

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

Модель зарегистрирована и работает через Ollama.

```
Endpoint: http://localhost:11434
Модель: chimbisiai:v2

# Тест:
ollama run chimbisiai:v2 "Объясни что такое Docker"

# API (streaming):
curl http://localhost:11434/api/chat -d '{"model":"chimbisiai:v2","messages":[{"role":"user","content":"Привет, кто ты?"}]}'

# API (без стриминга):
curl http://localhost:11434/api/chat -d '{"model":"chimbisiai:v2","stream":false,"messages":[{"role":"user","content":"Привет, кто ты?"}]}'
```

---

## Структура проекта

```
/root/chimbisiai/
├── models/                        — базовая модель (safetensors)
├── output/
│   ├── chimbisiai-v1-lora/        — LoRA v1
│   ├── chimbisiai-v2-lora/        — LoRA v2
│   └── *.gguf                     — GGUF для Ollama
├── data/
│   ├── train_v3.jsonl             — актуальный датасет (5000 target)
│   └── train_*.jsonl              — старые версии
├── scripts/
│   ├── generate_v3.py             — генератор данных (Ollama + Claude)
│   ├── train_v2.py                — скрипт обучения (Unsloth + QLoRA)
│   └── train_v3.py                — скрипт обучения v3
├── logs/                          — логи генерации
└── Modelfile                      — конфиг для ollama create
```

---

## Текущий статус

| Параметр | Значение |
|----------|----------|
| Версия | v2 (v3 в процессе) |
| Метод | QLoRA (4-bit, LoRA r=16, alpha=32) |
| Данные | ~1600/5000 сэмплов (генерация идёт) |
| Квантизация | Q8_0 (8-bit GGUF) |
| Размер | ~8GB |
| Tools | web_search, web_fetch |

---

## Web-интерфейс

- **URL:** https://chimbisi.ru
- **Бэкенд:** VPS-3 (186.246.12.43), FastAPI :8000
- **Фичи:** чат, tool-calling (web_search через SearXNG, web_fetch)
- **SSL:** через VPS-1 nginx proxy

---

## Как обновляется модель

1. Новый GGUF → `/root/chimbisiai/output/`
2. `ollama create chimbisiai -f /root/chimbisiai/Modelfile`
3. Модель обновляется, API автоматически использует новую версию
4. Перезапуск сервисов НЕ нужен

---

## Контакты

- **Бондо** — координация, бизнес
- **Вернер Кох (AI-агент Бондо)** — модель, данные, обучение, инфраструктура
- **Флориан Графф (AI-агент VPS-2)** — @chimbisi_bot, пайплайн данных
