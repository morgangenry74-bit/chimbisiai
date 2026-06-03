#!/usr/bin/env python3
"""
CHIMBISIAI Web API — FastAPI backend
Proxies chat requests to Ollama (local model)
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import httpx
import json
import time
import uuid

app = FastAPI(title="CHIMBISIAI API", version="1.0")

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "chimbisiai-v3"

SYSTEM_PROMPT = """You are CHIMBISIAI — an AI partner, not a chatbot. You think like a co-founder, not a servant.
You are direct, serious, practical. No fluff, no filler.
You have opinions and you share them without being asked.
You think deeply, reason step by step, and show your work.
Be concise but complete. Every sentence should add value.
Answer in the same language the user writes in."""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float = 0.7


@app.get("/api/health")
async def health():
    """Health check — also verifies Ollama is running"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"status": "ok", "model": MODEL_NAME, "available": MODEL_NAME in str(models)}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint — streams response from Ollama"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": request.stream,
        "options": {
            "temperature": request.temperature,
            "repeat_penalty": 1.2,
            "num_predict": 2048,
        }
    }

    if not request.stream:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                    timeout=120
                )
                data = resp.json()
                return {
                    "id": str(uuid.uuid4()),
                    "content": data.get("message", {}).get("content", ""),
                    "model": MODEL_NAME,
                    "done": True
                }
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    # Streaming response
    async def generate():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                    timeout=120
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                                if chunk.get("done"):
                                    yield f"data: {json.dumps({'done': True})}\n\n"
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# Maintenance banner control
from fastapi.responses import JSONResponse

@app.get("/api/maintenance")
async def maintenance():
    """Return maintenance banner state — disabled after v3 deploy"""
    return JSONResponse(
        content={"active": False},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )
