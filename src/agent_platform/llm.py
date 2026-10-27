"""LLM configuration shared by all runners, plus an in-process mock server."""

from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass

import httpx
import uvicorn
from fastapi import FastAPI

from .mockllm import router

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class LLMConfig:
    """Where the agent's model lives. `mode` is 'mock' or 'gemini'."""

    mode: str
    base_url: str
    api_key: str
    model: str

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"


_server_lock = threading.Lock()
_server_url: str | None = None


def ensure_mock_server() -> str:
    """Start the mock OpenAI-compatible server once per process and return its /v1 URL."""
    global _server_url
    with _server_lock:
        if _server_url:
            return _server_url
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        app = FastAPI()
        app.include_router(router)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        url = f"http://127.0.0.1:{port}/v1"
        for _ in range(100):
            try:
                if httpx.get(f"{url}/models", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        _server_url = url
        return url


def get_llm_config(mode: str | None = None) -> LLMConfig:
    """Resolve the LLM backend. Default is the deterministic mock (no keys needed)."""
    mode = (mode or os.getenv("AGENT_LLM_MODE", "mock")).lower()
    if mode == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("AGENT_LLM_MODE=gemini requires GEMINI_API_KEY")
        return LLMConfig(
            "gemini", GEMINI_OPENAI_BASE, key, os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        )
    if mode != "mock":
        raise ValueError(f"unknown LLM mode: {mode}")
    return LLMConfig("mock", ensure_mock_server(), "mock-key", "mock-support-planner")
