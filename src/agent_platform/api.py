"""FastAPI service: REST endpoints for running agents, the MCP server and the A2A agent."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .a2a_server import build_a2a_routes
from .llm import get_llm_config
from .mcp_server import mcp
from .ratelimit import RateLimiter
from .runners import RUNNERS, available_runners, execute
from .tools import EXAMPLE_PROMPTS

WEB_DIR = Path(__file__).parent / "web"


class RunRequest(BaseModel):
    framework: str = Field(examples=["langgraph"])
    prompt: str = Field(min_length=1, max_length=500)
    mode: str = Field(default="mock", pattern="^(mock|gemini)$")


def create_app(public_url: str | None = None) -> FastAPI:
    public_url = public_url or os.getenv("PUBLIC_URL", "http://127.0.0.1:8000")
    limiter = RateLimiter(
        per_key=int(os.getenv("GEMINI_DEMO_PER_IP_HOURLY", "5")),
        total=int(os.getenv("GEMINI_DEMO_TOTAL_HOURLY", "30")),
    )
    mcp_app = mcp.http_app(path="/")
    app = FastAPI(title="agent-platform", version=__version__, lifespan=mcp_app.lifespan)

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/frameworks")
    async def frameworks() -> dict:
        return {
            "available": available_runners(),
            "all": {k: v[1] for k, v in RUNNERS.items()},
            "examples": EXAMPLE_PROMPTS,
            "gemini_enabled": bool(os.getenv("GEMINI_API_KEY") and os.getenv("ENABLE_GEMINI_DEMO")),
        }

    @app.post("/api/run")
    async def run(req: RunRequest, request: Request) -> JSONResponse:
        if req.framework not in RUNNERS:
            raise HTTPException(404, f"unknown framework: {req.framework}")
        if req.framework not in available_runners():
            raise HTTPException(409, f"{req.framework} is not installed in this environment")
        if req.mode == "gemini" and not os.getenv("ENABLE_GEMINI_DEMO"):
            raise HTTPException(403, "live Gemini mode is disabled on this server")
        if req.mode == "gemini":
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                raise HTTPException(429, "live Gemini demo is rate limited; try again later")
        try:
            get_llm_config(req.mode)
            return JSONResponse(await execute(req.framework, req.prompt, req.mode))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    app.router.routes.extend(build_a2a_routes(public_url))
    app.mount("/mcp", mcp_app)
    if WEB_DIR.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

        @app.get("/recordings.json", include_in_schema=False)
        async def recordings() -> FileResponse:
            return FileResponse(WEB_DIR / "recordings.json")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
