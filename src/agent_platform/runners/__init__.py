"""Agent runners: the same support task implemented once per framework.

Every runner module exposes `NAME`, `LABEL` and `async def run(prompt, llm) -> str`.
Modules are imported lazily so a missing optional framework never breaks the others.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from typing import Any

from ..llm import get_llm_config
from ..trace import RECORDER

RUNNERS: dict[str, tuple[str, str]] = {
    "langgraph": ("langgraph_runner", "LangGraph"),
    "google_adk": ("adk_runner", "Google ADK"),
    "openai_agents": ("openai_agents_runner", "OpenAI Agents SDK"),
    "msaf": ("msaf_runner", "Microsoft Agent Framework"),
    "openai_agents_mcp": ("openai_agents_mcp_runner", "OpenAI Agents SDK + MCP"),
    "autogen": ("autogen_runner", "AutoGen"),
    "semantic_kernel": ("semantic_kernel_runner", "Semantic Kernel"),
    "crewai": ("crewai_runner", "CrewAI"),
}

_run_lock = asyncio.Lock()


def available_runners() -> dict[str, str]:
    """Map runner name -> label for runners whose framework is importable in this environment."""
    found: dict[str, str] = {}
    for name, (module, label) in RUNNERS.items():
        try:
            importlib.import_module(f"{__name__}.{module}")
        except ImportError:
            continue
        found[name] = label
    return found


async def execute(name: str, prompt: str, mode: str | None = None) -> dict[str, Any]:
    """Run one framework end to end and return the answer plus a trace of what happened."""
    if name not in RUNNERS:
        raise KeyError(f"unknown framework '{name}'. Known: {sorted(RUNNERS)}")
    module_name, label = RUNNERS[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    llm = get_llm_config(mode)
    async with _run_lock:  # the trace recorder is process-global, so runs are serialised
        RECORDER.reset()
        started = time.perf_counter()
        answer = await module.run(prompt, llm)
        elapsed = round((time.perf_counter() - started) * 1000)
        trace = RECORDER.snapshot()
    return {
        "framework": name,
        "label": label,
        "prompt": prompt,
        "answer": answer,
        "mode": llm.mode,
        "model": llm.model,
        "duration_ms": elapsed,
        "trace": trace,
    }
