"""CrewAI: Agent + Task + Crew with tools decorated via `crewai.tools.tool`."""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

from ..llm import LLMConfig  # noqa: E402
from ..tools import SYSTEM_PROMPT, TOOLS  # noqa: E402

NAME = "crewai"
LABEL = "CrewAI"


def _crew_tools():
    wrapped = []
    for fn in TOOLS:
        wrapped.append(tool(fn.__name__)(fn))
    return wrapped


def _run_sync(prompt: str, llm: LLMConfig) -> str:
    model = LLM(
        model=f"openai/{llm.model}",
        base_url=llm.base_url,
        api_key=llm.api_key,
        temperature=0,
    )
    agent = Agent(
        role="Customer support specialist",
        goal=SYSTEM_PROMPT,
        backstory="You answer customers precisely using the provided tools.",
        tools=_crew_tools(),
        llm=model,
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=prompt,
        expected_output="A concise, factual answer for the customer.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff().raw)


async def run(prompt: str, llm: LLMConfig) -> str:
    return await asyncio.to_thread(_run_sync, prompt, llm)
