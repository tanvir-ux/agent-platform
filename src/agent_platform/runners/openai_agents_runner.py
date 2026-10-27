"""OpenAI Agents SDK: Agent + function tools on the Chat Completions model backend."""

from __future__ import annotations

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI

from ..llm import LLMConfig
from ..tools import SYSTEM_PROMPT, TOOLS

NAME = "openai_agents"
LABEL = "OpenAI Agents SDK"


async def run(prompt: str, llm: LLMConfig) -> str:
    set_tracing_disabled(True)  # the SDK's built-in tracing exports to OpenAI's service
    client = AsyncOpenAI(base_url=llm.base_url, api_key=llm.api_key)
    agent = Agent(
        name="SupportAgent",
        instructions=SYSTEM_PROMPT,
        model=OpenAIChatCompletionsModel(model=llm.model, openai_client=client),
        tools=[function_tool(fn) for fn in TOOLS],
    )
    result = await Runner.run(agent, prompt)
    return str(result.final_output)
