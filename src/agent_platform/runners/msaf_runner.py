"""Microsoft Agent Framework: Agent over the OpenAI Chat Completions client with function tools."""

from __future__ import annotations

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

from ..llm import LLMConfig
from ..tools import SYSTEM_PROMPT, TOOLS

NAME = "msaf"
LABEL = "Microsoft Agent Framework"


async def run(prompt: str, llm: LLMConfig) -> str:
    client = OpenAIChatCompletionClient(model=llm.model, api_key=llm.api_key, base_url=llm.base_url)
    agent = Agent(
        client,
        instructions=SYSTEM_PROMPT,
        name="SupportAgent",
        tools=[tool(fn) for fn in TOOLS],
    )
    response = await agent.run(prompt)
    return str(response.text)
