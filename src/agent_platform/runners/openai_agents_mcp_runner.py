"""OpenAI Agents SDK consuming the FastMCP server over real MCP (stdio transport).

Unlike the other runners, the tools here are not Python callables inside the agent process.
The agent discovers them from the MCP server (`python -m agent_platform.mcp_server`) at run
time, which is how tools are shared between independent agents and desktop assistants.
"""

from __future__ import annotations

import os
import sys

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio
from openai import AsyncOpenAI

from ..llm import LLMConfig
from ..tools import SYSTEM_PROMPT
from ..trace import RECORDER

NAME = "openai_agents_mcp"
LABEL = "OpenAI Agents SDK + MCP"


async def run(prompt: str, llm: LLMConfig) -> str:
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=llm.base_url, api_key=llm.api_key)
    async with MCPServerStdio(
        params={
            "command": sys.executable,
            "args": ["-m", "agent_platform.mcp_server"],
            "env": dict(os.environ),
        },
        cache_tools_list=True,
    ) as server:
        agent = Agent(
            name="SupportAgentMCP",
            instructions=SYSTEM_PROMPT,
            model=OpenAIChatCompletionsModel(model=llm.model, openai_client=client),
            mcp_servers=[server],
        )
        result = await Runner.run(agent, prompt)
    # The tools ran in the MCP server subprocess, so mirror them into the local trace.
    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        if item.type == "tool_call_item":
            RECORDER.add(
                "tool", f"mcp.{raw.name}", input=str(raw.arguments), output="called over MCP"
            )
        elif item.type == "tool_call_output_item":
            RECORDER.add("tool", "mcp.result", input="", output=str(item.output)[:400])
    return str(result.final_output)
