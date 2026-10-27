"""AutoGen (AgentChat): AssistantAgent with tool functions and an OpenAI-compatible client."""

from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..llm import LLMConfig
from ..tools import SYSTEM_PROMPT, TOOLS

NAME = "autogen"
LABEL = "AutoGen"


async def run(prompt: str, llm: LLMConfig) -> str:
    client = OpenAIChatCompletionClient(
        model=llm.model,
        base_url=llm.base_url,
        api_key=llm.api_key,
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            family="unknown",
            structured_output=False,
        ),
    )
    agent = AssistantAgent(
        "support_agent",
        model_client=client,
        tools=list(TOOLS),
        system_message=SYSTEM_PROMPT,
        reflect_on_tool_use=True,
        max_tool_iterations=3,
    )
    try:
        result = await agent.run(task=prompt)
        return str(result.messages[-1].to_text())
    finally:
        await client.close()
