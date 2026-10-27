"""Semantic Kernel: ChatCompletionAgent with a native kernel plugin and auto tool invocation."""

from __future__ import annotations

from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

from .. import tools
from ..llm import LLMConfig

NAME = "semantic_kernel"
LABEL = "Semantic Kernel"


class SupportPlugin:
    """Kernel plugin delegating to the shared tool functions."""

    @kernel_function(description=tools.get_order_status.__doc__)
    def get_order_status(self, order_id: Annotated[str, "4-digit order id"]) -> str:
        return tools.get_order_status(order_id)

    @kernel_function(description=tools.search_policy.__doc__)
    def search_policy(
        self, topic: Annotated[str, "refund, shipping, warranty or cancellation"]
    ) -> str:
        return tools.search_policy(topic)

    @kernel_function(description=tools.calculate_refund.__doc__)
    def calculate_refund(self, order_id: Annotated[str, "4-digit order id"]) -> str:
        return tools.calculate_refund(order_id)

    @kernel_function(description=tools.convert_currency.__doc__)
    def convert_currency(
        self,
        amount: Annotated[float, "amount to convert"],
        from_currency: Annotated[str, "source currency code"],
        to_currency: Annotated[str, "target currency code"],
    ) -> str:
        return tools.convert_currency(amount, from_currency, to_currency)


async def run(prompt: str, llm: LLMConfig) -> str:
    client = AsyncOpenAI(base_url=llm.base_url, api_key=llm.api_key)
    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(ai_model_id=llm.model, async_client=client),
        name="SupportAgent",
        instructions=tools.SYSTEM_PROMPT,
        plugins=[SupportPlugin()],
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
    response = await agent.get_response(messages=prompt)
    return str(response.message.content)
