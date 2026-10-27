"""A2A (Agent2Agent) protocol server: publishes an agent card and answers JSON-RPC tasks.

Any A2A-compatible client can discover the agent at `/.well-known/agent-card.json` and send it
a message. Each message is executed by one of the framework runners (default: LangGraph, or the
framework named in the message metadata key `framework`).
"""

from __future__ import annotations

import os

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState

from .runners import execute
from .tools import EXAMPLE_PROMPTS

DEFAULT_FRAMEWORK = os.getenv("A2A_DEFAULT_FRAMEWORK", "langgraph")
RPC_PATH = "/a2a"


def build_agent_card(base_url: str) -> AgentCard:
    skill = AgentSkill(
        id="support_desk",
        name="Customer support desk",
        description=(
            "Answers order-status, refund-policy, refund-amount and currency questions "
            "using tools. Implemented by interchangeable agent frameworks."
        ),
        tags=["support", "tools", "mcp"],
        examples=EXAMPLE_PROMPTS[:2],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )
    return AgentCard(
        name="Support Desk Agent",
        description="A tool-using support agent from the agent-platform showcase.",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"{base_url.rstrip('/')}{RPC_PATH}",
                protocol_version="1.0",
            )
        ],
        skills=[skill],
    )


class SupportAgentExecutor(AgentExecutor):
    """Bridges A2A tasks to the framework runners."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        await updater.update_status(
            TaskState.TASK_STATE_WORKING, message=new_text_message("Working on it...")
        )
        query = get_message_text(context.message)
        meta = dict(context.message.metadata) if context.message.metadata else {}
        framework = str(meta.get("framework", DEFAULT_FRAMEWORK))
        try:
            result = await execute(framework, query)
            text = result["answer"]
        except Exception as exc:  # report failures to the caller as a failed task
            await updater.update_status(
                TaskState.TASK_STATE_FAILED, message=new_text_message(f"Agent error: {exc}")
            )
            return
        await updater.add_artifact(parts=[new_text_part(text=text, media_type="text/plain")])
        await updater.update_status(
            TaskState.TASK_STATE_COMPLETED, message=new_text_message("Done.")
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellation is not supported by this demo agent.")


def build_a2a_routes(base_url: str) -> list:
    """Starlette routes (agent card + JSON-RPC) ready to mount on any ASGI app."""
    card = build_agent_card(base_url)
    handler = DefaultRequestHandler(
        agent_executor=SupportAgentExecutor(), task_store=InMemoryTaskStore(), agent_card=card
    )
    routes: list = []
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, RPC_PATH))
    return routes
