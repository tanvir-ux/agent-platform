"""Google ADK: LlmAgent + Runner + in-memory sessions.

Mock mode plugs a custom `BaseLlm` into ADK (ADK speaks Gemini-style `Content` objects, not the
OpenAI wire format), reusing the same deterministic planner. Live mode uses the real
`gemini-*` model string via the `GOOGLE_API_KEY` environment variable.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..llm import LLMConfig
from ..mockllm import decide
from ..tools import SYSTEM_PROMPT, TOOLS
from ..trace import RECORDER

NAME = "google_adk"
LABEL = "Google ADK"


def _to_openai_messages(request: LlmRequest) -> list[dict]:
    messages: list[dict] = []
    for content in request.contents:
        for part in content.parts or []:
            if part.function_call:
                fc = part.function_call
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": fc.id or fc.name,
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(fc.args or {}),
                                },
                            }
                        ],
                    }
                )
            elif part.function_response:
                fr = part.function_response
                payload = fr.response or {}
                text = (
                    payload["result"]
                    if isinstance(payload.get("result"), str)
                    else json.dumps(payload)
                )
                messages.append({"role": "tool", "tool_call_id": fr.id or fr.name, "content": text})
            elif part.text:
                role = "assistant" if content.role == "model" else "user"
                messages.append({"role": role, "content": part.text})
    return messages


class MockPlannerLlm(BaseLlm):
    """Deterministic ADK model that reuses the shared rule-based planner."""

    model: str = "mock-support-planner"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"mock-support-planner"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        names = {
            d.name
            for t in (llm_request.config.tools or [] if llm_request.config else [])
            for d in (t.function_declarations or [])
        }
        decision = decide(_to_openai_messages(llm_request), names)
        if "tool_calls" in decision:
            parts = [
                types.Part(
                    function_call=types.FunctionCall(
                        name=c["name"], args=c["arguments"], id=uuid.uuid4().hex[:8]
                    )
                )
                for c in decision["tool_calls"]
            ]
            RECORDER.add(
                "llm",
                self.model,
                decision="tool_calls",
                detail=", ".join(
                    f"{c['name']}({json.dumps(c['arguments'])})" for c in decision["tool_calls"]
                ),
            )
        else:
            parts = [types.Part(text=decision["content"])]
            RECORDER.add("llm", self.model, decision="answer", detail=decision["content"])
        yield LlmResponse(content=types.Content(role="model", parts=parts))


async def run(prompt: str, llm: LLMConfig) -> str:
    if llm.is_mock:
        model: BaseLlm | str = MockPlannerLlm()
    else:
        os.environ["GOOGLE_API_KEY"] = llm.api_key
        model = llm.model
    agent = LlmAgent(
        name="support_agent", model=model, instruction=SYSTEM_PROMPT, tools=list(TOOLS)
    )
    sessions = InMemorySessionService()
    runner = Runner(app_name="agent-platform", agent=agent, session_service=sessions)
    session = await sessions.create_session(app_name="agent-platform", user_id="demo")
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    final = ""
    async for event in runner.run_async(user_id="demo", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts)
    return final
