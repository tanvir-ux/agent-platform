"""Deterministic mock LLM exposed as an OpenAI-compatible `/v1/chat/completions` endpoint.

It is *not* a language model. It is a small rule-based planner that:
1. reads the latest user message and decides which of the offered tools to call, then
2. once tool results are present, composes the final answer from them.

Because it speaks the OpenAI wire format, every framework in this repository (OpenAI Agents SDK,
AutoGen, Semantic Kernel, Microsoft Agent Framework, LangChain/LangGraph, CrewAI) exercises its
real agent loop, tool-calling and message plumbing against it, with no API keys and no
randomness. Swapping `base_url` to a real provider turns the same code into a live agent.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .trace import RECORDER

_CURRENCY = r"(usd|eur|bdt|gbp)"
_CONVERT_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*{_CURRENCY}\s*(?:to|in|into)\s*{_CURRENCY}", re.I)
_ORDER_RE = re.compile(r"\b(\d{4})\b")


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def plan_tool_calls(query: str, tool_names: set[str]) -> list[dict[str, Any]]:
    """Decide which tools to call for a user query (pure function, easy to unit test)."""
    q = query.lower()
    calls: list[dict[str, Any]] = []

    def add(name: str, **args: Any) -> None:
        if name in tool_names:
            calls.append({"name": name, "arguments": args})

    convert = _CONVERT_RE.search(query)
    order = _ORDER_RE.search(query)
    wants_refund = any(w in q for w in ("refund", "return", "money back", "get back"))

    if convert:
        add(
            "convert_currency",
            amount=float(convert[1]),
            from_currency=convert[2].upper(),
            to_currency=convert[3].upper(),
        )
    if order:
        add("get_order_status", order_id=order[1])
    if wants_refund:
        add("search_policy", topic="refund")
        if order:
            add("calculate_refund", order_id=order[1])
    if "cancel" in q:
        add("search_policy", topic="cancellation")
    if "warranty" in q:
        add("search_policy", topic="warranty")
    if "shipping" in q or "delivery time" in q:
        add("search_policy", topic="shipping")
    if not calls:
        add("search_policy", topic=query)
    return calls


def compose_answer(results: list[dict[str, Any]]) -> str:
    """Turn parsed tool results into a short, factual answer."""
    parts: list[str] = []
    for r in results:
        if "error" in r:
            parts.append(f"I could not complete a lookup: {r['error']}.")
        elif "status" in r and "item" in r:
            parts.append(
                f"Order {r['order_id']} ({r['item']}) is {r['status']}, expected {r['eta']}."
            )
        elif "policies" in r:
            body = " ".join(r["policies"].values()) or "No matching policy was found."
            parts.append(f"Policy: {body}")
        elif "refund_usd" in r:
            fee = r.get("restocking_fee_pct", 0)
            suffix = f" after a {fee}% restocking fee" if fee else ""
            parts.append(
                f"Estimated refund for order {r['order_id']}: ${r['refund_usd']:.2f}{suffix}."
            )
        elif "converted" in r:
            parts.append(f"{r['amount']:g} {r['from']} is {r['converted']:g} {r['to']}.")
    return " ".join(parts) or "I do not have enough information to answer."


def _base_name(name: str) -> str:
    """Strip framework prefixes such as 'SupportPlugin-' or 'server__' from a tool name."""
    return re.split(r"[-.:]|__", name)[-1]


def decide(messages: list[dict[str, Any]], tool_names: set[str]) -> dict[str, Any]:
    """Return {'tool_calls': [...]} or {'content': str} for an OpenAI-format message list.

    Some frameworks namespace tool names (Semantic Kernel sends 'Plugin-tool'); the planner
    matches on the base name and echoes the exact name it was offered.
    """
    actual = {_base_name(n): n for n in tool_names}
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    tool_msgs = [m for m in messages[last_user + 1 :] if m.get("role") == "tool"]
    if tool_msgs:
        results = []
        for m in tool_msgs:
            try:
                results.append(json.loads(_text(m.get("content"))))
            except json.JSONDecodeError:
                results.append({"error": _text(m.get("content"))[:120]})
        return {"content": compose_answer(results)}
    query = _text(messages[last_user].get("content")) if last_user >= 0 else ""
    calls = plan_tool_calls(query, set(actual))
    for call in calls:
        call["name"] = actual[call["name"]]
    return {"tool_calls": calls} if calls else {"content": "How can I help?"}


router = APIRouter()


def _completion(model: str, decision: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any]
    if "tool_calls" in decision:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                }
                for c in decision["tool_calls"]
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": decision["content"]}
        finish = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse(completion: dict[str, Any]):
    choice = completion["choices"][0]
    base = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
    }
    delta: dict[str, Any] = {"role": "assistant"}
    msg = choice["message"]
    if msg.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(msg["tool_calls"])]
    else:
        delta["content"] = msg["content"]
    first = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    last_choice = {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
    yield f"data: {json.dumps(first)}\n\n"
    yield f"data: {json.dumps({**base, 'choices': [last_choice]})}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": "mock-support-planner", "object": "model"}]}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    tools = body.get("tools") or []
    names = {t.get("function", {}).get("name") for t in tools if t.get("type") == "function"}
    decision = decide(body.get("messages", []), {n for n in names if n})
    kind = "tool_calls" if "tool_calls" in decision else "answer"
    RECORDER.add(
        "llm",
        "mock-support-planner",
        decision=kind,
        detail=(
            ", ".join(f"{c['name']}({json.dumps(c['arguments'])})" for c in decision["tool_calls"])
            if kind == "tool_calls"
            else decision["content"]
        ),
    )
    completion = _completion(body.get("model", "mock-support-planner"), decision)
    if body.get("stream"):
        return StreamingResponse(_sse(completion), media_type="text/event-stream")
    return JSONResponse(completion)
