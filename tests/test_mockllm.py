import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_platform import mockllm

NAMES = {"get_order_status", "search_policy", "calculate_refund", "convert_currency"}


def test_plan_order_and_refund():
    calls = mockllm.plan_tool_calls("Where is order 1042 and can I get a refund?", NAMES)
    assert [c["name"] for c in calls] == ["get_order_status", "search_policy", "calculate_refund"]


def test_plan_currency():
    (call,) = mockllm.plan_tool_calls("convert 50 eur to gbp", NAMES)
    assert call["arguments"] == {"amount": 50.0, "from_currency": "EUR", "to_currency": "GBP"}


def test_plan_respects_available_tools():
    assert mockllm.plan_tool_calls("order 1042", {"search_policy"})[0]["name"] == "search_policy"


def test_decide_two_phase():
    msgs = [{"role": "user", "content": "Convert 10 USD to EUR"}]
    first = mockllm.decide(msgs, NAMES)
    assert first["tool_calls"][0]["name"] == "convert_currency"
    msgs.append(
        {
            "role": "tool",
            "content": json.dumps({"amount": 10.0, "from": "USD", "to": "EUR", "converted": 9.2}),
        }
    )
    assert "9.2 EUR" in mockllm.decide(msgs, NAMES)["content"]


def test_decide_handles_namespaced_tool_names():
    names = {"SupportPlugin-get_order_status"}
    out = mockllm.decide([{"role": "user", "content": "status of order 1042"}], names)
    assert out["tool_calls"][0]["name"] == "SupportPlugin-get_order_status"


def test_compose_answer_reports_errors():
    assert "not found" in mockllm.compose_answer([{"error": "order not found"}])


def test_openai_compatible_endpoints():
    app = FastAPI()
    app.include_router(mockllm.router)
    client = TestClient(app)
    assert client.get("/v1/models").json()["data"]
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "refund for order 1042"}],
        "tools": [{"type": "function", "function": {"name": n, "parameters": {}}} for n in NAMES],
    }
    msg = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
    assert msg["tool_calls"]
    streamed = client.post("/v1/chat/completions", json={**body, "stream": True})
    assert "data: [DONE]" in streamed.text
