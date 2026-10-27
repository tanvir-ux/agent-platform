import json

import pytest
from fastapi.testclient import TestClient

from agent_platform.api import create_app
from agent_platform.runners import available_runners


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app("http://testserver")) as c:
        yield c


def test_health_and_frameworks(client):
    assert client.get("/api/health").json()["status"] == "ok"
    data = client.get("/api/frameworks").json()
    assert len(data["all"]) == 8 and data["examples"]


def test_run_and_validation(client):
    fw = next(iter(available_runners()))
    ok = client.post("/api/run", json={"framework": fw, "prompt": "Convert 120 USD to BDT."})
    assert ok.status_code == 200 and "14640" in ok.json()["answer"]
    assert client.post("/api/run", json={"framework": "nope", "prompt": "x"}).status_code == 404
    assert client.post("/api/run", json={"framework": fw, "prompt": ""}).status_code == 422


def test_gemini_mode_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("ENABLE_GEMINI_DEMO", raising=False)
    fw = next(iter(available_runners()))
    r = client.post("/api/run", json={"framework": fw, "prompt": "hi", "mode": "gemini"})
    assert r.status_code == 403


def test_gemini_mode_is_rate_limited(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_DEMO", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_DEMO_PER_IP_HOURLY", "1")
    fw = next(iter(available_runners()))
    with TestClient(create_app("http://testserver")) as c:
        body = {"framework": fw, "prompt": "hi", "mode": "gemini"}
        assert c.post("/api/run", json=body).status_code == 503  # no key configured
        assert c.post("/api/run", json=body).status_code == 429


def test_web_ui_and_recordings(client):
    assert "agent-platform" in client.get("/").text
    rec = client.get("/recordings.json").json()
    assert rec["frameworks"] and rec["runs"]


def test_a2a_agent_card(client):
    card = client.get("/.well-known/agent-card.json").json()
    assert card["name"] and card["skills"]


def test_a2a_message_send(client):
    fw = next(iter(available_runners()))
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": "m1",
                "role": "ROLE_USER",
                "parts": [{"text": "Convert 120 USD to BDT."}],
                "metadata": {"framework": fw},
            }
        },
    }
    r = client.post("/a2a", json=payload, headers={"A2A-Version": "1.0"})
    assert r.status_code == 200
    assert "14640" in json.dumps(r.json())
