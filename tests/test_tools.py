import json

from agent_platform import tools


def test_order_status_found_and_missing():
    assert json.loads(tools.get_order_status("1042"))["status"] == "shipped"
    assert "error" in json.loads(tools.get_order_status("9999"))


def test_refund_is_full_for_unopened_item():
    out = json.loads(tools.calculate_refund("1042"))
    assert out == {"order_id": "1042", "refund_usd": 120.0, "restocking_fee_pct": 0}


def test_refund_applies_restocking_fee_for_opened_electronics(monkeypatch):
    order = {**tools.ORDERS["1042"], "opened": True}
    monkeypatch.setitem(tools.ORDERS, "1042", order)
    out = json.loads(tools.calculate_refund("1042"))
    assert out["restocking_fee_pct"] == 15 and out["refund_usd"] == 102.0


def test_refund_outside_window_is_zero(monkeypatch):
    order = {**tools.ORDERS["1042"], "purchased": "2026-01-01"}
    monkeypatch.setitem(tools.ORDERS, "1042", order)
    assert json.loads(tools.calculate_refund("1042"))["refund_usd"] == 0.0


def test_policy_search_and_unknown_topic():
    assert "refund" in json.loads(tools.search_policy("refund"))["policies"]
    assert json.loads(tools.search_policy("zzzz-nothing"))["policies"] == {}


def test_convert_currency():
    assert json.loads(tools.convert_currency(120, "usd", "bdt"))["converted"] == 14640.0
    assert "error" in json.loads(tools.convert_currency(1, "USD", "XXX"))


def test_tool_specs_match_tools():
    assert {s["name"] for s in tools.TOOL_SPECS} == {f.__name__ for f in tools.TOOLS}
    assert len(tools.EXAMPLE_PROMPTS) == 4
