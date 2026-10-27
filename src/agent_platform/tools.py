"""Domain tools for the demo task: a small customer-support desk.

The same plain Python functions are (a) wrapped natively by every agent framework and
(b) exposed over MCP by `mcp_server.py`. All of them return JSON strings so every framework
passes identical text back to the model.
"""

from __future__ import annotations

import json
from datetime import date

from .trace import traced

ORDERS: dict[str, dict] = {
    "1042": {
        "item": "Wireless Headphones",
        "category": "electronics",
        "price_usd": 120.0,
        "status": "shipped",
        "eta": "2026-11-02",
        "opened": False,
        "purchased": "2026-10-20",
    },
    "1043": {
        "item": "Running Shoes",
        "category": "apparel",
        "price_usd": 80.0,
        "status": "delivered",
        "eta": "2026-10-18",
        "opened": True,
        "purchased": "2026-10-10",
    },
    "2077": {
        "item": "Laptop Stand",
        "category": "accessories",
        "price_usd": 45.0,
        "status": "processing",
        "eta": "2026-11-05",
        "opened": False,
        "purchased": "2026-10-25",
    },
}

POLICIES: dict[str, str] = {
    "refund": "Refunds are available within 30 days of purchase. Opened electronics carry a "
    "15% restocking fee. Unopened items are refunded in full.",
    "shipping": "Standard shipping takes 3 to 7 business days. Orders over 50 USD ship free.",
    "warranty": "Electronics include a 12-month limited warranty covering manufacturing defects.",
    "cancellation": "Orders in the processing state can be cancelled free of charge.",
}

RATES_PER_USD: dict[str, float] = {"USD": 1.0, "EUR": 0.92, "BDT": 122.0, "GBP": 0.79}

TODAY = date(2026, 10, 27)


@traced("tool.get_order_status", kind="tool")
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its 4-digit order id."""
    order = ORDERS.get(str(order_id).strip())
    if order is None:
        return json.dumps({"order_id": str(order_id), "error": "order not found"})
    return json.dumps(
        {
            "order_id": str(order_id),
            "item": order["item"],
            "status": order["status"],
            "eta": order["eta"],
        }
    )


@traced("tool.search_policy", kind="tool")
def search_policy(topic: str) -> str:
    """Search the support policy handbook (refund, shipping, warranty, cancellation)."""
    topic_l = topic.lower()
    hits = {k: v for k, v in POLICIES.items() if k in topic_l or topic_l in k}
    if not hits:
        hits = {k: v for k, v in POLICIES.items() if any(w in v.lower() for w in topic_l.split())}
    return json.dumps({"topic": topic, "policies": hits})


@traced("tool.calculate_refund", kind="tool")
def calculate_refund(order_id: str) -> str:
    """Compute the refund amount in USD for an order under the refund policy."""
    order = ORDERS.get(str(order_id).strip())
    if order is None:
        return json.dumps({"order_id": str(order_id), "error": "order not found"})
    age = (TODAY - date.fromisoformat(order["purchased"])).days
    if age > 30:
        return json.dumps(
            {"order_id": str(order_id), "refund_usd": 0.0, "reason": "outside window"}
        )
    fee = 0.15 if (order["category"] == "electronics" and order["opened"]) else 0.0
    refund = round(order["price_usd"] * (1 - fee), 2)
    return json.dumps(
        {"order_id": str(order_id), "refund_usd": refund, "restocking_fee_pct": int(fee * 100)}
    )


@traced("tool.convert_currency", kind="tool")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between USD, EUR, BDT and GBP using fixed demo rates."""
    src, dst = from_currency.upper(), to_currency.upper()
    if src not in RATES_PER_USD or dst not in RATES_PER_USD:
        return json.dumps({"error": f"unsupported currency pair {src}/{dst}"})
    value = round(float(amount) / RATES_PER_USD[src] * RATES_PER_USD[dst], 2)
    return json.dumps({"amount": float(amount), "from": src, "to": dst, "converted": value})


TOOLS = [get_order_status, search_policy, calculate_refund, convert_currency]

TOOL_SPECS: list[dict] = [
    {
        "name": "get_order_status",
        "description": get_order_status.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": search_policy.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
    {
        "name": "calculate_refund",
        "description": calculate_refund.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "convert_currency",
        "description": convert_currency.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a customer-support assistant. Use the tools to look up order status, policies, "
    "refund amounts and currency conversions, then answer concisely using only tool results."
)

EXAMPLE_PROMPTS = [
    "Where is order 1042, and how much would I get back if I return it?",
    "What is the refund policy, and can I get money back for order 1043?",
    "Convert 120 USD to BDT.",
    "Can I cancel order 2077?",
]
