import json

from fastmcp import Client

from agent_platform.mcp_server import mcp


async def test_mcp_lists_and_calls_tools():
    async with Client(mcp) as client:
        names = {t.name for t in await client.list_tools()}
        assert names == {
            "get_order_status",
            "search_policy",
            "calculate_refund",
            "convert_currency",
        }
        result = await client.call_tool("get_order_status", {"order_id": "1042"})
        assert json.loads(result.content[0].text)["status"] == "shipped"


async def test_mcp_policy_resource():
    async with Client(mcp) as client:
        text = (await client.read_resource("policy://handbook"))[0].text
        assert "refund" in text.lower()
