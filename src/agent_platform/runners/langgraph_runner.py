"""LangGraph: an explicit StateGraph (agent node -> tool node loop) over LangChain tools."""

from __future__ import annotations

from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm import LLMConfig
from ..tools import SYSTEM_PROMPT, TOOLS

NAME = "langgraph"
LABEL = "LangGraph"


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(llm: LLMConfig):
    """Compile the agent graph. Exposed separately so tests and docs can inspect it."""
    lc_tools = [StructuredTool.from_function(fn) for fn in TOOLS]
    model = ChatOpenAI(
        base_url=llm.base_url,
        api_key=llm.api_key,
        model=llm.model,
        temperature=0,
        # a fresh client per graph: langchain caches a global one that dies with its event loop
        http_async_client=httpx.AsyncClient(timeout=60),
    ).bind_tools(lc_tools)

    async def agent(state: State) -> dict:
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(lc_tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run(prompt: str, llm: LLMConfig) -> str:
    app = build_graph(llm)
    out = await app.ainvoke({"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]})
    return str(out["messages"][-1].content)
