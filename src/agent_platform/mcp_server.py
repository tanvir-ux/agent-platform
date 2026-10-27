"""FastMCP server exposing the support tools over the Model Context Protocol."""

from __future__ import annotations

from fastmcp import FastMCP

from . import tools

mcp = FastMCP("support-desk")

for _fn in tools.TOOLS:
    mcp.tool(_fn)


@mcp.resource("policy://handbook")
def handbook() -> str:
    """The full support policy handbook."""
    return "\n".join(f"{k}: {v}" for k, v in tools.POLICIES.items())


def main() -> None:
    """Run over stdio (for MCP clients such as desktop assistants)."""
    mcp.run()


if __name__ == "__main__":
    main()
