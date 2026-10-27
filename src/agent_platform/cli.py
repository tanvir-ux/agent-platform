"""Command line entry point: `agent-platform serve|run|mcp|list`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .runners import available_runners, execute
from .tools import EXAMPLE_PROMPTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-platform", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="start the FastAPI service (REST + A2A + MCP + web UI)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    run = sub.add_parser("run", help="run one framework (or `all`) on a prompt")
    run.add_argument("framework", help="framework name, or `all`")
    run.add_argument("prompt", nargs="?", default=EXAMPLE_PROMPTS[0])
    run.add_argument("--mode", choices=["mock", "gemini"], default="mock")
    run.add_argument("--trace", action="store_true", help="print the full trace as JSON")

    sub.add_parser("mcp", help="run the MCP server over stdio")
    sub.add_parser("list", help="list the frameworks installed in this environment")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, label in available_runners().items():
            print(f"{name:<18} {label}")
        return 0
    if args.cmd == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()
        return 0
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("agent_platform.api:app", host=args.host, port=args.port)
        return 0

    names = list(available_runners()) if args.framework == "all" else [args.framework]
    for name in names:
        try:
            result = asyncio.run(execute(name, args.prompt, args.mode))
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"[{result['label']}] ({result['duration_ms']} ms) {result['answer']}")
        if args.trace:
            print(json.dumps(result["trace"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
