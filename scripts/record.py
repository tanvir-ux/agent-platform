"""Record real framework runs for the static (GitHub Pages) demo.

Run once in the main environment and once in the CrewAI environment; results are merged:
    python scripts/record.py
    python scripts/record.py          # inside the CrewAI venv
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_platform.runners import available_runners, execute
from agent_platform.tools import EXAMPLE_PROMPTS

OUT = Path(__file__).resolve().parents[1] / "src" / "agent_platform" / "web" / "recordings.json"
ORDER = [
    "langgraph",
    "google_adk",
    "openai_agents",
    "msaf",
    "openai_agents_mcp",
    "autogen",
    "semantic_kernel",
    "crewai",
]


async def main() -> None:
    data = json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
    runs = {(r["framework"], r["prompt"]): r for r in data["runs"]}
    for name in available_runners():
        for prompt in EXAMPLE_PROMPTS:
            result = await execute(name, prompt, "mock")
            runs[(name, prompt)] = result
            print(f"recorded {name:<18} {prompt[:48]!r} -> {result['duration_ms']} ms")
    frameworks = [f for f in ORDER if any(k[0] == f for k in runs)]
    data = {
        "note": "Real framework runs against the deterministic mock model (scripts/record.py).",
        "prompts": EXAMPLE_PROMPTS,
        "frameworks": frameworks,
        "runs": [
            runs[k]
            for k in sorted(runs, key=lambda k: (ORDER.index(k[0]), EXAMPLE_PROMPTS.index(k[1])))
        ],
    }
    OUT.write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
