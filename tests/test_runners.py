import asyncio

import pytest

from agent_platform.runners import RUNNERS, execute

from .conftest import EXPECTED


@pytest.mark.parametrize("prompt", sorted(EXPECTED))
def test_framework_produces_expected_answer_and_trace(framework, installed, prompt):
    if framework not in installed:
        pytest.skip(f"{framework} is not installed in this environment")
    result = asyncio.run(execute(framework, prompt, "mock"))
    for fragment in EXPECTED[prompt]:
        assert fragment in result["answer"], result["answer"]
    kinds = [e["kind"] for e in result["trace"]]
    assert "llm" in kinds and "tool" in kinds
    assert result["label"] == RUNNERS[framework][1]


def test_unknown_framework():
    with pytest.raises(KeyError):
        asyncio.run(execute("nope", "hi", "mock"))
