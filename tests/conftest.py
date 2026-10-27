import pytest

from agent_platform.runners import RUNNERS, available_runners

EXPECTED = {
    "Where is order 1042, and how much would I get back if I return it?": [
        "1042",
        "shipped",
        "$120.00",
    ],
    "Convert 120 USD to BDT.": ["14640 BDT"],
    "Can I cancel order 2077?": ["2077", "Policy"],
}


@pytest.fixture(scope="session")
def installed() -> dict[str, str]:
    return available_runners()


def pytest_generate_tests(metafunc):
    if "framework" in metafunc.fixturenames:
        metafunc.parametrize("framework", sorted(RUNNERS))
