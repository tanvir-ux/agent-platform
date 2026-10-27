import json
from pathlib import Path

from agent_platform import cli
from agent_platform.runners import RUNNERS
from agent_platform.tools import EXAMPLE_PROMPTS


def test_cli_list_and_run(capsys):
    assert cli.main(["list"]) == 0
    first = capsys.readouterr().out.split()[0]
    assert cli.main(["run", first, "Convert 120 USD to BDT."]) == 0
    assert "14640" in capsys.readouterr().out


def test_cli_unknown_framework(capsys):
    assert cli.main(["run", "nope"]) == 2


def test_recordings_cover_every_framework_and_prompt():
    path = Path(cli.__file__).parent / "web" / "recordings.json"
    data = json.loads(path.read_text())
    assert set(data["frameworks"]) == set(RUNNERS)
    keys = {(r["framework"], r["prompt"]) for r in data["runs"]}
    assert keys == {(f, p) for f in RUNNERS for p in EXAMPLE_PROMPTS}
    assert all(r["answer"] and r["trace"] for r in data["runs"])


def test_pages_site_is_in_sync_with_packaged_ui():
    root = Path(__file__).resolve().parents[1]
    web = root / "src" / "agent_platform" / "web"
    for name in ("index.html", "recordings.json"):
        assert (root / "docs" / name).read_text() == (web / name).read_text(), name
