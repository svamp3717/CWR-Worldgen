"""Keep the manually-built Windows tools and tools-v release in sync."""
from __future__ import annotations

import re
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
TOOL_DIR = PROJECT / "tools"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from generate_pyinstaller_splash import TOOLS, WORLDGEN_NAME  # noqa: E402

BUILD = PROJECT / ".github" / "workflows" / "build-windows-tools.yml"
PUBLISH = PROJECT / ".github" / "workflows" / "build-and-publish-windows-tools.yml"


def test_tool_matrix_has_all_named_tools_and_entrypoints() -> None:
    workflow = BUILD.read_text(encoding="utf-8")
    entries = re.findall(
        r"(?m)^          - name: ([\w-]+)\n"
        r"            script: (tools/[\w_]+\.py)\n"
        r"            mode: (--[\w]+)$",
        workflow,
    )
    by_name = {name: (script, mode) for name, script, mode in entries}
    assert len(entries) == len(by_name) == 4
    assert set(by_name) == set(TOOLS)
    assert WORLDGEN_NAME not in by_name
    assert by_name["CWR-Island-Studio"] == (
        "tools/island_studio_entry.py", "--windowed",
    )
    assert by_name["WRP-Mod-Dependency-Scanner"][1] == "--console"
    for name, (script, mode) in by_name.items():
        assert (PROJECT / script).is_file(), name
        assert mode in ("--windowed", "--console")
    assert "matrix.name" in workflow
    assert "retention-days: 14" in workflow
    assert 'python tools/generate_pyinstaller_splash.py' in workflow
    assert 'python tools/configure_tool_splash_spec.py' in workflow
    assert "workflow_dispatch:" in workflow and "workflow_call:" in workflow


def test_manual_asset_release_verifies_and_publishes_every_tool() -> None:
    source = PUBLISH.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in source
    assert "on:\n  push:" not in source
    assert "uses: ./.github/workflows/build-windows-tools.yml" in source
    loop = re.search(r"for tool in ([^\n]+); do", source)
    assert loop is not None
    assert set(loop.group(1).split()) == set(TOOLS)
    for name in TOOLS:
        assert f'"release-assets/{name}.exe" \\' in source
    assert "gh release create" in source
    assert "--latest=false" in source
    assert "refs/heads/main" in source
