from __future__ import annotations

from pathlib import Path

from cwr_worldgen.gui_entry import (
    ROAD_INSPECTOR_CLI_MARKER,
    road_inspector_postbuild_command,
)


def test_source_gui_postbuild_command_uses_python_module(tmp_path: Path) -> None:
    command = road_inspector_postbuild_command(
        tmp_path / "build",
        "wg_demo",
        tmp_path / "source",
        frozen=False,
        executable="python-test",
    )

    assert command == [
        "python-test",
        "-m",
        "cwr_worldgen.road_inspector_postbuild",
        "--build-dir",
        str(tmp_path / "build"),
        "--world-name",
        "wg_demo",
        "--source-dir",
        str(tmp_path / "source"),
    ]


def test_frozen_gui_postbuild_command_reenters_executable(tmp_path: Path) -> None:
    command = road_inspector_postbuild_command(
        tmp_path / "build",
        "wg_demo",
        frozen=True,
        executable="CWR-Worldgen.exe",
    )

    assert command[:2] == ["CWR-Worldgen.exe", ROAD_INSPECTOR_CLI_MARKER]
    assert command[2:] == [
        "--build-dir",
        str(tmp_path / "build"),
        "--world-name",
        "wg_demo",
    ]


def test_gui_source_contains_postbuild_checkbox_and_pipeline_hook() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cwr_worldgen"
        / "gui_entry.py"
    ).read_text(encoding="utf-8")

    assert "Run Road Inspector after a successful build" in source
    assert '"run_road_inspector_after_build"' in source
    assert '"Running Road Inspector"' in source
    assert "road_inspector_postbuild_command(" in source
