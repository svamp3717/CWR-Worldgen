# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from unittest.mock import patch

from cwr_worldgen import build_cli


def test_worldgen_build_uses_current_production_pipeline() -> None:
    with patch("cwr_worldgen.build_cli._cli_main", return_value=0) as cli_main:
        result = build_cli.main([
            "build",
            "source/example",
            "--output",
            "build/example",
            "--name",
            "wg_example",
        ])

    assert result == 0
    cli_main.assert_called_once_with([
        "milestone9",
        "--source-dir",
        "source/example",
        "--output",
        "build/example",
        "--name",
        "wg_example",
    ])


def test_worldgen_build_accepts_source_alias() -> None:
    assert build_cli._translate_args([
        "build", "--source", "source/example", "--output", "build/example"
    ]) == [
        "milestone9", "--source-dir", "source/example", "--output", "build/example"
    ]


def test_worldgen_keeps_existing_commands_available() -> None:
    assert build_cli._translate_args(["fetch-sources", "--help"]) == ["fetch-sources", "--help"]
