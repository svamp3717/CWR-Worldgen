# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path, PureWindowsPath

import cwr_worldgen
from cwr_worldgen._version import GENERATOR_VERSION, __version__
from cwr_worldgen.gui import APP_TITLE
from cwr_worldgen.generator import _write_json


class VersionTests(unittest.TestCase):
    def test_package_and_gui_use_authoritative_version(self) -> None:
        self.assertEqual(cwr_worldgen.__version__, __version__)
        self.assertEqual(GENERATOR_VERSION, f"cwr-worldgen {__version__}")
        self.assertEqual(APP_TITLE, f"CWR Worldgen {__version__}")

    def test_cli_reports_authoritative_version(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "cwr_worldgen", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), GENERATOR_VERSION)

    def test_runtime_sources_do_not_embed_stale_release_versions(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        forbidden = (
            '"generator": "cwr-worldgen 0.',
            'User-Agent": "cwr-worldgen/0.',
            'User-Agent": "cwr-worldgen-location-example/0.',
        )
        offenders: list[str] = []
        for path in sorted((project_root / "src" / "cwr_worldgen").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_manifest_json_serializes_nested_windows_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "manifest.json"
            _write_json(
                output,
                {
                    "world": {
                        "surface_colour_reference_path": PureWindowsPath(
                            r"G:\source-data\reference-map.png"
                        )
                    }
                },
            )
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                document["world"]["surface_colour_reference_path"],
                r"G:\source-data\reference-map.png",
            )

    def test_packaging_reads_same_version_constant(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('dynamic = ["version"]', pyproject)
        self.assertIn(
            'version = {attr = "cwr_worldgen._version.__version__"}',
            pyproject,
        )


if __name__ == "__main__":
    unittest.main()
