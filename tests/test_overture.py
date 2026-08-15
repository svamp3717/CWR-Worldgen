# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cwr_worldgen.overture import OVERTURE_CLI_MARKER, overture_command_prefix


class OvertureExecutableDiscoveryTests(unittest.TestCase):
    def test_finds_overturemaps_beside_running_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            python_dir = root / "python"
            app_dir.mkdir()
            python_dir.mkdir()
            app = app_dir / "cwr-worldgen.exe"
            app.write_bytes(b"")
            overture = app_dir / "overturemaps.exe"
            overture.write_bytes(b"")
            python = python_dir / "python.exe"
            python.write_bytes(b"")
            with (
                patch("cwr_worldgen.overture.sys.argv", [str(app)]),
                patch("cwr_worldgen.overture.sys.executable", str(python)),
                patch("cwr_worldgen.overture.shutil.which", return_value=None),
            ):
                self.assertEqual(overture_command_prefix(), [str(overture)])

    def test_frozen_app_reenters_itself_for_bundled_overture_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app = root / "cwr-worldgen-gui.exe"
            app.write_bytes(b"")
            with (
                patch("cwr_worldgen.overture.sys.argv", [str(app)]),
                patch("cwr_worldgen.overture.sys.executable", str(app)),
                patch("cwr_worldgen.overture.sys.frozen", True, create=True),
                patch("cwr_worldgen.overture.shutil.which", return_value=None),
            ):
                self.assertEqual(
                    overture_command_prefix(),
                    [str(app), OVERTURE_CLI_MARKER],
                )

    def test_running_entrypoint_folder_has_priority_over_python_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            python_dir = root / "python"
            app_dir.mkdir()
            python_dir.mkdir()
            app_overture = app_dir / "overturemaps.exe"
            python_overture = python_dir / "overturemaps.exe"
            app_overture.write_bytes(b"")
            python_overture.write_bytes(b"")
            with (
                patch("cwr_worldgen.overture.sys.argv", [str(app_dir / "cwr-worldgen.exe")]),
                patch("cwr_worldgen.overture.sys.executable", str(python_dir / "python.exe")),
            ):
                self.assertEqual(overture_command_prefix(), [str(app_overture)])


if __name__ == "__main__":
    unittest.main()
