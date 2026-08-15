# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cwr_worldgen.overture import (
    OVERTURE_CLI_MARKER,
    _release_sort_key,
    fetch_overture_buildings_geojson,
    overture_command_prefix,
    run_overture_worker,
)


class OvertureWorkerTests(unittest.TestCase):
    def test_frozen_app_reenters_itself_for_internal_overture_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp) / "cwr-worldgen.exe"
            app.write_bytes(b"")
            with (
                patch("cwr_worldgen.overture.sys.executable", str(app)),
                patch("cwr_worldgen.overture.sys.frozen", True, create=True),
            ):
                self.assertEqual(
                    overture_command_prefix(),
                    [str(app.resolve()), OVERTURE_CLI_MARKER],
                )

    def test_source_app_reenters_cwr_worker_instead_of_overture_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python.exe"
            python.write_bytes(b"")
            with (
                patch("cwr_worldgen.overture.sys.executable", str(python)),
                patch("cwr_worldgen.overture.sys.frozen", False, create=True),
            ):
                self.assertEqual(
                    overture_command_prefix(),
                    [
                        str(python.resolve()),
                        "-m",
                        "cwr_worldgen.gui_entry",
                        OVERTURE_CLI_MARKER,
                    ],
                )

    def test_release_sort_key_handles_patch_numbers_numerically(self) -> None:
        releases = ["2026-07-22.0", "2026-07-22.10", "2026-06-17.9"]
        self.assertEqual(max(releases, key=_release_sort_key), "2026-07-22.10")

    def test_worker_converts_cli_bbox_to_internal_bbox(self) -> None:
        with patch("cwr_worldgen.overture.download_overture_buildings_direct", return_value="2026-07-22.0") as download:
            result = run_overture_worker(
                [
                    "--bbox",
                    "18.0,59.0,18.1,59.1",
                    "--output",
                    "buildings.geojson",
                    "--connect-timeout",
                    "10",
                    "--request-timeout",
                    "30",
                ]
            )
        self.assertEqual(result, 0)
        download.assert_called_once_with(
            (59.0, 18.0, 59.1, 18.1),
            Path("buildings.geojson"),
            connect_timeout=10,
            request_timeout=30,
        )

    def test_fetch_uses_internal_worker_and_preserves_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "buildings.geojson"

            def fake_run(command, **kwargs):
                temporary = Path(command[command.index("--output") + 1])
                temporary.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
                return None

            with (
                patch(
                    "cwr_worldgen.overture.overture_command_prefix",
                    return_value=["cwr-worldgen.exe", OVERTURE_CLI_MARKER],
                ),
                patch("cwr_worldgen.overture.subprocess.run", side_effect=fake_run) as run,
            ):
                result = fetch_overture_buildings_geojson(
                    (59.0, 18.0, 59.1, 18.1),
                    output,
                    refresh=True,
                )

            self.assertEqual(result, output)
            command = run.call_args.args[0]
            self.assertNotIn("download", command)
            self.assertNotIn("--no-stac", command)
            self.assertEqual(command[command.index("--connect-timeout") + 1], "10")
            self.assertEqual(command[command.index("--request-timeout") + 1], "30")
            self.assertEqual(run.call_args.kwargs["timeout"], 300)


if __name__ == "__main__":
    unittest.main()
