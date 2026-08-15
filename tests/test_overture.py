# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cwr_worldgen.overture import (
    DEFAULT_OVERTURE_RELEASE,
    OVERTURE_CLI_MARKER,
    _azure_href_from_stac_asset,
    fetch_overture_buildings_geojson,
    overture_command_prefix,
    run_overture_worker,
    selected_overture_release,
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
                    [str(python.resolve()), "-m", "cwr_worldgen.gui_entry", OVERTURE_CLI_MARKER],
                )

    def test_release_can_be_overridden_without_using_root_stac_catalog(self) -> None:
        with patch.dict(os.environ, {"CWR_OVERTURE_RELEASE": "2026-07-22.1"}):
            self.assertEqual(selected_overture_release(), "2026-07-22.1")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(selected_overture_release(), DEFAULT_OVERTURE_RELEASE)

    def test_invalid_release_override_is_rejected(self) -> None:
        with patch.dict(os.environ, {"CWR_OVERTURE_RELEASE": "latest"}):
            with self.assertRaises(ValueError):
                selected_overture_release()

    def test_stac_aws_asset_is_mapped_to_azure_mirror(self) -> None:
        asset = {
            "aws": {
                "alternate": {
                    "s3": {
                        "href": (
                            "s3://overturemaps-us-west-2/release/2026-06-17.0/"
                            "theme=buildings/type=building/part-00000.parquet"
                        )
                    }
                }
            }
        }
        self.assertEqual(
            _azure_href_from_stac_asset(asset),
            (
                "az://overturemapswestus2.blob.core.windows.net/release/2026-06-17.0/"
                "theme=buildings/type=building/part-00000.parquet"
            ),
        )

    def test_worker_converts_cli_bbox_to_internal_bbox(self) -> None:
        with patch(
            "cwr_worldgen.overture.download_overture_buildings_direct",
            return_value="2026-06-17.0",
        ) as download:
            result = run_overture_worker(
                [
                    "--bbox",
                    "18.0,59.0,18.1,59.1",
                    "--output",
                    "buildings.geojson",
                ]
            )
        self.assertEqual(result, 0)
        download.assert_called_once_with(
            (59.0, 18.0, 59.1, 18.1),
            Path("buildings.geojson"),
        )

    def test_fetch_uses_internal_worker_and_moves_completed_output(self) -> None:
        class CompletedProcess:
            def __init__(self, command):
                temporary = Path(command[command.index("--output") + 1])
                temporary.write_text(
                    '{"type":"FeatureCollection","features":[]}',
                    encoding="utf-8",
                )

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "buildings.geojson"
            with (
                patch(
                    "cwr_worldgen.overture.overture_command_prefix",
                    return_value=["cwr-worldgen.exe", OVERTURE_CLI_MARKER],
                ),
                patch(
                    "cwr_worldgen.overture.subprocess.Popen",
                    side_effect=lambda command, **kwargs: CompletedProcess(command),
                ) as popen,
            ):
                result = fetch_overture_buildings_geojson(
                    (59.0, 18.0, 59.1, 18.1),
                    output,
                    refresh=True,
                )

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            command = popen.call_args.args[0]
            self.assertEqual(command[0:2], ["cwr-worldgen.exe", OVERTURE_CLI_MARKER])
            self.assertIn("--bbox", command)
            self.assertIn("--output", command)


if __name__ == "__main__":
    unittest.main()
