from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.external_json_runtime import config_root, default_config_root, overlay_external_json
from tools.stage_external_json import JSON_DIRECTORIES, stage_external_json


class ExternalJsonRuntimeTests(unittest.TestCase):
    def test_default_config_root_is_beside_frozen_executable(self) -> None:
        root = default_config_root(executable=Path("/opt/cwr/CWR-Worldgen"), platform="linux")
        self.assertEqual(root, Path("/opt/cwr/config"))

    def test_macos_config_root_is_beside_app_bundle(self) -> None:
        root = default_config_root(
            executable=Path("/Applications/Test/CWR-Worldgen.app/Contents/MacOS/CWR-Worldgen"),
            platform="darwin",
        )
        self.assertEqual(root, Path("/Applications/Test/config"))

    def test_environment_override_wins(self) -> None:
        root = config_root(
            environ={"CWR_WORLDGEN_CONFIG_DIR": "/tmp/modded-cwr"},
            executable=Path("/opt/cwr/CWR-Worldgen"),
            platform="linux",
        )
        self.assertEqual(root, Path("/tmp/modded-cwr"))

    def test_external_json_replaces_bundled_directory_contents(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            config = root / "config"
            for relative in JSON_DIRECTORIES:
                bundled_dir = bundle / "cwr_worldgen" / relative
                external_dir = config / relative
                bundled_dir.mkdir(parents=True)
                external_dir.mkdir(parents=True)
                (bundled_dir / "old.json").write_text('{"source":"bundled"}', encoding="utf-8")
                (external_dir / "modded.json").write_text('{"source":"external"}', encoding="utf-8")

            copied = overlay_external_json(bundle_root=bundle, external_root=config)

            self.assertEqual(copied, len(JSON_DIRECTORIES))
            for relative in JSON_DIRECTORIES:
                target = bundle / "cwr_worldgen" / relative
                self.assertFalse((target / "old.json").exists())
                self.assertEqual(
                    json.loads((target / "modded.json").read_text(encoding="utf-8"))["source"],
                    "external",
                )

    def test_missing_external_directory_keeps_bundled_defaults(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            target = bundle / "cwr_worldgen" / "data"
            target.mkdir(parents=True)
            default = target / "default.json"
            default.write_text('{"ok":true}', encoding="utf-8")

            copied = overlay_external_json(bundle_root=bundle, external_root=root / "missing")

            self.assertEqual(copied, 0)
            self.assertTrue(default.exists())

    def test_staging_tool_copies_every_runtime_json_directory(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            destination = root / "config"
            expected = 0
            for index, relative in enumerate(JSON_DIRECTORIES):
                directory = source / relative
                directory.mkdir(parents=True)
                (directory / f"sample-{index}.json").write_text("{}", encoding="utf-8")
                (directory / "not-json.txt").write_text("ignored", encoding="utf-8")
                expected += 1

            copied = stage_external_json(destination, source_root=source)

            self.assertEqual(copied, expected)
            for index, relative in enumerate(JSON_DIRECTORIES):
                self.assertTrue((destination / relative / f"sample-{index}.json").is_file())
                self.assertFalse((destination / relative / "not-json.txt").exists())


if __name__ == "__main__":
    unittest.main()
