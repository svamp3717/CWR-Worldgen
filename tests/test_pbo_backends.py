from __future__ import annotations

import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from cwr_worldgen.pbo import pack_directory_cached, read_pbo
from cwr_worldgen.milestone9 import Milestone9Spec, build_milestone9
import json
import test_milestone8 as milestone8_tests


class PboBackendTests(unittest.TestCase):
    def _tool(self, root: Path, *, fail: bool = False) -> Path:
        tool = root / "PoseidonTools"
        if fail:
            body = "#!/usr/bin/env python3\nimport sys\nprint('simulated native failure')\nsys.exit(7)\n"
        else:
            body = textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import pathlib
                import struct
                import sys

                if len(sys.argv) != 5 or sys.argv[1:3] != ["pbo", "pack"]:
                    sys.exit(2)
                source = pathlib.Path(sys.argv[3])
                output = pathlib.Path(sys.argv[4])
                files = sorted((p for p in source.rglob("*") if p.is_file()), key=lambda p: p.as_posix().lower())
                entries = []
                for path in files:
                    name = path.relative_to(source).as_posix().replace("/", "\\\\").encode("ascii")
                    data = path.read_bytes()
                    entries.append((name, data, int(path.stat().st_mtime)))
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("wb") as stream:
                    for name, data, timestamp in entries:
                        stream.write(name + b"\\0")
                        stream.write(struct.pack("<IIIII", 0, 0, 0, timestamp, len(data)))
                    stream.write(b"\\0")
                    stream.write(struct.pack("<IIIII", 0, 0, 0, 0, 0))
                    for _name, data, _timestamp in entries:
                        stream.write(data)
                """
            )
        tool.write_text(body, encoding="utf-8")
        tool.chmod(0o755)
        return tool

    def test_poseidon_backend_packs_and_validates_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "config.cpp").write_text("class CfgPatches {};", encoding="ascii")
            (source / "data").mkdir()
            (source / "data" / "g.paa").write_bytes(b"texture")
            tool = self._tool(root)

            result = pack_directory_cached(
                source,
                root / "world.pbo",
                cache_dir=root / "cache",
                backend="poseidon",
                poseidon_tools_path=tool,
            )

            self.assertEqual(result.requested_backend, "poseidon")
            self.assertEqual(result.backend, "poseidon")
            self.assertEqual(Path(result.poseidon_tools_path or ""), tool.resolve())
            self.assertIsNone(result.fallback_reason)
            self.assertEqual(
                {entry.name for entry in read_pbo(root / "world.pbo")},
                {"config.cpp", "data\\g.paa"},
            )
            second = pack_directory_cached(
                source,
                root / "world-again.pbo",
                cache_dir=root / "cache",
                backend="poseidon",
                poseidon_tools_path=tool,
            )
            self.assertTrue(second.archive_hit)
            self.assertEqual((root / "world.pbo").read_bytes(), (root / "world-again.pbo").read_bytes())

    def test_auto_backend_falls_back_to_python_after_native_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "a.txt").write_bytes(b"alpha")
            tool = self._tool(root, fail=True)

            result = pack_directory_cached(
                source,
                root / "world.pbo",
                cache_dir=root / "cache",
                backend="auto",
                poseidon_tools_path=tool,
            )

            self.assertEqual(result.requested_backend, "auto")
            self.assertEqual(result.backend, "python")
            self.assertIn("simulated native failure", result.fallback_reason or "")
            self.assertEqual(read_pbo(root / "world.pbo")[0].data, b"alpha")

    def test_milestone9_uses_poseidon_for_primary_and_reproducibility_pbos(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = milestone8_tests.Milestone8BuildTests()._source(root / "source")
            tool = self._tool(root)
            result = build_milestone9(
                root / "build",
                Milestone9Spec(
                    source_dir=source,
                    name="cwr_poseidon",
                    display_name="CWR Poseidon",
                    solver_iterations=2,
                    world_edge_blend_cells=1,
                    max_forest_objects=0,
                    ground_texture_profile="generated",
                    surface_overview_size=128,
                    surface_texture_size=32,
                    pbo_backend="poseidon",
                    poseidon_tools_path=tool,
                    verify_regeneration=True,
                ),
            )
            cache = json.loads(result.cache_report_path.read_text(encoding="utf-8"))
            reproducibility = json.loads(result.reproducibility_path.read_text(encoding="utf-8"))
            self.assertEqual(cache["incremental_pbo"]["backend"], "poseidon")
            self.assertEqual(Path(cache["incremental_pbo"]["poseidon_tools_path"]), tool.resolve())
            self.assertTrue(reproducibility["pbo_byte_match"])

    def test_required_poseidon_backend_reports_missing_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PATH": "", "CWR_POSEIDON_TOOLS": ""},
            clear=False,
        ):
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "a.txt").write_bytes(b"alpha")
            with self.assertRaisesRegex(ValueError, "PoseidonTools executable was not found"):
                pack_directory_cached(
                    source,
                    root / "world.pbo",
                    cache_dir=root / "cache",
                    backend="poseidon",
                )


if __name__ == "__main__":
    unittest.main()
