# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import struct
import tempfile
import unittest

from cwr_worldgen.assets import AssetRecord, canonical_asset_path
from cwr_worldgen.legacy_proxy_models import (
    proxy_safe_model_path,
    write_proxy_safe_visual_clone,
)
from cwr_worldgen.procedural_buildings import inspect_mlod
from cwr_worldgen.procedural_forests import (
    ProceduralForestClusterLibrary,
    cluster_model_path,
)


def _synthetic_odol() -> bytes:
    points = (
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
    )
    flags = (
        0x0100 | 0x10000,  # LandOn + LightSky
        0x0800 | 0x20000,  # LandKeep + LightCloud
        0x0400 | 0x4000,   # LandAbove + FogDisable, preserved
    )
    uvs = ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))
    normals = ((0.0, 0.0, 1.0),) * 3

    out = bytearray()
    out += b"ODOL"
    out += struct.pack("<II", 7, 1)
    out += struct.pack("<I", len(flags))
    out += b"".join(struct.pack("<I", value) for value in flags)
    out += struct.pack("<I", len(uvs))
    out += b"".join(struct.pack("<ff", *value) for value in uvs)
    out += struct.pack("<I", len(points))
    out += b"".join(struct.pack("<fff", *value) for value in points)
    out += struct.pack("<I", len(normals))
    out += b"".join(struct.pack("<fff", *value) for value in normals)
    out += b"\0" * 48
    out += struct.pack("<I", 1)
    out += b"data\\leaf.paa\0"
    out += struct.pack("<I", 0)
    out += struct.pack("<I", 0)
    out += struct.pack("<II", 1, 0)
    out += struct.pack("<IhBHHH", 0x10, 0, 3, 0, 1, 2)
    return bytes(out)


def _first_lod_point_flags(data: bytes) -> tuple[int, ...]:
    if data[:4] != b"MLOD" or data[12:16] != b"SP3X":
        raise AssertionError("expected generated MLOD/SP3X")
    head_size = struct.unpack_from("<i", data, 16)[0]
    point_count = struct.unpack_from("<i", data, 24)[0]
    offset = 12 + head_size
    return tuple(
        struct.unpack_from("<I", data, offset + index * 16 + 12)[0]
        for index in range(point_count)
    )


class LegacyProxyModelTests(unittest.TestCase):
    def test_proxy_clone_clears_only_land_flags(self) -> None:
        source = _synthetic_odol()
        generated = r"testworld\f\p\safe.p3d"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "safe.p3d"
            info = write_proxy_safe_visual_clone(
                path,
                source,
                source_model=r"data3d\ker test.p3d",
                generated_model=generated,
            )

            self.assertEqual(info.source_format, "ODOL")
            self.assertEqual(info.land_flagged_points, 2)
            self.assertEqual(info.point_count, 3)
            self.assertEqual(info.face_count, 1)
            self.assertIn(r"data\leaf.paa", info.texture_paths)

            flags = _first_lod_point_flags(path.read_bytes())
            self.assertEqual(flags, (0x10000, 0x20000, 0x4400))
            self.assertTrue(all((value & 0x0900) == 0 for value in flags))

            summary = inspect_mlod(path)
            self.assertEqual(summary.lod_count, 1)
            self.assertIn(r"data\leaf.paa", summary.textures)
            self.assertIn(("class", "bushsoft"), summary.named_properties[0])

    def test_cwa_safe_clusters_reject_missing_proxy_source_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralForestClusterLibrary(
                "testworld",
                proxy_profile="everon-safe",
                cache_enabled=False,
                require_proxy_safe_clones=True,
            )
            library.register_model(
                cluster_model_path("testworld", "border_thicket", 0.15)
            )

            with self.assertRaisesRegex(
                ValueError,
                "CWA 1\\.99-safe generated vegetation requires readable source P3Ds",
            ):
                library.write_assets(
                    root / "world",
                    root / "catalogue.json",
                    asset_records=(),
                )

    def test_ce_clusters_report_incomplete_proxy_clone_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralForestClusterLibrary(
                "testworld",
                proxy_profile="everon-safe",
                cache_enabled=False,
            )
            library.register_model(
                cluster_model_path("testworld", "border_thicket", 0.15)
            )

            result = library.write_assets(
                root / "world",
                root / "catalogue.json",
                asset_records=(),
            )
            self.assertFalse(result.to_manifest()["proxy_safe_complete"])
            self.assertEqual(
                set(result.proxy_safe_missing_models),
                set(library.required_proxy_models()),
            )
            self.assertEqual(result.proxy_safe_cloned_models, ())

    def test_forest_carriers_proxy_generated_safe_clones(self) -> None:
        source_bytes = _synthetic_odol()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_file = root / "stock.p3d"
            source_file.write_bytes(source_bytes)

            library = ProceduralForestClusterLibrary(
                "testworld",
                proxy_profile="everon-safe",
                cache_enabled=False,
            )
            model = cluster_model_path("testworld", "border_thicket", 0.15)
            library.register_model(model)

            records = tuple(
                AssetRecord(
                    path=canonical_asset_path(source_model),
                    source=str(source_file),
                    size=len(source_bytes),
                    sha256=sha256(source_bytes).hexdigest(),
                    dependencies=(r"data\leaf.paa",),
                    readable=True,
                )
                for source_model in library.required_proxy_models()
            )

            result = library.write_assets(
                root / "world",
                root / "catalogue.json",
                asset_records=records,
            )
            self.assertGreaterEqual(len(result.model_files), 2)
            self.assertTrue(result.to_manifest()["proxy_safe_complete"])
            self.assertFalse(result.proxy_safe_missing_models)
            self.assertEqual(
                set(result.proxy_safe_cloned_models),
                set(library.required_proxy_models()),
            )

            cluster_file = root / "world" / "f" / "b_border_thicket_15.p3d"
            summary = inspect_mlod(cluster_file)
            proxies = tuple(
                name
                for lod in summary.selection_names
                for name in lod
                if name.casefold().startswith("proxy:")
            )
            self.assertTrue(proxies)
            self.assertTrue(
                all("testworld\\f\\p\\" in name.casefold() for name in proxies)
            )
            self.assertFalse(any("data3d" in name.casefold() for name in proxies))

            for source_model in library.required_proxy_models():
                generated_model = proxy_safe_model_path("testworld", source_model)
                relative = generated_model.split("\\", 1)[1].replace("\\", "/")
                clone = root / "world" / relative
                self.assertTrue(clone.is_file())
                self.assertTrue(
                    all(
                        (flag & 0x0900) == 0
                        for flag in _first_lod_point_flags(clone.read_bytes())
                    )
                )


if __name__ == "__main__":
    unittest.main()
