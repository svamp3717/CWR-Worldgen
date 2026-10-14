# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import struct
import tempfile
import unittest
from unittest.mock import patch

from cwr_worldgen.assets import AssetRecord, canonical_asset_path
from cwr_worldgen import generator
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen.legacy_proxy_models import ProxyCloneError
from cwr_worldgen.legacy_proxy_models import (
    inspect_roadway_surface_style,
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



def _synthetic_road_odol(
    *,
    roadway_texture: str = r"o\road\contact.paa",
) -> bytes:
    def lod(texture: str, point_flag: int, face_flag: int) -> bytes:
        points = (
            (-2.0, 0.0, -3.0),
            (2.0, 0.0, -3.0),
            (2.0, 0.0, 3.0),
            (-2.0, 0.0, 3.0),
        )
        uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        normals = ((0.0, -1.0, 0.0),) * 4
        out = bytearray()
        out += struct.pack("<I", 4)
        out += b"".join(struct.pack("<I", point_flag) for _ in range(4))
        out += struct.pack("<I", 4)
        out += b"".join(struct.pack("<ff", *value) for value in uvs)
        out += struct.pack("<I", 4)
        out += b"".join(struct.pack("<fff", *value) for value in points)
        out += struct.pack("<I", 4)
        out += b"".join(struct.pack("<fff", *value) for value in normals)
        out += b"\0" * 48
        out += struct.pack("<I", 1)
        out += texture.encode("ascii") + b"\0"
        out += struct.pack("<I", 0)  # point-to-vertex
        out += struct.pack("<I", 0)  # vertex-to-point
        out += struct.pack("<II", 1, 10)
        out += struct.pack("<IhBHHHH", face_flag, 0, 4, 0, 1, 2, 3)
        out += struct.pack("<I", 0)  # sections
        out += struct.pack("<I", 0)  # named selections
        out += struct.pack("<I", 0)  # named properties
        out += struct.pack("<I", 0)  # animation phases
        out += b"\0" * 12            # colors + flags
        out += struct.pack("<I", 0)  # proxies
        return bytes(out)

    out = bytearray(b"ODOL")
    out += struct.pack("<II", 7, 2)
    out += lod(r"o\road\visual.paa", 0x13F, 0x24102)
    out += lod(roadway_texture, 0x55AA, 0x12345678)
    out += struct.pack("<ff", 1.0, 3.0e15)
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
            self.assertIn(r"data\leaf.paa", summary.texture_paths)
            self.assertIn(("class", "bushsoft"), summary.named_properties[0])

    def test_roadway_surface_style_reads_actual_odol_roadway_lod(self) -> None:
        style = inspect_roadway_surface_style(_synthetic_road_odol())

        self.assertEqual(style.source_format, "ODOL")
        self.assertEqual(style.texture_path, r"o\road\contact.paa")
        self.assertEqual(style.point_flag, 0x55AA)
        self.assertEqual(style.face_flag, 0x12345678)

    def test_roadway_surface_style_preserves_blank_donor_texture(self) -> None:
        style = inspect_roadway_surface_style(
            _synthetic_road_odol(roadway_texture="")
        )

        self.assertEqual(style.source_format, "ODOL")
        self.assertEqual(style.texture_path, "")
        self.assertEqual(style.point_flag, 0x55AA)
        self.assertEqual(style.face_flag, 0x12345678)

    def test_blank_donor_roadway_surface_survives_generated_p3d(self) -> None:
        """Untextured native contact faces must not become asphalt contact."""
        model = r"o\road\sil25.p3d"
        visual = r"o\road\sil_new.paa"
        donor = _synthetic_road_odol(roadway_texture="")
        style, report = generator._road_surface_style_from_donor(
            model, visual, donor
        )
        self.assertEqual(style.roadway_texture, "")
        self.assertEqual(style.roadway_point_flag, 0x55AA)
        self.assertEqual(style.roadway_face_flag, 0x12345678)
        self.assertEqual(report["roadway_inspected_texture"], "")
        self.assertEqual(report["roadway_texture_source"], "donor-roadway-lod")

        key = infrastructure.InfrastructureModelKey(
            "road", "road_paved_w091_l0063", 91, 63,
        )
        visual_lod, _geometry, contact_lod, _land = (
            infrastructure._road_lods(key, visual, style)
        )
        self.assertTrue(visual_lod.faces)
        self.assertTrue(all(face.texture == visual for face in visual_lod.faces))
        self.assertTrue(contact_lod.faces)
        self.assertTrue(all(face.texture == "" for face in contact_lod.faces))
        with tempfile.TemporaryDirectory() as temp:
            generated = Path(temp) / "paved.p3d"
            infrastructure.write_infrastructure_mlod(
                generated, key, visual, road_surface_style=style
            )
            result = inspect_roadway_surface_style(generated.read_bytes())
        self.assertEqual(result.texture_path, "")
        self.assertEqual(result.point_flag, 0x55AA)
        self.assertEqual(result.face_flag, 0x12345678)

    def test_unreadable_visual_lod_does_not_erase_donor_roadway_metadata(self) -> None:
        """Visual parsing errors must not suppress separately readable contact."""
        donor = _synthetic_road_odol()
        with patch.object(
            generator, "inspect_visual_surface_style",
            side_effect=ProxyCloneError("visual parse failed"),
        ):
            style, report = generator._road_surface_style_from_donor(
                r"o\road\sil25.p3d", r"o\road\sil_new.paa", donor,
            )
        self.assertEqual(style.roadway_texture, r"o\road\contact.paa")
        self.assertEqual(style.roadway_point_flag, 0x55AA)
        self.assertEqual(style.roadway_face_flag, 0x12345678)
        self.assertEqual(report["source_format"], None)
        self.assertEqual(report["roadway_source_format"], "ODOL")
        self.assertEqual(report["roadway_texture_source"], "donor-roadway-lod")

    def test_missing_donor_contact_uses_explicit_visual_fallback(self) -> None:
        visual = r"modroads\paved.paa"
        style, report = generator._road_surface_style_from_donor(
            r"modroads\paved.p3d", visual, None,
        )
        self.assertEqual(style.roadway_texture, visual)
        self.assertEqual(style.roadway_point_flag, 0)
        self.assertEqual(style.roadway_face_flag, 0)
        self.assertEqual(report["roadway_source_format"], "visual-fallback")
        self.assertEqual(report["roadway_texture_source"], "visual-fallback")

    def test_generated_road_families_serialize_real_donor_contact_styles(self) -> None:
        """Paved, dirt, gravel and junction contact cannot revert to visuals."""
        fixtures = (
            ("road_paved_w091_l0063", r"o\\road\\sil_new.paa",
             r"stock\\asphalt_contact.paa"),
            ("road_dirt_w035_l0063", r"o\\road\\ces_hned.paa",
             r"stock\\earth_contact.paa"),
            ("road_gravel_w046_l0063", r"addon\\gravel_visual.paa",
             r"addon\\gravel_contact.paa"),
            ("road_j3_paved_w091_h000_090_180", r"o\\road\\sil_new.paa",
             r"stock\\asphalt_contact.paa"),
        )
        with tempfile.TemporaryDirectory() as temp:
            for subtype, visual_texture, contact_texture in fixtures:
                with self.subTest(subtype=subtype):
                    donor = _synthetic_road_odol(
                        roadway_texture=contact_texture
                    )
                    style, report = generator._road_surface_style_from_donor(
                        r"addon\\donor.p3d", visual_texture, donor
                    )
                    self.assertEqual(report["roadway_texture_source"], "donor-roadway-lod")
                    key = infrastructure.InfrastructureModelKey(
                        "road", subtype, 91, 63,
                    )
                    dest = Path(temp) / f"{subtype}.p3d"
                    task = infrastructure._InfrastructureAssetTask(
                        key=key,
                        wire=rf"world\\i\\{subtype}.p3d",
                        relative=f"i/{subtype}.p3d",
                        destination=dest,
                        texture=visual_texture,
                        surface_style=style,
                        cache_path=None,
                        cache_enabled=False,
                        cache_refresh=False,
                        usage_count=1,
                    )
                    entry, hit = infrastructure._write_infrastructure_asset_task(task)
                    self.assertFalse(hit)
                    self.assertTrue(entry["roadway_contact_verified"])
                    self.assertEqual(entry["roadway_texture"], contact_texture)
                    self.assertEqual(entry["roadway_point_flag"], "0x000055aa")
                    self.assertEqual(entry["roadway_face_flag"], "0x12345678")
                    parsed = inspect_roadway_surface_style(dest.read_bytes())
                    self.assertEqual(parsed.texture_path, contact_texture)
                    self.assertEqual(parsed.point_flag, 0x55AA)
                    self.assertEqual(parsed.face_flag, 0x12345678)

    def test_generated_road_contact_audit_rejects_wrong_emitted_metadata(self) -> None:
        """Do not silently package road models with the wrong driving surface."""
        source = _synthetic_road_odol(roadway_texture=r"stock\\asphalt_contact.paa")
        style, _ = generator._road_surface_style_from_donor(
            r"stock\\sil25.p3d", r"stock\\visual.paa", source
        )
        original_writer = infrastructure.write_infrastructure_mlod

        def write_without_donor_style(path, key, texture, *, road_surface_style):
            return original_writer(
                path, key, texture,
                road_surface_style=infrastructure.RoadSurfaceStyle(),
            )

        with tempfile.TemporaryDirectory() as temp:
            task = infrastructure._InfrastructureAssetTask(
                key=infrastructure.InfrastructureModelKey(
                    "road", "road_paved_w091_l0063", 91, 63
                ),
                wire=r"world\\i\\road_paved_w091_l0063.p3d",
                relative="i/road_paved_w091_l0063.p3d",
                destination=Path(temp) / "bad.p3d",
                texture=r"stock\\visual.paa",
                surface_style=style,
                cache_path=None,
                cache_enabled=False,
                cache_refresh=False,
                usage_count=1,
            )
            with patch.object(
                infrastructure, "write_infrastructure_mlod",
                side_effect=write_without_donor_style,
            ):
                with self.assertRaisesRegex(ValueError, "did not preserve donor Roadway"):
                    infrastructure._write_infrastructure_asset_task(task)

    def test_generated_road_audit_rejects_one_wrong_contact_face(self) -> None:
        """A minority offroad patch must not evade a dominant-face audit."""
        contact_texture = r"stock\\asphalt_contact.paa"
        donor = _synthetic_road_odol(roadway_texture=contact_texture)
        style, _report = generator._road_surface_style_from_donor(
            r"stock\\sil25.p3d", r"stock\\visual.paa", donor
        )
        original_writer = infrastructure.write_infrastructure_mlod

        def write_bad_last_face(path, key, texture, *, road_surface_style):
            original_writer(
                path, key, texture, road_surface_style=road_surface_style,
            )
            blob = path.read_bytes()
            last = blob.rfind(contact_texture.encode("ascii"))
            self.assertGreaterEqual(last, 0)
            path.write_bytes(blob[:last] + b"x" + blob[last + 1:])

        with tempfile.TemporaryDirectory() as temp:
            task = infrastructure._InfrastructureAssetTask(
                key=infrastructure.InfrastructureModelKey(
                    "road", "road_paved_w091_l0063", 91, 63
                ),
                wire=r"world\\i\\road_paved_w091_l0063.p3d",
                relative="i/road_paved_w091_l0063.p3d",
                destination=Path(temp) / "partial-bad.p3d",
                texture=r"stock\\visual.paa",
                surface_style=style,
                cache_path=None,
                cache_enabled=False,
                cache_refresh=False,
                usage_count=1,
            )
            with patch.object(
                infrastructure, "write_infrastructure_mlod",
                side_effect=write_bad_last_face,
            ):
                with self.assertRaisesRegex(
                    ValueError, "did not preserve donor Roadway contact on faces"
                ):
                    infrastructure._write_infrastructure_asset_task(task)

    def test_cwa_safe_clusters_reject_missing_proxy_source_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralForestClusterLibrary(
                "testworld",
                proxy_profile="everon",
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

    def test_ce_clusters_keep_stock_proxy_children_without_cloning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralForestClusterLibrary(
                "testworld",
                proxy_profile="everon",
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
            self.assertTrue(result.to_manifest()["proxy_safe_complete"])
            self.assertEqual(result.proxy_safe_missing_models, ())
            self.assertEqual(result.proxy_safe_cloned_models, ())

            cluster_file = root / "world" / "f" / "b_border_thicket_15.p3d"
            summary = inspect_mlod(cluster_file)
            proxies = tuple(
                name
                for lod in summary.selection_names
                for name in lod
                if name.casefold().startswith("proxy:")
            )
            self.assertTrue(proxies)
            self.assertTrue(any("data3d" in name.casefold() for name in proxies))
            self.assertFalse(any("testworld\\f\\p\\" in name.casefold() for name in proxies))

    def test_forest_carriers_proxy_generated_safe_clones(self) -> None:
        source_bytes = _synthetic_odol()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_file = root / "stock.p3d"
            source_file.write_bytes(source_bytes)

            library = ProceduralForestClusterLibrary(
                "testworld",
                proxy_profile="everon",
                cache_enabled=False,
                require_proxy_safe_clones=True,
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
