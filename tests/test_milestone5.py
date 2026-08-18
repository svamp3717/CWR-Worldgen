# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from cwr_worldgen.images import _image_values
from cwr_worldgen.source_pipeline import (
    Milestone5Spec,
    SourceFetchSpec,
    SourceRegridSpec,
    _heightmap_from_raster,
    build_milestone5,
    fetch_sources,
    regrid_sources,
    validate_source_bundle,
)


FIXTURES = Path(__file__).parent / "fixtures"


class Milestone5Tests(unittest.TestCase):

    def test_new_source_defaults_use_25m_cells_for_a_6_4km_world(self) -> None:
        spec = SourceFetchSpec(source_dir=Path("source"), center=(59.45, 17.0))
        self.assertEqual(spec.cells, 256)
        self.assertEqual(spec.cell_size, 25.0)
        self.assertEqual(spec.world_size, 6400.0)

    def test_image_values_supports_pre_12_1_pillow_api(self) -> None:
        class LegacyImage:
            def getdata(self):
                return (1.0, 2.0, 3.0)

        self.assertEqual(tuple(_image_values(LegacyImage())), (1.0, 2.0, 3.0))

    def test_image_values_prefers_modern_pillow_api(self) -> None:
        class ModernImage:
            def get_flattened_data(self):
                return (4.0, 5.0)

            def getdata(self):
                raise AssertionError("legacy fallback should not be used")

        self.assertEqual(tuple(_image_values(ModernImage())), (4.0, 5.0))


    def test_georeferenced_dem_resampling_uses_requested_bbox_and_north_up(self) -> None:
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            self.skipTest("optional DEM source dependencies are not installed")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.tif"
            output = root / "heightmap.tif"
            transform = from_bounds(0.0, 0.0, 1.0, 1.0, 10, 10)
            rows, columns = np.indices((10, 10))
            longitudes = transform.c + (columns + 0.5) * transform.a
            latitudes = transform.f + (rows + 0.5) * transform.e
            values = (longitudes * 100.0 + latitudes * 10.0).astype("float32")
            with rasterio.open(
                raw,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as dataset:
                dataset.write(values, 1)

            minimum, maximum, metadata = _heightmap_from_raster(
                raw,
                output,
                4,
                (0.25, 0.25, 0.75, 0.75),
            )
            with rasterio.open(output) as dataset:
                actual = dataset.read(1)
                self.assertEqual(dataset.width, 4)
                self.assertEqual(dataset.height, 4)
                self.assertEqual(dataset.tags().get("CWR_GRID"), "game-terrain-vertices")

            # The stored north-up image contains WRP vertices: first is
            # lon=0.25, lat=0.625 and final is lon=0.625, lat=0.25.
            self.assertAlmostEqual(float(actual[0, 0]), 31.25, places=3)
            self.assertAlmostEqual(float(actual[-1, -1]), 65.0, places=3)
            self.assertGreater(float(actual[0].mean()), float(actual[-1].mean()))
            self.assertAlmostEqual(minimum, 27.5, places=3)
            self.assertAlmostEqual(maximum, 68.75, places=3)
            self.assertEqual(metadata["method"], "georeferenced-bilinear-sampling")
            self.assertEqual(metadata["target_grid"], "game-terrain-vertices")
            self.assertEqual(metadata["raw_bounds_west_south_east_north"], [0.0, 0.0, 1.0, 1.0])

    def test_georeferenced_dem_resampling_interpolates_nodata_edges(self) -> None:
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            self.skipTest("optional DEM source dependencies are not installed")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw-nodata.tif"
            output = root / "heightmap.tif"
            values = np.arange(64, dtype="float32").reshape(8, 8)
            values[:, 0] = np.nan
            with rasterio.open(
                raw,
                "w",
                driver="GTiff",
                width=8,
                height=8,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_bounds(0.0, 0.0, 1.0, 1.0, 8, 8),
                nodata=np.nan,
            ) as dataset:
                dataset.write(values, 1)

            _, _, metadata = _heightmap_from_raster(raw, output, 8, (0.0, 0.0, 1.0, 1.0))
            with rasterio.open(output) as dataset:
                actual = dataset.read(1)
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreater(metadata["missing_target_samples_filled"], 0)
            # The filled edge should follow nearby terrain rather than becoming
            # one global-median shelf.
            self.assertGreater(len(set(round(float(value), 3) for value in actual[:, 0])), 1)

    def _cached_source(self, root: Path) -> SourceFetchSpec:
        elevation = root / "elevation" / "raw"
        elevation.mkdir(parents=True)
        # A tiny north-up HGT fixture. All terrain remains safely above sea level
        # before the OSM water pass lowers explicit water cells.
        values = (72, 73, 74, 71, 72, 73, 70, 71, 72)
        payload = b"".join(value.to_bytes(2, "big", signed=True) for value in values)
        with ZipFile(elevation / "N00E000.hgt.zip", "w") as archive:
            archive.writestr("N00E000.hgt", payload)

        osm = root / "osm"
        osm.mkdir(parents=True)
        (osm / "raw-overpass.json").write_bytes((FIXTURES / "osm-playability.json").read_bytes())
        overture = root / "overture"
        overture.mkdir(parents=True)
        (overture / "buildings.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}\n',
            encoding="utf-8",
        )
        return SourceFetchSpec(
            source_dir=root,
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=64,
            cell_size=50.0,
            dem_provider="hgt",
            reference_map=False,
        )

    def test_fetch_sources_reports_granular_osm_dem_and_validation_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            spec = self._cached_source(root)
            events: list[tuple[int, str]] = []
            with (
                patch("urllib.request.urlopen", side_effect=AssertionError("network access")),
                patch("cwr_worldgen.source_pipeline.report_progress", side_effect=lambda percent, stage: events.append((percent, stage))),
            ):
                fetch_sources(spec)

            stages = [stage for _percent, stage in events]
            self.assertTrue(any("Overpass JSON" in stage for stage in stages))
            self.assertTrue(any("HGT elevation tiles" in stage for stage in stages))
            self.assertTrue(any("Sampling HGT heightmap rows" in stage for stage in stages))
            self.assertTrue(any("Hashing frozen source file" in stage for stage in stages))
            self.assertTrue(any("Validating completed frozen source bundle" in stage for stage in stages))
            self.assertEqual(events[-1], (100, "OSM and heightmap source fetch complete"))
            self.assertGreater(len(events), 20)

    def test_fetch_sources_uses_persistent_overture_tile_cache_outside_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            spec = self._cached_source(root)
            overture_path = root / "overture" / "buildings.geojson"
            with (
                patch("urllib.request.urlopen", side_effect=AssertionError("network access")),
                patch(
                    "cwr_worldgen.source_pipeline.fetch_overture_buildings_geojson",
                    return_value=overture_path,
                ) as fetch_overture,
            ):
                fetch_sources(spec)
            _args, kwargs = fetch_overture.call_args
            self.assertEqual(
                kwargs["tile_cache_dir"],
                root.parent / ".cwr-worldgen-cache" / "overture",
            )

    def test_fetch_sources_can_skip_overture_download_completely(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            spec = replace(self._cached_source(root), overture_buildings_enabled=False)
            with (
                patch("urllib.request.urlopen", side_effect=AssertionError("network access")),
                patch("cwr_worldgen.source_pipeline.fetch_overture_buildings_geojson") as fetch_overture,
            ):
                bundle = fetch_sources(spec)
            fetch_overture.assert_not_called()
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(manifest["overture"])
            self.assertIsNone(bundle.overture_buildings_geojson_path)

    def test_fetches_from_cached_raw_files_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            spec = self._cached_source(root)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
                bundle = fetch_sources(spec)
            report = validate_source_bundle(root)
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(report.valid)
            self.assertEqual(bundle.bbox, (0.0, 0.0, 0.01, 0.01))
            self.assertEqual(bundle.cells, 64)
            self.assertEqual(manifest["osm"]["element_count"], 14)
            self.assertEqual(manifest["overture"]["provider"], "Overture Maps")
            self.assertEqual(manifest["overture"]["buildings_geojson"], "overture/buildings.geojson")
            self.assertEqual(manifest["elevation"]["product"], "SRTM3 HGT")
            self.assertTrue(bundle.heightmap_path.is_file())
            self.assertTrue(bundle.overture_buildings_geojson_path.is_file())
            self.assertTrue(bundle.checksum_path.is_file())

    def test_bbox_selection_can_write_reference_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            original = self._cached_source(root)
            spec = SourceFetchSpec(
                source_dir=original.source_dir,
                bbox=original.bbox,
                cells=original.cells,
                cell_size=original.cell_size,
                dem_provider=original.dem_provider,
                reference_map=True,
            )

            def fake_reference(path, bbox, zoom, cache_dir, *, refresh, progress_callback=None):
                self.assertEqual(bbox, spec.bbox)
                self.assertGreaterEqual(zoom, 2)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"reference-map")

            with patch("cwr_worldgen.source_pipeline._reference_map", side_effect=fake_reference):
                bundle = fetch_sources(spec)
            manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(bundle.reference_map_path.is_file())
            self.assertEqual(manifest["reference_map"]["provider"], "OpenTopoMap")

    def test_regrid_sources_reuses_frozen_osm_and_raw_hgt_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "source-50m"
            target_root = Path(temp) / "source-25m"
            source = fetch_sources(self._cached_source(source_root))
            source_manifest_before = source.manifest_path.read_bytes()
            source_osm_before = source.osm_json_path.read_bytes()
            source_raw = source_root / "elevation" / "raw" / "N00E000.hgt.zip"
            source_raw_before = source_raw.read_bytes()
            (source_root / "normalized").mkdir()
            (source_root / "normalized" / "stale.txt").write_text("do not copy", encoding="utf-8")

            with patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
                target = regrid_sources(SourceRegridSpec(
                    source_dir=source_root,
                    output_source_dir=target_root,
                    cell_size=25.0,
                ))

            self.assertEqual(target.cells, 128)
            self.assertEqual(target.cell_size, 25.0)
            self.assertEqual(target.cells * target.cell_size, source.cells * source.cell_size)
            self.assertEqual(target.osm_json_path.read_bytes(), source_osm_before)
            self.assertEqual(target.overture_buildings_geojson_path.read_bytes(), source.overture_buildings_geojson_path.read_bytes())
            self.assertEqual((target_root / "elevation" / "raw" / "N00E000.hgt.zip").read_bytes(), source_raw_before)
            self.assertFalse((target_root / "normalized").exists())
            self.assertEqual(source.manifest_path.read_bytes(), source_manifest_before)
            self.assertTrue(validate_source_bundle(target_root).valid)
            manifest = json.loads(target.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["regridded_from"]["cells"], 64)
            self.assertEqual(manifest["selection"]["cells"], 128)
            self.assertEqual(manifest["selection"]["cell_size_metres"], 25.0)
            self.assertEqual(manifest["overture"]["buildings_geojson"], "overture/buildings.geojson")
            self.assertTrue(manifest["elevation"]["resampling"]["reused_local_raw_dem"])
            with Image.open(target.heightmap_path) as image:
                self.assertEqual(image.size, (128, 128))

    def test_regrid_sources_rejects_grid_that_changes_world_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "source"
            source = fetch_sources(self._cached_source(source_root))
            with self.assertRaisesRegex(ValueError, "preserve the source world size"):
                regrid_sources(SourceRegridSpec(
                    source_dir=source.root,
                    output_source_dir=Path(temp) / "bad",
                    cells=64,
                    cell_size=25.0,
                ))

    def test_source_snapshot_remains_unchanged_without_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            spec = self._cached_source(root)
            first = fetch_sources(spec)
            before = first.manifest_path.read_bytes()
            before_checksums = first.checksum_path.read_bytes()
            with patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
                second = fetch_sources(spec)
            self.assertEqual(before, second.manifest_path.read_bytes())
            self.assertEqual(before_checksums, second.checksum_path.read_bytes())
            self.assertEqual(first.fingerprint, second.fingerprint)

    def test_corruption_is_rejected_before_world_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            bundle = fetch_sources(self._cached_source(root))
            bundle.osm_json_path.write_text('{"elements": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen file osm/raw-overpass.json"):
                validate_source_bundle(root)
            with self.assertRaisesRegex(ValueError, "source bundle validation failed"):
                build_milestone5(Path(temp) / "build", Milestone5Spec(source_dir=root))

    def test_offline_build_is_deterministic_and_carries_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            bundle = fetch_sources(self._cached_source(root))
            spec = Milestone5Spec(
                source_dir=root,
                name="cwr_m5_test",
                display_name="CWR M5 Test",
                asset_roots=(FIXTURES / "assets",),
                strict_assets=True,
            )
            with patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
                first = build_milestone5(Path(temp) / "one", spec)
                second = build_milestone5(Path(temp) / "two", spec)
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.pbo_path.read_bytes(), second.pbo_path.read_bytes())
            self.assertEqual(first.source_manifest_path.read_bytes(), bundle.manifest_path.read_bytes())
            runtime = first.pbo_path.parent.parent
            self.assertEqual(runtime.name, "@CWR-Milestone5")
            self.assertTrue((runtime / "SOURCE-PROVENANCE.json").is_file())
            self.assertTrue((runtime / "OSM-ATTRIBUTION.txt").is_file())
            self.assertTrue((runtime / "DEM-ATTRIBUTION.txt").is_file())
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["milestone"], 5)
            self.assertEqual(manifest["source_bundle"]["manifest_sha256"], bundle.fingerprint)


    def test_failed_refresh_preserves_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            bundle = fetch_sources(self._cached_source(root))
            before = bundle.manifest_path.read_bytes()
            refresh = SourceFetchSpec(
                source_dir=root,
                bbox=(0.0, 0.0, 0.01, 0.01),
                cells=64,
                cell_size=50.0,
                dem_provider="hgt",
                refresh=True,
            )
            with (
                patch("urllib.request.urlopen", side_effect=OSError("offline")),
                patch("cwr_worldgen.location_example.time.sleep", return_value=None),
            ):
                with self.assertRaisesRegex(RuntimeError, "Overpass endpoints failed"):
                    fetch_sources(refresh)
            self.assertEqual(before, bundle.manifest_path.read_bytes())
            self.assertTrue(validate_source_bundle(root).valid)

    def test_existing_source_directory_must_match_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sources"
            fetch_sources(self._cached_source(root))
            changed = SourceFetchSpec(
                source_dir=root,
                bbox=(0.0, 0.0, 0.02, 0.02),
                cells=64,
                cell_size=50.0,
                dem_provider="hgt",
            )
            with self.assertRaisesRegex(ValueError, "different selection"):
                fetch_sources(changed)


if __name__ == "__main__":
    unittest.main()
