# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from cwr_worldgen.assets import _CATALOGUE_MEMORY, scan_assets
from cwr_worldgen.normalization import (
    NormalizationSpec,
    load_normalized_dataset,
    normalize_source_bundle,
)
from cwr_worldgen.osm import (
    BboxProjection,
    OsmDataset,
    _SPATIAL_INDEX_REGISTRY,
    prepare_spatial_index,
)
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary
from cwr_worldgen.generator import _load_processed_dem
from cwr_worldgen.model import PlayabilitySpec
from cwr_worldgen.pbo import pack_directory_cached
import test_milestone8 as milestone8_tests


class PersistentCacheTests(unittest.TestCase):
    def test_asset_catalogue_cache_invalidates_on_asset_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            model = assets / "Data3D" / "cached.p3d"
            texture = assets / "Data" / "cached.paa"
            model.parent.mkdir(parents=True)
            texture.parent.mkdir(parents=True)
            model.write_bytes(b"MLOD data\\cached.paa\0")
            texture.write_bytes(b"texture-a")
            cache_dir = root / "cache"

            first = scan_assets(
                (assets,), (r"data3d\cached.p3d",), cache_dir=cache_dir
            )
            self.assertFalse(first.cache_hit)
            self.assertTrue(first.verified)
            self.assertTrue(Path(first.cache_path).is_file())

            _CATALOGUE_MEMORY.clear()
            second = scan_assets(
                (assets,), (r"data3d\cached.p3d",), cache_dir=cache_dir
            )
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.catalogue_sha256, second.catalogue_sha256)

            texture.write_bytes(b"texture-b-with-new-size")
            _CATALOGUE_MEMORY.clear()
            changed = scan_assets(
                (assets,), (r"data3d\cached.p3d",), cache_dir=cache_dir
            )
            self.assertFalse(changed.cache_hit)
            self.assertNotEqual(first.catalogue_sha256, changed.catalogue_sha256)

    def test_procedural_assets_use_content_addressed_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache_dir = root / "cache"
            dataset = OsmDataset(
                source_generator="cache-test",
                element_count=0,
                coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            )
            projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

            def build(destination: Path):
                library = ProceduralBuildingLibrary(
                    world_name="cache_world", cache_dir=cache_dir
                )
                library.prepare(dataset, projection, 12.0)
                library.place_point({"building": "house"}, 12.0, 0.0)
                return library.write_assets(destination, destination / "catalogue.json")

            first = build(root / "first")
            second = build(root / "second")
            self.assertGreater(first.cache_misses, 0)
            self.assertEqual(first.cache_hits, 0)
            self.assertEqual(second.cache_misses, 0)
            self.assertEqual(second.cache_hits, first.cache_misses)
            self.assertEqual(
                (root / "first" / first.model_assets[0].relative_path).read_bytes(),
                (root / "second" / second.model_assets[0].relative_path).read_bytes(),
            )

    def test_normalized_dataset_and_spatial_index_persist_between_process_like_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = milestone8_tests.Milestone8BuildTests()._source(root / "source")
            normalized = normalize_source_bundle(NormalizationSpec(source_dir=source))
            cache_dir = root / "cache"

            first_dataset = load_normalized_dataset(normalized, cache_dir=cache_dir)
            second_dataset = load_normalized_dataset(normalized, cache_dir=cache_dir)
            self.assertFalse(first_dataset.parsed_cache_hit)
            self.assertTrue(second_dataset.parsed_cache_hit)
            self.assertEqual(
                replace(second_dataset, parsed_cache_hit=False), first_dataset
            )

            projection = BboxProjection.create(normalized.bbox, normalized.world_size)
            _SPATIAL_INDEX_REGISTRY.clear()
            first_index = prepare_spatial_index(
                first_dataset, projection, cache_dir=cache_dir
            )
            self.assertFalse(first_index.cache_hit)
            _SPATIAL_INDEX_REGISTRY.clear()
            second_index = prepare_spatial_index(
                second_dataset, projection, cache_dir=cache_dir
            )
            self.assertTrue(second_index.cache_hit)
            self.assertEqual(first_index.road_segments, second_index.road_segments)
            self.assertEqual(first_index.road_buckets, second_index.road_buckets)

    def test_processed_dem_cache_invalidates_when_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heightmap = root / "height.png"
            Image.new("L", (16, 16), 64).save(heightmap)
            spec = PlayabilitySpec(
                heightmap_path=heightmap,
                name="cache_dem",
                display_name="Cache DEM",
                cells=16,
                cell_size=25.0,
                input_mode="meters",
                bbox=(0.0, 0.0, 0.01, 0.01),
                cache_dir=root / "cache",
            )
            first, first_hit, first_key, _ = _load_processed_dem(spec)
            second, second_hit, second_key, _ = _load_processed_dem(spec)
            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(first_key, second_key)
            self.assertEqual(first, second)

            Image.new("L", (16, 16), 96).save(heightmap)
            changed, changed_hit, changed_key, _ = _load_processed_dem(spec)
            self.assertFalse(changed_hit)
            self.assertNotEqual(first_key, changed_key)
            self.assertNotEqual(first.elevations, changed.elevations)

    def test_incremental_pbo_cache_reuses_archive_and_unchanged_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "a.txt").write_bytes(b"alpha")
            (source / "b.txt").write_bytes(b"beta")
            cache_dir = root / "cache"
            first = pack_directory_cached(source, root / "one.pbo", cache_dir=cache_dir)
            second = pack_directory_cached(source, root / "two.pbo", cache_dir=cache_dir)
            self.assertFalse(first.archive_hit)
            self.assertTrue(second.archive_hit)
            self.assertEqual((root / "one.pbo").read_bytes(), (root / "two.pbo").read_bytes())

            (source / "b.txt").write_bytes(b"beta changed")
            third = pack_directory_cached(source, root / "three.pbo", cache_dir=cache_dir)
            self.assertFalse(third.archive_hit)
            self.assertGreaterEqual(third.reused_blob_entries, 1)
            self.assertGreaterEqual(third.new_blob_entries, 1)
            self.assertNotEqual(first.archive_key, third.archive_key)



if __name__ == "__main__":
    unittest.main()
