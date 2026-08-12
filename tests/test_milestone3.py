# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cwr_worldgen.generator import build_milestone3
from cwr_worldgen.model import OsmSpec
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPolygonFeature,
    OsmRaster,
    build_overpass_query,
    generate_world_objects,
    parse_overpass_json,
    rasterize_osm,
)
from cwr_worldgen.pbo import read_pbo
from cwr_worldgen.wrp import inspect_rvw4


FIXTURES = Path(__file__).parent / "fixtures"


def _spec(**overrides) -> OsmSpec:
    values = {
        "heightmap_path": FIXTURES / "osm-height.png",
        "osm_json_path": FIXTURES / "osm-sample.json",
        "bbox": (0.0, 0.0, 0.01, 0.01),
        "cells": 64,
        "cell_size": 20.0,
        "elevation_minimum": 10.0,
        "elevation_maximum": 40.0,
        "rock_height": 1000.0,
        "rock_slope_degrees": 89.0,
        "max_road_objects": 1000,
        "max_buildings": 100,
        "max_forest_objects": 200,
        "forest_tree_spacing": 50.0,
    }
    values.update(overrides)
    return OsmSpec(**values)


class Milestone3Tests(unittest.TestCase):
    def test_builds_osm_geography_world(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = build_milestone3(Path(temp) / "build", _spec())
            self.assertTrue(result.pbo_path.is_file())
            self.assertTrue(result.osm_source_path and result.osm_source_path.is_file())
            self.assertTrue(result.osm_query_path and result.osm_query_path.is_file())
            self.assertTrue(result.osm_preview_path and result.osm_preview_path.is_file())
            self.assertTrue(result.attribution_path and result.attribution_path.is_file())

            summary = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertEqual((summary.width, summary.height), (64, 64))
            self.assertGreater(summary.object_count, 0)
            self.assertEqual(summary.object_ids, tuple(range(1, summary.object_count + 1)))
            self.assertIn(r"o\road\sil25.p3d", summary.object_models)
            self.assertIn(r"o\road\ces25.p3d", summary.object_models)
            self.assertIn(r"O\Hous\domek_sedy.p3d", summary.object_models)
            self.assertIn(r"data3d\dum_mesto2.p3d", summary.object_models)
            self.assertIn(r"O\Hous\hangar_2.p3d", summary.object_models)
            self.assertIn(r"data3d\les_su_ctver_pruhozi.p3d", summary.object_models)
            self.assertTrue(summary.has_object_terminator)
            self.assertEqual(summary.texture_slots[0], r"cwr_milestone3\data\d.paa")
            self.assertEqual(summary.texture_index_counts[0], 0)

            pbo = {entry.name.casefold(): entry.data for entry in read_pbo(result.pbo_path)}
            self.assertEqual(
                set(pbo),
                {
                    "config.cpp",
                    "cwr_milestone3.wrp",
                    r"data\d.paa",
                    r"data\w.paa",
                    r"data\s.paa",
                    r"data\g.paa",
                    r"data\r.paa",
                    r"data\f.paa",
                    r"data\a.paa",
                    r"data\u.paa",
                    r"data\p.paa",
                },
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["milestone"], 3)
            self.assertEqual(manifest["osm"]["element_count"], 12)
            self.assertGreater(manifest["osm"]["raster_cell_counts"]["water"], 0)
            self.assertGreater(manifest["osm"]["raster_cell_counts"]["forest"], 0)
            self.assertGreater(manifest["osm"]["raster_cell_counts"]["farmland"], 0)
            self.assertGreater(manifest["osm"]["raster_cell_counts"]["urban"], 0)
            self.assertGreater(manifest["objects"]["roads"], 0)
            self.assertEqual(manifest["objects"]["buildings"], 4)
            self.assertGreater(manifest["objects"]["forest"], 0)
            self.assertIn("OpenStreetMap contributors", result.attribution_path.read_text(encoding="utf-8"))
            self.assertIn("Failures: 0", result.report_path.read_text(encoding="utf-8"))

    def test_multipolygon_hole_is_not_forest(self) -> None:
        data = (FIXTURES / "osm-sample.json").read_bytes()
        dataset = parse_overpass_json(data)
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1280.0)
        raster = rasterize_osm(dataset, projection, cells=64, include_minor_roads=False)

        def cell_for(lat: float, lon: float) -> int:
            x, z = projection.to_world((lat, lon))
            cx = min(63, int(x / 1280.0 * 64))
            cz = min(63, int(z / 1280.0 * 64))
            return cz * 64 + cx

        self.assertTrue(raster.forest[cell_for(0.0085, 0.0012)])
        self.assertFalse(raster.forest[cell_for(0.0069, 0.0022)])
        # Coastline is south->north with water on its right/east side.
        self.assertTrue(raster.water[cell_for(0.006, 0.009)])
        self.assertFalse(raster.water[cell_for(0.006, 0.005)])

    def test_forest_block_full_footprint_avoids_road_corridor(self) -> None:
        spec = _spec(
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=2,
            cell_size=25.0,
            forest_tree_spacing=50.0,
            forest_road_clearance=0.0,
            max_forest_objects=10,
        )
        dataset = OsmDataset(
            source_generator="test",
            element_count=1,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(
                OsmLineFeature(
                    "way/1",
                    {"highway": "residential"},
                    ((0.0, 0.98), (1.0, 0.98)),
                ),
            ),
            building_polygons=(),
            building_points=(),
            places=(),
        )
        raster = OsmRaster(
            cells=2,
            water=(False,) * 4,
            forest=(True,) * 4,
            farmland=(False,) * 4,
            urban=(False,) * 4,
            roads=(False,) * 4,
            buildings=(False,) * 4,
            high_resolution=8,
            coastline_seed_count=0,
        )
        projection = BboxProjection.create(spec.bbox, spec.world_size)
        result = generate_world_objects(
            dataset, projection, raster, (10.0,) * 4, spec, include_roads=False
        )
        self.assertEqual(result.forest_objects, 0)
        self.assertEqual(result.forest_road_rejections, 1)


    def test_building_anchor_uses_highest_footprint_terrain(self) -> None:
        spec = _spec(
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=3,
            cell_size=25.0,
            max_road_objects=0,
            max_buildings=10,
            max_forest_objects=0,
            building_ground_clearance=0.20,
        )
        dataset = OsmDataset(
            source_generator="test",
            element_count=1,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(),
            building_polygons=(
                OsmPolygonFeature(
                    "way/1",
                    {"building": "yes"},
                    (GeoPolygon(((0.4, 0.2), (0.4, 0.8), (0.6, 0.8), (0.6, 0.2), (0.4, 0.2))),),
                ),
            ),
            building_points=(),
            places=(),
        )
        raster = OsmRaster(
            cells=3,
            water=(False,) * 9,
            forest=(False,) * 9,
            farmland=(False,) * 9,
            urban=(False,) * 9,
            roads=(False,) * 9,
            buildings=(True,) * 9,
            high_resolution=8,
            coastline_seed_count=0,
        )
        projection = BboxProjection.create(spec.bbox, spec.world_size)
        # Runtime vertices are at x=0, 25 and 50 m. The polygon centroid is
        # around 5 m, but the selected foundation reaches the 10 m east vertex.
        result = generate_world_objects(
            dataset, projection, raster, (0.0, 0.0, 10.0) * 3, spec, include_roads=False
        )
        self.assertEqual(result.building_objects, 1)
        building = result.objects[0]
        self.assertAlmostEqual(building.y, 10.20, places=3)
        self.assertGreater(result.maximum_building_grounding_raise, 4.9)

    def test_forest_block_anchor_uses_highest_full_block_terrain(self) -> None:
        spec = _spec(
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=3,
            cell_size=25.0,
            forest_tree_spacing=50.0,
            max_road_objects=0,
            max_buildings=0,
            max_forest_objects=1,
            forest_ground_clearance=0.15,
        )
        dataset = OsmDataset(
            source_generator="test", element_count=0, coastlines=(), water=(), forests=(),
            farmland=(), urban=(), roads=(), building_polygons=(), building_points=(), places=(),
        )
        raster = OsmRaster(
            cells=3, water=(False,) * 9, forest=(True,) * 9, farmland=(False,) * 9,
            urban=(False,) * 9, roads=(False,) * 9, buildings=(False,) * 9,
            high_resolution=8, coastline_seed_count=0,
        )
        projection = BboxProjection.create(spec.bbox, spec.world_size)
        result = generate_world_objects(
            dataset, projection, raster, (0.0, 0.0, 10.0) * 3, spec, include_roads=False
        )
        self.assertEqual(result.forest_objects, 1)
        forest = result.objects[0]
        self.assertAlmostEqual(forest.y, 10.15, places=3)
        self.assertGreater(result.maximum_forest_grounding_raise, 4.9)

    def test_minor_roads_are_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            without = build_milestone3(root / "without", _spec(include_minor_roads=False))
            with_minor = build_milestone3(root / "with", _spec(include_minor_roads=True))
            without_manifest = json.loads(without.manifest_path.read_text(encoding="utf-8"))
            with_manifest = json.loads(with_minor.manifest_path.read_text(encoding="utf-8"))
            self.assertGreater(with_manifest["objects"]["roads"], without_manifest["objects"]["roads"])

    def test_build_is_deterministic_from_saved_osm_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = build_milestone3(root / "first", _spec())
            second = build_milestone3(root / "second", _spec())
            self.assertEqual(
                hashlib.sha256(first.pbo_path.read_bytes()).digest(),
                hashlib.sha256(second.pbo_path.read_bytes()).digest(),
            )
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.osm_preview_path.read_bytes(), second.osm_preview_path.read_bytes())

    def test_rejects_invalid_bbox(self) -> None:
        with self.assertRaisesRegex(ValueError, "bbox latitude order"):
            _spec(bbox=(1.0, 0.0, 0.0, 1.0)).validate()

    def test_overpass_query_contains_bbox_and_required_feature_classes(self) -> None:
        query = build_overpass_query((1.0, 2.0, 3.0, 4.0), timeout_seconds=45)
        self.assertIn("[bbox:1.0000000,2.0000000,3.0000000,4.0000000]", query)
        self.assertIn('["natural"="coastline"]', query)
        self.assertIn('["natural"="water"]', query)
        self.assertIn('["landuse"~', query)
        self.assertIn('["highway"]', query)
        self.assertIn('["building"]', query)
        self.assertIn("out body geom;", query)


if __name__ == "__main__":
    unittest.main()
