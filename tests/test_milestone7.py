# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from cwr_worldgen.milestone7 import Milestone7Spec, build_milestone7
from cwr_worldgen.model import ConstraintPlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPolygonFeature,
    OsmRaster,
)
from cwr_worldgen.source_pipeline import SourceFetchSpec, fetch_sources
from cwr_worldgen.terrain_solver import (
    _effective_terrain_smoothing,
    solve_terrain_constraints,
)


def _polygon_feature(projection: BboxProjection, key: str, tags: dict[str, str], points: tuple[tuple[float, float], ...]) -> OsmPolygonFeature:
    ring = tuple(projection.to_latlon(point) for point in points)
    return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))


class ConstraintSolverTests(unittest.TestCase):
    def test_512_world_scales_terrain_smoothing_from_256_reference(self) -> None:
        base = ConstraintPlayabilitySpec(
            heightmap_path=Path("unused.tif"), cells=256, cell_size=25.0,
            shoreline_transition_cells=3, lake_shore_smoothing_cells=8,
            world_edge_blend_cells=3, natural_smoothing_strength=0.16,
            solver_iterations=20,
        )
        large = ConstraintPlayabilitySpec(
            heightmap_path=Path("unused.tif"), cells=512, cell_size=25.0,
            shoreline_transition_cells=3, lake_shore_smoothing_cells=8,
            world_edge_blend_cells=3, natural_smoothing_strength=0.16,
            solver_iterations=20,
        )
        base_effective = _effective_terrain_smoothing(base)
        large_effective = _effective_terrain_smoothing(large)
        self.assertEqual(base_effective.scale, 1.0)
        self.assertEqual(base_effective.shoreline_transition_cells, 3)
        self.assertEqual(base_effective.lake_shore_smoothing_cells, 8)
        self.assertEqual(base_effective.world_edge_blend_cells, 3)
        self.assertEqual(base_effective.natural_smoothing_strength, 0.16)
        self.assertEqual(base_effective.solver_iterations, 20)
        self.assertEqual(large_effective.scale, 2.0)
        self.assertEqual(large_effective.shoreline_transition_cells, 6)
        self.assertEqual(large_effective.lake_shore_smoothing_cells, 16)
        self.assertEqual(large_effective.world_edge_blend_cells, 6)
        self.assertAlmostEqual(large_effective.natural_smoothing_strength, 0.64)
        self.assertEqual(large_effective.solver_iterations, 20)

    def test_larger_world_effective_relaxation_removes_more_dem_jaggedness(self) -> None:
        cells = 16
        bbox = (0.0, 0.0, 0.01, 0.01)
        original = tuple(
            12.0 if (x + z) % 2 else 0.0
            for z in range(cells)
            for x in range(cells)
        )
        empty = (False,) * (cells * cells)
        raster = OsmRaster(
            cells=cells, water=empty, forest=empty, farmland=empty,
            urban=empty, roads=empty, buildings=empty,
            high_resolution=cells, coastline_seed_count=0,
        )
        dataset = OsmDataset(
            source_generator="smoothing-scale", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )

        def solve(cell_size: float):
            spec = ConstraintPlayabilitySpec(
                heightmap_path=Path("unused.tif"), bbox=bbox,
                cells=cells, cell_size=cell_size,
                shoreline_transition_cells=0, lake_shore_smoothing_cells=0,
                world_edge_blend_cells=0, natural_smoothing_strength=0.16,
                solver_iterations=20,
            )
            return solve_terrain_constraints(
                original,
                dataset,
                BboxProjection.create(bbox, cells * cell_size),
                raster,
                spec,
            )

        reference = solve(400.0)  # 6.4 km world
        doubled = solve(800.0)    # 12.8 km world

        def neighbour_roughness(values: tuple[float, ...]) -> float:
            differences = []
            for z in range(cells):
                for x in range(cells):
                    index = z * cells + x
                    if x + 1 < cells:
                        differences.append(abs(values[index] - values[index + 1]))
                    if z + 1 < cells:
                        differences.append(abs(values[index] - values[index + cells]))
            return sum(differences) / len(differences)

        self.assertEqual(reference.smoothing_reference_scale, 1.0)
        self.assertEqual(doubled.smoothing_reference_scale, 2.0)
        self.assertLess(
            neighbour_roughness(doubled.elevations),
            neighbour_roughness(reference.elevations),
        )

    def test_road_grading_does_not_worsen_roads_or_report_merged_building_masks(self) -> None:
        cells = 64
        cell_size = 25.0
        bbox = (0.0, 0.0, 0.01, 0.01)
        projection = BboxProjection.create(bbox, cells * cell_size)
        road = OsmLineFeature(
            "way/road-through-pad",
            {"highway": "primary", "surface": "asphalt"},
            tuple(
                projection.to_latlon(point)
                for point in ((100.0, 800.0), (500.0, 800.0), (800.0, 800.0), (1500.0, 800.0))
            ),
        )
        building = _polygon_feature(
            projection,
            "way/building",
            {"building": "yes"},
            ((700.0, 700.0), (900.0, 700.0), (900.0, 900.0), (700.0, 900.0), (700.0, 700.0)),
        )
        dataset = OsmDataset(
            source_generator="road-building-grade-test",
            element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(road,), building_polygons=(building,),
        )
        building_mask = [False] * (cells * cells)
        for z in range(28, 36):
            for x in range(28, 36):
                building_mask[z * cells + x] = True
        empty = (False,) * (cells * cells)
        raster = OsmRaster(
            cells=cells,
            water=empty,
            forest=empty,
            farmland=empty,
            urban=empty,
            roads=empty,
            buildings=tuple(building_mask),
            high_resolution=256,
            coastline_seed_count=0,
        )
        original: list[float] = []
        for z in range(cells):
            for x in range(cells):
                world_x = (x + 0.5) * cell_size
                height = 10.0
                if world_x < 650.0:
                    height = 10.0 + (650.0 - world_x) * 0.20
                elif world_x > 950.0:
                    height = 10.0 - (world_x - 950.0) * 0.20
                original.append(height)
        spec = ConstraintPlayabilitySpec(
            heightmap_path=Path("unused.tif"),
            bbox=bbox,
            cells=cells,
            cell_size=cell_size,
            maximum_road_grade_percent=12.0,
            maximum_grade_adjustment=40.0,
            road_grade_radius=100.0,
            building_grade_radius=25.0,
            solver_iterations=20,
            world_edge_blend_cells=0,
        )
        result = solve_terrain_constraints(original, dataset, projection, raster, spec)
        self.assertLessEqual(
            result.maximum_road_slope_after_percent,
            max(spec.maximum_road_grade_percent, result.maximum_road_slope_before_percent + 0.10) + 1e-6,
        )
        self.assertLessEqual(result.building_roughness_after, result.building_roughness_before + 1e-6)

    def test_inland_lake_bank_is_widened_and_grade_limited(self) -> None:
        cells = 32
        cell_size = 50.0
        bbox = (0.0, 0.0, 0.01, 0.01)
        original = [30.0] * (cells * cells)
        water_mask = [False] * (cells * cells)
        for z in range(14, 18):
            for x in range(14, 18):
                water_mask[z * cells + x] = True
        raster = OsmRaster(
            cells=cells,
            water=tuple(water_mask),
            forest=(False,) * (cells * cells),
            farmland=(False,) * (cells * cells),
            urban=(False,) * (cells * cells),
            roads=(False,) * (cells * cells),
            buildings=(False,) * (cells * cells),
            high_resolution=128,
            coastline_seed_count=0,
        )
        dataset = OsmDataset(
            source_generator="test",
            element_count=0,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(),
            watercourses=(),
            building_polygons=(),
            building_points=(),
            places=(),
        )
        spec = ConstraintPlayabilitySpec(
            heightmap_path=Path("unused.tif"),
            bbox=bbox,
            cells=cells,
            cell_size=cell_size,
            lake_shore_smoothing_cells=8,
            lake_shore_maximum_slope_percent=8.0,
            solver_iterations=6,
            world_edge_blend_cells=0,
        )
        result = solve_terrain_constraints(
            original,
            dataset,
            BboxProjection.create(bbox, cells * cell_size),
            raster,
            spec,
        )
        # CWA has one global water plane. A synthetic lake at 30 m is therefore
        # preserved at DEM height rather than excavated to sea level.
        self.assertEqual(result.coastal_water_components, 0)
        self.assertEqual(result.inland_water_components, 0)
        self.assertEqual(result.water_cells, 0)
        self.assertEqual(result.deep_water_cells, 0)
        self.assertEqual(result.uncertain_water_cells_preserved, 16)
        self.assertEqual(result.lake_shore_cells, 0)
        self.assertEqual(result.elevations[15 * cells + 14], 30.0)
        self.assertEqual(result.elevations[15 * cells + 13], 30.0)

    def test_unified_solver_enforces_priority_constraints(self) -> None:
        cells = 32
        cell_size = 10.0
        world_size = cells * cell_size
        bbox = (0.0, 0.0, 0.01, 0.01)
        projection = BboxProjection.create(bbox, world_size)
        original = []
        for z in range(cells):
            for x in range(cells):
                original.append(15.0 + z * 0.7 + ((x % 3) - 1) * 1.4)

        def line(key: str, tags: dict[str, str], points: tuple[tuple[float, float], ...]) -> OsmLineFeature:
            return OsmLineFeature(key, tags, tuple(projection.to_latlon(point) for point in points))

        roads = (
            line("road/major", {"highway": "primary", "width": "30"}, ((20.0, 180.0), (300.0, 180.0))),
            line("road/bridge", {"highway": "service", "bridge": "yes", "width": "12"}, ((80.0, 20.0), (80.0, 300.0))),
            line("road/tunnel", {"highway": "service", "tunnel": "yes"}, ((120.0, 20.0), (120.0, 300.0))),
            line("road/embankment", {"highway": "track", "embankment": "yes"}, ((20.0, 250.0), (300.0, 250.0))),
        )
        stream = line("waterway/stream", {"waterway": "stream"}, ((250.0, 300.0), (250.0, 20.0)))
        building = _polygon_feature(
            projection,
            "building/1",
            {"building": "house"},
            ((190.0, 80.0), (230.0, 80.0), (230.0, 120.0), (190.0, 120.0), (190.0, 80.0)),
        )
        water = _polygon_feature(
            projection,
            "water/1",
            {"natural": "water"},
            ((20.0, 20.0), (70.0, 20.0), (70.0, 70.0), (20.0, 70.0), (20.0, 20.0)),
        )
        dataset = OsmDataset(
            source_generator="test",
            element_count=7,
            coastlines=(),
            water=(water,),
            forests=(),
            farmland=(),
            urban=(),
            roads=roads,
            watercourses=(stream,),
            building_polygons=(building,),
            building_points=(),
            places=(),
        )
        water_mask = [False] * (cells * cells)
        building_mask = [False] * (cells * cells)
        for z in range(2, 7):
            for x in range(2, 7):
                water_mask[z * cells + x] = True
        for z in range(8, 12):
            for x in range(19, 23):
                building_mask[z * cells + x] = True
        raster = OsmRaster(
            cells=cells,
            water=tuple(water_mask),
            forest=(False,) * (cells * cells),
            farmland=(False,) * (cells * cells),
            urban=(False,) * (cells * cells),
            roads=(False,) * (cells * cells),
            buildings=tuple(building_mask),
            high_resolution=256,
            coastline_seed_count=0,
        )
        spec = ConstraintPlayabilitySpec(
            heightmap_path=Path("unused.tif"),
            bbox=bbox,
            cells=cells,
            cell_size=cell_size,
            maximum_road_grade_percent=12.0,
            major_road_grade_percent=8.0,
            road_grade_radius=30.0,
            building_grade_radius=20.0,
            maximum_grade_adjustment=20.0,
            shoreline_transition_cells=3,
            solver_iterations=12,
            world_edge_blend_cells=2,
        )
        first = solve_terrain_constraints(original, dataset, projection, raster, spec)
        second = solve_terrain_constraints(original, dataset, projection, raster, spec)
        self.assertEqual(first.elevations, second.elevations)
        # The mapped water in this synthetic fixture sits ~15-20 m above sea
        # level, so it is preserved instead of being flattened into CWA water.
        self.assertEqual(first.water_cells, 0)
        self.assertGreater(first.uncertain_water_cells_preserved, 0)
        self.assertLessEqual(first.water_roughness_after, first.water_roughness_before + 1e-6)
        self.assertLessEqual(first.building_roughness_after, 0.1)
        self.assertLessEqual(first.downhill_violations_after, first.downhill_violations_before)
        # The synthetic bridge-tagged service road never reaches the global
        # water plane, so it is intentionally treated as an ordinary road.
        self.assertEqual(first.bridge_segments, 0)
        self.assertEqual(first.tunnel_segments_excluded, 1)
        self.assertEqual(first.embankment_segments, 1)
        self.assertGreater(first.major_road_cells, 0)
        self.assertGreater(first.watercourse_cells, 0)
        self.assertGreater(first.total_cut_volume_m3 + first.total_fill_volume_m3, 0.0)
        self.assertEqual(first.priority_order[0], "water-bodies")
        self.assertEqual(first.priority_order[-1], "world-boundaries")

    def test_invalid_solver_controls_are_rejected(self) -> None:
        spec = ConstraintPlayabilitySpec(
            heightmap_path=Path(__file__),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=32,
            cell_size=50.0,
            major_road_grade_percent=13.0,
            maximum_road_grade_percent=12.0,
        )
        with self.assertRaisesRegex(ValueError, "major road grade"):
            spec.validate()

        invalid_lake = ConstraintPlayabilitySpec(
            heightmap_path=Path(__file__),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=32,
            cell_size=50.0,
            lake_shore_maximum_slope_percent=101.0,
        )
        with self.assertRaisesRegex(ValueError, "lake shore maximum slope"):
            invalid_lake.validate()


class Milestone7BuildTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        elevation = root / "elevation" / "raw"
        elevation.mkdir(parents=True)
        values = (30, 35, 40, 25, 30, 35, 20, 25, 30)
        payload = b"".join(value.to_bytes(2, "big", signed=True) for value in values)
        with ZipFile(elevation / "N00E000.hgt.zip", "w") as archive:
            archive.writestr("N00E000.hgt", payload)
        osm = root / "osm"
        osm.mkdir(parents=True)
        document = {
            "version": 0.6,
            "generator": "milestone7-test",
            "elements": [
                {"type": "way", "id": 1, "tags": {"natural": "water"}, "geometry": [
                    {"lat": 0.001, "lon": 0.001}, {"lat": 0.001, "lon": 0.003},
                    {"lat": 0.003, "lon": 0.003}, {"lat": 0.003, "lon": 0.001},
                    {"lat": 0.001, "lon": 0.001},
                ]},
                {"type": "way", "id": 2, "tags": {"highway": "primary"}, "geometry": [
                    {"lat": 0.005, "lon": 0.001}, {"lat": 0.005, "lon": 0.009},
                ]},
                {"type": "way", "id": 3, "tags": {"highway": "service", "bridge": "yes"}, "geometry": [
                    {"lat": 0.001, "lon": 0.006}, {"lat": 0.009, "lon": 0.006},
                ]},
                {"type": "way", "id": 4, "tags": {"highway": "service", "tunnel": "yes"}, "geometry": [
                    {"lat": 0.001, "lon": 0.007}, {"lat": 0.009, "lon": 0.007},
                ]},
                {"type": "way", "id": 5, "tags": {"highway": "track", "embankment": "yes"}, "geometry": [
                    {"lat": 0.008, "lon": 0.001}, {"lat": 0.008, "lon": 0.009},
                ]},
                {"type": "way", "id": 6, "tags": {"building": "house"}, "geometry": [
                    {"lat": 0.006, "lon": 0.002}, {"lat": 0.006, "lon": 0.003},
                    {"lat": 0.007, "lon": 0.003}, {"lat": 0.007, "lon": 0.002},
                    {"lat": 0.006, "lon": 0.002},
                ]},
                {"type": "way", "id": 7, "tags": {"waterway": "stream"}, "geometry": [
                    {"lat": 0.009, "lon": 0.009}, {"lat": 0.001, "lon": 0.009},
                ]},
                {"type": "way", "id": 8, "tags": {"landuse": "forest"}, "geometry": [
                    {"lat": 0.0005, "lon": 0.0005}, {"lat": 0.0005, "lon": 0.0095},
                    {"lat": 0.0095, "lon": 0.0095}, {"lat": 0.0095, "lon": 0.0005},
                    {"lat": 0.0005, "lon": 0.0005},
                ]},
                {"type": "node", "id": 9, "lat": 0.005, "lon": 0.005, "tags": {"place": "village", "name": "Testby"}},
            ],
        }
        (osm / "raw-overpass.json").write_text(json.dumps(document), encoding="utf-8")
        overture = root / "overture"
        overture.mkdir(parents=True)
        (overture / "buildings.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}\n',
            encoding="utf-8",
        )
        fetch_sources(SourceFetchSpec(
            source_dir=root,
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=64,
            cell_size=50.0,
            dem_provider="hgt",
            reference_map=False,
        ))
        return root

    def test_milestone7_build_is_offline_deterministic_and_emits_solver_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            spec = Milestone7Spec(
                source_dir=source,
                name="cwr_m7_test",
                display_name="CWR M7 Test",
                solver_iterations=8,
                world_edge_blend_cells=2,
            )
            first = build_milestone7(root / "one", spec)
            second = build_milestone7(root / "two", spec)
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.pbo_path.read_bytes(), second.pbo_path.read_bytes())
            self.assertEqual(first.pbo_path.parent.parent.name, "@CWR-Milestone7")
            self.assertTrue(first.solver_heightmap_path.is_file())
            report = json.loads(first.grading_report_path.read_text(encoding="utf-8"))
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(report["solver"], "unified-priority-constraint-relaxation")
            self.assertEqual(report["water_cells"], 0)
            self.assertGreater(report["uncertain_water_cells_preserved"], 0)
            self.assertLessEqual(
                report["water_roughness_after"],
                report["water_roughness_before"] + 1e-6,
            )
            self.assertIn("category_adjustments", report)
            self.assertEqual(manifest["milestone"], 7)
            self.assertIn("constraint_terrain_solver", manifest)
            self.assertTrue((first.pbo_path.parent.parent / "TERRAIN-SOLVER-REPORT.json").is_file())


if __name__ == "__main__":
    unittest.main()
