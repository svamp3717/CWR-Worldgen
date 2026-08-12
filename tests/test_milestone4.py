from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cwr_worldgen.assets import scan_assets
from cwr_worldgen.generator import build_milestone4
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.model import PlayabilitySpec
from cwr_worldgen.wrp import inspect_rvw4


FIXTURES = Path(__file__).parent / "fixtures"


class Milestone4Tests(unittest.TestCase):
    def spec(self, **overrides) -> PlayabilitySpec:
        values = {
            "heightmap_path": FIXTURES / "osm-height.png",
            "osm_json_path": FIXTURES / "osm-playability.json",
            "bbox": (0.0, 0.0, 0.01, 0.01),
            "cells": 64,
            "cell_size": 50.0,
            "elevation_minimum": -12.0,
            "elevation_maximum": 120.0,
            "sea_level": 0.0,
            "asset_roots": (FIXTURES / "assets",),
            "strict_assets": True,
        }
        values.update(overrides)
        return PlayabilitySpec(**values)

    def test_builds_playability_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = build_milestone4(Path(temp) / "build", self.spec(verify_regeneration=True))
            manifest = json.loads(result.manifest_path.read_text())
            road = json.loads(result.road_report_path.read_text())
            grading = json.loads(result.grading_report_path.read_text())
            assets = json.loads(result.asset_catalogue_path.read_text())
            reproducibility = json.loads(result.reproducibility_path.read_text())
            config = (result.source_dir / "config.cpp").read_text(encoding="ascii")
            wrp = inspect_rvw4(result.wrp_path, height_scale=self.spec().height_scale)

            self.assertEqual(manifest["milestone"], 4)
            self.assertEqual(manifest["playability"]["town_names"][0]["name"], "Testville")
            self.assertIn("class Names", config)
            self.assertIn('name = "Testville";', config)
            self.assertEqual(road["failed_connections"], 0)
            self.assertLessEqual(road["maximum_connection_gap"], self.spec().road_connection_tolerance)
            self.assertGreater(grading["changed_cells"], 0)
            self.assertGreater(grading["transitions"]["shoreline_cells"], 0)
            self.assertTrue(assets["verified"])
            self.assertTrue(reproducibility["pipeline_repeat_match"])
            self.assertTrue(reproducibility["wrp_byte_match"])
            self.assertTrue(reproducibility["pbo_byte_match"])
            self.assertGreater(wrp.object_count, 0)


    def test_regeneration_verification_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = build_milestone4(Path(temp) / "build", self.spec())
            reproducibility = json.loads(result.reproducibility_path.read_text())
            self.assertFalse(reproducibility["verification_enabled"])
            self.assertEqual(reproducibility["verification_status"], "skipped")
            self.assertNotIn("pipeline_repeat_match", reproducibility)
            self.assertFalse((result.output_dir / ".regeneration-check").exists())

    def test_regeneration_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = build_milestone4(Path(temp) / "a", self.spec())
            second = build_milestone4(Path(temp) / "b", self.spec())
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.pbo_path.read_bytes(), second.pbo_path.read_bytes())
            a = json.loads(first.reproducibility_path.read_text())
            b = json.loads(second.reproducibility_path.read_text())
            self.assertEqual(a["generation_fingerprint"], b["generation_fingerprint"])

    def test_strict_asset_scan_rejects_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "assets"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "strict asset validation failed"):
                build_milestone4(Path(temp) / "build", self.spec(asset_roots=(empty,)))

    def test_asset_scan_detects_p3d_texture_dependencies(self) -> None:
        scan = scan_assets((FIXTURES / "assets",), (r"data3d\les_su_ctver_pruhozi.p3d",))
        self.assertTrue(scan.verified)
        forest = next(record for record in scan.records if record.path.endswith("les_su_ctver_pruhozi.p3d"))
        self.assertTrue(forest.dependencies)
        self.assertFalse(scan.missing_dependencies)

    def test_asset_scan_resolves_bare_texture_from_data_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "data3d" / "forest.p3d"
            texture = root / "data" / "str_fikovnik.paa"
            model.parent.mkdir(parents=True)
            texture.parent.mkdir(parents=True)
            model.write_bytes(b"str_fikovnik.paa\0")
            texture.write_bytes(b"PAA")

            scan = scan_assets((root,), (r"data3d\forest.p3d",))

            self.assertTrue(scan.verified)
            self.assertFalse(scan.missing_dependencies)

    def test_asset_scan_resolves_paa_reference_to_pac_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "data3d" / "forest.p3d"
            texture = root / "data" / "str_fikovnik.pac"
            model.parent.mkdir(parents=True)
            texture.parent.mkdir(parents=True)
            model.write_bytes(b"str_fikovnik.paa\0")
            texture.write_bytes(b"PAC")

            scan = scan_assets((root,), (r"data3d\forest.p3d",))

            self.assertTrue(scan.verified)
            self.assertFalse(scan.missing_dependencies)

    def test_asset_scan_keeps_ambiguous_bare_texture_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "data3d" / "forest.p3d"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"shared.paa\0")
            for folder in ("addon_a", "addon_b"):
                texture = root / folder / "shared.paa"
                texture.parent.mkdir(parents=True)
                texture.write_bytes(folder.encode("ascii"))

            scan = scan_assets((root,), (r"data3d\forest.p3d",))

            self.assertEqual(scan.missing_dependencies, ("shared.paa",))

    def test_asset_catalogue_scans_pbo_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pbo = Path(temp) / "data3d.pbo"
            write_pbo(
                pbo,
                (
                    PboEntry("les_su_ctver_pruhozi.p3d", b"data3d\\forest.pac\0"),
                    PboEntry("forest.pac", b"PAC"),
                ),
            )
            scan = scan_assets((pbo,), (r"data3d\les_su_ctver_pruhozi.p3d",))
            self.assertTrue(scan.verified)
            self.assertEqual(len(scan.records), 2)

    def test_transition_seed_changes_dither_but_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = build_milestone4(Path(temp) / "one", self.spec(deterministic_seed="one"))
            again = build_milestone4(Path(temp) / "again", self.spec(deterministic_seed="one"))
            second = build_milestone4(Path(temp) / "two", self.spec(deterministic_seed="two"))
            self.assertEqual(first.wrp_path.read_bytes(), again.wrp_path.read_bytes())
            self.assertNotEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())

    def test_invalid_playability_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "road connection tolerance"):
            self.spec(road_connection_tolerance=-1).validate()
        with self.assertRaisesRegex(ValueError, "transition cells"):
            self.spec(transition_cells=99).validate()
        with self.assertRaisesRegex(ValueError, "forest road clearance"):
            self.spec(forest_road_clearance=-0.1).validate()


class RoadFittingRegressionTests(unittest.TestCase):
    @staticmethod
    def _dataset(*roads):
        from cwr_worldgen.osm import OsmDataset

        return OsmDataset(
            source_generator="test",
            element_count=len(roads),
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=tuple(roads),
            building_polygons=(),
            building_points=(),
            places=(),
        )

    def test_short_osm_vertex_intervals_do_not_create_false_road_gaps(self) -> None:
        from cwr_worldgen.osm import BboxProjection, OsmLineFeature
        from cwr_worldgen.playability import fit_road_objects

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 3200.0)
        points = tuple(projection.to_latlon((100.0 + index, 500.0)) for index in range(101))
        road = OsmLineFeature("road/short-vertices", {"highway": "primary", "surface": "asphalt"}, points)
        spec = PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=64,
            cell_size=50.0,
        )
        report = fit_road_objects(self._dataset(road), projection, [0.0] * (64 * 64), spec)
        self.assertLessEqual(len(report.objects), 5)
        self.assertEqual(report.failed_connections, 0)
        self.assertLessEqual(report.maximum_chain_gap, 0.251)

    def test_merged_junction_branches_cover_the_shared_node(self) -> None:
        from cwr_worldgen.osm import BboxProjection, OsmLineFeature
        from cwr_worldgen.playability import fit_road_objects

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 3200.0)
        node = (800.0, 800.0)
        roads = (
            OsmLineFeature("road/west", {"highway": "primary"}, (
                projection.to_latlon((600.0, 800.0)), projection.to_latlon(node))),
            OsmLineFeature("road/east", {"highway": "primary"}, (
                projection.to_latlon(node), projection.to_latlon((1000.0, 800.0)))),
            OsmLineFeature("road/north", {"highway": "primary"}, (
                projection.to_latlon(node), projection.to_latlon((800.0, 1000.0)))),
        )
        spec = PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=64,
            cell_size=50.0,
        )
        report = fit_road_objects(self._dataset(*roads), projection, [0.0] * (64 * 64), spec)
        self.assertEqual(report.connection_count, 1)
        self.assertEqual(report.failed_connections, 0)
        self.assertLessEqual(report.maximum_connection_gap, 1e-6)

    def test_junction_models_stop_before_the_central_texture_patch(self) -> None:
        from cwr_worldgen.osm import BboxProjection, OsmLineFeature
        from cwr_worldgen.playability import _model_axis, _point_segment_distance, fit_road_objects

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 3200.0)
        node = (800.0, 800.0)
        roads = (
            OsmLineFeature("road/west", {"highway": "primary"}, (
                projection.to_latlon((600.0, 800.0)), projection.to_latlon(node))),
            OsmLineFeature("road/east", {"highway": "primary"}, (
                projection.to_latlon(node), projection.to_latlon((1000.0, 800.0)))),
            OsmLineFeature("road/north", {"highway": "primary"}, (
                projection.to_latlon(node), projection.to_latlon((800.0, 1000.0)))),
        )
        spec = PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 0.01, 0.01),
            cells=64, cell_size=50.0,
        )
        report = fit_road_objects(self._dataset(*roads), projection, [0.0] * (64 * 64), spec)
        nearest = min(
            _point_segment_distance(node, *_model_axis(obj, spec.road_segment_length))
            for obj in report.objects
        )
        self.assertGreaterEqual(nearest, 5.0)
        self.assertLessEqual(nearest, 5.5)
        self.assertEqual(report.maximum_model_overlap_metres, 0.0)
        self.assertEqual(report.terrain_filled_junctions, 1)

    def test_road_anchor_uses_highest_full_model_footprint(self) -> None:
        from cwr_worldgen.osm import BboxProjection, OsmLineFeature
        from cwr_worldgen.playability import fit_road_objects

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        road = OsmLineFeature(
            "road/ridge", {"highway": "residential"},
            (projection.to_latlon((37.75, 50.0)), projection.to_latlon((62.25, 50.0))),
        )
        spec = PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 1.0, 1.0),
            cells=4, cell_size=25.0,
        )
        elevations = [0.0] * 16
        elevations[1 * 4 + 2] = 10.0
        elevations[2 * 4 + 2] = 10.0
        report = fit_road_objects(self._dataset(road), projection, elevations, spec)
        self.assertEqual(len(report.objects), 1)
        self.assertGreaterEqual(report.objects[0].y, 9.99)

    def test_sharp_turn_is_split_without_leaving_an_uncovered_seam(self) -> None:
        from cwr_worldgen.osm import BboxProjection, OsmLineFeature
        from cwr_worldgen.playability import fit_road_objects

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 3200.0)
        world_points = ((500.0, 500.0), (520.0, 500.0), (520.0, 520.0), (540.0, 520.0))
        road = OsmLineFeature(
            "road/corner",
            {"highway": "secondary"},
            tuple(projection.to_latlon(point) for point in world_points),
        )
        spec = PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=64,
            cell_size=50.0,
        )
        report = fit_road_objects(self._dataset(road), projection, [0.0] * (64 * 64), spec)
        self.assertGreaterEqual(report.chain_count, 2)
        self.assertLessEqual(report.maximum_chain_gap, spec.road_connection_tolerance)


if __name__ == "__main__":
    unittest.main()
