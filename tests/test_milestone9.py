# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import re
import tempfile
import unittest

from PIL import Image

from cwr_worldgen import procedural_buildings as building_models
from cwr_worldgen._version import GENERATOR_VERSION
from cwr_worldgen.cli import _parser
from cwr_worldgen.location_example import square_bbox
from cwr_worldgen.milestone9 import (
    MALDEN_BUSH_MODELS,
    MALDEN_FOREST_BLOCK_MODEL,
    MALDEN_ROADSIDE_TREE_MODELS,
    MALDEN_SINGLE_TREE_MODEL,
    NOGOVA_BUSH_MODELS,
    NOGOVA_ROADSIDE_TREE_MODEL,
    NOGOVA_ROADSIDE_TREE_MODELS,
    NOGOVA_SINGLE_TREE_MODEL,
    Milestone9Spec,
    _Milestone9PlayabilitySpec,
    _resolved_forest_profile_models,
    build_milestone9,
)
from cwr_worldgen.generator import (
    _assemble_world_objects,
    _placement_driven_surface_overlay,
    _verify_single_world_pbo_layout,
)
from cwr_worldgen.osm import (
    BboxProjection,
    BuildingPlacementPlan,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPointFeature,
    OsmPolygonFeature,
    OsmRaster,
    generate_world_objects,
    rasterize_osm,
    plan_building_placements,
    road_bridge_crosses_ditch_only,
    road_is_dirt,
    road_is_gravel,
    road_model_for_tags,
    write_meadow_grass_placement_preview,
    _sample_elevation,
    _triangle_elevation_bounds,
    _maximum_polygon_elevation,
    _minimum_polygon_elevation,
    _square_elevation_samples,
    _oriented_footprint_elevation_samples,
    _non_buried_vegetation_anchor,
    _non_buried_vegetation_fit,
    _distributed_grid_indices,
    _forest_single_tree_candidates,
    _forest_single_tree_rank,
    _geographic_forest_single_tree_cells,
    _scaled_synthetic_tree_limit,
    _bridge_module_chunks,
    _demote_dense_garage_clusters_to_sheds,
    ObjectGenerationResult,
    augment_dataset_with_overture_buildings,
    NOGOVA_BRIDGE_MODEL,
    NOGOVA_BRIDGE_MODULE_LENGTH_METRES,
    NOGOVA_BRIDGE_APPROACH_OFFSET_METRES,
    NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES,
    CHURCH_EXTRA_GROUND_CLEARANCE_METRES,
    BUILDING_TERRAIN_EDGE_MARGIN_METRES,
    OSM_INDIVIDUAL_TREE_MODELS,
    STOCK_HEDGE_MODELS,
    STOCK_STONE_MODELS,
    ROADSIDE_BARRIER_CLEARANCE_METRES,
    line_intersects_road_corridors,
    project_road_corridors,
    plan_iterative_grounding_objects,
    refine_iterative_grounding_terrain,
    forest_block_intersects_road_corridors,
    STOCK_METAL_FENCE_MODELS,
    STOCK_WALL_MODELS,
)
from cwr_worldgen.model import WorldObject
from cwr_worldgen.paa import inspect_paa, write_rgb_dxt1_paa
from cwr_worldgen.pbo import read_pbo
from cwr_worldgen.playability import (
    fit_road_objects,
    _curved_gravel_model_for_run,
    _generated_gravel_terrain_raise,
    _road_object_on_slope,
    road_model_variants,
    town_locations,
)
from cwr_worldgen.procedural_buildings import (
    BuildingPlacement,
    BuildingVariantKey,
    ProceduralBuildingLibrary,
    _visual_lod,
    inspect_mlod,
)
from cwr_worldgen.procedural_forests import DITCH_GRASS_VARIANTS
from cwr_worldgen.procedural_infrastructure import (
    GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES,
    GENERATED_GRAVEL_HALF_WIDTH_METRES,
    GENERATED_GRAVEL_SURFACE_CLEARANCE_METRES,
    GENERATED_GRAVEL_VISUAL_TOP_METRES,
    InfrastructureModelKey,
    ProceduralInfrastructureLibrary,
    create_gravel_road_texture_image,
    gravel_road_model_path,
    _road_lods,
)
from cwr_worldgen.surface_pass import (
    EVERON_SURFACE_TEXTURES,
    MATERIAL_INDEX,
    MILESTONE9_MATERIALS,
    NOGOVA_SURFACE_TEXTURES,
    build_surface_pass,
    external_surface_texture_paths,
    render_building_source_reference,
    render_overview_map,
    surface_texture_wire_paths,
    write_surface_textures,
)
from cwr_worldgen.terrain import NOGOVA_GROUND_TEXTURE_CYCLE, OSM_MATERIALS
from cwr_worldgen.wrp import inspect_rvw4
import test_milestone8 as milestone8_tests


class WorldObjectOrderingTests(unittest.TestCase):
    @staticmethod
    def _object(object_id: int, model: str) -> WorldObject:
        return WorldObject(object_id, model, float(object_id), 0.0, 0.0)

    def test_forests_and_rural_trees_are_serialized_before_buildings(self) -> None:
        road = self._object(1, r"o\road\sil25.p3d")
        buildings = (
            self._object(2, r"cwr_test\g\house_a.p3d"),
            self._object(3, r"cwr_test\g\house_b.p3d"),
        )
        forests = (
            self._object(4, r"data3d\les ctverec pruchozi_T1.p3d"),
            self._object(5, r"cwr_test\f\forest_under.p3d"),
            self._object(6, r"cwr_test\f\forest_border.p3d"),
        )
        ditch = (self._object(7, r"cwr_test\f\ditch_grass.p3d"),)
        barrier = (self._object(8, r"cwr_test\i\fence.p3d"),)
        bridge = (self._object(9, r"cwr_test\i\bridge.p3d"),)
        rural = (
            self._object(10, r"cwr_test\f\tree_row.p3d"),
            self._object(11, r"cwr_test\i\rock_group.p3d"),
        )
        mapped_tree = self._object(12, r"data3d\str briza.p3d")
        utility = self._object(13, r"cwr_test\i\utility_power_pole.p3d")
        nonroads = ObjectGenerationResult(
            objects=buildings + forests + ditch + barrier + bridge + rural + (mapped_tree, utility),
            road_objects=0,
            building_objects=2,
            forest_objects=1,
            road_objects_truncated=False,
            building_objects_truncated=False,
            forest_objects_truncated=False,
            forest_undergrowth_objects=1,
            forest_border_objects=1,
            ditch_grass_objects=1,
            barrier_objects=1,
            bridge_objects=1,
            tree_row_objects=1,
            rural_rock_objects=1,
            mapped_tree_objects=1,
            utility_objects=1,
        )
        semantic = self._object(14, r"cwr_test\s\school.p3d")

        ordered = _assemble_world_objects((road,), nonroads, (semantic,))

        self.assertEqual(
            [obj.model_path for obj in ordered],
            [
                road.model_path,
                *[obj.model_path for obj in forests],
                mapped_tree.model_path,
                *[obj.model_path for obj in rural],
                *[obj.model_path for obj in buildings],
                ditch[0].model_path,
                barrier[0].model_path,
                bridge[0].model_path,
                utility.model_path,
                semantic.model_path,
            ],
        )
        self.assertEqual([obj.object_id for obj in ordered], list(range(1, len(ordered) + 1)))
        self.assertEqual(buildings[0].object_id, 2)

    def test_object_ordering_rejects_category_count_drift(self) -> None:
        nonroads = ObjectGenerationResult(
            objects=(self._object(1, r"cwr_test\g\house.p3d"),),
            road_objects=0,
            building_objects=0,
            forest_objects=0,
            road_objects_truncated=False,
            building_objects_truncated=False,
            forest_objects_truncated=False,
        )
        with self.assertRaisesRegex(ValueError, "category counts"):
            _assemble_world_objects((), nonroads, ())


class SurfacePassTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> tuple[OsmDataset, BboxProjection]:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def polygon(key: str, tags: dict[str, str], points: tuple[tuple[float, float], ...]) -> OsmPolygonFeature:
            ring = tuple(projection.to_latlon(point) for point in (*points, points[0]))
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        def line(key: str, tags: dict[str, str], points: tuple[tuple[float, float], ...]) -> OsmLineFeature:
            return OsmLineFeature(key, tags, tuple(projection.to_latlon(point) for point in points))

        water = polygon("way/1", {"natural": "water"}, ((0, 0), (210, 0), (210, 1000), (0, 1000)))
        forest = polygon("way/2", {"landuse": "forest"}, ((260, 80), (470, 80), (470, 430), (260, 430)))
        farmland = polygon("way/3", {"landuse": "farmland"}, ((510, 70), (920, 70), (920, 430), (510, 430)))
        urban = polygon("way/4", {"landuse": "residential", "category": "urban"}, ((250, 590), (500, 590), (500, 920), (250, 920)))
        industrial = polygon("way/5", {"landuse": "industrial", "category": "industrial"}, ((560, 590), (920, 590), (920, 920), (560, 920)))
        paved = line("way/6", {"highway": "primary", "surface": "asphalt"}, ((220, 500), (980, 500)))
        dirt = line("way/7", {"highway": "track", "surface": "dirt"}, ((220, 540), (980, 540)))
        dataset = OsmDataset(
            source_generator="milestone9-test",
            element_count=7,
            coastlines=(),
            water=(water,),
            forests=(forest,),
            farmland=(farmland,),
            urban=(urban, industrial),
            roads=(paved, dirt),
        )
        return dataset, projection

    def _spec(self, seed: str, reference: Path | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            cells=32,
            cell_size=31.25,
            include_minor_roads=True,
            sea_level=0.0,
            rock_height=110.0,
            rock_slope_degrees=30.0,
            deterministic_seed=seed,
            surface_shoreline_wet_cells=1,
            surface_shoreline_sand_cells=2,
            surface_transition_cells=2,
            surface_forest_edge_cells=1,
            surface_farmland_strip_cells=3,
            surface_road_shoulder_metres=24.0,
            surface_dirt_blend_metres=28.0,
            surface_steep_slope_degrees=38.0,
            surface_colour_reference_path=reference,
            surface_colour_reference_strength=0.55 if reference else 0.0,
        )

    def test_seeded_surface_classes_are_deterministic_and_feature_owned(self) -> None:
        dataset, projection = self._dataset()
        raster = rasterize_osm(dataset, projection, cells=32, include_minor_roads=True)
        elevations = [60.0] * (32 * 32)
        slopes = [4.0] * (32 * 32)
        elevations[31 * 32 + 31] = 130.0
        slopes[30 * 32 + 31] = 33.0
        slopes[29 * 32 + 31] = 48.0

        first = build_surface_pass(dataset, projection, raster, elevations, slopes, self._spec("world-seed-a"))
        second = build_surface_pass(dataset, projection, raster, elevations, slopes, self._spec("world-seed-a"))
        changed_seed = build_surface_pass(dataset, projection, raster, elevations, slopes, self._spec("world-seed-b"))

        self.assertEqual(first, second)
        self.assertNotEqual(first.indices, changed_seed.indices)
        self.assertGreater(first.wet_shoreline_cells, 0)
        self.assertGreater(first.dry_shoreline_cells, 0)
        self.assertGreater(first.forest_edge_cells, 0)
        self.assertGreater(first.farmland_light_cells, 0)
        self.assertGreater(first.farmland_dark_cells, 0)
        self.assertGreater(first.field_boundary_cells, 0)
        self.assertGreater(first.urban_cells, 0)
        self.assertGreater(first.industrial_cells, 0)
        self.assertGreater(first.paved_road_cells, 0)
        self.assertGreater(first.paved_shoulder_cells, 0)
        self.assertGreater(first.dirt_road_cells, 0)
        self.assertGreater(first.dirt_blend_cells, 0)
        self.assertGreater(first.rock_cells, 0)
        self.assertGreater(first.steep_rock_cells, 0)
        self.assertGreaterEqual(first.feature_seed_count, 4)
        self.assertEqual(len(MILESTONE9_MATERIALS), 22)
        for code in "wqsghrkfeabcuipodtvjx":
            self.assertIn(code, MATERIAL_INDEX)

    def test_explicit_osm_surfaces_and_aeroways_override_broad_landuse(self) -> None:
        dataset, projection = self._dataset()

        def polygon(key: str, tags: dict[str, str], points: tuple[tuple[float, float], ...]) -> OsmPolygonFeature:
            ring = tuple(projection.to_latlon(point) for point in (*points, points[0]))
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        grassland = polygon("way/grassland", {"natural": "grassland", "surface_kind": "grassland"}, ((600, 650), (700, 650), (700, 750), (600, 750)))
        park = polygon("way/park", {"leisure": "park", "surface_kind": "park"}, ((300, 650), (400, 650), (400, 750), (300, 750)))
        pitch = polygon("way/pitch", {"leisure": "pitch", "sport": "soccer", "surface_kind": "sports_pitch"}, ((420, 650), (520, 650), (520, 750), (420, 750)))
        sand = polygon("way/beach", {"natural": "beach", "surface_kind": "beach"}, ((600, 200), (700, 200), (700, 300), (600, 300)))
        runway = OsmLineFeature(
            "way/runway", {"aeroway": "runway", "width": "35"},
            tuple(projection.to_latlon(point) for point in ((550, 470), (900, 470))),
        )
        dataset = replace(
            dataset, surface_areas=(grassland, park, pitch, sand), aeroway_lines=(runway,), aeroway_areas=(),
        )
        raster = rasterize_osm(dataset, projection, cells=32, include_minor_roads=True)
        report = build_surface_pass(dataset, projection, raster, [60.0] * 1024, [3.0] * 1024, self._spec("osm-surfaces"))
        self.assertGreater(report.mapped_grassland_cells, 0)
        self.assertGreater(report.mapped_park_cells, 0)
        self.assertGreater(report.mapped_sand_cells, 0)
        self.assertGreater(report.indices.count(MATERIAL_INDEX["j"]), 0)
        self.assertGreater(report.indices.count(MATERIAL_INDEX["x"]), 0)
        self.assertGreater(report.aeroway_surface_cells, 0)

    def test_osm_desert_and_sand_landcover_are_mapped_to_sand_surface(self) -> None:
        dataset, projection = self._dataset()

        def polygon(key: str, tags: dict[str, str], points: tuple[tuple[float, float], ...]) -> OsmPolygonFeature:
            ring = tuple(projection.to_latlon(point) for point in (*points, points[0]))
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        desert = polygon("way/desert", {"natural": "desert", "surface_kind": "sand"}, ((600, 200), (700, 200), (700, 300), (600, 300)))
        landcover = polygon("way/sand-cover", {"landcover": "sand", "surface_kind": "sand"}, ((720, 200), (820, 200), (820, 300), (720, 300)))
        dataset = replace(dataset, surface_areas=(desert, landcover))
        raster = rasterize_osm(dataset, projection, cells=32, include_minor_roads=True)
        report = build_surface_pass(dataset, projection, raster, [60.0] * 1024, [3.0] * 1024, self._spec("osm-desert"))
        self.assertGreater(report.mapped_sand_cells, 0)
        self.assertGreater(report.indices.count(MATERIAL_INDEX["s"]), 0)

    def test_asphalt_surface_wins_over_gravel_on_overlapping_ways(self) -> None:
        dataset, projection = self._dataset()
        asphalt = OsmLineFeature(
            "way/asphalt-over-gravel", {"highway": "primary", "surface": "asphalt"},
            tuple(projection.to_latlon(point) for point in ((220.0, 565.0), (980.0, 565.0))),
        )
        gravel = OsmLineFeature(
            "way/gravel-under-asphalt", {"highway": "service", "surface": "gravel"},
            tuple(projection.to_latlon(point) for point in ((220.0, 565.0), (980.0, 565.0))),
        )
        dataset = replace(dataset, roads=(asphalt, gravel), gravel_roads=(gravel,))
        raster = rasterize_osm(dataset, projection, cells=32, include_minor_roads=True)
        report = build_surface_pass(
            dataset, projection, raster, [60.0] * (32 * 32), [3.0] * (32 * 32),
            self._spec("asphalt-over-gravel"),
        )
        self.assertGreater(report.paved_road_cells, 0)
        self.assertEqual(report.gravel_road_cells, 0)
        self.assertNotIn(MATERIAL_INDEX["v"], report.indices)

    def test_gravel_roads_keep_underlying_world_surface_material(self) -> None:
        dataset, projection = self._dataset()
        gravel = OsmLineFeature(
            "way/gravel-surface", {"highway": "service", "surface": "gravel"},
            tuple(projection.to_latlon(point) for point in ((220.0, 565.0), (980.0, 565.0))),
        )
        dataset = replace(dataset, roads=(*dataset.roads, gravel), gravel_roads=(gravel,))
        raster = rasterize_osm(dataset, projection, cells=32, include_minor_roads=True)
        report = build_surface_pass(
            dataset, projection, raster, [60.0] * (32 * 32), [3.0] * (32 * 32),
            self._spec("gravel-surface"),
        )
        self.assertGreater(report.gravel_road_cells, 0)
        self.assertEqual(report.indices.count(MATERIAL_INDEX["v"]), 0)
        self.assertEqual(report.gravel_blend_cells, 0)

    def test_nogova_ice_hockey_pitch_uses_standard_grass_without_generated_site_slab(self) -> None:
        dataset, projection = self._dataset()
        ring = tuple(projection.to_latlon(point) for point in (
            (420.0, 650.0), (520.0, 650.0), (520.0, 750.0), (420.0, 750.0), (420.0, 650.0)
        ))
        pitch = OsmPolygonFeature(
            "way/239731757",
            {"leisure": "pitch", "sport": "ice_hockey", "surface": "asphalt", "surface_kind": "sports_pitch"},
            (GeoPolygon(ring),),
        )
        dataset = replace(dataset, surface_areas=(*dataset.surface_areas, pitch), sites=(*dataset.sites, replace(pitch, tags={**pitch.tags, "site": "sports_pitch"})))
        raster = rasterize_osm(dataset, projection, cells=32, include_minor_roads=True)
        report = build_surface_pass(
            dataset, projection, raster, [60.0] * (32 * 32), [3.0] * (32 * 32),
            self._spec("ice-hockey-pitch"),
        )
        self.assertGreater(report.mapped_sports_cells, 0)
        self.assertIn(MATERIAL_INDEX["y"], report.indices)
        paths = surface_texture_wire_paths("ice_hockey", "nogova")
        self.assertEqual(paths[MATERIAL_INDEX["y"]], paths[MATERIAL_INDEX["g"]])
        self.assertEqual(paths[MATERIAL_INDEX["y"]], r"o\t1.paa")

    def test_optional_colour_reference_is_deterministic(self) -> None:
        dataset, projection = self._dataset()
        raster = rasterize_osm(dataset, projection, cells=32, include_minor_roads=True)
        elevations = [55.0] * (32 * 32)
        slopes = [3.0] * (32 * 32)
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference.png"
            Image.new("RGB", (64, 64), (180, 165, 105)).save(reference)
            spec = self._spec("reference-seed", reference)
            first = build_surface_pass(dataset, projection, raster, elevations, slopes, spec)
            second = build_surface_pass(dataset, projection, raster, elevations, slopes, spec)
            self.assertEqual(first.indices, second.indices)
            self.assertGreater(first.colour_reference_cells, 0)

    def test_everon_profile_uses_only_verified_stock_paths(self) -> None:
        world_name = "abcdefghijklmnopqrst"
        paths = surface_texture_wire_paths(world_name, "everon")
        external = set(external_surface_texture_paths("everon"))
        stock = {
            r"Eden\tn.paa", r"Eden\zbh.paa", r"Eden\bak\bah.pac",
            r"o\l1.paa", r"o\lom2.paa",
            r"o\pole1.paa", r"o\pole2.paa",
        }
        self.assertEqual(external, stock)
        self.assertEqual(len(paths), len(MILESTONE9_MATERIALS))
        self.assertTrue(all(path in stock for path in paths))
        self.assertTrue(all(not path.startswith(world_name + r"\data") for path in paths))
        self.assertEqual(paths[MATERIAL_INDEX["a"]], r"o\pole1.paa")
        self.assertEqual(paths[MATERIAL_INDEX["b"]], r"o\pole2.paa")
        self.assertEqual(paths[MATERIAL_INDEX["c"]], r"Eden\zbh.paa")
        self.assertEqual(paths[MATERIAL_INDEX["g"]], r"Eden\zbh.paa")
        self.assertEqual(paths[MATERIAL_INDEX["f"]], r"Eden\zbh.paa")
        self.assertEqual(paths[MATERIAL_INDEX["e"]], r"Eden\zbh.paa")
        self.assertEqual(paths[MATERIAL_INDEX["r"]], r"o\l1.paa")
        self.assertEqual(paths[MATERIAL_INDEX["k"]], r"o\lom2.paa")
        self.assertEqual(paths[MATERIAL_INDEX["p"]], r"Eden\tn.paa")

    def test_nogova_profile_uses_farmland_tiles_for_farmland(self) -> None:
        world_name = "abcdefghijklmnopqrst"
        nogova_paths = surface_texture_wire_paths(world_name, "nogova")
        everon_paths = surface_texture_wire_paths(world_name, "everon")
        self.assertNotEqual(nogova_paths, everon_paths)
        self.assertIsNot(NOGOVA_SURFACE_TEXTURES, EVERON_SURFACE_TEXTURES)
        expected = [
            NOGOVA_GROUND_TEXTURE_CYCLE[index % len(NOGOVA_GROUND_TEXTURE_CYCLE)]
            for index, _material in enumerate(MILESTONE9_MATERIALS)
        ]
        expected[MATERIAL_INDEX["a"]] = r"o\pole1.paa"
        expected[MATERIAL_INDEX["b"]] = r"o\pole2.paa"
        expected[MATERIAL_INDEX["y"]] = expected[MATERIAL_INDEX["g"]]
        expected[MATERIAL_INDEX["x"]] = r"o\ps.paa"
        self.assertEqual(nogova_paths, tuple(expected))
        self.assertEqual(nogova_paths[MATERIAL_INDEX["a"]], r"o\pole1.paa")
        self.assertEqual(nogova_paths[MATERIAL_INDEX["b"]], r"o\pole2.paa")
        self.assertEqual(
            nogova_paths[MATERIAL_INDEX["j"]],
            NOGOVA_GROUND_TEXTURE_CYCLE[MATERIAL_INDEX["j"] % len(NOGOVA_GROUND_TEXTURE_CYCLE)],
        )
        self.assertEqual(nogova_paths[MATERIAL_INDEX["y"]], nogova_paths[MATERIAL_INDEX["g"]])
        self.assertEqual(nogova_paths[MATERIAL_INDEX["y"]], r"o\t1.paa")
        self.assertEqual(nogova_paths[MATERIAL_INDEX["x"]], r"o\ps.paa")
        self.assertEqual(
            set(external_surface_texture_paths("nogova")),
            {*NOGOVA_GROUND_TEXTURE_CYCLE, r"o\pole1.paa", r"o\pole2.paa", r"o\ps.paa"},
        )
        self.assertNotIn(r"o\b1.paa", external_surface_texture_paths("nogova"))

    def test_malden_profile_reuses_basic_ground_for_farmland(self) -> None:
        world_name = "abcdefghijklmnopqrst"
        paths = surface_texture_wire_paths(world_name, "malden")
        self.assertFalse(external_surface_texture_paths("malden"))
        self.assertEqual(len(paths), len(MILESTONE9_MATERIALS))
        self.assertTrue(all(path.startswith(world_name + r"\data") for path in paths))
        grass = paths[MATERIAL_INDEX["g"]]
        self.assertEqual(paths[MATERIAL_INDEX["a"]], grass)
        self.assertEqual(paths[MATERIAL_INDEX["b"]], grass)
        self.assertEqual(paths[MATERIAL_INDEX["c"]], grass)

    def test_malden_forest_profile_resolves_original_cwc_vegetation_defaults(self) -> None:
        spec = Milestone9Spec(source_dir=Path("unused"), forest_profile="malden")
        resolved = _resolved_forest_profile_models(spec)
        self.assertEqual(resolved["forest_tree_model"], MALDEN_FOREST_BLOCK_MODEL)
        self.assertEqual(resolved["forest_single_tree_model"], MALDEN_SINGLE_TREE_MODEL)
        self.assertEqual(tuple(resolved["forest_roadside_tree_models"]), MALDEN_ROADSIDE_TREE_MODELS)
        self.assertEqual(tuple(resolved["forest_roadside_bush_models"]), MALDEN_BUSH_MODELS)
        self.assertEqual(tuple(resolved["steep_hill_bush_models"]), MALDEN_BUSH_MODELS)

    def test_nogova_resistance_blocks_resolve_nogova_individual_vegetation(self) -> None:
        spec = Milestone9Spec(
            source_dir=Path("unused"),
            forest_profile="everon",
            forest_tree_model=r"o\tree\les_nw_ctver_pruhozi_T1.p3d",
        )
        resolved = _resolved_forest_profile_models(spec)
        self.assertEqual(resolved["forest_single_tree_model"], NOGOVA_SINGLE_TREE_MODEL)
        self.assertEqual(resolved["forest_roadside_tree_model"], NOGOVA_ROADSIDE_TREE_MODEL)
        self.assertEqual(tuple(resolved["forest_roadside_tree_models"]), NOGOVA_ROADSIDE_TREE_MODELS)
        self.assertEqual(tuple(resolved["forest_roadside_bush_models"]), NOGOVA_BUSH_MODELS)
        self.assertEqual(tuple(resolved["steep_hill_bush_models"]), NOGOVA_BUSH_MODELS)

    def test_desert_profile_is_packaged_without_changing_everon_paths(self) -> None:
        world_name = "abcdefghijklmnopqrst"
        desert_paths = surface_texture_wire_paths(world_name, "desert")
        everon_paths = surface_texture_wire_paths(world_name, "everon")
        self.assertFalse(external_surface_texture_paths("desert"))
        self.assertEqual(len(desert_paths), len(MILESTONE9_MATERIALS))
        self.assertTrue(all(path.startswith(world_name + r"\data") for path in desert_paths))
        self.assertEqual(everon_paths[MATERIAL_INDEX["s"]], r"Eden\bak\bah.pac")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generated = write_surface_textures(root / "generated", "cwr_tex", "generated", "seed", 32)
            desert = write_surface_textures(root / "desert", "cwr_tex", "desert", "seed", 32)
            generated_sand = next(path for path in generated if path.name == "s.paa")
            desert_sand = next(path for path in desert if path.name == "s.paa")
            self.assertNotEqual(generated_sand.read_bytes(), desert_sand.read_bytes())

    def test_placement_overlay_uses_the_active_wrp_material_table(self) -> None:
        raster = SimpleNamespace(forest=(True, True, True, True))
        nonroads = SimpleNamespace(objects=(
            SimpleNamespace(model_path=r"data3d\forest.p3d", x=25.0, z=25.0),
            SimpleNamespace(model_path=r"world\i\rock_a.p3d", x=75.0, z=25.0),
        ))
        base_spec = dict(
            cells=2, cell_size=50.0, surface_pass_enabled=True,
            forest_tree_model=r"data3d\forest.p3d", forest_everon_steep_model="",
            forest_tree_spacing=50.0, name="world",
        )

        legacy_index = {material.code: index for index, material in enumerate(OSM_MATERIALS)}
        legacy, _report, _counts = _placement_driven_surface_overlay(
            (legacy_index["g"],) * 4, None, nonroads, raster,
            SimpleNamespace(**base_spec, surface_ground_mode="milestone8", rock_slope_degrees=44.0),
            slopes=(55.0,) * 4,
        )
        self.assertEqual(legacy[0], legacy_index["f"])
        self.assertEqual(legacy[1], legacy_index["r"])
        self.assertTrue(all(index < len(OSM_MATERIALS) for index in legacy))

        expanded, _report, _counts = _placement_driven_surface_overlay(
            (MATERIAL_INDEX["g"],) * 4, None, nonroads, raster,
            SimpleNamespace(**base_spec, surface_ground_mode="milestone9", rock_slope_degrees=44.0, surface_steep_slope_degrees=52.0),
            slopes=(55.0,) * 4,
        )
        self.assertEqual(expanded[0], MATERIAL_INDEX["f"])
        self.assertEqual(expanded[1], MATERIAL_INDEX["k"])

    def test_surface_pass_does_not_use_python_random_module(self) -> None:
        module_path = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "surface_pass.py"
        source = module_path.read_text(encoding="utf-8")
        self.assertNotIn("import random", source)
        self.assertNotIn("from random", source)
        self.assertIn("blake2s", source)




class RoadPieceFittingTests(unittest.TestCase):

    def test_default_object_budgets_cover_dense_full_sized_worlds(self) -> None:
        source_spec = Milestone9Spec(source_dir=Path("unused"))
        runtime_spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 1.0, 1.0)
        )
        self.assertEqual(source_spec.max_road_objects, 1024000)
        self.assertEqual(runtime_spec.max_road_objects, 1024000)
        self.assertEqual(source_spec.max_buildings, 100000)
        self.assertEqual(runtime_spec.max_buildings, 100000)
        self.assertEqual(source_spec.max_forest_objects, 500000)
        self.assertEqual(runtime_spec.max_forest_objects, 500000)
        self.assertEqual(source_spec.forest_undergrowth_maximum_objects, 120000)
        self.assertEqual(runtime_spec.forest_undergrowth_maximum_objects, 120000)
        self.assertTrue(source_spec.include_minor_roads)
        self.assertTrue(runtime_spec.include_minor_roads)
        self.assertEqual(source_spec.ground_texture_profile, "nogova")
        self.assertEqual(runtime_spec.ground_texture_profile, "nogova")
        self.assertTrue(source_spec.bus_stops_enabled)
        self.assertTrue(runtime_spec.bus_stops_enabled)
        self.assertTrue(source_spec.cemeteries_enabled)
        self.assertTrue(runtime_spec.cemeteries_enabled)
        self.assertEqual(source_spec.maximum_grave_objects, 12000)
        self.assertEqual(runtime_spec.maximum_grave_objects, 12000)
        self.assertEqual(source_spec.church_ground_clearance, 3.00)
        self.assertEqual(runtime_spec.church_ground_clearance, 3.00)
        self.assertEqual(source_spec.maximum_steep_hill_bush_objects, 80000)
        self.assertEqual(runtime_spec.maximum_steep_hill_bush_objects, 80000)
        self.assertEqual(source_spec.maximum_wetland_reed_objects, 100000)
        self.assertEqual(runtime_spec.maximum_wetland_reed_objects, 100000)
        self.assertEqual(source_spec.maximum_residential_infill_buildings, 1500)
        self.assertEqual(runtime_spec.maximum_residential_infill_buildings, 1500)
        self.assertEqual(source_spec.residential_infill_spacing, 68.0)
        self.assertEqual(runtime_spec.residential_infill_spacing, 68.0)
        self.assertEqual(source_spec.residential_infill_road_clearance, 0.5)
        self.assertEqual(runtime_spec.residential_infill_road_clearance, 0.5)
        self.assertTrue(source_spec.overture_buildings_enabled)
        self.assertTrue(runtime_spec.overture_buildings_enabled)
        self.assertEqual(source_spec.bus_stop_model, r"o\misc\aut_z_st.p3d")
        self.assertEqual(source_spec.bus_stop_footprint, 1.6)
        self.assertEqual(runtime_spec.bus_stop_footprint, 1.6)
        self.assertEqual(source_spec.bus_stop_ground_clearance, 0.12)
        self.assertEqual(runtime_spec.bus_stop_ground_clearance, 0.12)
        self.assertEqual(source_spec.grave_ground_clearance, 0.12)
        self.assertEqual(runtime_spec.grave_ground_clearance, 0.12)
        self.assertEqual(source_spec.stock_hedge_models, STOCK_HEDGE_MODELS)
        self.assertEqual(runtime_spec.stock_hedge_models, STOCK_HEDGE_MODELS)
        self.assertEqual(source_spec.stock_wall_models, STOCK_WALL_MODELS)
        self.assertEqual(runtime_spec.stock_wall_models, STOCK_WALL_MODELS)
        self.assertEqual(source_spec.stock_metal_fence_models, STOCK_METAL_FENCE_MODELS)
        self.assertEqual(runtime_spec.stock_metal_fence_models, STOCK_METAL_FENCE_MODELS)
        self.assertEqual(source_spec.forest_road_clearance, 0.0)
        self.assertEqual(runtime_spec.forest_road_clearance, 0.0)
        self.assertEqual(source_spec.forest_building_clearance, 1.0)
        self.assertEqual(source_spec.forest_single_tree_footprint, 2.0)
        self.assertEqual(runtime_spec.forest_single_tree_footprint, 2.0)
        self.assertEqual(source_spec.forest_single_tree_maximum_float, 0.5)
        self.assertEqual(runtime_spec.forest_single_tree_maximum_float, 0.5)
        self.assertFalse(source_spec.procedural_bridges)
        self.assertFalse(runtime_spec.procedural_bridges)

    def test_milestone9_cli_uses_the_same_expanded_object_budgets(self) -> None:
        args = _parser().parse_args([
            "milestone9",
            "--output",
            "build/test",
            "--source-dir",
            "source-data/test",
        ])
        self.assertEqual(args.max_road_objects, 1024000)
        self.assertEqual(args.max_buildings, 100000)
        self.assertEqual(args.max_forest_objects, 500000)
        self.assertEqual(args.forest_undergrowth_max_objects, 120000)
        self.assertTrue(args.include_minor_roads)
        self.assertEqual(args.ground_textures, "nogova")
        self.assertTrue(args.bus_stops_enabled)
        self.assertIsNone(args.osm_asset_map)
        self.assertTrue(args.cemeteries_enabled)
        self.assertEqual(args.max_grave_objects, 12000)
        self.assertEqual(args.church_ground_clearance, 3.00)
        self.assertEqual(args.max_steep_hill_bush_objects, 80000)
        self.assertEqual(args.max_wetland_reed_objects, 100000)
        self.assertEqual(args.max_residential_infill_buildings, 1500)
        self.assertEqual(args.residential_infill_spacing, 68.0)
        self.assertEqual(args.residential_infill_road_clearance, 0.5)
        self.assertTrue(args.overture_buildings_enabled)
        self.assertIsNone(args.overture_buildings_geojson)
        self.assertEqual(args.bus_stop_model, r"o\misc\aut_z_st.p3d")
        self.assertEqual(args.bus_stop_footprint, 1.6)
        self.assertEqual(args.bus_stop_ground_clearance, 0.12)
        self.assertEqual(args.grave_ground_clearance, 0.12)
        self.assertEqual(args.forest_road_clearance, 0.0)
        self.assertEqual(args.forest_building_clearance, 1.0)
        self.assertEqual(args.forest_single_tree_footprint, 2.0)
        self.assertEqual(args.forest_single_tree_max_float, 0.5)
        self.assertTrue(args.forest_severe_hill_fallback)
        self.assertEqual(args.forest_severe_hill_relief, 5.0)
        self.assertEqual(args.forest_severe_hill_trees_per_block, 10)
        self.assertEqual(args.forest_polygon_sink_fraction, 0.5)
        self.assertEqual(args.bridge_module_length, 30.0)
        self.assertFalse(args.procedural_bridges)
        self.assertAlmostEqual(args.bridge_deck_clearance, 1.25)
        self.assertFalse(args.procedural_building_interiors)

        interiors = _parser().parse_args([
            "milestone9",
            "--output", "build/test",
            "--source-dir", "source-data/test",
            "--procedural-building-interiors",
        ])
        self.assertTrue(interiors.procedural_building_interiors)

        procedural_bridges = _parser().parse_args([
            "milestone9",
            "--output", "build/test",
            "--source-dir", "source-data/test",
            "--procedural-bridges",
            "--bridge-module-length", "18",
        ])
        self.assertTrue(procedural_bridges.procedural_bridges)
        self.assertEqual(procedural_bridges.bridge_module_length, 18.0)
        stock_bridges = _parser().parse_args([
            "milestone9",
            "--output", "build/test",
            "--source-dir", "source-data/test",
            "--stock-bridges",
        ])
        self.assertFalse(stock_bridges.procedural_bridges)

    def test_milestone9_cli_accepts_osm_asset_mapping_file(self) -> None:
        args = _parser().parse_args([
            "milestone9", "--output", "build/test", "--source-dir", "source-data/test",
            "--osm-asset-map", "asset-map.json",
        ])
        self.assertEqual(args.osm_asset_map, Path("asset-map.json"))

    def test_milestone9_cli_accepts_default_and_disable_names(self) -> None:
        defaults = _parser().parse_args([
            "milestone9",
            "--output", "build/test",
            "--source-dir", "source-data/test",
        ])
        self.assertTrue(defaults.bus_stops_enabled)
        self.assertTrue(defaults.include_minor_roads)

        disabled = _parser().parse_args([
            "milestone9",
            "--output", "build/test",
            "--source-dir", "source-data/test",
            "--no-bus-stop-signs",
            "--no-minor-roads",
            "--no-overture-buildings",
            "--no-severe-hill-forest-fallback",
        ])
        self.assertFalse(disabled.bus_stops_enabled)
        self.assertFalse(disabled.include_minor_roads)
        self.assertFalse(disabled.overture_buildings_enabled)
        self.assertFalse(disabled.forest_severe_hill_fallback)

    def test_stock_nogova_bridge_mode_omits_ordinary_road_pieces_on_bridge_way(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/stock-bridge-road",
            {"highway": "secondary", "bridge": "yes", "surface": "asphalt"},
            tuple(projection.to_latlon(point) for point in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="stock-bridge-road", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,),
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_stock_bridge_road", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=8, cell_size=25.0,
            max_road_objects=100, procedural_bridges=False, strict_assets=False,
        )
        stock = fit_road_objects(dataset, projection, (0.0,) * 64, spec)
        self.assertEqual(stock.objects, ())
        procedural = fit_road_objects(
            dataset, projection, (0.0,) * 64, replace(spec, procedural_bridges=True)
        )
        self.assertGreater(len(procedural.objects), 0)

    def test_stock_roads_reject_a_budget_that_would_emit_a_partial_network(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        road = OsmLineFeature(
            "way/complete-network",
            {"highway": "primary", "surface": "asphalt"},
            tuple(
                projection.to_latlon(point)
                for point in ((100.0, 500.0), (900.0, 500.0))
            ),
        )
        dataset = OsmDataset(
            source_generator="road-budget",
            element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40,
            cell_size=25.0,
            max_road_objects=2,
            strict_assets=False,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"road object budget is too small.*requires .*increase --max-road-objects",
        ):
            fit_road_objects(
                dataset,
                projection,
                [0.0] * (spec.cells * spec.cells),
                spec,
            )

    def test_zero_road_budget_disables_roads_without_truncation(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        road = OsmLineFeature(
            "way/disabled",
            {"highway": "primary", "surface": "asphalt"},
            tuple(
                projection.to_latlon(point)
                for point in ((100.0, 500.0), (900.0, 500.0))
            ),
        )
        dataset = OsmDataset(
            source_generator="road-disabled",
            element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40,
            cell_size=25.0,
            max_road_objects=0,
            strict_assets=False,
        )
        result = fit_road_objects(
            dataset,
            projection,
            [0.0] * (spec.cells * spec.cells),
            spec,
        )
        self.assertEqual(result.objects, ())
        self.assertFalse(result.truncated)

    def test_gravel_surfaces_use_generated_gravel_road_family(self) -> None:
        spec = _Milestone9PlayabilitySpec(
            name="cwr_gravel",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=40,
            cell_size=25.0,
            strict_assets=False,
        )
        self.assertTrue(spec.procedural_gravel_roads)
        for surface in ("gravel", "fine_gravel", "compacted", "pebblestone", "unpaved"):
            tags = {"highway": "service", "surface": surface}
            self.assertTrue(road_is_gravel(tags))
            self.assertEqual(
                road_model_for_tags(spec, tags),
                rf"{spec.name}\i\gravel25.p3d",
            )
        self.assertEqual(
            road_model_for_tags(spec, {"highway": "service", "surface": "earth"}),
            spec.dirt_road_model,
        )
        self.assertEqual(
            road_model_for_tags(spec, {"highway": "service", "surface": "asphalt"}),
            spec.paved_road_model,
        )

    def test_generated_gravel_assets_are_embedded_in_the_world_pbo(self) -> None:
        from cwr_worldgen.pbo import pack_directory
        from cwr_worldgen.procedural_infrastructure import ProceduralInfrastructureLibrary

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "config.cpp").write_text("class CfgPatches {};\n", encoding="ascii")
            (source / "cwr_gravel_bundle.wrp").write_bytes(b"4WVR")

            library = ProceduralInfrastructureLibrary("cwr_gravel_bundle")
            library.register_models((
                r"cwr_gravel_bundle\i\gravel25.p3d",
                r"cwr_gravel_bundle\i\gravel12.p3d",
                r"cwr_gravel_bundle\i\gravel6.p3d",
            ))
            generation = library.write_assets(
                source, root / "infrastructure-asset-catalogue.json"
            )
            pbo_path = root / "cwr_gravel_bundle.pbo"
            pack_directory(source, pbo_path)

            layout = _verify_single_world_pbo_layout(
                pbo_path, "cwr_gravel_bundle", generation
            )
            entries = {entry.name for entry in read_pbo(pbo_path)}
            self.assertEqual(layout["mode"], "single_world_pbo")
            self.assertFalse(layout["separate_road_pbo"] )
            self.assertEqual(len(layout["generated_road_models"]), 3)
            self.assertEqual(layout["generated_road_textures"], [r"i\g.paa"] )
            self.assertIn("cwr_gravel_bundle.wrp", entries)
            self.assertIn(r"i\gravel25.p3d", entries)
            self.assertIn(r"i\gravel12.p3d", entries)
            self.assertIn(r"i\gravel6.p3d", entries)
            self.assertIn(r"i\g.paa", entries)
            self.assertIn(r"i\infrastructure.json", entries)
            self.assertEqual(list(root.glob("*.pbo")), [pbo_path])

    def test_procedural_rocks_reuse_o_pbo_l1_and_lom2_textures(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralInfrastructureLibrary("cwr_rock_textures")
            first = library.rock_model("rock_group_0", 9.0, 9.0)
            second = library.rock_model("rock_group_1", 9.0, 9.0)
            assets = library.write_assets(root, root / "infrastructure.json")
            self.assertNotIn("i/rock.paa", assets.texture_files)
            first_path = root / first.split("\\", 1)[1].replace("\\", "/")
            second_path = root / second.split("\\", 1)[1].replace("\\", "/")
            self.assertIn(r"o\l1.paa", inspect_mlod(first_path).texture_paths)
            self.assertIn(r"o\lom2.paa", inspect_mlod(second_path).texture_paths)

    def test_world_generation_uses_stock_ingame_stones_not_generated_rock_p3ds(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        rock_polygon = OsmPolygonFeature(
            "way/rock", {"natural": "rock"},
            (GeoPolygon(tuple(projection.to_latlon(point) for point in (
                (40.0, 40.0), (160.0, 40.0), (160.0, 160.0), (40.0, 160.0), (40.0, 40.0)
            ))),),
        )
        dataset = OsmDataset(
            source_generator="stock-stones", element_count=1, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(), rural_vegetation=(rock_polygon,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="stock_stones", heightmap_path=Path("unused.png"), bbox=(0,0,1,1),
            cells=8, cell_size=25.0, max_road_objects=0, max_buildings=0, max_forest_objects=0,
            rural_vegetation_enabled=True, maximum_rural_vegetation_objects=100,
            meadow_grass_enabled=False, wetland_reeds_enabled=False, barriers_enabled=False,
            bridges_enabled=False, forest_undergrowth_enabled=False, forest_border_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(dataset, projection, raster, (5.0,) * 64, spec, include_roads=False)
        stone_paths = {path.casefold() for path in STOCK_STONE_MODELS}
        placed = [obj for obj in result.objects if obj.model_path.casefold() in stone_paths]
        self.assertTrue(placed)
        self.assertFalse(any(r"\i\rock_" in obj.model_path.casefold() for obj in result.objects))

    def test_generated_infrastructure_texture_paths_fit_full_world_name(self) -> None:
        world_name = "abcdefghijklmnopqrst"
        self.assertEqual(len(world_name), 20)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralInfrastructureLibrary(world_name)
            library.register_model(rf"{world_name}\i\gravel6.p3d")
            library.barrier_model("fence", 6.0)
            library.barrier_model("wall", 6.0)
            library.barrier_model("hedge", 6.0)
            library.bridge_model("single", 7.0, 18.0)
            for subtype in ("power_pole", "power_tower", "water_tower"):
                library.utility_model(subtype)
            assets = library.write_assets(root, root / "infrastructure.json")
            self.assertIn("i/g.paa", assets.texture_files)
            self.assertIn("i/b.paa", assets.texture_files)
            self.assertIn("i/f.paa", assets.texture_files)
            self.assertIn("i/w.paa", assets.texture_files)
            self.assertIn("i/h.paa", assets.texture_files)
            for relative in assets.model_files:
                summary = inspect_mlod(root / relative)
                for texture_path in summary.texture_paths:
                    self.assertLessEqual(len(texture_path.encode("ascii")), 31)

    def test_generated_osm_utility_models_are_packable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralInfrastructureLibrary("cwr_osm_utilities")
            for subtype in ("power_pole", "power_tower", "water_tower"):
                library.register_model(library.utility_model(subtype))
            assets = library.write_assets(root, root / "infrastructure.json")
            self.assertEqual(assets.generated_variants, 3)
            self.assertTrue((root / "i" / "util_power_pole.p3d").is_file())
            self.assertTrue((root / "i" / "util_power_tower.p3d").is_file())
            self.assertTrue((root / "i" / "util_water_tower.p3d").is_file())
            for relative in assets.model_files:
                if "/util_" not in relative:
                    continue
                summary = inspect_mlod(root / relative)
                self.assertGreaterEqual(len(summary.resolutions), 3)

    def test_default_gravel_roads_use_generated_gravel_assets(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)
        road = OsmLineFeature(
            "way/gravel",
            {"highway": "service", "surface": "gravel"},
            tuple(projection.to_latlon(point) for point in ((100.0, 500.0), (900.0, 500.0))),
        )
        dataset = OsmDataset(
            source_generator="gravel-road",
            element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_gravel_assets",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=40,
            cell_size=25.0,
            strict_assets=False,
        )
        report = fit_road_objects(dataset, projection, (2.0,) * (spec.cells * spec.cells), spec)
        self.assertTrue(report.objects)
        paths = {obj.model_path.casefold() for obj in report.objects}
        self.assertTrue(all(path.startswith(r"cwr_gravel_assets\i\gravel") for path in paths))
        self.assertTrue(any(path.endswith("gravel6.p3d") or path.endswith("gravel12.p3d") or path.endswith("gravel25.p3d") for path in paths))

    def test_road_piece_origins_never_extend_outside_world_border(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)
        roads = (
            OsmLineFeature(
                "way/gravel-border", {"highway": "service", "surface": "gravel"},
                tuple(projection.to_latlon(point) for point in ((-12.0, 500.0), (45.0, 500.0))),
            ),
            OsmLineFeature(
                "way/paved-border", {"highway": "secondary", "surface": "asphalt"},
                tuple(projection.to_latlon(point) for point in ((500.0, 970.0), (500.0, 1015.0))),
            ),
        )
        dataset = OsmDataset(
            source_generator="road-world-border", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=roads,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_border_roads", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=40, cell_size=25.0,
            strict_assets=False,
        )
        report = fit_road_objects(
            dataset, projection, (2.0,) * (spec.cells * spec.cells), spec
        )
        self.assertTrue(report.objects)
        self.assertTrue(all(
            0.0 <= obj.x < spec.world_size and 0.0 <= obj.z < spec.world_size
            for obj in report.objects
        ))

    def test_generated_gravel_uses_a_deterministic_procedural_texture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = ProceduralInfrastructureLibrary("cwr_proc_gravel")
            first.register_model(r"cwr_proc_gravel\i\gravel6.p3d")
            first_assets = first.write_assets(root / "a", root / "a" / "infrastructure.json")
            second = ProceduralInfrastructureLibrary("cwr_proc_gravel")
            second.register_model(r"cwr_proc_gravel\i\gravel6.p3d")
            second.write_assets(root / "b", root / "b" / "infrastructure.json")
            texture_a = root / "a" / "i" / "g.paa"
            texture_b = root / "b" / "i" / "g.paa"
            self.assertIn("i/g.paa", first_assets.texture_files)
            self.assertEqual(hashlib.sha256(texture_a.read_bytes()).hexdigest(), hashlib.sha256(texture_b.read_bytes()).hexdigest())
            info = inspect_paa(texture_a)
            self.assertEqual((info.width, info.height), (512, 512))
            self.assertEqual(info.minimum_mip_width, 4)
            self.assertTrue((root / "a" / "i" / "gravel-source-surfaces.txt").is_file())
            catalogue = json.loads((root / "a" / "infrastructure.json").read_text(encoding="utf-8"))
            source = catalogue["gravel_texture_source"]
            self.assertEqual(source["type"], "bundled-reference")
            self.assertEqual(source["texture_recipe"], "reference-gravel-photo-clean-edge-v3")
            self.assertEqual(source["edge_blend"], "clean DXT1 cutout plus smoothly irregular model edge")
            self.assertEqual(source["texture_size"], 512)
            self.assertEqual(source["map_symbol"], "road")
            self.assertEqual(source["surface_rule_values"]["default"], 0.20)
            self.assertEqual(source["surface_rule_values"]["st??????"], 0.50)

        image = create_gravel_road_texture_image(128)
        self.assertEqual(image.mode, "RGBA")
        alpha = image.getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        self.assertEqual(alpha_min, 0)
        self.assertEqual(alpha_max, 255)
        # The road centre stays opaque while both outer verges contain holes for
        # the world's own terrain texture to show through.
        self.assertEqual(alpha.getpixel((32, 32)), 255)
        self.assertEqual(alpha.getpixel((0, 32)), 0)
        # No ordered-dither fringe: each row has one contiguous transparent
        # strip at each outside edge, then stays opaque across the road.
        row = [alpha.getpixel((x, 64)) for x in range(128)]
        first_opaque = next(index for index, value in enumerate(row) if value == 255)
        last_opaque = len(row) - 1 - next(index for index, value in enumerate(reversed(row)) if value == 255)
        self.assertTrue(all(value == 0 for value in row[:first_opaque]))
        self.assertTrue(all(value == 255 for value in row[first_opaque:last_opaque + 1]))
        self.assertTrue(all(value == 0 for value in row[last_opaque + 1:]))

    def test_generated_gravel_visual_is_a_terrain_hugging_ribbon(self) -> None:
        key = InfrastructureModelKey("road", "gravel25", 60, 245)
        visual, map_geometry, roadway, land = _road_lods(
            key, r"cwr_gravel\i\gravel.paa"
        )
        self.assertGreater(len(visual.faces), 2)
        self.assertGreater(len(visual.points), 4)
        interior_heights = [point[1] for point in visual.points[2:-2]]
        self.assertTrue(all(
            abs(value - GENERATED_GRAVEL_VISUAL_TOP_METRES) < 1e-9
            for value in interior_heights
        ))
        self.assertTrue(all(point[1] < GENERATED_GRAVEL_VISUAL_TOP_METRES for point in visual.points[:2]))
        self.assertTrue(all(point[1] < GENERATED_GRAVEL_VISUAL_TOP_METRES for point in visual.points[-2:]))
        self.assertEqual({face.texture for face in visual.faces}, {r"cwr_gravel\i\gravel.paa"})
        # Alpha-bearing gravel may repeat along the road, but never across its
        # width; otherwise the transparent texture edge repeats at U=1 and cuts
        # a grass strip down the middle of a two-repeat road.
        visual_u = [vertex[2] for face in visual.faces for vertex in face.vertices]
        self.assertGreaterEqual(min(visual_u), 0.0)
        self.assertLessEqual(max(visual_u), 1.0)
        self.assertTrue(all(
            abs(point[1] - GENERATED_GRAVEL_VISUAL_TOP_METRES) < 1e-9
            for point in roadway.points
        ))
        self.assertTrue(land.points)
        self.assertEqual(dict(visual.properties).get("class"), "road")
        self.assertEqual(dict(visual.properties).get("map"), "road")
        self.assertEqual(dict(map_geometry.properties).get("map"), "road")
        self.assertEqual(len(map_geometry.faces), 0)

    def test_generated_gravel_surface_sits_exactly_on_flat_ground(self) -> None:
        spec = _Milestone9PlayabilitySpec(
            name="cwr_gravel_flat",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=3,
            cell_size=25.0,
            strict_assets=False,
        )
        obj = _road_object_on_slope(
            1,
            r"cwr_gravel_flat\i\gravel6.p3d",
            (25.0, 37.5),
            (31.0, 37.5),
            (4.0,) * 9,
            spec,
            vertical_offset=0.035,
        )
        self.assertAlmostEqual(
            obj.y + GENERATED_GRAVEL_VISUAL_TOP_METRES,
            4.0,
            places=6,
        )

    def test_generated_gravel_piece_is_not_raised_over_a_terrain_ridge(self) -> None:
        spec = _Milestone9PlayabilitySpec(
            name="cwr_gravel_ridge",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=3,
            cell_size=25.0,
            strict_assets=False,
        )
        # Terrain samples are centred at 12.5, 37.5 and 62.5 metres. The road
        # endpoints sit on the shoulders while its centre crosses the 5 m ridge.
        elevations = (
            0.0, 5.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 5.0, 0.0,
        )
        start = (25.0, 37.5)
        end = (50.0, 37.5)
        obj = _road_object_on_slope(
            1,
            r"cwr_gravel_ridge\i\gravel25.p3d",
            start,
            end,
            elevations,
            spec,
            vertical_offset=0.035,
        )
        centre_terrain = _sample_elevation(
            elevations, spec.cells, spec.cell_size, obj.x, obj.z
        )
        visible_surface = obj.y + GENERATED_GRAVEL_VISUAL_TOP_METRES * math.cos(
            math.radians(obj.pitch_degrees)
        )
        self.assertLessEqual(visible_surface, centre_terrain + 1e-6)

    def test_default_gravel_uses_generated_piece_family(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)
        road = OsmLineFeature(
            "way/gravel-short-pieces",
            {"highway": "service", "surface": "gravel"},
            tuple(
                projection.to_latlon(point)
                for point in ((100.0, 500.0), (900.0, 500.0))
            ),
        )
        dataset = OsmDataset(
            source_generator="gravel-short-pieces",
            element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_gravel_short",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=40,
            cell_size=25.0,
            strict_assets=False,
        )
        report = fit_road_objects(
            dataset,
            projection,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )
        gravel = [obj for obj in report.objects if r"\i\gravel" in obj.model_path.casefold()]
        self.assertGreater(len(gravel), 20)
        self.assertTrue(all(obj.model_path.casefold().startswith(r"cwr_gravel_short\i\gravel") for obj in gravel))

    def test_generated_gravel_does_not_raise_for_a_high_roadside_shoulder(self) -> None:
        spec = _Milestone9PlayabilitySpec(
            name="cwr_gravel_cross_slope",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=3,
            cell_size=25.0,
            strict_assets=False,
        )
        # The road centreline follows the zero-height middle column. The right
        # shoulder rises sharply, which 0.9.84 used to lift the entire slab.
        elevations = (
            0.0, 0.0, 20.0,
            0.0, 0.0, 20.0,
            0.0, 0.0, 20.0,
        )
        raise_metres = _generated_gravel_terrain_raise(
            (25.0, 25.0),
            (25.0, 50.0),
            0.0,
            0.0,
            elevations,
            spec,
            vertical_offset=0.035,
            pitch_degrees=0.0,
        )
        self.assertAlmostEqual(raise_metres, 0.0)

    def test_gravel_is_lower_and_emitted_before_asphalt(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        asphalt = OsmLineFeature(
            "way/asphalt-later", {"highway": "primary", "surface": "asphalt"},
            tuple(projection.to_latlon(point) for point in ((100.0, 600.0), (900.0, 600.0))),
        )
        gravel = OsmLineFeature(
            "way/gravel-earlier", {"highway": "service", "surface": "gravel"},
            tuple(projection.to_latlon(point) for point in ((100.0, 400.0), (900.0, 400.0))),
        )
        dataset = OsmDataset(
            source_generator="road-layer-priority", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(asphalt, gravel), gravel_roads=(gravel,),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40, cell_size=25.0, strict_assets=False,
        )
        report = fit_road_objects(
            dataset, projection, [0.0] * (spec.cells * spec.cells), spec
        )
        gravel_objects = [obj for obj in report.objects if r"\i\gravel" in obj.model_path.casefold()]
        asphalt_objects = [obj for obj in report.objects if r"\road\sil" in obj.model_path.casefold()]
        self.assertTrue(gravel_objects)
        self.assertTrue(asphalt_objects)
        self.assertLess(max(obj.y for obj in gravel_objects), min(obj.y for obj in asphalt_objects))
        self.assertLess(
            min(report.objects.index(obj) for obj in gravel_objects),
            min(report.objects.index(obj) for obj in asphalt_objects),
        )

    def test_service_and_unclassified_default_to_dirt_with_paved_override(self) -> None:
        self.assertTrue(road_is_dirt({"highway": "service"}))
        self.assertTrue(road_is_dirt({"highway": "unclassified"}))
        self.assertTrue(road_is_dirt({"highway": "service", "surface": "gravel"}))
        self.assertTrue(road_is_gravel({"highway": "service", "surface": "unpaved"}))
        self.assertTrue(road_is_dirt({"highway": "service", "surface": "unpaved"}))
        self.assertFalse(road_is_gravel({"highway": "service", "surface": "earth"}))
        self.assertFalse(road_is_dirt({"highway": "service", "surface": "asphalt"}))
        self.assertFalse(road_is_dirt({"highway": "unclassified", "surface": "paved"}))
        self.assertFalse(road_is_dirt({"highway": "residential"}))

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        roads = (
            OsmLineFeature(
                "way/service",
                {"highway": "service"},
                tuple(projection.to_latlon(point) for point in ((100.0, 200.0), (250.0, 200.0))),
            ),
            OsmLineFeature(
                "way/unclassified",
                {"highway": "unclassified"},
                tuple(projection.to_latlon(point) for point in ((100.0, 400.0), (250.0, 400.0))),
            ),
            OsmLineFeature(
                "way/service-paved",
                {"highway": "service", "surface": "asphalt"},
                tuple(projection.to_latlon(point) for point in ((100.0, 600.0), (250.0, 600.0))),
            ),
        )
        dataset = OsmDataset(
            source_generator="milestone9-road-surface-test",
            element_count=len(roads),
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=roads,
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40, cell_size=25.0, strict_assets=False, bus_stops_enabled=True,
        )
        result = fit_road_objects(dataset, projection, [0.0] * (spec.cells * spec.cells), spec)
        model_paths = {obj.model_path.casefold() for obj in result.objects}
        self.assertTrue(any(r"\ces" in path for path in model_paths))
        self.assertTrue(any(r"\sil" in path for path in model_paths))

    def test_short_dirt_service_branch_survives_asphalt_junction_cap(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        node = (500.0, 500.0)

        def road(key: str, tags: dict[str, str], points: tuple[tuple[float, float], ...]) -> OsmLineFeature:
            return OsmLineFeature(
                key, tags, tuple(projection.to_latlon(point) for point in points)
            )

        roads = (
            road("way/asphalt-west", {"highway": "primary", "surface": "asphalt"}, ((250.0, 500.0), node)),
            road("way/asphalt-east", {"highway": "primary", "surface": "asphalt"}, (node, (750.0, 500.0))),
            road("way/service-1295713279", {"highway": "service"}, (node, (500.0, 502.35))),
        )
        dataset = OsmDataset(
            source_generator="short-mixed-junction",
            element_count=len(roads),
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=roads,
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40,
            cell_size=25.0,
            strict_assets=False,
        )
        result = fit_road_objects(
            dataset, projection, [0.0] * (spec.cells * spec.cells), spec
        )
        dirt_objects = [obj for obj in result.objects if r"\ces" in obj.model_path.casefold()]
        self.assertTrue(dirt_objects)
        self.assertTrue(any(obj.model_path.casefold().endswith(r"ces6.p3d") for obj in dirt_objects))

    def test_stock_short_pieces_caps_and_pitch_remove_gaps_and_overlap(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        node = (500.0, 500.0)

        def road(key: str, points: tuple[tuple[float, float], ...]) -> OsmLineFeature:
            return OsmLineFeature(
                key,
                {"highway": "primary", "surface": "asphalt"},
                tuple(projection.to_latlon(point) for point in points),
            )

        roads = (
            road("way/west", ((250.0, 500.0), node)),
            road("way/east", (node, (750.0, 500.0))),
            road("way/north", (node, (500.0, 760.0))),
            road("way/short", ((100.0, 200.0), (118.0, 200.0))),
        )
        dataset = OsmDataset(
            source_generator="milestone9-road-test",
            element_count=len(roads),
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=roads,
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40,
            cell_size=25.0,
            strict_assets=False,
        )
        elevations = [float(x) * 0.35 for _z in range(spec.cells) for x in range(spec.cells)]

        first = fit_road_objects(dataset, projection, elevations, spec)
        second = fit_road_objects(dataset, projection, elevations, spec)

        self.assertEqual(first, second)
        self.assertEqual(first.failed_connections, 0)
        self.assertEqual(first.junction_cap_objects, 1)
        self.assertEqual(first.terrain_filled_junctions, 0)
        self.assertLessEqual(first.maximum_connection_gap, 1e-6)
        self.assertLessEqual(first.maximum_model_overlap_metres, 1e-6)
        self.assertLessEqual(first.maximum_chain_gap, 1e-6)
        self.assertGreater(first.short_piece_objects, 0)
        self.assertGreater(first.maximum_road_pitch_degrees, 0.1)
        models = {obj.model_path.casefold() for obj in first.objects}
        self.assertIn(r"o\road\sil6.p3d", models)
        self.assertTrue(any(path.endswith((r"sil6.p3d", r"sil12.p3d")) for path in models))
        self.assertTrue(any(abs(obj.pitch_degrees) > 0.1 for obj in first.objects))


    def test_road_cut_forest_blocks_use_individual_trees_clear_of_the_road(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        road = OsmLineFeature(
            "way/service-1295713279",
            {"highway": "service"},
            tuple(projection.to_latlon(point) for point in ((0.0, 25.0), (100.0, 25.0))),
        )
        dataset = OsmDataset(
            source_generator="road-cut-forest",
            element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        cells = 8
        forest_mask = [True] * (cells * cells)
        road_mask = [False] * (cells * cells)
        for column in range(cells):
            forest_mask[2 * cells + column] = False
            road_mask[2 * cells + column] = True
        raster = OsmRaster(
            cells=cells,
            water=(False,) * (cells * cells),
            forest=tuple(forest_mask),
            farmland=(False,) * (cells * cells),
            urban=(False,) * (cells * cells),
            roads=tuple(road_mask),
            buildings=(False,) * (cells * cells),
            high_resolution=cells,
            coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_road_cut_forest",
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=cells,
            cell_size=12.5,
            forest_tree_spacing=50.0,
            max_road_objects=0,
            max_buildings=0,
            max_forest_objects=200,
            forest_single_tree_enabled=False,
            forest_undergrowth_enabled=False,
            forest_border_enabled=False,
            steep_hill_bushes_enabled=False,
            ditch_grass_enabled=False,
            barriers_enabled=False,
            bridges_enabled=False,
            rural_vegetation_enabled=False,
            wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (5.0,) * (cells * cells), spec, include_roads=False
        )
        tree_models = {model.casefold() for model in spec.forest_roadside_tree_models}
        bush_models = {model.casefold() for model in spec.forest_roadside_bush_models}
        roadside_trees = [
            obj for obj in result.objects if obj.model_path.casefold() in tree_models
        ]
        roadside_bushes = [
            obj for obj in result.objects if obj.model_path.casefold() in bush_models
        ]
        self.assertEqual(spec.forest_roadside_trees_per_cut_block, 20)
        self.assertEqual(spec.forest_roadside_bushes_per_cut_block, 16)
        self.assertEqual(spec.forest_single_tree_footprint, 2.0)
        self.assertEqual(len(roadside_trees), 40)
        self.assertEqual(len(roadside_bushes), 32)
        self.assertEqual(len(roadside_trees), result.forest_single_tree_objects)
        self.assertGreaterEqual(len({obj.model_path.casefold() for obj in roadside_trees}), 3)
        self.assertGreaterEqual(len({obj.model_path.casefold() for obj in roadside_bushes}), 3)
        corridors = project_road_corridors(dataset, projection, spec)
        self.assertFalse(
            forest_block_intersects_road_corridors(corridors, 50.0, 21.5, block_size=2.0)
        )
        self.assertTrue(
            forest_block_intersects_road_corridors(corridors, 50.0, 21.5, block_size=4.0)
        )
        self.assertTrue(all(
            not forest_block_intersects_road_corridors(
                corridors, obj.x, obj.z, block_size=spec.forest_single_tree_footprint
            )
            for obj in roadside_trees
        ))
        self.assertTrue(all(
            not forest_block_intersects_road_corridors(
                corridors, obj.x, obj.z, block_size=spec.forest_roadside_bush_footprint
            )
            for obj in roadside_bushes
        ))
        self.assertTrue(any(obj.z < 20.0 for obj in roadside_trees))
        self.assertTrue(any(obj.z > 35.0 for obj in roadside_trees))
        nearest_distances = []
        for index, obj in enumerate(roadside_trees):
            nearest_distances.append(min(
                ((obj.x - other.x) ** 2 + (obj.z - other.z) ** 2) ** 0.5
                for other_index, other in enumerate(roadside_trees)
                if other_index != index
            ))
        self.assertGreater(len({round(distance, 2) for distance in nearest_distances}), 12)
        repeated = generate_world_objects(
            dataset, projection, raster, (5.0,) * (cells * cells), spec, include_roads=False
        )
        self.assertEqual(result.objects, repeated.objects)

    def test_long_collinear_osm_way_does_not_create_a_cap_per_vertex(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        points = tuple((100.0 + index * 2.0, 500.0) for index in range(401))
        road = OsmLineFeature(
            "way/many-vertices",
            {"highway": "primary", "surface": "asphalt"},
            tuple(projection.to_latlon(point) for point in points),
        )
        dataset = OsmDataset(
            source_generator="milestone9-many-road-nodes",
            element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40,
            cell_size=25.0,
            strict_assets=False,
        )
        result = fit_road_objects(dataset, projection, [0.0] * (spec.cells * spec.cells), spec)
        self.assertEqual(result.junction_cap_objects, 0)
        self.assertGreaterEqual(result.suppressed_degree_two_caps, 399)
        self.assertEqual(result.road_connection_slot_risk_nodes, 0)
        self.assertEqual(result.failed_connections, 0)
        self.assertLessEqual(result.maximum_chain_gap, spec.road_connection_tolerance)

    def test_nearby_real_junctions_are_not_suppressed(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        first_node = (496.0, 500.0)
        second_node = (504.0, 500.0)

        def road(key: str, points: tuple[tuple[float, float], ...]) -> OsmLineFeature:
            return OsmLineFeature(
                key,
                {"highway": "primary", "surface": "asphalt"},
                tuple(projection.to_latlon(point) for point in points),
            )

        roads = (
            road("way/main", ((250.0, 500.0), first_node, second_node, (750.0, 500.0))),
            road("way/first-branch", (first_node, (496.0, 750.0))),
            road("way/second-branch", (second_node, (504.0, 250.0))),
        )
        dataset = OsmDataset(
            source_generator="milestone9-nearby-junctions",
            element_count=len(roads),
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=roads,
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40, cell_size=25.0, strict_assets=False,
        )
        result = fit_road_objects(dataset, projection, [0.0] * (spec.cells * spec.cells), spec)
        self.assertEqual(result.junction_cap_objects, 2)
        self.assertEqual(result.failed_connections, 0)
        self.assertLessEqual(result.maximum_connection_gap, 1e-6)

    def test_degree_two_city_corner_uses_rounded_short_piece_chain_without_cap(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        road = OsmLineFeature(
            "way/city-corner",
            {"highway": "residential", "surface": "asphalt"},
            tuple(
                projection.to_latlon(point)
                for point in ((250.0, 400.0), (500.0, 400.0), (500.0, 650.0))
            ),
        )
        dataset = OsmDataset(
            source_generator="milestone9-rounded-city-corner",
            element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40, cell_size=25.0, strict_assets=False,
        )
        result = fit_road_objects(dataset, projection, [0.0] * (spec.cells * spec.cells), spec)

        self.assertEqual(result.junction_cap_objects, 0)
        self.assertGreaterEqual(result.short_piece_objects, 3)
        corner_headings = [
            obj.heading_degrees
            for obj in result.objects
            if 485.0 <= obj.x <= 502.0 and 398.0 <= obj.z <= 415.0
        ]
        self.assertTrue(any(15.0 < heading < 80.0 for heading in corner_headings))
        self.assertLessEqual(result.maximum_chain_gap, 1e-6)
        self.assertLessEqual(result.maximum_model_overlap_metres, 1e-6)

    def test_curved_stock_road_uses_true_model_chords_without_overlap(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        points = (
            (350.0, 400.0),
            (362.0, 377.0),
            (381.0, 358.0),
            (404.0, 346.0),
            (430.0, 342.0),
            (456.0, 347.0),
            (479.0, 360.0),
            (497.0, 380.0),
            (508.0, 404.0),
        )
        road = OsmLineFeature(
            "way/curved",
            {"highway": "primary", "surface": "asphalt"},
            tuple(projection.to_latlon(point) for point in points),
        )
        dataset = OsmDataset(
            source_generator="milestone9-curved-road",
            element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40, cell_size=25.0, strict_assets=False,
        )
        result = fit_road_objects(dataset, projection, [0.0] * (spec.cells * spec.cells), spec)
        self.assertGreater(len(result.objects), 3)
        self.assertLessEqual(result.maximum_chain_gap, 1e-6)
        self.assertLessEqual(result.maximum_model_overlap_metres, 1e-6)

    def test_generated_gravel_has_three_metre_turn_piece(self) -> None:
        path = gravel_road_model_path("cwr_tight", 3, 15)
        self.assertEqual(path.casefold(), r"cwr_tight\i\gravel3_r15.p3d")
        variants = road_model_variants(r"cwr_tight\i\gravel25.p3d", 25.0)
        self.assertIn(3, {piece.nominal_length for piece in variants})
        visual, _map_geometry, roadway, _land = _road_lods(
            InfrastructureModelKey("road", "gravel3_r15", 46, 30),
            r"cwr_tight\i\g.paa",
        )
        self.assertTrue(visual.faces)
        self.assertTrue(roadway.faces)

    def test_generated_gravel_has_extra_turn_and_junction_variants(self) -> None:
        self.assertEqual(
            gravel_road_model_path("cwr_tight", 3, 45).casefold(),
            r"cwr_tight\i\gravel3_r45.p3d",
        )
        self.assertEqual(
            gravel_road_model_path("cwr_tight", 6, -30).casefold(),
            r"cwr_tight\i\gravel6_l30.p3d",
        )
        for subtype in ("gravel3_r45", "gravel6_l30", "gravel12_r20", "gravel_j3", "gravel_j4"):
            visual, _map_geometry, roadway, _land = _road_lods(
                InfrastructureModelKey("road", subtype, 46, 54 if subtype == "gravel_j3" else 60),
                r"cwr_tight\i\g.paa",
            )
            self.assertTrue(visual.faces)
            self.assertTrue(roadway.faces)
            if subtype in {"gravel_j3", "gravel_j4"}:
                for face in visual.faces:
                    uv = [(vertex[2], vertex[3]) for vertex in face.vertices]
                    area2 = sum(
                        u0 * v1 - u1 * v0
                        for (u0, v0), (u1, v1) in zip(uv, uv[1:] + uv[:1])
                    )
                    self.assertGreater(abs(area2), 1.0e-6)

    def test_all_gravel_junctions_use_generated_intersection_hubs(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        centre = (500.0, 500.0)
        endpoints = ((500.0, 430.0), (500.0, 570.0), (430.0, 500.0), (570.0, 500.0))
        roads = tuple(
            OsmLineFeature(
                f"way/gravel-junction-{index}", {"highway": "service", "surface": "gravel"},
                tuple(projection.to_latlon(point) for point in (centre, endpoint)),
            )
            for index, endpoint in enumerate(endpoints)
        )
        dataset = OsmDataset(
            source_generator="gravel-junction-hub", element_count=len(roads),
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=roads, gravel_roads=roads,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_gravel_hub", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01), cells=40, cell_size=25.0, strict_assets=False,
        )
        report = fit_road_objects(dataset, projection, [0.0] * (40 * 40), spec)
        self.assertTrue(any(
            obj.model_path.casefold().endswith(r"\i\gravel_j4.p3d")
            for obj in report.objects
        ))

    def test_generated_gravel_selects_curved_ribbon_variants_on_bends(self) -> None:
        run = ((100.0, 100.0), (102.8, 105.2), (106.1, 110.0), (110.0, 114.3))
        selected = _curved_gravel_model_for_run(
            r"cwr_curve\i\gravel6.p3d", run, (100.5, 101.0), (108.5, 112.7)
        )
        self.assertRegex(selected.casefold(), r"gravel6_[lr](?:05|10|15|20|30)\.p3d$")

        subtype = selected.rsplit("\\", 1)[-1][:-4].casefold()
        visual, _map_geometry, roadway, _land = _road_lods(
            InfrastructureModelKey("road", subtype, 60, 59), r"cwr_curve\i\gravel.paa"
        )
        visual_centres = [
            (visual.points[index][0] + visual.points[index + 1][0]) * 0.5
            for index in range(0, len(visual.points), 2)
        ]
        roadway_centres = [
            (roadway.points[index][0] + roadway.points[index + 1][0]) * 0.5
            for index in range(0, len(roadway.points), 2)
        ]
        self.assertGreater(max(abs(value) for value in visual_centres), 0.01)
        self.assertGreater(max(abs(value) for value in roadway_centres), 0.01)

    def test_gravel_road_fit_uses_generated_gravel_family_on_a_gentle_arc(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        points = []
        for index in range(13):
            angle = math.radians(-15.0 + index * 2.5)
            points.append((450.0 + 120.0 * math.sin(angle), 450.0 + 120.0 * math.cos(angle)))
        road = OsmLineFeature(
            "way/gravel-arc", {"highway": "service", "surface": "gravel"},
            tuple(projection.to_latlon(point) for point in points),
        )
        dataset = OsmDataset(
            source_generator="gravel-arc", element_count=1, coastlines=(), water=(), forests=(),
            farmland=(), urban=(), roads=(road,), gravel_roads=(road,),
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_curve_fit", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01), cells=40, cell_size=25.0, strict_assets=False,
        )
        report = fit_road_objects(dataset, projection, [0.0] * (40 * 40), spec)
        gravel_models = [obj.model_path.casefold() for obj in report.objects]
        self.assertTrue(gravel_models)
        self.assertTrue(all(model.startswith(r"cwr_curve_fit\i\gravel") for model in gravel_models))
        # Gentle gravel bends may use 12 m curved ribbons. Keeping them out of
        # the old six-metre-only chain roughly halves the number of visible
        # piece joins without bringing back the terrain-bridging 25 m slab.
        self.assertLessEqual(len(gravel_models), 7)
        self.assertTrue(any("gravel12" in model for model in gravel_models))
        self.assertFalse(any("gravel25" in model for model in gravel_models))

    def test_gravel_visual_overlap_extends_past_nominal_piece_ends(self) -> None:
        key = InfrastructureModelKey("road", "gravel6", 60, 59)
        visual, _map_geometry, roadway, _land = _road_lods(key, r"cwr_overlap\i\gravel.paa")
        visual_z = [point[2] for point in visual.points]
        roadway_z = [point[2] for point in roadway.points]
        self.assertLess(min(visual_z), min(roadway_z))
        self.assertGreater(max(visual_z), max(roadway_z))


    def test_gravel_visual_join_overlap_is_full_width_and_buried(self) -> None:
        key = InfrastructureModelKey("road", "gravel6", 46, 59)
        visual, _map_geometry, roadway, _land = _road_lods(key, r"cwr_join\i\gravel.paa")
        first_width = math.dist(
            (visual.points[0][0], visual.points[0][2]),
            (visual.points[1][0], visual.points[1][2]),
        )
        nominal_width = math.dist(
            (visual.points[2][0], visual.points[2][2]),
            (visual.points[3][0], visual.points[3][2]),
        )
        self.assertGreaterEqual(first_width, nominal_width)
        self.assertAlmostEqual(first_width, 4.6, places=5)
        self.assertLess(visual.points[0][1], visual.points[2][1])
        self.assertAlmostEqual(
            math.dist((roadway.points[0][0], roadway.points[0][2]),
                      (roadway.points[1][0], roadway.points[1][2])),
            4.6, places=5,
        )

    def test_church_uses_same_object_origin_clearance_as_houses(self) -> None:
        self.assertAlmostEqual(CHURCH_EXTRA_GROUND_CLEARANCE_METRES, 0.0, places=6)

    def test_church_and_house_use_identical_final_grounding_rule(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(source_generator="same-building-grounding", element_count=0, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=())
        raster = OsmRaster(cells=4, water=(False,)*16, forest=(False,)*16, farmland=(False,)*16, urban=(False,)*16, roads=(False,)*16, buildings=(False,)*16, high_resolution=4, coastline_seed_count=0)
        elevations = tuple(float(v) for v in (2,3,4,5, 3,4,5,6, 4,5,6,7, 5,6,7,8))
        footprint = ((25.0,25.0),(50.0,25.0),(50.0,50.0),(25.0,50.0))
        house = BuildingPlacementPlan("way/house",0,"polygon",37.5,37.5,0.0,r"same\g\house.p3d",footprint,building_family="residential")
        church = BuildingPlacementPlan("way/church",0,"polygon",37.5,37.5,0.0,r"same\g\church.p3d",footprint,building_family="church")
        spec = _Milestone9PlayabilitySpec(name="same_grounding",heightmap_path=Path("unused.png"),bbox=(0,0,1,1),cells=4,cell_size=25.0,max_road_objects=0,max_buildings=2,max_forest_objects=0,forest_undergrowth_enabled=False,forest_border_enabled=False,ditch_grass_enabled=False,barriers_enabled=False,bridges_enabled=False,rural_vegetation_enabled=False,strict_assets=False)
        house_result = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False, building_placement_plans=(house,))
        church_result = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False, building_placement_plans=(church,))
        self.assertEqual(house_result.building_objects, 1)
        self.assertEqual(church_result.building_objects, 1)
        self.assertAlmostEqual(house_result.objects[0].y, church_result.objects[0].y, places=6)

    def test_unnamed_isolated_dwelling_is_not_emitted_as_map_town_name(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        unnamed = OsmPointFeature(
            "way/isolated", {"place": "isolated_dwelling", "name": ""},
            projection.to_latlon((300.0, 300.0)),
        )
        legacy_unnamed = OsmPointFeature(
            "way/legacy-isolated",
            {"place": "isolated_dwelling", "name": "Unnamed isolated dwelling"},
            projection.to_latlon((350.0, 350.0)),
        )
        named = OsmPointFeature(
            "node/hamlet", {"place": "hamlet", "name": "Named Hamlet"},
            projection.to_latlon((600.0, 600.0)),
        )
        dataset = OsmDataset(
            source_generator="unnamed-place-label", element_count=3,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            places=(unnamed, legacy_unnamed, named),
        )
        locations = town_locations(dataset, projection, 64)
        self.assertEqual(len(locations), 1)
        self.assertEqual(locations[0].name, "Named Hamlet")

    def test_large_church_near_world_edge_is_nudged_inside_sampled_terrain(self) -> None:
        # Reproduce the supplied Vansö kyrka failure: its 0.9.206 procedural
        # footprint finished almost exactly at x=6400, while the last stored
        # vertex of a 256x25 m RVW4 grid is at x=6375.
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 6400.0)
        ring_world = (
            (6382.0, 696.0), (6399.8, 696.0), (6399.8, 720.0),
            (6382.0, 720.0), (6382.0, 696.0),
        )
        ring = tuple(projection.to_latlon(point) for point in ring_world)
        church = OsmPolygonFeature(
            "way/126830999",
            {"building": "church", "amenity": "place_of_worship"},
            (GeoPolygon(ring),),
        )
        dataset = OsmDataset(
            source_generator="vanso-edge-church", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(church,),
        )
        raster = OsmRaster(
            cells=256, water=(False,) * (256 * 256), forest=(False,) * (256 * 256),
            farmland=(False,) * (256 * 256), urban=(False,) * (256 * 256),
            roads=(False,) * (256 * 256), buildings=(False,) * (256 * 256),
            high_resolution=256, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="vanso_edge", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01), cells=256, cell_size=25.0,
            max_buildings=10, max_road_objects=0, max_forest_objects=0,
            strict_assets=False,
        )
        library = ProceduralBuildingLibrary(world_name=spec.name, maximum_variants=16)
        library.prepare(dataset, projection, spec.point_building_footprint)
        plans, truncated = plan_building_placements(dataset, projection, raster, spec, library)
        self.assertFalse(truncated)
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.building_family, "church")
        safe_maximum = (spec.cells - 1) * spec.cell_size - BUILDING_TERRAIN_EDGE_MARGIN_METRES
        self.assertLessEqual(max(point[0] for point in plan.support_polygon), safe_maximum + 1e-6)
        self.assertLess(plan.x, 6390.0)

    def test_complex_five_way_node_is_not_given_a_four_slot_hub(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        node = (500.0, 500.0)
        endpoints = ((250.0, 500.0), (750.0, 500.0), (500.0, 250.0), (500.0, 750.0), (700.0, 700.0))
        roads = tuple(
            OsmLineFeature(
                f"way/branch-{index}",
                {"highway": "primary", "surface": "asphalt"},
                tuple(projection.to_latlon(point) for point in (node, endpoint)),
            )
            for index, endpoint in enumerate(endpoints)
        )
        dataset = OsmDataset(
            source_generator="milestone9-five-way",
            element_count=len(roads),
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=roads,
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40, cell_size=25.0, strict_assets=False,
        )
        result = fit_road_objects(dataset, projection, [0.0] * (spec.cells * spec.cells), spec)
        self.assertEqual(result.complex_junctions_without_caps, 1)
        self.assertEqual(result.junction_cap_objects, 0)
        self.assertEqual(result.road_connection_slot_risk_nodes, 0)

    def test_milestone9_forest_ladder_uses_terrain_fit_and_reusable_clusters(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(
            source_generator="forest-grounding",
            element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(), building_points=(), places=(),
        )
        raster = OsmRaster(
            cells=4, water=(False,) * 16, forest=(True,) * 16, farmland=(False,) * 16,
            urban=(False,) * 16, roads=(False,) * 16, buildings=(False,) * 16,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_forest_ladder",
            heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 1.0, 1.0),
            cells=4, cell_size=25.0, forest_tree_spacing=50.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=4,
            forest_maximum_block_relief=2.5, forest_block_maximum_burial=1.25,
            forest_block_maximum_float=2.25, forest_everon_steep_maximum_relief=4.0,
            forest_everon_steep_maximum_burial=2.0, forest_everon_steep_maximum_float=4.0,
            forest_severe_hill_fallback=False,
            forest_undergrowth_enabled=False, forest_single_tree_enabled=False,
            forest_border_enabled=False,
            steep_hill_bushes_enabled=False, ditch_grass_enabled=False, strict_assets=False,
        )

        flat = generate_world_objects(
            dataset, projection, raster, (0.0,) * 16, spec, include_roads=False
        )
        self.assertEqual(flat.forest_block_objects, 4)
        self.assertEqual(flat.forest_everon_steep_objects, 0)
        self.assertEqual(flat.forest_cluster_objects, 0)
        self.assertTrue(all(obj.model_path == spec.forest_tree_model for obj in flat.objects))
        self.assertTrue(all(abs(obj.y - spec.forest_ground_clearance) < 1e-6 for obj in flat.objects))

        gentle_heights = tuple(float(x) for _z in range(4) for x in range(4))
        gentle = generate_world_objects(
            dataset, projection, raster, gentle_heights, spec, include_roads=False
        )
        self.assertEqual(gentle.forest_block_objects, 4)
        self.assertGreater(gentle.objects[0].y, spec.forest_ground_clearance)
        self.assertLessEqual(gentle.maximum_forest_burial, spec.forest_block_maximum_burial + 1e-6)
        self.assertLessEqual(gentle.maximum_forest_float, spec.forest_block_maximum_float + 1e-6)

        moderate_heights = tuple(float(x * 3) for _z in range(4) for x in range(4))
        moderate = generate_world_objects(
            dataset, projection, raster, moderate_heights, spec, include_roads=False
        )
        self.assertEqual(moderate.forest_block_objects, 0)
        self.assertEqual(moderate.forest_everon_steep_objects, 4)
        self.assertEqual(moderate.forest_cluster_objects, 0)
        self.assertTrue(all(obj.model_path == spec.forest_everon_steep_model for obj in moderate.objects))

        steep_heights = tuple(float(x * 8) for _z in range(4) for x in range(4))
        steep = generate_world_objects(
            dataset, projection, raster, steep_heights, spec, include_roads=False
        )
        self.assertEqual(steep.forest_block_objects, 0)
        self.assertEqual(steep.forest_everon_steep_objects, 2)
        self.assertEqual(steep.forest_cluster_objects, 2)
        self.assertEqual(steep.forest_hillside_tree_objects, 0)
        self.assertEqual(steep.forest_cluster_rejections, 0)
        self.assertTrue(any(obj.model_path.startswith(r"cwr_forest_ladder\f\c_") for obj in steep.objects))
        self.assertLessEqual(steep.forest_cluster_maximum_burial, spec.forest_cluster_maximum_burial + 1e-6)
        self.assertLessEqual(steep.forest_cluster_maximum_float, spec.forest_cluster_maximum_float + 1e-6)
        repeated = generate_world_objects(
            dataset, projection, raster, steep_heights, spec, include_roads=False
        )
        self.assertEqual(steep.objects, repeated.objects)

    def test_custom_primary_forest_model_is_not_replaced(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(
            source_generator="custom-forest-model", element_count=0, coastlines=(),
            water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=4, water=(False,) * 16, forest=(True,) * 16,
            farmland=(False,) * 16, urban=(False,) * 16, roads=(False,) * 16,
            buildings=(False,) * 16, high_resolution=4, coastline_seed_count=0,
        )
        custom_model = r"myaddon\forest\custom_square.p3d"
        spec = _Milestone9PlayabilitySpec(
            name="cwr_custom_forest", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=4, cell_size=25.0,
            forest_tree_spacing=50.0, forest_tree_model=custom_model,
            max_road_objects=0, max_buildings=0, max_forest_objects=4,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            steep_hill_bushes_enabled=False, ditch_grass_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (0.0,) * 16, spec, include_roads=False
        )
        self.assertEqual({obj.model_path for obj in result.objects}, {custom_model})

    def test_default_hill_limits_keep_square_blocks_on_rolling_ground(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(
            source_generator="rolling-forest", element_count=0, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=4, water=(False,) * 16, forest=(True,) * 16, farmland=(False,) * 16,
            urban=(False,) * 16, roads=(False,) * 16, buildings=(False,) * 16,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_rolling_forest", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=4, cell_size=25.0,
            forest_tree_spacing=50.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=4, forest_undergrowth_enabled=False,
            forest_single_tree_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, strict_assets=False,
        )
        rolling_heights = tuple(float(x * 2.0) for _z in range(4) for x in range(4))
        result = generate_world_objects(
            dataset, projection, raster, rolling_heights, spec, include_roads=False
        )
        self.assertEqual(result.forest_block_objects, 4)
        self.assertEqual(result.forest_everon_steep_objects, 0)
        self.assertEqual(result.forest_sunk_polygon_objects, 0)
        self.assertLessEqual(result.maximum_forest_burial, spec.forest_block_maximum_burial + 1e-6)
        self.assertLessEqual(result.maximum_forest_float, spec.forest_block_maximum_float + 1e-6)

    def test_32_metre_hill_uses_sunk_polygon_before_individual_tree_fallback(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(
            source_generator="32m-forest-hill", element_count=0, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=4, water=(False,) * 16, forest=(True,) * 16,
            farmland=(False,) * 16, urban=(False,) * 16, roads=(False,) * 16,
            buildings=(False,) * 16, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_32m_forest_hill", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=4, cell_size=25.0,
            forest_tree_spacing=50.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=40, forest_undergrowth_enabled=False,
            forest_single_tree_enabled=False,
            forest_border_enabled=False, steep_hill_bushes_enabled=False,
            ditch_grass_enabled=False, strict_assets=False,
        )
        hill = tuple(float(x) * (32.0 / 3.0) for _z in range(4) for x in range(4))
        grounded = generate_world_objects(
            dataset, projection, raster, hill, spec, include_roads=False
        )
        self.assertEqual(grounded.forest_block_objects, 0)
        self.assertEqual(grounded.forest_everon_steep_objects, 4)
        self.assertEqual(grounded.forest_sunk_polygon_objects, 4)
        self.assertEqual(grounded.forest_cluster_objects, 0)
        self.assertEqual(grounded.forest_undergrowth_objects, 0)
        self.assertEqual(grounded.forest_hillside_tree_objects, 0)
        self.assertTrue(all(obj.model_path == spec.forest_everon_steep_model for obj in grounded.objects))

        normal_triangle = generate_world_objects(
            dataset,
            projection,
            raster,
            hill,
            replace(spec, forest_severe_hill_fallback=False),
            include_roads=False,
        )
        self.assertEqual(normal_triangle.forest_sunk_polygon_objects, 0)
        maximum_expected_sink = grounded.maximum_hillside_tree_relief * 0.5
        for normal, sunk in zip(normal_triangle.objects, grounded.objects, strict=True):
            self.assertGreater(normal.y - sunk.y, 0.0)
            self.assertLessEqual(normal.y - sunk.y, maximum_expected_sink + 1e-6)

        below_old_severe_threshold = generate_world_objects(
            dataset,
            projection,
            raster,
            hill,
            replace(spec, forest_severe_hill_relief=100.0),
            include_roads=False,
        )
        self.assertEqual(below_old_severe_threshold.forest_sunk_polygon_objects, 4)
        self.assertEqual(below_old_severe_threshold.objects, grounded.objects)

        grounding_plan = plan_iterative_grounding_objects(
            dataset, projection, raster, hill, spec, ()
        )
        self.assertEqual(grounding_plan.objects, grounded.objects)
        repeated = generate_world_objects(
            dataset, projection, raster, hill, spec, include_roads=False
        )
        self.assertEqual(grounded.objects, repeated.objects)

        too_steep_for_polygon = generate_world_objects(
            dataset,
            projection,
            raster,
            hill,
            replace(
                spec,
                forest_severe_hill_relief=100.0,
                forest_everon_steep_maximum_relief=1.0,
            ),
            include_roads=False,
        )
        self.assertEqual(too_steep_for_polygon.forest_everon_steep_objects, 0)
        self.assertEqual(too_steep_for_polygon.forest_sunk_polygon_objects, 0)
        self.assertEqual(too_steep_for_polygon.forest_cluster_objects, 0)
        self.assertEqual(too_steep_for_polygon.forest_undergrowth_objects, 4)
        self.assertEqual(too_steep_for_polygon.forest_hillside_tree_objects, 40)

    def test_35_and_43_metre_hills_allow_configured_polygon_tiers(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(
            source_generator="43m-forest-hill", element_count=0, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=4, water=(False,) * 16, forest=(True,) * 16,
            farmland=(False,) * 16, urban=(False,) * 16, roads=(False,) * 16,
            buildings=(False,) * 16, high_resolution=8, coastline_seed_count=0,
        )
        base = _Milestone9PlayabilitySpec(
            name="cwr_43m_forest_hill", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=4, cell_size=25.0,
            forest_tree_spacing=50.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=4, forest_undergrowth_enabled=False,
            forest_single_tree_enabled=False, forest_border_enabled=False,
            steep_hill_bushes_enabled=False,
            ditch_grass_enabled=False, strict_assets=False,
        )
        for hill_height in (35.0, 43.0):
            with self.subTest(hill_height=hill_height):
                hill = tuple(
                    float(x) * (hill_height / 3.0)
                    for _z in range(4) for x in range(4)
                )

                square = generate_world_objects(
                    dataset,
                    projection,
                    raster,
                    hill,
                    replace(
                        base,
                        forest_maximum_block_relief=100.0,
                        forest_block_maximum_burial=100.0,
                    ),
                    include_roads=False,
                )
                self.assertEqual(square.forest_block_objects, 4)

                triangle = generate_world_objects(
                    dataset,
                    projection,
                    raster,
                    hill,
                    replace(
                        base,
                        forest_maximum_block_relief=0.0,
                        forest_everon_steep_maximum_relief=100.0,
                        forest_everon_steep_maximum_burial=100.0,
                        forest_severe_hill_fallback=False,
                    ),
                    include_roads=False,
                )
                self.assertEqual(triangle.forest_block_objects, 0)
                self.assertEqual(triangle.forest_everon_steep_objects, 4)

    def test_forest_borders_and_ditch_grass_use_reusable_cluster_models(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)

        def ll(point: tuple[float, float]) -> tuple[float, float]:
            return projection.to_latlon(point)

        forest_ring = tuple(
            ll(point)
            for point in ((20.0, 20.0), (180.0, 20.0), (180.0, 180.0), (20.0, 180.0), (20.0, 20.0))
        )
        forest = OsmPolygonFeature(
            "way/forest", {"landuse": "forest"}, (GeoPolygon(forest_ring),)
        )
        ditch = OsmLineFeature(
            "way/ditch", {"waterway": "ditch"},
            tuple(ll(point) for point in ((10.0, 100.0), (190.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="forest-edge-ditch", element_count=2,
            coastlines=(), water=(), forests=(forest,), farmland=(), urban=(), roads=(),
            watercourses=(ditch,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(True,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_vegetation_edges", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=8, cell_size=25.0,
            forest_tree_spacing=1000.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=100, forest_single_tree_enabled=False, forest_undergrowth_maximum_objects=20,
            forest_undergrowth_spacing=35.0, forest_border_maximum_objects=20,
            maximum_ditch_grass_objects=20, strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (0.0,) * 64, spec, include_roads=False
        )
        self.assertGreater(result.forest_undergrowth_objects, 0)
        self.assertGreater(result.forest_border_objects, 0)
        self.assertGreater(result.ditch_grass_objects, 0)
        undergrowth_models = [obj.model_path for obj in result.objects if r"\f\u_" in obj.model_path]
        border_models = [obj.model_path for obj in result.objects if r"\f\b_" in obj.model_path]
        ditch_models = [obj.model_path for obj in result.objects if r"\f\g_" in obj.model_path]
        self.assertEqual(len(undergrowth_models), result.forest_undergrowth_objects)
        self.assertEqual(len(border_models), result.forest_border_objects)
        self.assertEqual(len(ditch_models), result.ditch_grass_objects)
        repeated = generate_world_objects(
            dataset, projection, raster, (0.0,) * 64, spec, include_roads=False
        )
        self.assertEqual(result.objects, repeated.objects)

    def test_forest_undergrowth_keeps_every_other_placeable_cluster_by_default(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        dataset = OsmDataset(
            source_generator="half-undergrowth", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(True,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        full_spec = _Milestone9PlayabilitySpec(
            name="cwr_half_undergrowth", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=100,
            forest_tree_spacing=1000.0,
            forest_undergrowth_maximum_objects=100,
            forest_undergrowth_spacing=25.0,
            forest_undergrowth_maximum_relief=100.0,
            forest_undergrowth_maximum_burial=100.0,
            forest_undergrowth_maximum_float=100.0,
            forest_border_enabled=False, forest_single_tree_enabled=False,
            steep_hill_bushes_enabled=False, ditch_grass_enabled=False,
            barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, strict_assets=False,
        )
        forced_half = generate_world_objects(
            dataset, projection, raster, (0.0,) * 64, full_spec, include_roads=False
        )
        self.assertGreater(forced_half.forest_undergrowth_objects, 0)
        self.assertLessEqual(forced_half.forest_undergrowth_objects, 50)

    def test_full_build_embeds_reusable_forest_cluster_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = milestone8_tests.Milestone8BuildTests()._source(root / "source")
            spec = Milestone9Spec(
                source_dir=source, name="cwr_m9_clusters",
                display_name="CWR M9 Forest Clusters", solver_iterations=2,
                world_edge_blend_cells=1, max_buildings=0, max_forest_objects=8,
                forest_maximum_block_relief=1.0e-9,
                forest_everon_steep_maximum_relief=1.0e-9, forest_severe_hill_fallback=False,
                forest_block_maximum_burial=0.0, forest_block_maximum_float=0.0,
                forest_everon_steep_maximum_burial=0.0,
                forest_everon_steep_maximum_float=0.0,
                forest_undergrowth_enabled=True, forest_undergrowth_maximum_objects=4,
                forest_undergrowth_spacing=40.0, forest_border_enabled=False, ditch_grass_enabled=False,
                semantic_landmarks=False, ground_texture_profile="everon",
                surface_overview_size=128, surface_texture_size=32,
                strict_assets=False, verify_regeneration=True,
            )
            result = build_milestone9(root / "build", spec)
            self.assertIsNotNone(result.forest_cluster_catalogue_path)
            catalogue = json.loads(
                result.forest_cluster_catalogue_path.read_text(encoding="utf-8")
            )
            self.assertGreater(catalogue["placements"], 0)
            self.assertGreater(catalogue["generated_variants"], 0)
            packed_entries = {entry.name: entry.data for entry in read_pbo(result.pbo_path)}
            entries = set(packed_entries)
            self.assertIn(r"f\clusters.json", entries)
            self.assertNotIn(r"f\v.paa", entries)
            self.assertNotIn(r"f\t.paa", entries)
            self.assertNotIn(r"f\g.paa", entries)
            self.assertTrue(any(name.startswith(r"f\c_") and name.endswith(".p3d") for name in entries))
            self.assertTrue(any(name.startswith(r"f\u_") and name.endswith(".p3d") for name in entries))
            generated_name = next(
                name for name in entries if name.startswith(r"f\c_") and name.endswith(".p3d")
            )
            generated_path = root / "generated-forest.p3d"
            generated_path.write_bytes(packed_entries[generated_name])
            generated_summary = inspect_mlod(generated_path)
            proxy_names = tuple(
                name for names in generated_summary.selection_names for name in names
                if name.casefold().startswith("proxy:")
            )
            self.assertTrue(proxy_names)
            self.assertFalse(generated_summary.texture_paths)
            wrp = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertTrue(any(path.startswith(r"cwr_m9_clusters\f\c_") for path in wrp.object_models))
            self.assertTrue(any(path.startswith(r"cwr_m9_clusters\f\u_") for path in wrp.object_models))
            stock_ground = {
                r"Eden\tn.paa", r"Eden\zbh.paa", r"Eden\bak\bah.pac",
                r"o\l1.paa", r"o\lom2.paa",
                r"o\pole1.paa", r"o\pole2.paa",
            }
            self.assertTrue(all(path in stock_ground for path in wrp.texture_slots[1:1 + len(MILESTONE9_MATERIALS)]))
            generated_ground_entries = {
                rf"data\{material.code}.paa"
                for material in MILESTONE9_MATERIALS
                if material.code != "d"  # data\d.paa is the reserved slot-zero dummy.
            }
            self.assertTrue(entries.isdisjoint(generated_ground_entries))
            forest_slot = 1 + MATERIAL_INDEX["f"]
            self.assertEqual(wrp.texture_slots[forest_slot], r"Eden\zbh.paa")
            self.assertGreater(wrp.texture_index_counts[forest_slot], 0)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pbo_layout"]["mode"], "single_world_pbo")
            self.assertTrue(manifest["pbo_layout"]["verified"])
            self.assertFalse(manifest["pbo_layout"]["separate_road_pbo"])
            self.assertIsInstance(manifest["pbo_layout"]["generated_infrastructure_entries"], list)
            for relative in manifest["pbo_layout"]["generated_infrastructure_entries"]:
                self.assertIn(relative.replace("/", "\\"), entries)
            self.assertGreater(manifest["objects"]["accepted_forest_material_cells"], 0)
            grounding = manifest["objects"]["grounding"]
            self.assertTrue(grounding["terrain_quantized_before_object_placement"])
            self.assertTrue(grounding["church_final_footprint_validation"])
            self.assertEqual(len(grounding["grave_model_specific_profiles"]), 5)
            self.assertLessEqual(
                grounding["maximum_solver_to_wrp_height_change_metres"],
                manifest["iterative_grounding"]["maximum_adjustment"] + grounding["wrp_height_scale_metres"] + 1e-9,
            )
            reproducibility = json.loads(result.reproducibility_path.read_text(encoding="utf-8"))
            self.assertTrue(reproducibility["forest_cluster_assets_byte_match"])

    def test_build_is_reproducible_and_embeds_surface_visual_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = milestone8_tests.Milestone8BuildTests()._source(root / "source")
            spec = Milestone9Spec(
                source_dir=source,
                name="cwr_m9_test",
                display_name="CWR M9 Test",
                solver_iterations=4,
                world_edge_blend_cells=2,
                max_forest_objects=0,
                ground_texture_profile="generated",
                surface_overview_size=128,
                surface_texture_size=32,
                verify_regeneration=True,
            )
            first = build_milestone9(root / "one", spec)
            second = build_milestone9(root / "two", spec)

            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.pbo_path.read_bytes(), second.pbo_path.read_bytes())
            self.assertEqual(first.pbo_path.parent.parent.name, "@CWR-Milestone9")
            self.assertIsNotNone(first.surface_report_path)
            self.assertIsNotNone(first.overview_map_path)
            self.assertIsNotNone(first.overview_paa_path)
            self.assertIsNotNone(first.world_icon_path)
            self.assertTrue(first.surface_report_path.is_file())
            self.assertTrue(first.overview_map_path.is_file())
            self.assertTrue(first.overview_paa_path.is_file())
            self.assertTrue(first.world_icon_path.is_file())

            overview_info = inspect_paa(first.overview_paa_path)
            icon_info = inspect_paa(first.world_icon_path)
            self.assertEqual((overview_info.width, overview_info.height), (128, 128))
            self.assertEqual((icon_info.width, icon_info.height), (128, 128))

            entries = {entry.name: entry.data for entry in read_pbo(first.pbo_path)}
            procedural_models = {
                name: data for name, data in entries.items()
                if name.startswith("g\\b_") and name.endswith(".p3d")
            }
            self.assertTrue(procedural_models)
            for name, data in procedural_models.items():
                model_path = root / "inspect" / name.replace("\\", "/")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(data)
                properties = dict(inspect_mlod(model_path).named_properties[1])
                self.assertIn(properties.get("map"), {"house", "building"}, name)
            self.assertIn(r"data\overview.paa", entries)
            self.assertIn(r"data\icon.paa", entries)
            for material in MILESTONE9_MATERIALS:
                self.assertIn(rf"data\{material.code}.paa", entries)
            config = entries["config.cpp"].decode("ascii")
            self.assertIn(r"\cwr_m9_test\data\icon.paa", config)

            wrp = inspect_rvw4(first.wrp_path, height_scale=0.05)
            self.assertEqual(
                set(wrp.texture_paths),
                {rf"{spec.name}\data\d.paa", *(rf"{spec.name}\data\{material.code}.paa" for material in MILESTONE9_MATERIALS)},
            )

            surface = json.loads(first.surface_report_path.read_text(encoding="utf-8"))
            self.assertGreater(surface["wet_shoreline_cells"] + surface["dry_shoreline_cells"], 0)
            self.assertGreater(surface["forest_edge_cells"], 0)
            self.assertGreater(surface["paved_road_cells"] + surface["dirt_road_cells"], 0)
            self.assertEqual(set(surface["material_cells"]), {material.code for material in MILESTONE9_MATERIALS})
            self.assertEqual(surface["ground_application_mode"], "milestone9")
            self.assertEqual(set(surface["wrp_material_cells"]), {material.code for material in MILESTONE9_MATERIALS})

            road_fit = json.loads(first.road_report_path.read_text(encoding="utf-8"))
            self.assertEqual(road_fit["terrain_filled_junctions"], 0)
            self.assertLessEqual(road_fit["maximum_model_overlap_metres"], 1e-6)
            self.assertIn("junction_cap_objects", road_fit)
            self.assertIn("maximum_road_pitch_degrees", road_fit)

            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest, second_manifest)
            self.assertEqual(manifest["schema"], 9)
            self.assertEqual(manifest["milestone"], 9)
            self.assertEqual(manifest["generator"], GENERATOR_VERSION)
            self.assertEqual(manifest["surface_visual_pass"]["ground_application_mode"], "milestone9")
            self.assertIn("surface_visual_pass", manifest)
            self.assertNotIn("cache", manifest)
            first_cache = json.loads(first.cache_report_path.read_text(encoding="utf-8"))
            second_cache = json.loads(second.cache_report_path.read_text(encoding="utf-8"))
            self.assertFalse(first_cache["parsed_source"]["hit"])
            self.assertTrue(second_cache["parsed_source"]["hit"])
            self.assertTrue(second_cache["spatial_index"]["hit"])
            self.assertTrue(second_cache["processed_dem"]["hit"])
            self.assertTrue(second_cache["osm_raster"]["hit"])
            self.assertTrue(second_cache["terrain_solution"]["hit"])
            self.assertTrue(second_cache["surface_pipeline"]["hit"])
            self.assertTrue(second_cache["forest_and_building_placement"]["hit"])
            self.assertTrue(second_cache["surface_textures"]["hit"])
            self.assertTrue(second_cache["overview_assets"]["hit"])
            self.assertTrue(second_cache["incremental_pbo"]["archive_hit"])
            self.assertEqual(second_cache["incremental_pbo"]["requested_backend"], "auto")
            self.assertEqual(second_cache["incremental_pbo"]["backend"], "python")
            self.assertGreater(second_cache["procedural_assets"]["building_hits"], 0)

            overview_changed = build_milestone9(
                root / "three", replace(spec, surface_overview_size=256)
            )
            overview_cache = json.loads(overview_changed.cache_report_path.read_text(encoding="utf-8"))
            self.assertTrue(overview_cache["processed_dem"]["hit"])
            self.assertTrue(overview_cache["terrain_solution"]["hit"])
            self.assertTrue(overview_cache["surface_pipeline"]["hit"])
            self.assertTrue(overview_cache["forest_and_building_placement"]["hit"])
            self.assertFalse(overview_cache["overview_assets"]["hit"])

            forest_changed = build_milestone9(
                root / "four", replace(spec, forest_maximum_block_relief=4.0)
            )
            forest_cache = json.loads(forest_changed.cache_report_path.read_text(encoding="utf-8"))
            self.assertTrue(forest_cache["processed_dem"]["hit"])
            self.assertTrue(forest_cache["terrain_solution"]["hit"])
            self.assertTrue(forest_cache["surface_pipeline"]["hit"])
            self.assertFalse(forest_cache["forest_and_building_placement"]["hit"])

            reproducibility = json.loads(first.reproducibility_path.read_text(encoding="utf-8"))
            self.assertTrue(reproducibility["pipeline_repeat_match"])
            self.assertTrue(reproducibility["wrp_byte_match"])
            self.assertTrue(reproducibility["pbo_byte_match"])
            self.assertTrue(reproducibility["surface_assets_byte_match"])

            report = first.report_path.read_text(encoding="utf-8")
            self.assertIn("Milestone 9 surface and visual checks", report)
            self.assertIn("Transition decisions derive from deterministic seed", report)


    def test_legacy_milestone8_ground_palette_remains_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = milestone8_tests.Milestone8BuildTests()._source(root / "source")
            spec = Milestone9Spec(
                source_dir=source,
                name="cwr_m9_legacy",
                display_name="CWR M9 Legacy Palette",
                solver_iterations=2,
                world_edge_blend_cells=1,
                max_forest_objects=0,
                ground_texture_profile="generated",
                surface_ground_mode="milestone8",
                surface_overview_size=128,
                surface_texture_size=32,
            )
            result = build_milestone9(root / "build", spec)
            entries = {entry.name for entry in read_pbo(result.pbo_path)}
            for material in OSM_MATERIALS:
                self.assertIn(rf"data\{material.code}.paa", entries)
            milestone9_only_codes = (
                {material.code for material in MILESTONE9_MATERIALS}
                - {material.code for material in OSM_MATERIALS}
                - {"d"}  # mandatory dummy texture exists in every build
            )
            for code in milestone9_only_codes:
                self.assertNotIn(rf"data\{code}.paa", entries)
            wrp = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertEqual(len(set(wrp.texture_paths)), len(OSM_MATERIALS) + 1)
            report = json.loads(result.surface_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["ground_application_mode"], "milestone8")



if __name__ == "__main__":
    unittest.main()

class SemanticFeatureTests(unittest.TestCase):
    def test_vegetation_grounding_stays_above_rvw4_saddle_diagonals(self) -> None:
        cells = 3
        cell_size = 25.0
        elevations = (
            0.0, 20.0, 20.0,
            20.0, 0.0, 0.0,
            20.0, 0.0, 0.0,
        )
        square_supports = _square_elevation_samples(
            elevations, cells, cell_size, 12.5, 12.5, 2.0
        )
        triangle_supports = _oriented_footprint_elevation_samples(
            elevations, cells, cell_size, 12.5, 12.5, 1.2, 2.0, 37.0
        )
        for supports in (square_supports, triangle_supports):
            self.assertAlmostEqual(min(supports), 0.0)
            self.assertAlmostEqual(max(supports), 20.0)
            self.assertAlmostEqual(
                _non_buried_vegetation_anchor(supports, clearance=0.15),
                20.15,
            )
            self.assertAlmostEqual(
                _non_buried_vegetation_anchor(supports, clearance=-2.0),
                20.0,
            )
            self.assertIsNone(_non_buried_vegetation_fit(
                supports, clearance=0.15, maximum_float=0.5
            ))
        gentle_fit = _non_buried_vegetation_fit(
            (10.0, 10.2), clearance=0.15, maximum_float=0.5
        )
        self.assertIsNotNone(gentle_fit)
        self.assertAlmostEqual(gentle_fit[0], 10.35)
        self.assertAlmostEqual(gentle_fit[1], 0.35)

    def test_building_grounding_brackets_rvw4_saddle_diagonals(self) -> None:
        cells = 3
        cell_size = 25.0
        # The first quad alternates low/high corners. Bilinear interpolation is
        # 10 m at its centre, while its two possible triangle diagonals render
        # that same point at either 0 m or 20 m.
        elevations = (
            0.0, 20.0, 20.0,
            20.0, 0.0, 0.0,
            20.0, 0.0, 0.0,
        )
        footprint = ((11.5, 11.5), (13.5, 11.5), (13.5, 13.5), (11.5, 13.5))
        self.assertAlmostEqual(
            _sample_elevation(elevations, cells, cell_size, 12.5, 12.5), 10.0
        )
        self.assertEqual(
            _triangle_elevation_bounds(elevations, cells, cell_size, 12.5, 12.5),
            (0.0, 20.0),
        )
        self.assertAlmostEqual(
            _minimum_polygon_elevation(elevations, cells, cell_size, footprint),
            0.0,
        )
        self.assertAlmostEqual(
            _maximum_polygon_elevation(elevations, cells, cell_size, footprint),
            20.0,
        )

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 75.0)
        dataset = OsmDataset(
            source_generator="saddle-grounding", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=cells, water=(False,) * 9, forest=(False,) * 9,
            farmland=(False,) * 9, urban=(False,) * 9, roads=(False,) * 9,
            buildings=(False,) * 9, high_resolution=cells, coastline_seed_count=0,
        )
        plan = BuildingPlacementPlan(
            osm_key="way/saddle-house", geometry_index=0, geometry_kind="polygon",
            x=25.0, z=25.0, heading_degrees=0.0,
            model_path=r"saddle\g\house.p3d", support_polygon=footprint,
            building_family="residential",
        )
        spec = _Milestone9PlayabilitySpec(
            name="saddle_grounding", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=cells, cell_size=cell_size,
            max_road_objects=0, max_buildings=1, max_forest_objects=0,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, elevations, spec, include_roads=False,
            building_placement_plans=(plan,),
        )
        self.assertEqual(result.building_objects, 1)
        self.assertGreaterEqual(
            result.objects[0].y,
            20.0 + spec.building_ground_clearance - 1e-9,
        )

    def test_optional_procedural_building_interiors_are_enterable_and_bounded(self) -> None:
        from cwr_worldgen.procedural_buildings import (
            BuildingVariantKey,
            _door_dimensions,
            _geometry_lod,
            _interior_paths_lod,
            _interior_roadway_lod,
            _interior_storey_count,
            _placement_uses_second_storey,
            _interior_window_openings,
            _window_openings,
            write_building_mlod,
        )

        enabled = ProceduralBuildingLibrary(
            world_name="interior_test", generate_interiors=True
        )
        disabled = ProceduralBuildingLibrary(
            world_name="interior_test", generate_interiors=False
        )
        key = enabled.key_for({"building": "house", "building:levels": "2"}, 12.0, 16.0)
        self.assertTrue(key.interiors)
        self.assertEqual(_interior_storey_count(key), 2)
        self.assertEqual(
            _interior_storey_count(
                enabled.key_for({"building": "house"}, 5.0, 7.0)
            ),
            1,
        )
        self.assertFalse(
            disabled.key_for({"building": "house"}, 12.0, 16.0).interiors
        )
        utility_cases = (
            ({"building": "barn"}, 14.0, 42.0, "agricultural"),
            ({"building": "shed"}, 6.0, 8.0, "outbuilding"),
            ({"building": "garage"}, 7.0, 9.0, "outbuilding"),
            ({"building": "warehouse"}, 24.0, 60.0, "industrial"),
        )
        for tags, width, length, family in utility_cases:
            with self.subTest(utility_family=family, tags=tags):
                utility_key = enabled.key_for(tags, width, length)
                self.assertEqual(utility_key.family, family)
                self.assertTrue(utility_key.interiors)

        # Barns/warehouses and car-capable outbuildings use vehicle-scale
        # openings. Outbuildings too small to contain a car are sheds with a
        # pedestrian-size entrance, regardless of whether OSM called them a
        # garage or shed.
        house_door_half, house_door_height, _ = _door_dimensions(key)
        barn_key = enabled.key_for({"building": "barn"}, 14.0, 42.0)
        garage_key = enabled.key_for({"building": "shed"}, 6.0, 8.0)
        small_shed_key = enabled.key_for({"building": "garage"}, 2.0, 3.5)
        warehouse_key = enabled.key_for({"building": "warehouse"}, 24.0, 60.0)
        self.assertEqual(garage_key.outbuilding_kind, "garage")
        self.assertEqual(small_shed_key.outbuilding_kind, "shed")
        for utility_key in (barn_key, garage_key, warehouse_key):
            utility_half, utility_height, _ = _door_dimensions(utility_key)
            self.assertGreater(utility_half, house_door_half)
            self.assertGreater(utility_height, house_door_height)
        shed_half, shed_height, _ = _door_dimensions(small_shed_key)
        self.assertLess(shed_half, garage_key.width_m * 0.25)
        self.assertLess(shed_height, _door_dimensions(garage_key)[1])
        self.assertLess(shed_half, _door_dimensions(garage_key)[0])

        # Untagged/default-height houses get an upper floor most, but not all,
        # of the time. Selection is stable from tags and placement coordinates.
        generic_house = enabled.key_for({"building": "house"}, 12.0, 16.0)
        mixed = [
            _placement_uses_second_storey(
                {"building": "house"}, ((float(index) * 17.0, 25.0),), generic_house
            )
            for index in range(40)
        ]
        self.assertGreater(sum(mixed), 20)
        self.assertLess(sum(mixed), 40)
        self.assertEqual(
            mixed,
            [
                _placement_uses_second_storey(
                    {"building": "house"}, ((float(index) * 17.0, 25.0),), generic_house
                )
                for index in range(40)
            ],
        )
        self.assertFalse(
            enabled.key_for(
                {"building": "house"}, 32.0, 38.0, settlement_context="village"
            ).interiors
        )

        plain = replace(key, interiors=False)
        visual = _visual_lod(
            key,
            r"interior_test\d\wall.paa",
            r"interior_test\d\roof.paa",
            35.0,
            r"interior_test\d\front.paa",
            r"interior_test\d\floor.paa",
            0.5,
            interior_texture=r"interior_test\d\inside.paa",
        )
        plain_visual = _visual_lod(
            plain,
            r"interior_test\d\wall.paa",
            r"interior_test\d\roof.paa",
            35.0,
            r"interior_test\d\front.paa",
            r"interior_test\d\floor.paa",
            0.5,
        )
        geometry = _geometry_lod(key)
        self.assertGreater(len(visual.faces), len(plain_visual.faces))
        self.assertGreaterEqual(len(geometry.selections), 30)
        self.assertEqual(dict(geometry.properties)["class"], "house")
        stair_key = replace(key, foundation_depth_m=2.0)
        roadway = _interior_roadway_lod(stair_key, 2.0)
        self.assertIsNotNone(roadway)
        assert roadway is not None
        self.assertTrue(math.isclose(roadway.resolution, 3.0e15, rel_tol=1e-6))
        self.assertLess(min(point[1] for point in roadway.points), -1.5)
        self.assertLess(
            min(point[2] for point in roadway.points),
            -stair_key.length_m * 0.5 - 2.0,
        )
        self.assertTrue(roadway.faces)
        self.assertGreater(max(point[1] for point in roadway.points), 2.5)
        # Upper stairs now use a segmented continuous Roadway slope. A thin
        # solid stepped Geometry staircase sits just below matching horizontal
        # Roadway treads, so infantry has both walkable and physical support.
        stair_layout = building_models._second_storey_layout(stair_key)
        self.assertIsNotNone(stair_layout)
        horizontal_upper_faces = [
            face for face in roadway.faces
            if (
                max(roadway.points[index][1] for index, _normal, _u, _v in face.vertices)
                - min(roadway.points[index][1] for index, _normal, _u, _v in face.vertices)
            ) < 1e-6
            and min(roadway.points[index][1] for index, _normal, _u, _v in face.vertices)
            > building_models.INTERIOR_ROADWAY_Y_M + 0.10
        ]
        self.assertGreaterEqual(
            len(horizontal_upper_faces),
            building_models.INTERIOR_SECOND_STOREY_STAIR_STEPS,
        )

        stair_levels = sorted({
            round(roadway.points[index][1], 3)
            for face in horizontal_upper_faces
            for index, _normal, _u, _v in face.vertices
            if 0.05 <= roadway.points[index][1] <= 2.65
        })
        self.assertGreaterEqual(len(stair_levels), 12)

        paths = _interior_paths_lod(stair_key, 2.0)
        self.assertIsNotNone(paths)
        assert paths is not None
        self.assertIn("Pos3", {selection.name for selection in paths.selections})
        upstairs_positions = []
        for selection in paths.selections:
            if not selection.name.startswith("Pos"):
                continue
            for point_index, weight in enumerate(selection.point_weights):
                if weight:
                    upstairs_positions.append(paths.points[point_index][1])
        self.assertGreater(max(upstairs_positions), 2.5)

        def assert_collision_free(x: float, y: float, z: float) -> None:
            for start in range(0, len(geometry.points), 8):
                component = geometry.points[start:start + 8]
                xs = [point[0] for point in component]
                ys = [point[1] for point in component]
                zs = [point[2] for point in component]
                self.assertFalse(
                    min(xs) <= x <= max(xs)
                    and min(ys) <= y <= max(ys)
                    and min(zs) <= z <= max(zs)
                )

        # The enterable collision shell no longer duplicates the Roadway floor
        # or uses a low solid ceiling. Centre-floor and head-height space remain
        # clear, including through the partition doorway.
        assert_collision_free(0.0, 0.02, 1.0)
        assert_collision_free(0.0, 2.55, 0.0)

        roadway_wall_clearance = 0.12
        wall_thickness = min(0.30, max(0.18, min(key.width_m, key.length_m) * 0.025))
        self.assertLessEqual(
            max(abs(point[0]) for point in roadway.points if point[2] > -key.length_m * 0.5),
            key.width_m * 0.5 - wall_thickness - roadway_wall_clearance + 1e-6,
        )

        # The entrance is intentionally occupied by the closed animated door.
        # Its collision component shares the visual ``door1`` selection so it
        # rotates out of the doorway when a player opens it.
        self.assertIn("door1", {selection.name for selection in geometry.selections})
        door_half = min(0.8, max(0.6, key.width_m * 0.5 * 0.18))
        door_z = -key.length_m * 0.5 + 0.05
        door_components = []
        for start in range(0, len(geometry.points), 8):
            component = geometry.points[start:start + 8]
            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            zs = [point[2] for point in component]
            if (
                min(xs) <= 0.0 <= max(xs)
                and min(ys) <= 1.0 <= max(ys)
                and min(zs) <= door_z <= max(zs)
            ):
                door_components.append(component)
        self.assertEqual(len(door_components), 1)
        self.assertLessEqual(max(abs(point[0]) for point in door_components[0]), door_half + 1e-6)
        back_openings = _interior_window_openings(
            key, -key.width_m * 0.5, key.width_m * 0.5, key.height_m
        )
        side_openings = _interior_window_openings(
            key, -key.length_m * 0.5, key.length_m * 0.5, key.height_m
        )
        self.assertGreater(len(back_openings), 1)
        self.assertGreater(len(side_openings), 1)
        self.assertTrue(any(opening_bottom > 3.0 for _a, _b, opening_bottom, _top in back_openings))
        for opening_min, opening_max, opening_bottom, opening_top in back_openings:
            assert_collision_free(
                (opening_min + opening_max) * 0.5,
                (opening_bottom + opening_top) * 0.5,
                key.length_m * 0.5 - 0.05,
            )
        for opening_min, opening_max, opening_bottom, opening_top in side_openings:
            window_z = (opening_min + opening_max) * 0.5
            window_y = (opening_bottom + opening_top) * 0.5
            assert_collision_free(key.width_m * 0.5 - 0.05, window_y, window_z)
            assert_collision_free(-key.width_m * 0.5 + 0.05, window_y, window_z)

        self.assertFalse(any("glass" in face.texture.lower() for face in visual.faces))
        self.assertIn(
            r"interior_test\d\inside.paa",
            {face.texture for face in visual.faces},
        )
        swedish_visual = _visual_lod(
            replace(key, regional_style="sweden_red"),
            r"interior_test\d\open_wall.paa",
            r"interior_test\d\roof.paa",
            35.0,
            foundation_texture=r"interior_test\d\floor.paa",
            interior_texture=r"interior_test\d\inside.paa",
            window_trim_texture=r"interior_test\d\white_trim.paa",
        )
        self.assertIn(
            r"interior_test\d\white_trim.paa",
            {face.texture for face in swedish_visual.faces},
        )
        for whitewash_style in (
            "eastern_whitewash",
            "africa_whitewash",
            "middle_east_whitewash",
        ):
            whitewash_visual = _visual_lod(
                replace(key, regional_style=whitewash_style),
                r"interior_test\d\open_wall.paa",
                r"interior_test\d\roof.paa",
                35.0,
                foundation_texture=r"interior_test\d\floor.paa",
                interior_texture=r"interior_test\d\inside.paa",
                window_trim_texture=r"interior_test\d\white_trim.paa",
            )
            self.assertIn(
                r"interior_test\d\white_trim.paa",
                {face.texture for face in whitewash_visual.faces},
            )
        plain_with_trim_argument = _visual_lod(
            key,
            r"interior_test\d\open_wall.paa",
            r"interior_test\d\roof.paa",
            35.0,
            foundation_texture=r"interior_test\d\floor.paa",
            interior_texture=r"interior_test\d\inside.paa",
            window_trim_texture=r"interior_test\d\white_trim.paa",
        )
        self.assertNotIn(
            r"interior_test\d\white_trim.paa",
            {face.texture for face in plain_with_trim_argument.faces},
        )
        self.assertIn(
            r"\d\o",
            enabled._open_wall_texture(
                key.family, key.regional_style, key.texture_variant
            ),
        )

        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "enterable.p3d"
            write_building_mlod(
                model,
                BuildingVariantKey(
                    key.family,
                    key.roof_style,
                    key.width_m,
                    key.length_m,
                    key.height_m,
                    interiors=True,
                ),
                wall_texture=r"interior_test\d\wall.paa",
                roof_texture=r"interior_test\d\roof.paa",
                front_texture=r"interior_test\d\front.paa",
                foundation_texture=r"interior_test\d\floor.paa",
                foundation_depth=0.5,
                interior_texture=r"interior_test\d\inside.paa",
            )
            summary = inspect_mlod(model)
            self.assertEqual(summary.lod_count, 7)
            self.assertTrue(any(
                math.isclose(value, 20.0, rel_tol=1e-6)
                for value in summary.resolutions
            ))
            self.assertTrue(any(
                math.isclose(value, 3.0e15, rel_tol=1e-6)
                for value in summary.resolutions
            ))
            geometry_index = next(
                index for index, value in enumerate(summary.resolutions)
                if math.isclose(value, 1.0e13, rel_tol=1e-6)
            )
            memory_index = next(
                index for index, value in enumerate(summary.resolutions)
                if math.isclose(value, 1.0e15, rel_tol=1e-6)
            )
            paths_index = next(
                index for index, value in enumerate(summary.resolutions)
                if math.isclose(value, 4.0e15, rel_tol=1e-6)
            )
            self.assertEqual(dict(summary.named_properties[geometry_index])["class"], "house")
            self.assertIn("door1", summary.selection_names[0])
            self.assertIn("door1", summary.selection_names[geometry_index])
            self.assertIn("door1_axis", summary.selection_names[memory_index])
            self.assertIn("door1_action", summary.selection_names[memory_index])
            self.assertIn("In1", summary.selection_names[paths_index])
            self.assertIn("In2", summary.selection_names[paths_index])
            self.assertIn("Pos1", summary.selection_names[paths_index])
            self.assertIn("Pos3", summary.selection_names[paths_index])
            self.assertIn(r"interior_test\d\inside.paa", summary.texture_paths)
            # The distance shell should be tiny compared with the detailed
            # enterable model and omit all interior/window-cut complexity.
            distance_index = summary.resolutions.index(20.0)
            self.assertLess(summary.face_counts[distance_index], 50)
            self.assertLess(summary.face_counts[0], 750)

    def test_dense_garage_cluster_keeps_road_nearest_garage_and_demotes_farther_members(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        road = OsmLineFeature(
            "way/service",
            {"highway": "service"},
            tuple(projection.to_latlon(point) for point in ((10.0, 20.0), (190.0, 20.0))),
        )
        dataset = OsmDataset(
            source_generator="garage-cluster", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        library = ProceduralBuildingLibrary(world_name="garage_cluster")
        key = BuildingVariantKey(
            "outbuilding", "gabled", 6.0, 8.0, 3.0,
            outbuilding_kind="garage",
        )
        plans = []
        for index, (x, z) in enumerate(((45.0, 25.0), (55.0, 34.0), (65.0, 43.0), (75.0, 52.0), (85.0, 61.0))):
            placement = BuildingPlacement(
                library.model_path(key), 0.0, key, key
            )
            plans.append(BuildingPlacementPlan(
                osm_key=f"way/garage-{index}", geometry_index=0,
                geometry_kind="polygon", x=x, z=z, heading_degrees=0.0,
                model_path=placement.model_path,
                support_polygon=((x-3.0, z-4.0), (x+3.0, z-4.0), (x+3.0, z+4.0), (x-3.0, z+4.0)),
                procedural_placement=placement, building_family="outbuilding",
            ))

        updated = _demote_dense_garage_clusters_to_sheds(
            plans, dataset, projection, library
        )
        kinds = [plan.procedural_placement.selected.outbuilding_kind for plan in updated]
        self.assertEqual(kinds[0], "garage")
        self.assertEqual(kinds.count("shed"), 2)
        self.assertEqual(kinds.count("garage"), 3)
        # The two farthest members should be the ones treated as likely sheds;
        # road access is the strongest evidence that an accessory building is a
        # genuine vehicle garage.
        self.assertEqual(kinds[-2:], ["shed", "shed"])

    def test_utility_interiors_use_open_halls_and_bounded_collision(self) -> None:
        from cwr_worldgen import procedural_buildings as building_models

        library = ProceduralBuildingLibrary(
            world_name="utility_interiors", generate_interiors=True
        )
        cases = (
            ({"building": "barn"}, 14.0, 42.0, "agricultural"),
            ({"building": "shed"}, 6.0, 8.0, "outbuilding"),
            ({"building": "garage"}, 7.0, 9.0, "outbuilding"),
            ({"building": "warehouse"}, 24.0, 60.0, "industrial"),
        )
        for tags, width, length, family in cases:
            with self.subTest(family=family, tags=tags):
                key = library.key_for(tags, width, length)
                self.assertTrue(key.interiors)
                self.assertEqual(key.family, family)
                self.assertEqual(
                    building_models._interior_window_openings(
                        key, -key.width_m * 0.5, key.width_m * 0.5, key.height_m
                    ),
                    (),
                )
                geometry = building_models._geometry_lod(key)
                for start in range(0, len(geometry.points), 8):
                    component = geometry.points[start:start + 8]
                    self.assertLessEqual(
                        max(point[0] for point in component)
                        - min(point[0] for point in component),
                        40.000001,
                    )
                    self.assertLessEqual(
                        max(point[2] for point in component)
                        - min(point[2] for point in component),
                        40.000001,
                    )
                    xs = [point[0] for point in component]
                    ys = [point[1] for point in component]
                    zs = [point[2] for point in component]
                    self.assertFalse(
                        min(xs) <= 0.0 <= max(xs)
                        and min(ys) <= 0.02 <= max(ys)
                        and min(zs) <= 0.0 <= max(zs)
                    )
                roadway = building_models._interior_roadway_lod(key, 0.5)
                self.assertIsNotNone(roadway)
                assert roadway is not None
                self.assertTrue(roadway.faces)

        warehouse = library.key_for({"building": "warehouse"}, 24.0, 60.0)
        warehouse_roadway = building_models._interior_roadway_lod(warehouse, 0.5)
        assert warehouse_roadway is not None
        self.assertGreater(len(warehouse_roadway.faces), 5)

    def test_enterable_procedural_windows_have_visual_cross_mullions(self) -> None:
        from cwr_worldgen.procedural_buildings import (
            BuildingVariantKey,
            _gabled_profile,
            _interior_window_openings,
            _visual_lod,
        )

        cross_texture = r"interior_test\d\white_trim.paa"
        for roof_style in ("flat", "gabled"):
            with self.subTest(roof_style=roof_style):
                key = BuildingVariantKey(
                    "residential", roof_style, 12.0, 16.0, 6.0,
                    regional_style="sweden_red", interiors=True,
                )
                visual = _visual_lod(
                    key,
                    r"interior_test\d\open_wall.paa",
                    r"interior_test\d\roof.paa",
                    35.0,
                    foundation_texture=r"interior_test\d\floor.paa",
                    interior_texture=r"interior_test\d\inside.paa",
                    window_trim_texture=cross_texture,
                )

                if roof_style == "flat":
                    wall_top = key.height_m
                else:
                    wall_top, _roof_rise, _slope_length = _gabled_profile(key, 35.0)

                half_width = key.width_m * 0.5
                half_length = key.length_m * 0.5
                door_half = min(0.8, max(0.6, half_width * 0.18))
                front_windows = _interior_window_openings(
                    key, -half_width, half_width, wall_top,
                    ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
                )
                back_windows = _interior_window_openings(key, -half_width, half_width, wall_top)
                side_windows = _interior_window_openings(key, -half_length, half_length, wall_top)
                window_count = len(front_windows) + len(back_windows) + 2 * len(side_windows)

                trim_faces = [
                    face for face in visual.faces if face.texture == cross_texture
                ]
                # Each window has four flat surround strips plus two flat
                # mullion strips; the visual shell is double-sided. This keeps
                # the same silhouette with a fraction of the old box geometry.
                self.assertEqual(len(trim_faces), window_count * 6 * 2)

                # The crossbars occupy only window-height bands. No cross may
                # descend to the Y=0 entrance threshold.
                cross_start = window_count * 4 * 2
                for face in trim_faces[cross_start:]:
                    heights = [
                        visual.points[index][1]
                        for index, _normal, _u, _v in face.vertices
                    ]
                    self.assertGreaterEqual(min(heights), 0.9 - 1e-6)

    def test_two_outbuildings_prefer_the_road_nearest_one_as_garage(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        road = OsmLineFeature(
            "way/service-pair", {"highway": "service"},
            tuple(projection.to_latlon(point) for point in ((10.0, 20.0), (190.0, 20.0))),
        )
        dataset = OsmDataset(
            source_generator="garage-pair", element_count=1, coastlines=(),
            water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )
        library = ProceduralBuildingLibrary(world_name="garage_pair")
        key = BuildingVariantKey(
            "outbuilding", "gabled", 6.0, 8.0, 3.0,
            outbuilding_kind="garage",
        )
        plans = []
        for index, (x, z) in enumerate(((60.0, 28.0), (65.0, 52.0))):
            placement = BuildingPlacement(library.model_path(key), 0.0, key, key)
            plans.append(BuildingPlacementPlan(
                osm_key=f"way/pair-{index}", geometry_index=0, geometry_kind="polygon",
                x=x, z=z, heading_degrees=0.0, model_path=placement.model_path,
                support_polygon=((x-3.0, z-4.0), (x+3.0, z-4.0), (x+3.0, z+4.0), (x-3.0, z+4.0)),
                procedural_placement=placement, building_family="outbuilding",
            ))
        updated = _demote_dense_garage_clusters_to_sheds(
            plans, dataset, projection, library
        )
        kinds = [plan.procedural_placement.selected.outbuilding_kind for plan in updated]
        self.assertEqual(kinds, ["garage", "shed"])

    def test_enterable_building_uses_house_grounding_and_foundation_stairs(self) -> None:
        cells = 20
        cell_size = 10.0
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), cells * cell_size)
        dataset = OsmDataset(
            source_generator="door-fallback", element_count=0, coastlines=(),
            water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=cells, water=(False,) * (cells * cells),
            forest=(False,) * (cells * cells), farmland=(False,) * (cells * cells),
            urban=(False,) * (cells * cells), roads=(False,) * (cells * cells),
            buildings=(False,) * (cells * cells), high_resolution=cells,
            coastline_seed_count=0,
        )
        elevations = tuple(
            20.0 - row for row in range(cells) for _column in range(cells)
        )
        key = BuildingVariantKey(
            "residential", "gabled", 10.0, 10.0, 6.0,
            regional_style="eastern_whitewash", interiors=True,
        )
        placement = BuildingPlacement(
            r"door_fallback\g\enterable.p3d", 0.0, key, key
        )
        plan = BuildingPlacementPlan(
            osm_key="way/door", geometry_index=0, geometry_kind="polygon",
            x=100.0, z=100.0, heading_degrees=0.0,
            model_path=placement.model_path,
            support_polygon=((95.0, 95.0), (105.0, 95.0), (105.0, 105.0), (95.0, 105.0)),
            procedural_placement=placement, building_family="residential",
        )
        spec = _Milestone9PlayabilitySpec(
            name="door_fallback", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=cells, cell_size=cell_size,
            max_road_objects=0, max_buildings=1, max_forest_objects=0,
            building_minimum_area=1.0, building_foundation_maximum_depth=0.5,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, strict_assets=False,
        )
        enterable_library = ProceduralBuildingLibrary(
            world_name=spec.name, maximum_foundation_depth=0.5,
            cache_enabled=False,
        )
        enterable = generate_world_objects(
            dataset, projection, raster, elevations, spec, include_roads=False,
            building_asset_library=enterable_library,
            building_placement_plans=(plan,),
        )
        self.assertEqual(enterable.building_objects, 1)
        self.assertEqual(enterable.building_foundation_rejections, 0)
        self.assertEqual(enterable.building_interior_fallbacks, 0)
        self.assertTrue(enterable_library._usage)
        used_key = next(iter(enterable_library._usage))
        self.assertTrue(used_key.interiors)
        self.assertGreater(used_key.foundation_depth_m, 0.5)

        closed_key = replace(key, interiors=False)
        closed_placement = replace(
            placement, selected=closed_key, requested=closed_key,
            model_path=r"door_fallback\g\closed.p3d",
        )
        closed_plan = replace(
            plan, model_path=closed_placement.model_path,
            procedural_placement=closed_placement,
        )
        closed_library = ProceduralBuildingLibrary(
            world_name=spec.name, maximum_foundation_depth=0.5, cache_enabled=False
        )
        closed = generate_world_objects(
            dataset, projection, raster, elevations, spec, include_roads=False,
            building_asset_library=closed_library,
            building_placement_plans=(closed_plan,),
        )
        self.assertEqual(closed.building_objects, 1)
        self.assertAlmostEqual(enterable.objects[0].y, closed.objects[0].y, places=6)
        self.assertAlmostEqual(
            enterable.maximum_building_foundation_depth,
            closed.maximum_building_foundation_depth,
            places=6,
        )

    def test_six_stage_grounding_refines_building_and_tree_supports(self) -> None:
        cells = 4
        raster = OsmRaster(
            cells=cells, water=(False,) * 16, forest=(True,) * 16,
            farmland=(False,) * 16, urban=(False,) * 16, roads=(False,) * 16,
            buildings=(False,) * 16, high_resolution=cells,
            coastline_seed_count=0,
        )
        elevations = tuple(
            float(row * cells + column)
            for row in range(cells)
            for column in range(cells)
        )
        building = BuildingPlacementPlan(
            osm_key="way/building", geometry_index=0, geometry_kind="polygon",
            x=15.0, z=15.0, heading_degrees=0.0,
            model_path=r"six_stage\g\building.p3d",
            support_polygon=((0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)),
        )
        square_model = r"data3d\les ctverec pruchozi_T1.p3d"
        provisional = ObjectGenerationResult(
            objects=(
                WorldObject(1, building.model_path, 15.0, 10.1, 15.0),
                WorldObject(2, square_model, 30.0, 10.15, 30.0),
            ),
            road_objects=0, building_objects=1, forest_objects=1,
            road_objects_truncated=False, building_objects_truncated=False,
            forest_objects_truncated=False,
        )
        spec = SimpleNamespace(
            cells=cells, cell_size=10.0, name="six_stage",
            forest_tree_model=square_model,
            forest_everon_steep_model=r"data3d\les trojuhelnik pruchozi.p3d",
            forest_tree_spacing=20.0, forest_everon_steep_footprint=18.0,
            forest_single_tree_footprint=2.0, forest_ground_clearance=0.15,
            iterative_grounding_maximum_adjustment=2.0,
            iterative_grounding_strength=0.70,
        )
        refined, report = refine_iterative_grounding_terrain(
            elevations, provisional, (building,), raster, spec
        )
        self.assertEqual(report.building_supports, 1)
        self.assertEqual(report.tree_supports, 1)
        self.assertGreater(report.adjusted_cells, 0)
        self.assertLessEqual(report.maximum_adjustment, 1.4 + 1e-9)
        self.assertNotEqual(refined, elevations)

    def test_lightweight_grounding_planner_only_builds_rigid_supports(self) -> None:
        cells = 4
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        raster = OsmRaster(
            cells=cells, water=(False,) * 16, forest=(True,) * 16,
            farmland=(False,) * 16, urban=(False,) * 16, roads=(False,) * 16,
            buildings=(False,) * 16, high_resolution=cells,
            coastline_seed_count=0,
        )
        dataset = OsmDataset(
            source_generator="grounding-plan", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        building = BuildingPlacementPlan(
            osm_key="way/building", geometry_index=0, geometry_kind="polygon",
            x=12.5, z=12.5, heading_degrees=0.0,
            model_path=r"grounding\g\building.p3d",
            support_polygon=((2.0, 2.0), (23.0, 2.0), (23.0, 23.0), (2.0, 23.0)),
        )
        spec = _Milestone9PlayabilitySpec(
            name="grounding", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=cells, cell_size=25.0,
            max_road_objects=0, max_buildings=1, max_forest_objects=20,
            forest_tree_spacing=50.0, forest_low_anchor=False,
            forest_undergrowth_enabled=True, forest_border_enabled=True,
            ditch_grass_enabled=True, barriers_enabled=True, bridges_enabled=True,
            rural_vegetation_enabled=True, strict_assets=False,
        )
        events: list[tuple[int, str]] = []
        result = plan_iterative_grounding_objects(
            dataset, projection, raster, (2.0,) * 16, spec, (building,),
            progress_callback=lambda percent, stage: events.append((percent, stage)),
        )
        self.assertEqual(result.building_objects, 1)
        self.assertGreater(result.forest_objects, 0)
        self.assertEqual(len(result.objects), result.building_objects + result.forest_objects)
        self.assertEqual(result.forest_undergrowth_objects, 0)
        self.assertEqual(result.bridge_objects, 0)
        self.assertTrue(any("Planning primary forest supports" in stage for _, stage in events))
        self.assertFalse(any(stage == "Placing primary forest blocks" for _, stage in events))

    def test_only_fully_water_covered_below_sea_buildings_are_omitted(self) -> None:
        cells = 4
        cell_size = 10.0
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 40.0)
        dataset = OsmDataset(
            source_generator="submerged-buildings", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        water = [False] * 16
        water[0] = True
        water[15] = True
        raster = OsmRaster(
            cells=cells, water=tuple(water), forest=(False,) * 16,
            farmland=(False,) * 16, urban=(False,) * 16, roads=(False,) * 16,
            buildings=(False,) * 16, high_resolution=cells,
            coastline_seed_count=0,
        )
        elevations = [-5.0] * 16
        for index in (10, 11, 14, 15):
            elevations[index] = 5.0

        def plan(key: str, polygon: tuple[tuple[float, float], ...]) -> BuildingPlacementPlan:
            x = sum(point[0] for point in polygon) / len(polygon)
            z = sum(point[1] for point in polygon) / len(polygon)
            return BuildingPlacementPlan(
                osm_key=key, geometry_index=0, geometry_kind="polygon",
                x=x, z=z, heading_degrees=0.0,
                model_path=r"submerged\house.p3d", support_polygon=polygon,
                building_family="residential",
            )

        fully_below = plan(
            "way/fully-below", ((1.0, 1.0), (9.0, 1.0), (9.0, 9.0), (1.0, 9.0))
        )
        partly_submerged = plan(
            "way/partial", ((5.0, 1.0), (15.0, 1.0), (15.0, 9.0), (5.0, 9.0))
        )
        fully_water_above_sea = plan(
            "way/above-sea", ((31.0, 31.0), (39.0, 31.0), (39.0, 39.0), (31.0, 39.0))
        )
        spec = _Milestone9PlayabilitySpec(
            name="submerged", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0), cells=cells, cell_size=cell_size,
            sea_level=0.0, max_road_objects=0, max_buildings=3,
            max_forest_objects=0, building_minimum_area=1.0,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, tuple(elevations), spec,
            include_roads=False,
            building_placement_plans=(
                fully_below, partly_submerged, fully_water_above_sea,
            ),
        )
        self.assertEqual(result.building_objects, 2)
        self.assertEqual(result.building_fully_submerged_rejections, 1)
        self.assertEqual(
            {(round(obj.x, 1), round(obj.z, 1)) for obj in result.objects},
            {(10.0, 5.0), (35.0, 35.0)},
        )

    def test_semantic_building_families_and_church_tower(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary

        library = ProceduralBuildingLibrary(world_name="semantic_test", maximum_variants=16)
        church = library.key_for({"building": "church", "amenity": "place_of_worship"}, 14.0, 24.0)
        school = library.key_for({"building": "school", "amenity": "school"}, 18.0, 32.0)
        shop = library.key_for({"building": "retail", "shop": "convenience"}, 10.0, 14.0)
        self.assertEqual(church.family, "church")
        self.assertEqual(school.family, "school")
        self.assertEqual(shop.family, "shop")
        social = library.key_for({"building": "warehouse", "amenity": "social_facility", "social_facility": "group_home"}, 20.0, 36.0)
        self.assertEqual(social.family, "urban")
        church_visual = _visual_lod(
            church, r"semantic_test\g\w_church.paa", r"semantic_test\g\r_gabled.paa",
            35.0, r"semantic_test\g\f_church.paa",
        )
        self.assertGreaterEqual(max(point[1] for point in church_visual.points), 24.0)
        self.assertGreater(max(point[1] for point in church_visual.points), church.height_m + 10.0)

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        def building(key: str, tags: dict[str, str], bounds: tuple[float, float, float, float]) -> OsmPolygonFeature:
            x0, z0, x1, z1 = bounds
            ring = tuple(projection.to_latlon(point) for point in ((x0,z0),(x1,z0),(x1,z1),(x0,z1),(x0,z0)))
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))
        dataset = OsmDataset(
            source_generator="semantic-buildings", element_count=3,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(
                building("way/church", {"building":"church", "amenity":"place_of_worship"}, (100,100,124,114)),
                building("way/school", {"building":"school", "amenity":"school"}, (200,100,232,118)),
                building("way/shop", {"building":"retail", "shop":"convenience"}, (300,100,314,110)),
            ),
        )
        library.prepare(dataset, projection, 12.0)
        for feature in dataset.building_polygons:
            points = [projection.to_world(point) for point in feature.polygons[0].outer[:-1]]
            library.place_polygon(feature.tags, points)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = library.write_assets(root, root / "catalogue.json")
            by_family = {asset.key.family: asset for asset in result.model_assets}
            self.assertEqual(set(by_family), {"church", "school", "shop"})
            self.assertGreater(by_family["church"].visual_face_count, by_family["school"].visual_face_count)

    def test_church_model_requires_explicit_christian_church_semantics(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary

        library = ProceduralBuildingLibrary(world_name="strict_church", maximum_variants=16)
        explicit = library.key_for({"building": "church"}, 14.0, 24.0)
        christian = library.key_for(
            {"building": "yes", "amenity": "place_of_worship", "religion": "christian"},
            14.0, 24.0,
        )
        generic = library.key_for(
            {"building": "yes", "amenity": "place_of_worship"}, 14.0, 24.0
        )
        mosque = library.key_for(
            {"building": "mosque", "amenity": "place_of_worship", "religion": "muslim"},
            14.0, 24.0,
        )
        self.assertEqual(explicit.family, "church")
        self.assertEqual(christian.family, "church")
        self.assertNotEqual(generic.family, "church")
        self.assertNotEqual(mosque.family, "church")

    def test_town_context_uses_townhouses_and_apartments_not_farm_buildings(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary

        projection = BboxProjection.create((59.20, 17.90, 59.30, 18.10), 1000.0)

        def polygon_feature(key: str, tags: dict[str, str], bounds: tuple[float, float, float, float]) -> OsmPolygonFeature:
            x0, z0, x1, z1 = bounds
            ring = tuple(
                projection.to_latlon(point)
                for point in ((x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0))
            )
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        urban = polygon_feature(
            "way/urban", {"landuse": "residential", "category": "urban"},
            (50.0, 50.0, 950.0, 950.0),
        )
        small = polygon_feature("way/townhouse", {"building": "yes"}, (100.0, 100.0, 112.0, 120.0))
        large = polygon_feature("way/apartments", {"building": "yes"}, (200.0, 100.0, 240.0, 130.0))
        town = OsmPointFeature(
            "node/town", {"place": "town", "name": "Test Town"},
            projection.to_latlon((150.0, 120.0)),
        )
        dataset = OsmDataset(
            source_generator="town-context", element_count=4,
            coastlines=(), water=(), forests=(), farmland=(), urban=(urban,), roads=(),
            building_polygons=(small, large), places=(town,),
        )
        library = ProceduralBuildingLibrary(world_name="town_context", maximum_variants=16)
        library.prepare(dataset, projection, 12.0)
        small_points = [projection.to_world(point) for point in small.polygons[0].outer[:-1]]
        large_points = [projection.to_world(point) for point in large.polygons[0].outer[:-1]]
        self.assertEqual(library.plan_polygon(small.tags, small_points).selected.family, "townhouse")
        self.assertEqual(library.plan_polygon(large.tags, large_points).selected.family, "urban")

    def test_residential_landuse_alone_does_not_create_town_buildings(self) -> None:
        projection = BboxProjection.create((59.20, 17.90, 59.30, 18.10), 1000.0)

        def polygon_feature(key: str, tags: dict[str, str], bounds: tuple[float, float, float, float]) -> OsmPolygonFeature:
            x0, z0, x1, z1 = bounds
            ring = tuple(
                projection.to_latlon(point)
                for point in ((x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0))
            )
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        residential = polygon_feature(
            "way/residential", {"landuse": "residential", "category": "urban"},
            (0.0, 0.0, 1000.0, 1000.0),
        )
        building = polygon_feature("way/building", {"building": "yes"}, (100.0, 100.0, 112.0, 120.0))
        dataset = OsmDataset(
            source_generator="residential-only", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(residential,), roads=(),
            building_polygons=(building,),
        )
        library = ProceduralBuildingLibrary(world_name="residential_only", maximum_variants=16)
        library.prepare(dataset, projection, 12.0)
        points = [projection.to_world(point) for point in building.polygons[0].outer[:-1]]
        self.assertNotIn(library.plan_polygon(building.tags, points).selected.family, {"townhouse", "urban"})

    def test_town_building_context_stops_after_one_source_kilometre(self) -> None:
        projection = BboxProjection.create((59.20, 17.90, 59.30, 18.10), 1000.0)
        ring = tuple(
            projection.to_latlon(point)
            for point in ((100.0, 100.0), (112.0, 100.0), (112.0, 120.0), (100.0, 120.0), (100.0, 100.0))
        )
        building = OsmPolygonFeature("way/building", {"building": "yes"}, (GeoPolygon(ring),))
        distant_town = OsmPointFeature(
            "node/town", {"place": "town", "name": "Distant Town"},
            projection.to_latlon((260.0, 110.0)),
        )
        dataset = OsmDataset(
            source_generator="town-distance", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(building,), places=(distant_town,),
        )
        library = ProceduralBuildingLibrary(world_name="town_distance", maximum_variants=16)
        library.prepare(dataset, projection, 12.0)
        points = [projection.to_world(point) for point in building.polygons[0].outer[:-1]]
        self.assertNotIn(library.plan_polygon(building.tags, points).selected.family, {"townhouse", "urban"})

    def test_village_context_keeps_large_generic_building_residential(self) -> None:
        projection = BboxProjection.create((59.20, 17.90, 59.30, 18.10), 1000.0)
        ring = tuple(
            projection.to_latlon(point)
            for point in ((100.0, 100.0), (125.0, 100.0), (125.0, 130.0), (100.0, 130.0), (100.0, 100.0))
        )
        building = OsmPolygonFeature("way/village-building", {"building": "yes"}, (GeoPolygon(ring),))
        village = OsmPointFeature(
            "node/village", {"place": "village", "name": "Test Village"},
            projection.to_latlon((112.0, 115.0)),
        )
        dataset = OsmDataset(
            source_generator="village-context", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(building,), places=(village,),
        )
        library = ProceduralBuildingLibrary(world_name="village_context", maximum_variants=16)
        library.prepare(dataset, projection, 12.0)
        points = [projection.to_world(point) for point in building.polygons[0].outer[:-1]]
        placement = library.plan_polygon(building.tags, points)
        self.assertEqual(library._settlement_context(112.0, 115.0), "village")
        self.assertEqual(placement.selected.family, "residential")

    def test_city_promotes_medium_generic_building_before_town(self) -> None:
        from cwr_worldgen.procedural_buildings import _family

        tags = {"building": "yes", "building:levels": "2"}
        self.assertEqual(_family(tags, 15.0, 22.0, settlement_context="town"), "townhouse")
        self.assertEqual(_family(tags, 15.0, 22.0, settlement_context="city"), "urban")

    def test_variant_reuse_prefers_physical_fit_envelope_and_aspect(self) -> None:
        from cwr_worldgen.procedural_buildings import BuildingVariantKey

        library = ProceduralBuildingLibrary(world_name="fit_score", maximum_variants=2)
        requested = BuildingVariantKey("townhouse", "gable", 10.0, 30.0, 6.0)
        # The narrow candidate violates the stricter 70%-115% dimension
        # envelope.  The close 11x29 replacement remains physically plausible.
        too_narrow = BuildingVariantKey("townhouse", "gable", 6.0, 30.0, 6.0)
        plausible = BuildingVariantKey("townhouse", "gable", 11.0, 29.0, 6.0)
        self.assertFalse(library._variant_within_fit_envelope(requested, too_narrow))
        self.assertTrue(library._variant_within_fit_envelope(requested, plausible))
        self.assertEqual(library._best_variant(requested, (too_narrow, plausible)), plausible)

    def test_variant_reuse_prefers_size_over_matching_palette(self) -> None:
        from cwr_worldgen.procedural_buildings import BuildingVariantKey

        library = ProceduralBuildingLibrary(world_name="fit_before_palette", maximum_variants=2)
        requested = BuildingVariantKey(
            "residential", "gable", 10.0, 20.0, 6.0, regional_style="sweden_red"
        )
        exact_palette_wrong_size = BuildingVariantKey(
            "residential", "gable", 5.0, 20.0, 6.0, regional_style="sweden_red"
        )
        close_size_other_palette = BuildingVariantKey(
            "residential", "gable", 10.0, 19.0, 6.0, regional_style="sweden_yellow"
        )
        pool = library._reuse_candidates(
            requested, (exact_palette_wrong_size, close_size_other_palette)
        )
        self.assertEqual(pool, [close_size_other_palette])
        self.assertEqual(library._best_variant(requested, pool), close_size_other_palette)

    def test_sweden_region_biases_houses_toward_red_timber_styles(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary

        projection = BboxProjection.create((59.20, 17.90, 59.30, 18.10), 1000.0)
        dataset = OsmDataset(
            source_generator="sweden-region", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        library = ProceduralBuildingLibrary(world_name="sweden_region", maximum_variants=64)
        library.prepare(dataset, projection, 12.0)
        self.assertEqual(library.region_identifier, "sweden")
        styles = [
            library.key_for(
                {"building": "house", "name": f"House {index}"},
                8.0 + (index % 5) * 2.0,
                12.0 + (index % 7) * 2.0,
            ).regional_style
            for index in range(40)
        ]
        self.assertEqual(styles.count("sweden_red"), 19)
        self.assertLess(styles.count("sweden_red"), 24)
        explicit_red = library.key_for(
            {"building": "house", "building:colour": "red"}, 10.0, 16.0
        )
        self.assertEqual(explicit_red.regional_style, "sweden_red")
        apartments = [
            library.key_for(
                {"building": "apartments", "name": f"Block {index}"},
                18.0 + (index % 4) * 4.0,
                24.0 + (index % 5) * 5.0,
            ).regional_style
            for index in range(20)
        ]
        self.assertNotIn("sweden_red", apartments)

    def test_eastern_europe_region_adds_masonry_and_panel_variants(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary

        projection = BboxProjection.create((52.10, 20.90, 52.30, 21.10), 1000.0)
        dataset = OsmDataset(
            source_generator="eastern-europe-region", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        library = ProceduralBuildingLibrary(world_name="eastern_region", maximum_variants=128)
        library.prepare(dataset, projection, 12.0)
        self.assertEqual(library.region_identifier, "eastern_europe")

        house_styles = {
            library.key_for(
                {"building": "house", "name": f"House {index}"},
                8.0 + (index % 5) * 2.0,
                12.0 + (index % 7) * 2.0,
            ).regional_style
            for index in range(80)
        }
        self.assertTrue({"eastern_plaster", "eastern_brick", "eastern_whitewash"}.issubset(house_styles))

        explicit_brick = library.key_for(
            {"building": "house", "building:material": "brick"}, 10.0, 16.0
        )
        self.assertEqual(explicit_brick.regional_style, "eastern_brick")
        explicit_white = library.key_for(
            {"building": "house", "building:colour": "white"}, 10.0, 16.0
        )
        self.assertEqual(explicit_white.regional_style, "eastern_whitewash")
        concrete_apartments = library.key_for(
            {"building": "apartments", "building:material": "concrete"}, 22.0, 38.0,
            settlement_context="city",
        )
        self.assertEqual(concrete_apartments.family, "urban")
        self.assertEqual(concrete_apartments.regional_style, "eastern_panel")

        from cwr_worldgen.procedural_buildings import _wall_texture_image
        for family, style in (
            ("residential", "eastern_plaster"),
            ("townhouse", "eastern_brick"),
            ("agricultural", "eastern_whitewash"),
            ("urban", "eastern_panel"),
        ):
            image = _wall_texture_image(family, regional_style=style)
            self.assertEqual(image.size, (128, 128))
            self.assertGreater(len(image.getcolors(maxcolors=65536) or ()), 4)

    def test_africa_region_adds_earth_whitewash_block_and_colour_variants(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary, _wall_texture_image

        projection = BboxProjection.create((-1.40, 36.70, -1.20, 36.90), 1000.0)
        dataset = OsmDataset(
            source_generator="africa-region", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        library = ProceduralBuildingLibrary(world_name="africa_region", maximum_variants=128)
        library.prepare(dataset, projection, 12.0)
        self.assertEqual(library.region_identifier, "africa")

        house_styles = {
            library.key_for(
                {"building": "house", "name": f"House {index}"},
                8.0 + (index % 5) * 2.0,
                12.0 + (index % 7) * 2.0,
            ).regional_style
            for index in range(120)
        }
        self.assertTrue({
            "africa_earth", "africa_whitewash", "africa_block", "africa_colour"
        }.issubset(house_styles))
        self.assertEqual(
            library.key_for(
                {"building": "house", "building:material": "adobe"}, 10.0, 16.0
            ).regional_style,
            "africa_earth",
        )
        block = library.key_for(
            {"building": "apartments", "building:material": "concrete"},
            24.0, 40.0, settlement_context="city",
        )
        self.assertEqual(block.family, "urban")
        self.assertEqual(block.regional_style, "africa_block")
        colourful = library.key_for(
            {"building": "house", "building:colour": "turquoise"}, 10.0, 16.0
        )
        self.assertEqual(colourful.regional_style, "africa_colour")
        for family, style in (
            ("residential", "africa_earth"),
            ("townhouse", "africa_whitewash"),
            ("urban", "africa_block"),
            ("shop", "africa_colour"),
        ):
            image = _wall_texture_image(family, regional_style=style)
            self.assertEqual(image.size, (128, 128))
            self.assertGreater(len(image.getcolors(maxcolors=65536) or ()), 4)

    def test_western_europe_region_adds_stucco_brick_stone_and_half_timber(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary, _wall_texture_image

        projection = BboxProjection.create((48.70, 2.10, 49.00, 2.50), 1000.0)
        dataset = OsmDataset(
            source_generator="western-europe-region", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        library = ProceduralBuildingLibrary(
            world_name="western_region", maximum_variants=128,
            generate_interiors=True,
        )
        library.prepare(dataset, projection, 12.0)
        self.assertEqual(library.region_identifier, "western_europe")

        house_styles = {
            library.key_for(
                {"building": "house", "name": f"House {index}"},
                8.0 + (index % 5) * 2.0,
                12.0 + (index % 7) * 2.0,
            ).regional_style
            for index in range(160)
        }
        self.assertTrue({
            "western_stucco", "western_brick", "western_stone",
            "western_half_timber",
        }.issubset(house_styles))
        self.assertEqual(
            library.key_for(
                {"building": "house", "building:material": "brick"}, 10.0, 16.0
            ).regional_style,
            "western_brick",
        )
        self.assertEqual(
            library.key_for(
                {"building": "house", "building:material": "limestone"}, 10.0, 16.0
            ).regional_style,
            "western_stone",
        )
        enterable = library.key_for(
            {"building": "house", "building:material": "stucco"}, 10.0, 16.0
        )
        self.assertEqual(enterable.regional_style, "western_stucco")
        self.assertTrue(enterable.interiors)
        half_timbered = library.key_for(
            {"building": "house", "building:material": "half_timbered"}, 10.0, 16.0
        )
        self.assertEqual(half_timbered.regional_style, "western_half_timber")
        for family, style in (
            ("residential", "western_stucco"),
            ("townhouse", "western_brick"),
            ("agricultural", "western_stone"),
            ("shop", "western_half_timber"),
        ):
            image = _wall_texture_image(family, regional_style=style)
            self.assertEqual(image.size, (128, 128))
            self.assertGreater(len(image.getcolors(maxcolors=65536) or ()), 4)

    def test_middle_east_region_adds_sandstone_adobe_whitewash_and_concrete(self) -> None:
        from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary, _wall_texture_image

        projection = BboxProjection.create((24.60, 46.60, 24.80, 46.80), 1000.0)
        dataset = OsmDataset(
            source_generator="middle-east-region", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        library = ProceduralBuildingLibrary(world_name="middle_east_region", maximum_variants=128)
        library.prepare(dataset, projection, 12.0)
        self.assertEqual(library.region_identifier, "middle_east")

        house_styles = {
            library.key_for(
                {"building": "house", "name": f"House {index}"},
                8.0 + (index % 5) * 2.0,
                12.0 + (index % 7) * 2.0,
            ).regional_style
            for index in range(120)
        }
        self.assertTrue({
            "middle_east_sandstone", "middle_east_adobe",
            "middle_east_whitewash", "middle_east_concrete",
        }.issubset(house_styles))
        sandstone = library.key_for(
            {"building": "house", "building:material": "limestone"}, 10.0, 16.0
        )
        self.assertEqual(sandstone.regional_style, "middle_east_sandstone")
        self.assertEqual(sandstone.roof_style, "flat")
        adobe = library.key_for(
            {"building": "house", "building:material": "mud"}, 10.0, 16.0
        )
        self.assertEqual(adobe.regional_style, "middle_east_adobe")
        self.assertEqual(adobe.roof_style, "flat")
        concrete = library.key_for(
            {"building": "apartments", "building:material": "concrete"},
            24.0, 40.0, settlement_context="city",
        )
        self.assertEqual(concrete.regional_style, "middle_east_concrete")
        self.assertEqual(concrete.roof_style, "flat")
        for family, style in (
            ("residential", "middle_east_sandstone"),
            ("townhouse", "middle_east_whitewash"),
            ("agricultural", "middle_east_adobe"),
            ("urban", "middle_east_concrete"),
        ):
            image = _wall_texture_image(family, regional_style=style)
            self.assertEqual(image.size, (128, 128))
            self.assertGreater(len(image.getcolors(maxcolors=65536) or ()), 4)

    def test_explicit_country_tag_controls_regional_detection(self) -> None:
        from cwr_worldgen.building_semantics import detect_region

        eastern = detect_region((0.0, 0.0, 0.1, 0.1), ({"addr:country": "RO"},))
        self.assertIsNotNone(eastern)
        self.assertEqual(eastern.identifier, "eastern_europe")
        africa = detect_region((0.0, 0.0, 0.1, 0.1), ({"addr:country": "NG"},))
        self.assertIsNotNone(africa)
        self.assertEqual(africa.identifier, "africa")
        middle_east = detect_region((0.0, 0.0, 0.1, 0.1), ({"addr:country": "AE"},))
        self.assertIsNotNone(middle_east)
        self.assertEqual(middle_east.identifier, "middle_east")
        western = detect_region((47.9, 16.3, 48.4, 16.8), ({"addr:country": "AT"},))
        self.assertIsNotNone(western)
        self.assertEqual(western.identifier, "western_europe")

    def test_semantic_buildings_are_prioritised_before_the_building_cap(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def building(key: str, tags: dict[str, str], x: float) -> OsmPolygonFeature:
            ring = tuple(
                projection.to_latlon(point)
                for point in ((x, 100.0), (x + 20.0, 100.0), (x + 20.0, 120.0), (x, 120.0), (x, 100.0))
            )
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        dataset = OsmDataset(
            source_generator="semantic-priority",
            element_count=8,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(
                *(building(f"way/ordinary-{index:02d}", {"building": "house"}, 50.0 + index * 40.0) for index in range(5)),
                building("way/zz-church", {"building": "church", "amenity": "place_of_worship"}, 600.0),
                building("way/zz-school", {"building": "school", "amenity": "school"}, 650.0),
                building("way/zz-shop", {"building": "retail", "shop": "convenience"}, 700.0),
            ),
        )

        class FakeLibrary:
            @staticmethod
            def _family(tags: dict[str, str]) -> str:
                if tags.get("amenity") == "place_of_worship":
                    return "church"
                if tags.get("amenity") == "school":
                    return "school"
                if tags.get("shop"):
                    return "shop"
                return "ordinary"

            def place_polygon(self, tags, points, *, road_point=None):
                return SimpleNamespace(
                    heading_degrees=0.0,
                    model_path=f"{self._family(tags)}.p3d",
                    selected=SimpleNamespace(width_m=20.0, length_m=20.0),
                )

            def place_point(self, tags, footprint, heading):
                return SimpleNamespace(
                    heading_degrees=heading,
                    model_path=f"{self._family(tags)}.p3d",
                    selected=SimpleNamespace(width_m=footprint, length_m=footprint),
                )

        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01),
            cells=40,
            cell_size=25.0,
            max_road_objects=0,
            max_buildings=3,
            max_forest_objects=0,
            strict_assets=False,
        )
        raster = OsmRaster(
            cells=40,
            water=(False,) * 1600,
            forest=(False,) * 1600,
            farmland=(False,) * 1600,
            urban=(False,) * 1600,
            roads=(False,) * 1600,
            buildings=(False,) * 1600,
            high_resolution=40,
            coastline_seed_count=0,
        )
        result = generate_world_objects(
            dataset,
            projection,
            raster,
            [0.0] * 1600,
            spec,
            include_roads=False,
            building_asset_library=FakeLibrary(),
        )
        self.assertEqual(result.building_objects, 3)
        self.assertTrue(result.building_objects_truncated)
        self.assertEqual(
            {obj.model_path for obj in result.objects},
            {"church.p3d", "school.p3d", "shop.p3d"},
        )

    def test_variant_cap_reserves_semantic_building_families(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def building(key: str, tags: dict[str, str], x: float, width: float, length: float) -> OsmPolygonFeature:
            ring = tuple(
                projection.to_latlon(point)
                for point in ((x, 300.0), (x + width, 300.0), (x + width, 300.0 + length), (x, 300.0 + length), (x, 300.0))
            )
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        ordinary = tuple(
            building(f"way/house-{index:02d}", {"building": "house"}, 20.0 + index * 25.0, 8.0 + index, 10.0 + index)
            for index in range(12)
        )
        semantic = (
            building("way/church", {"building": "church", "amenity": "place_of_worship"}, 500.0, 18.0, 30.0),
            building("way/school", {"building": "school", "amenity": "school"}, 600.0, 24.0, 40.0),
            building("way/shop", {"building": "retail", "shop": "convenience"}, 700.0, 14.0, 18.0),
        )
        dataset = OsmDataset(
            source_generator="semantic-variant-cap",
            element_count=len(ordinary) + len(semantic),
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=ordinary + semantic,
        )
        library = ProceduralBuildingLibrary(world_name="semantic_cap", maximum_variants=4)
        library.prepare(dataset, projection, 12.0)
        selected_families = {key.family for key in library._mapping.values()}
        self.assertEqual(selected_families, {"residential", "church", "school", "shop"})

    def test_school_variants_never_reuse_barns_when_variant_cap_is_tight(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

        def building(key: str, tags: dict[str, str], x: float, width: float, length: float) -> OsmPolygonFeature:
            ring = tuple(projection.to_latlon(point) for point in (
                (x, 250.0), (x + width, 250.0), (x + width, 250.0 + length),
                (x, 250.0 + length), (x, 250.0),
            ))
            return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

        features = (
            building("way/house", {"building": "house"}, 30.0, 10.0, 14.0),
            building("way/barn", {"building": "barn"}, 100.0, 12.0, 44.0),
            building("way/school-small", {"building": "school", "amenity": "school"}, 220.0, 8.0, 12.0),
            building("way/school-medium", {"building": "school", "amenity": "school"}, 350.0, 12.0, 26.0),
            building("way/school-large", {"building": "school", "amenity": "school"}, 520.0, 12.0, 46.0),
        )
        dataset = OsmDataset(
            source_generator="school-cap-family", element_count=len(features),
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=features,
        )
        library = ProceduralBuildingLibrary(world_name="school_cap", maximum_variants=3)
        library.prepare(dataset, projection, 12.0)
        for feature in features[2:]:
            polygon = feature.polygons[0]
            points = [projection.to_world(point) for point in polygon.outer[:-1]]
            placement = library.plan_polygon(feature.tags, points)
            self.assertEqual(placement.requested.family, "school")
            self.assertEqual(placement.selected.family, "school")

    def test_bus_stop_and_procedural_sites_are_deterministic(self) -> None:
        from cwr_worldgen.osm import OsmPointFeature
        from cwr_worldgen.semantic_features import ProceduralSiteLibrary, generate_semantic_objects

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        road = OsmLineFeature(
            "way/road", {"highway":"primary"},
            tuple(projection.to_latlon(point) for point in ((100.0, 500.0), (900.0, 500.0))),
        )
        bus = OsmPointFeature("node/bus", {"landmark":"bus_stop"}, projection.to_latlon((300.0, 506.0)))
        def site(key: str, kind: str, bounds: tuple[float,float,float,float]) -> OsmPolygonFeature:
            x0,z0,x1,z1=bounds
            ring=tuple(projection.to_latlon(point) for point in ((x0,z0),(x1,z0),(x1,z1),(x0,z1),(x0,z0)))
            return OsmPolygonFeature(key, {"site":kind}, (GeoPolygon(ring),))
        dataset = OsmDataset(
            source_generator="semantic-sites", element_count=4,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
            landmarks=(bus,),
            sites=(
                site("way/pitch","sports_pitch",(100,100,180,150)),
                site("way/parking","parking",(220,100,280,140)),
            ),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0,0.0,0.01,0.01),
            cells=40, cell_size=25.0, strict_assets=False, bus_stops_enabled=True,
        )
        elevations=[10.0]*(spec.cells*spec.cells)
        first_library=ProceduralSiteLibrary("semantic_world")
        first_library.prepare(dataset, projection)
        first=generate_semantic_objects(dataset, projection, elevations, spec, first_library, starting_object_id=1)
        second_library=ProceduralSiteLibrary("semantic_world")
        second_library.prepare(dataset, projection)
        second=generate_semantic_objects(dataset, projection, elevations, spec, second_library, starting_object_id=1)
        self.assertEqual(first, second)
        self.assertEqual(first.bus_stop_objects, 1)
        self.assertEqual(first.sports_pitch_objects, 0)
        self.assertEqual(first.parking_objects, 0)
        self.assertIn(r"o\misc\aut_z_st.p3d", {obj.model_path for obj in first.objects})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            assets=first_library.write_assets(root, root/"sites.json")
            self.assertEqual(assets.generated_variants, 0)
            self.assertFalse((root/"s"/"f.paa").exists())
            self.assertFalse((root/"s"/"p.paa").exists())

    def test_cemeteries_receive_deterministic_stock_gravestones(self) -> None:
        from cwr_worldgen.semantic_features import (
            GRAVE_MODELS, ProceduralSiteLibrary, generate_semantic_objects,
        )

        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        ring = tuple(
            projection.to_latlon(point)
            for point in ((200.0, 200.0), (300.0, 200.0), (300.0, 300.0), (200.0, 300.0), (200.0, 200.0))
        )
        cemetery = OsmPolygonFeature(
            "way/cemetery", {"site": "cemetery", "landuse": "cemetery"},
            (GeoPolygon(ring),),
        )
        dataset = OsmDataset(
            source_generator="cemetery-test", element_count=1, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(), sites=(cemetery,),
        )
        spec = _Milestone9PlayabilitySpec(
            name="cemetery_test", heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 0.01, 0.01), cells=40, cell_size=25.0,
            strict_assets=False, maximum_grave_objects=200, grave_spacing=5.0,
        )
        raster = OsmRaster(
            cells=40, water=(False,) * 1600, forest=(False,) * 1600,
            farmland=(False,) * 1600, urban=(False,) * 1600, roads=(False,) * 1600,
            buildings=(False,) * 1600, high_resolution=40, coastline_seed_count=0,
        )
        library = ProceduralSiteLibrary(spec.name)
        library.prepare(dataset, projection)
        elevations = tuple(
            float(row * 8 + column * 2)
            for row in range(spec.cells)
            for column in range(spec.cells)
        )
        first = generate_semantic_objects(
            dataset, projection, elevations, spec, library,
            starting_object_id=1, raster=raster,
        )
        second_library = ProceduralSiteLibrary(spec.name)
        second_library.prepare(dataset, projection)
        second = generate_semantic_objects(
            dataset, projection, elevations, spec, second_library,
            starting_object_id=1, raster=raster,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.cemetery_sites, 1)
        self.assertGreater(first.grave_objects, 20)
        self.assertLessEqual(first.grave_objects, 200)
        self.assertTrue({obj.model_path for obj in first.objects}.issubset(set(GRAVE_MODELS)))
        from cwr_worldgen.osm import _oriented_rectangle, _polygon_elevation_extrema
        from cwr_worldgen.semantic_features import grave_grounding_profile
        profile_lifts = {grave_grounding_profile(model).origin_lift_metres for model in GRAVE_MODELS}
        self.assertGreater(len(profile_lifts), 1)
        for grave in first.objects:
            profile = grave_grounding_profile(grave.model_path)
            support_polygon = _oriented_rectangle(
                grave.x, grave.z,
                max(spec.grave_footprint, profile.width_metres),
                max(spec.grave_footprint, profile.length_metres),
                grave.heading_degrees, margin=0.08,
            )
            _minimum, support = _polygon_elevation_extrema(
                elevations, spec.cells, spec.cell_size, support_polygon
            )
            self.assertAlmostEqual(
                grave.y,
                support + profile.origin_lift_metres + spec.grave_ground_clearance,
                places=6,
            )
            self.assertGreaterEqual(
                grave.y - profile.origin_lift_metres + 1e-7,
                support + spec.grave_ground_clearance,
            )

    def test_cemetery_graves_avoid_final_building_footprints_and_road_corridors(self) -> None:
        from cwr_worldgen.osm import BuildingPlacementPlan
        from cwr_worldgen.semantic_features import ProceduralSiteLibrary, generate_semantic_objects

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        ll = projection.to_latlon
        cemetery_ring = tuple(
            ll(point) for point in ((20.0, 20.0), (180.0, 20.0), (180.0, 180.0), (20.0, 180.0), (20.0, 20.0))
        )
        cemetery = OsmPolygonFeature(
            "way/cemetery", {"site": "cemetery"}, (GeoPolygon(cemetery_ring),)
        )
        road = OsmLineFeature(
            "way/road", {"highway": "service"}, tuple(ll(point) for point in ((20.0, 100.0), (180.0, 100.0)))
        )
        dataset = OsmDataset(
            source_generator="cemetery-exclusions", element_count=2, coastlines=(),
            water=(), forests=(), farmland=(), urban=(), roads=(road,), sites=(cemetery,),
        )
        building_polygon = ((75.0, 75.0), (125.0, 75.0), (125.0, 125.0), (75.0, 125.0))
        plan = BuildingPlacementPlan(
            osm_key="way/church", geometry_index=0, geometry_kind="polygon",
            x=100.0, z=100.0, heading_degrees=0.0, model_path="test.p3d",
            support_polygon=building_polygon, building_family="church",
        )
        spec = _Milestone9PlayabilitySpec(
            name="cemetery_exclusions", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0, strict_assets=False,
            maximum_grave_objects=1000, grave_spacing=4.0,
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        library = ProceduralSiteLibrary(spec.name)
        library.prepare(dataset, projection)
        result = generate_semantic_objects(
            dataset, projection, (5.0,) * 64, spec, library, starting_object_id=1,
            raster=raster, building_placement_plans=(plan,),
        )
        self.assertGreater(result.grave_objects, 0)
        for grave in result.objects:
            self.assertFalse(73.0 <= grave.x <= 127.0 and 73.0 <= grave.z <= 127.0)
            self.assertGreater(abs(grave.z - 100.0), 3.0)

    def test_everon_forests_are_trusted_runtime_assets_in_milestone9(self) -> None:
        from cwr_worldgen.generator import _trusted_legacy_asset_paths
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=(0.0,0.0,0.01,0.01),
            strict_assets=True,
        )
        trusted = set(_trusted_legacy_asset_paths(spec, 9))
        self.assertIn(r"data3d\les ctverec pruchozi_t1.p3d", trusted)
        self.assertIn(r"data3d\les trojuhelnik pruchozi.p3d", trusted)
        self.assertNotIn(r"data3d\str_fikovnik.p3d", trusted)
        self.assertIn(r"data3d\str smrk_medium.p3d", trusted)
        self.assertTrue(all(path.casefold().startswith("data3d\\") for path in spec.forest_roadside_tree_models))
        self.assertTrue(all(path.casefold().startswith("data3d\\") for path in spec.forest_roadside_bush_models))
        self.assertTrue(all(path.casefold().startswith("data3d\\") for path in spec.steep_hill_bush_models))
        self.assertFalse(any(path.casefold().startswith("o\\tree\\") for path in trusted if "rakosi" not in path.casefold()))
        self.assertIn(r"o\misc\aut_z_st.p3d", trusted)
        self.assertTrue(set(path.casefold() for path in STOCK_HEDGE_MODELS).issubset(trusted))
        self.assertTrue(set(path.casefold() for path in STOCK_WALL_MODELS).issubset(trusted))
        self.assertTrue(set(path.casefold() for path in STOCK_METAL_FENCE_MODELS).issubset(trusted))
        from cwr_worldgen.semantic_features import GRAVE_MODELS
        self.assertTrue(set(path.casefold() for path in GRAVE_MODELS).issubset(trusted))
        bus_disabled = replace(spec, bus_stops_enabled=False)
        self.assertNotIn(r"o\misc\aut_z_st.p3d", set(_trusted_legacy_asset_paths(bus_disabled, 9)))

        malden = replace(spec, forest_profile="malden", forest_hillside_fallback=True)
        malden_trusted = set(_trusted_legacy_asset_paths(malden, 9))
        self.assertIn(r"data3d\str_fikovnik.p3d", malden_trusted)

class DeploymentTests(unittest.TestCase):
    def test_deploy_copies_complete_runtime_into_existing_mod_without_new_wrapper(self) -> None:
        from types import SimpleNamespace
        from cwr_worldgen.milestone9 import _deploy_runtime_to_existing_mod

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "build"
            source_mod = output / "@CWR-Milestone9"
            pbo = source_mod / "Addons" / "cwr_test.pbo"
            addon_metadata = source_mod / "Addons" / "cwr_test.sha256"
            intro = source_mod / "Anims" / "intro.cwr_test"
            pbo.parent.mkdir(parents=True)
            intro.mkdir(parents=True)
            pbo.write_bytes(b"pbo")
            addon_metadata.write_text("hash", encoding="utf-8")
            (intro / "mission.sqm").write_text("mission", encoding="utf-8")
            (intro / "intro.sqs").write_text("camera", encoding="utf-8")
            target = root / "@ExistingMod"
            target.mkdir()
            # Existing mods are not always consistent about directory casing.
            selected_addons = target / "addons"
            (target / "anims").mkdir()
            selected_addons.mkdir()
            result = SimpleNamespace(
                output_dir=output, pbo_path=pbo, intro_mission_path=intro / "mission.sqm"
            )
            # Selecting Addons itself is recovered to the enclosing mod root.
            report = _deploy_runtime_to_existing_mod(result, selected_addons)
            self.assertEqual((target / "addons" / "cwr_test.pbo").read_bytes(), b"pbo")
            self.assertEqual(
                (target / "addons" / "cwr_test.sha256").read_text(encoding="utf-8"),
                "hash",
            )
            self.assertEqual(
                (target / "anims" / "intro.cwr_test" / "mission.sqm").read_text(encoding="utf-8"),
                "mission",
            )
            self.assertEqual(
                (target / "anims" / "intro.cwr_test" / "intro.sqs").read_text(encoding="utf-8"),
                "camera",
            )
            self.assertFalse((target / "@CWR-Milestone9").exists())
            self.assertFalse((selected_addons / "Addons").exists())
            self.assertEqual(report["mod_folder"], str(target.resolve()))
            self.assertEqual(report["requested_folder"], str(selected_addons.resolve()))
            self.assertTrue(report["verified"])
            self.assertEqual(report["file_count"], 4)

    def test_deploy_runs_even_when_final_validation_report_fails(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = milestone8_tests.Milestone8BuildTests()._source(root / "source")
            target = root / "@ExistingMod"
            (target / "Addons").mkdir(parents=True)
            (target / "Anims").mkdir()
            spec = Milestone9Spec(
                source_dir=source,
                name="cwr_deploy_fail",
                display_name="CWR Deploy Fail",
                solver_iterations=1,
                world_edge_blend_cells=1,
                max_buildings=0,
                max_forest_objects=0,
                forest_undergrowth_enabled=False,
                forest_border_enabled=False,
                forest_single_tree_enabled=False,
                steep_hill_bushes_enabled=False,
                ditch_grass_enabled=False,
                barriers_enabled=False,
                bridges_enabled=False,
                rural_vegetation_enabled=False,
                wetland_reeds_enabled=False,
                rocky_forest_fallback_enabled=False,
                semantic_landmarks=False,
                surface_overview_size=128,
                surface_texture_size=32,
                strict_assets=False,
                deploy_mod_dir=target,
            )
            with patch(
                "cwr_worldgen.generator._validate_milestone4",
                side_effect=RuntimeError("forced final validation failure"),
            ):
                result = build_milestone9(root / "build", spec)
            deployed_pbo = target / "Addons" / "cwr_deploy_fail.pbo"
            self.assertTrue(deployed_pbo.is_file())
            self.assertEqual(deployed_pbo.read_bytes(), result.pbo_path.read_bytes())
            self.assertIn(
                "Final validation checks raised an exception",
                result.report_path.read_text(encoding="utf-8"),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["final_validation"]["status"], "failed")
            self.assertTrue(manifest["deployment"]["verified"])



class SemanticImportTests(unittest.TestCase):
    def test_overpass_and_parser_include_semantic_features(self) -> None:
        from cwr_worldgen.osm import build_overpass_query, parse_overpass_json
        query = build_overpass_query((0.0, 0.0, 0.01, 0.01))
        self.assertIn('["amenity"~"^(place_of_worship|school|social_facility|parking)$"]', query)
        self.assertIn('["social_facility"]', query)
        self.assertIn('["amenity"="grave_yard"]', query)
        self.assertIn('["landuse"="cemetery"]', query)
        self.assertIn('["shop"]', query)
        self.assertIn('["leisure"="pitch"]', query)
        self.assertIn('["sport"="soccer"]', query)
        self.assertIn('["highway"="bus_stop"]', query)
        document = {
            "generator": "semantic-import-test",
            "elements": [
                {"type":"node","id":1,"lat":0.005,"lon":0.005,"tags":{"highway":"bus_stop"}},
                {"type":"way","id":2,"tags":{"amenity":"parking"},"geometry":[
                    {"lat":0.001,"lon":0.001},{"lat":0.001,"lon":0.002},
                    {"lat":0.002,"lon":0.002},{"lat":0.002,"lon":0.001},{"lat":0.001,"lon":0.001},
                ]},
                {"type":"way","id":3,"tags":{"leisure":"pitch","sport":"soccer"},"geometry":[
                    {"lat":0.003,"lon":0.003},{"lat":0.003,"lon":0.004},
                    {"lat":0.004,"lon":0.004},{"lat":0.004,"lon":0.003},{"lat":0.003,"lon":0.003},
                ]},
                {"type":"way","id":4,"tags":{"building":"church","amenity":"place_of_worship"},"geometry":[
                    {"lat":0.006,"lon":0.006},{"lat":0.006,"lon":0.007},
                    {"lat":0.007,"lon":0.007},{"lat":0.007,"lon":0.006},{"lat":0.006,"lon":0.006},
                ]},
                {"type":"way","id":5,"tags":{"landuse":"cemetery","name":"Test Cemetery"},"geometry":[
                    {"lat":0.0075,"lon":0.001},{"lat":0.0075,"lon":0.003},
                    {"lat":0.0095,"lon":0.003},{"lat":0.0095,"lon":0.001},{"lat":0.0075,"lon":0.001},
                ]},
            ],
        }
        dataset = parse_overpass_json(json.dumps(document).encode("utf-8"))
        self.assertEqual(len(dataset.landmarks), 1)
        self.assertEqual(
            {feature.tags["site"] for feature in dataset.sites},
            {"parking", "sports_pitch", "cemetery"},
        )
        self.assertEqual(dataset.building_polygons[0].tags["amenity"], "place_of_worship")

    def test_generated_gravel_is_slightly_narrower_again(self) -> None:
        self.assertAlmostEqual(GENERATED_GRAVEL_HALF_WIDTH_METRES * 2.0, 4.60, places=6)


class InfrastructureAndRuralTests(unittest.TestCase):
    @staticmethod
    def _polygon(projection: BboxProjection, key: str, tags: dict[str, str], bounds: tuple[float, float, float, float]) -> OsmPolygonFeature:
        x0, z0, x1, z1 = bounds
        ring = tuple(projection.to_latlon(point) for point in ((x0,z0),(x1,z0),(x1,z1),(x0,z1),(x0,z0)))
        return OsmPolygonFeature(key, tags, (GeoPolygon(ring),))

    def test_new_osm_classes_are_queried_and_parsed(self) -> None:
        from cwr_worldgen.osm import build_overpass_query, parse_overpass_json
        query = build_overpass_query((0.0, 0.0, 0.01, 0.01))
        self.assertIn('["barrier"~"^(fence|wall|hedge|retaining_wall)$"]', query)
        self.assertIn('["man_made"="cutline"]', query)
        self.assertIn('["natural"="tree_row"]', query)
        self.assertIn('["natural"~"^(scrub|bare_rock|rock|scree)$"]', query)
        self.assertIn('["natural"="wetland"]', query)
        self.assertIn('["natural"="tree"]', query)
        self.assertIn('["aeroway"~"^(aerodrome|runway|taxiway|apron|helipad)$"]', query)
        self.assertIn('["power"~"^(pole|tower)$"]', query)
        self.assertIn('["man_made"="water_tower"]', query)
        self.assertIn('["natural"~"^(grassland|sand|beach|desert|dune)$"]', query)
        self.assertIn('["landcover"="sand"]', query)
        self.assertIn('["surface"="sand"]', query)
        self.assertIn('["leisure"="park"]', query)
        document = {"generator":"m14", "elements":[
            {"type":"way","id":1,"tags":{"barrier":"hedge"},"geometry":[{"lat":0.001,"lon":0.001},{"lat":0.001,"lon":0.005}]},
            {"type":"way","id":2,"tags":{"man_made":"cutline","cutline":"firebreak"},"geometry":[{"lat":0.002,"lon":0.001},{"lat":0.002,"lon":0.005}]},
            {"type":"way","id":3,"tags":{"natural":"tree_row"},"geometry":[{"lat":0.003,"lon":0.001},{"lat":0.003,"lon":0.005}]},
            {"type":"way","id":4,"tags":{"natural":"scrub"},"geometry":[{"lat":0.004,"lon":0.001},{"lat":0.004,"lon":0.005},{"lat":0.006,"lon":0.005},{"lat":0.006,"lon":0.001},{"lat":0.004,"lon":0.001}]},
            {"type":"way","id":5,"tags":{"natural":"wetland"},"geometry":[{"lat":0.006,"lon":0.006},{"lat":0.006,"lon":0.009},{"lat":0.009,"lon":0.009},{"lat":0.009,"lon":0.006},{"lat":0.006,"lon":0.006}]},
            {"type":"node","id":6,"lat":0.0025,"lon":0.007,"tags":{"natural":"tree","leaf_type":"broadleaved"}},
            {"type":"way","id":7,"tags":{"aeroway":"runway","width":"30"},"geometry":[{"lat":0.001,"lon":0.007},{"lat":0.009,"lon":0.007}]},
            {"type":"way","id":8,"tags":{"aeroway":"apron"},"geometry":[{"lat":0.001,"lon":0.0075},{"lat":0.001,"lon":0.009},{"lat":0.002,"lon":0.009},{"lat":0.002,"lon":0.0075},{"lat":0.001,"lon":0.0075}]},
            {"type":"node","id":9,"lat":0.0035,"lon":0.0075,"tags":{"power":"pole"}},
            {"type":"node","id":10,"lat":0.0045,"lon":0.0075,"tags":{"man_made":"water_tower"}},
            {"type":"way","id":11,"tags":{"natural":"grassland"},"geometry":[{"lat":0.001,"lon":0.0002},{"lat":0.001,"lon":0.0008},{"lat":0.002,"lon":0.0008},{"lat":0.002,"lon":0.0002},{"lat":0.001,"lon":0.0002}]},
            {"type":"way","id":12,"tags":{"leisure":"park"},"geometry":[{"lat":0.0022,"lon":0.0002},{"lat":0.0022,"lon":0.0008},{"lat":0.0032,"lon":0.0008},{"lat":0.0032,"lon":0.0002},{"lat":0.0022,"lon":0.0002}]},
            {"type":"way","id":13,"tags":{"natural":"beach"},"geometry":[{"lat":0.0034,"lon":0.0002},{"lat":0.0034,"lon":0.0008},{"lat":0.0044,"lon":0.0008},{"lat":0.0044,"lon":0.0002},{"lat":0.0034,"lon":0.0002}]},
            {"type":"way","id":14,"tags":{"natural":"desert"},"geometry":[{"lat":0.0046,"lon":0.0002},{"lat":0.0046,"lon":0.0008},{"lat":0.0056,"lon":0.0008},{"lat":0.0056,"lon":0.0002},{"lat":0.0046,"lon":0.0002}]},
        ]}
        dataset = parse_overpass_json(json.dumps(document).encode("utf-8"))
        self.assertEqual(len(dataset.barriers), 1)
        self.assertEqual(len(dataset.cutlines), 1)
        self.assertEqual(len(dataset.tree_rows), 1)
        self.assertEqual(len(dataset.rural_vegetation), 2)
        self.assertIn("wetland", {feature.tags.get("natural") for feature in dataset.rural_vegetation})
        self.assertEqual(len(dataset.individual_trees), 1)
        self.assertEqual(len(dataset.aeroway_lines), 1)
        self.assertEqual(len(dataset.aeroway_areas), 1)
        self.assertEqual({item.tags.get("utility") for item in dataset.utility_points}, {"power_pole", "water_tower"})
        self.assertEqual({item.tags.get("surface_kind") for item in dataset.surface_areas}, {"grassland", "park", "beach", "sand"})

    def test_individual_trees_and_utilities_are_placed_from_osm_points(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        dataset = OsmDataset(
            source_generator="mapped-points", element_count=6,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            individual_trees=(
                OsmPointFeature("node/tree-a", {"natural": "tree", "leaf_type": "broadleaved"}, projection.to_latlon((100.0, 100.0))),
                OsmPointFeature("node/tree-b", {"natural": "tree", "leaf_type": "broadleaved"}, projection.to_latlon((105.0, 100.0))),
                OsmPointFeature("node/tree-c", {"natural": "tree", "leaf_type": "broadleaved"}, projection.to_latlon((12.5, 12.5))),
            ),
            utility_points=(
                OsmPointFeature("node/pole", {"power": "pole", "utility": "power_pole"}, projection.to_latlon((80.0, 40.0))),
                OsmPointFeature("node/tower", {"power": "tower", "utility": "power_tower"}, projection.to_latlon((120.0, 40.0))),
                OsmPointFeature("node/water", {"man_made": "water_tower", "utility": "water_tower"}, projection.to_latlon((160.0, 40.0))),
            ),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_mapped_points", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
            cells=8, cell_size=25.0, max_road_objects=0, max_buildings=0, max_forest_objects=0,
            forest_single_tree_enabled=False, forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False, rural_vegetation_enabled=False,
            wetland_reeds_enabled=False, rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False,
            strict_assets=False,
        )
        elevations = [10.0] * 64
        elevations[0] = 0.0
        elevations[1] = 20.0
        elevations[8] = 20.0
        elevations[9] = 0.0
        result = generate_world_objects(
            dataset, projection, raster, tuple(elevations), spec, include_roads=False
        )
        self.assertEqual(result.mapped_tree_objects, 1)
        self.assertEqual(result.mapped_tree_rejections, 2)
        self.assertEqual(result.utility_objects, 3)
        mapped_trees = [
            obj for obj in result.objects
            if obj.model_path in OSM_INDIVIDUAL_TREE_MODELS
        ]
        self.assertEqual(len(mapped_trees), 1)
        self.assertAlmostEqual(mapped_trees[0].y, 10.04)
        self.assertTrue(any(obj.model_path.casefold().endswith(r"\i\util_power_pole.p3d") for obj in result.objects))
        self.assertTrue(any(obj.model_path.casefold().endswith(r"\i\util_power_tower.p3d") for obj in result.objects))
        self.assertTrue(any(obj.model_path.casefold().endswith(r"\i\util_water_tower.p3d") for obj in result.objects))

    def test_capped_interior_undergrowth_walk_is_spatially_distributed(self) -> None:
        columns = 64
        indices = list(_distributed_grid_indices(columns, "distribution-seed", "forest-undergrowth"))[:128]
        self.assertEqual(len(indices), len(set(indices)))
        rows = {index // columns for index in indices}
        cols = {index % columns for index in indices}
        self.assertGreaterEqual(len(rows), 24)
        self.assertGreater(len(cols), 24)

    def test_steep_forest_hills_receive_stock_bush_models(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        dataset = OsmDataset(
            source_generator="steep-bushes", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(True,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_steep_bushes", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=1,
            forest_tree_spacing=1000.0, forest_single_tree_enabled=False, forest_undergrowth_enabled=False,
            forest_border_enabled=False, ditch_grass_enabled=False,
            barriers_enabled=False, bridges_enabled=False, rural_vegetation_enabled=False,
            wetland_reeds_enabled=False, rocky_forest_fallback_enabled=False,
            steep_hill_bushes_enabled=True, maximum_steep_hill_bush_objects=20,
            steep_hill_bush_spacing=20.0, steep_hill_bush_minimum_slope_degrees=5.0,
            steep_hill_bush_maximum_relief=100.0, steep_hill_bush_maximum_burial=100.0,
            steep_hill_bush_maximum_float=100.0, strict_assets=False,
        )
        elevations = tuple(float(x * 8) for _z in range(8) for x in range(8))
        first = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        second = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        self.assertEqual(first.objects, second.objects)
        self.assertEqual(first.steep_hill_bush_objects, 20)
        bush_objects = [obj for obj in first.objects if obj.model_path in spec.steep_hill_bush_models]
        self.assertEqual(len(bush_objects), 20)
        for bush in bush_objects:
            required_ground = max(_triangle_elevation_bounds(
                elevations, spec.cells, spec.cell_size, bush.x, bush.z
            ))
            self.assertGreaterEqual(
                bush.y,
                required_ground + max(0.0, spec.steep_hill_bush_ground_clearance) - 1e-6,
            )

    def test_wetlands_use_stock_reeds_even_when_rural_clusters_are_disabled(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        wetland = self._polygon(
            projection, "way/wetland", {"rural_kind": "wetland"}, (20.0, 20.0, 180.0, 180.0)
        )
        dataset = OsmDataset(
            source_generator="wetland-reeds", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            rural_vegetation=(wetland,),
        )
        raster = OsmRaster(
            cells=8, water=(True,) * 64, forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_wetland_reeds", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            forest_single_tree_enabled=False, forest_undergrowth_enabled=False,
            forest_border_enabled=False, steep_hill_bushes_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=True,
            maximum_wetland_reed_objects=30, wetland_reed_spacing=22.0,
            wetland_reed_maximum_relief=100.0, wetland_reed_maximum_burial=100.0,
            wetland_reed_maximum_float=100.0, rocky_forest_fallback_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (2.0,) * 64, spec, include_roads=False
        )
        self.assertGreater(result.wetland_reed_objects, 0)
        self.assertLessEqual(result.wetland_reed_objects, 30)
        self.assertTrue({obj.model_path for obj in result.objects}.issubset(set(spec.wetland_reed_models)))

    def test_osm_meadows_get_deterministic_randomized_tall_grass_only(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        meadow = self._polygon(
            projection, "way/meadow", {"landuse": "meadow"}, (20.0, 20.0, 90.0, 180.0)
        )
        ordinary_farmland = self._polygon(
            projection, "way/farmland", {"landuse": "farmland"}, (110.0, 20.0, 180.0, 180.0)
        )
        dataset = OsmDataset(
            source_generator="meadow-grass", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(meadow, ordinary_farmland),
            urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64,
            farmland=(True,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_meadow_grass", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            forest_single_tree_enabled=False, forest_undergrowth_enabled=False,
            forest_border_enabled=False, steep_hill_bushes_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, meadow_grass_enabled=True,
            maximum_meadow_grass_objects=12, meadow_grass_spacing=20.0,
            strict_assets=False,
        )
        elevations = tuple(float(x) * 0.2 for _z in range(8) for x in range(8))
        first = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        second = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        self.assertEqual(first.objects, second.objects)
        self.assertEqual(first.meadow_grass_objects, 12)
        self.assertTrue(all("ditch_grass" in obj.model_path for obj in first.objects))
        self.assertTrue(all(obj.x < 100.0 for obj in first.objects))
        self.assertGreater(len({round(obj.heading_degrees, 3) for obj in first.objects}), 1)
        meadow_variant = DITCH_GRASS_VARIANTS[0]
        for grass in first.objects:
            grade_label = grass.model_path.rsplit("_", 1)[1].split(".", 1)[0]
            grade = int(grade_label) / 100.0
            angle = math.radians(grass.heading_degrees)
            width_axis = (math.cos(angle), -math.sin(angle))
            length_axis = (math.sin(angle), math.cos(angle))
            for _model, local_x, local_z, _scale, _heading in meadow_variant.proxy_layout:
                proxy_x = grass.x + local_x * width_axis[0] + local_z * length_axis[0]
                proxy_z = grass.z + local_x * width_axis[1] + local_z * length_axis[1]
                proxy_y = grass.y + grade * local_z
                required_ground = max(_triangle_elevation_bounds(
                    elevations, spec.cells, spec.cell_size, proxy_x, proxy_z
                ))
                self.assertGreaterEqual(proxy_y, required_ground - 1e-6)
        with tempfile.TemporaryDirectory() as temp:
            preview = Path(temp) / "meadow-grass-placement.png"
            write_meadow_grass_placement_preview(
                preview, dataset, projection, raster, first, spec, size=512
            )
            self.assertTrue(preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(preview) as image:
                self.assertEqual(image.size, (512, 608))

    def test_scrubland_bush_amount_is_reduced_without_lowering_into_ground(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        scrub = self._polygon(
            projection, "way/scrub", {"rural_kind": "scrub"}, (20.0, 20.0, 180.0, 180.0)
        )
        dataset = OsmDataset(
            source_generator="scrub-density", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            rural_vegetation=(scrub,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_scrub_density", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=True, maximum_rural_vegetation_objects=10000,
            rural_vegetation_spacing=28.0, strict_assets=False,
        )
        result = generate_world_objects(dataset, projection, raster, (5.0,) * 64, spec, include_roads=False)
        scrub_objects = [
            obj for obj in result.objects
            if "ditch_grass" in obj.model_path or "scrub_patch" in obj.model_path
        ]
        self.assertEqual(len(scrub_objects), result.scrub_objects)
        self.assertGreater(result.scrub_objects, 0)
        self.assertLessEqual(result.scrub_objects, 35)
        self.assertTrue(all(abs(obj.y - 5.03) < 1e-6 for obj in scrub_objects))

    def test_empty_residential_area_gets_deterministic_infill_but_mapped_area_does_not(self) -> None:
        from cwr_worldgen.osm import plan_building_placements

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        residential = self._polygon(
            projection, "way/residential", {"landuse": "residential"}, (20.0, 20.0, 280.0, 280.0)
        )
        road = OsmLineFeature(
            "way/road", {"highway": "residential"},
            tuple(projection.to_latlon(p) for p in ((30.0, 150.0), (270.0, 150.0))),
        )
        empty = OsmDataset(
            source_generator="infill", element_count=2, coastlines=(), water=(), forests=(), farmland=(),
            urban=(residential,), roads=(road,),
        )
        raster = rasterize_osm(empty, projection, cells=12, include_minor_roads=True, supersample=2)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_infill", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
            cells=12, cell_size=25.0, max_road_objects=0, max_buildings=100, max_forest_objects=0,
            residential_infill_enabled=True, maximum_residential_infill_buildings=20,
            residential_infill_spacing=55.0, residential_infill_minimum_area=1000.0,
            strict_assets=False,
        )
        library = ProceduralBuildingLibrary(world_name=spec.name, maximum_variants=16)
        library.prepare(empty, projection, spec.point_building_footprint)
        first, _ = plan_building_placements(empty, projection, raster, spec, library)
        second, _ = plan_building_placements(empty, projection, raster, spec, library)
        self.assertEqual(first, second)
        infill = [plan for plan in first if plan.synthetic_infill]
        self.assertGreater(len(infill), 0)
        self.assertLessEqual(len(infill), 20)
        self.assertTrue(all(plan.osm_key.startswith("infill/way/residential/") for plan in infill))

        mapped_house = self._polygon(
            projection, "way/house", {"building": "house"}, (45.0, 45.0, 58.0, 62.0)
        )
        mapped = replace(empty, building_polygons=(mapped_house,), element_count=3)
        mapped_raster = rasterize_osm(mapped, projection, cells=12, include_minor_roads=True, supersample=2)
        mapped_library = ProceduralBuildingLibrary(world_name="cwr_infill_mapped", maximum_variants=16)
        mapped_library.prepare(mapped, projection, spec.point_building_footprint)
        mapped_plans, _ = plan_building_placements(mapped, projection, mapped_raster, spec, mapped_library)
        self.assertFalse(any(plan.synthetic_infill for plan in mapped_plans))

    def test_overture_buildings_are_used_before_random_infill(self) -> None:
        from cwr_worldgen.osm import plan_building_placements

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        residential = self._polygon(
            projection, "way/residential", {"landuse": "residential"}, (20.0, 20.0, 280.0, 280.0)
        )
        road = OsmLineFeature(
            "way/road", {"highway": "residential"},
            tuple(projection.to_latlon(p) for p in ((30.0, 150.0), (270.0, 150.0))),
        )
        dataset = OsmDataset(
            source_generator="overture-infill", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(residential,), roads=(road,),
            places=(OsmPointFeature(
                "node/hamlet", {"place": "hamlet", "name": "Tiny"},
                projection.to_latlon((290.0, 290.0)),
            ),),
        )
        raster = rasterize_osm(dataset, projection, cells=12, include_minor_roads=True, supersample=2)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_overture_infill", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=12, cell_size=25.0,
            max_road_objects=0, max_buildings=100, max_forest_objects=0,
            residential_infill_enabled=True, maximum_residential_infill_buildings=20,
            residential_infill_spacing=55.0, residential_infill_minimum_area=1000.0,
            strict_assets=False,
        )

        def geojson_ring(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
            return [[lon, lat] for lat, lon in (projection.to_latlon(point) for point in points)]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overture-buildings.geojson"
            path.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "id": "building-a",
                    "properties": {"id": "building-a"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geojson_ring((
                            (70.0, 130.0), (82.0, 130.0), (82.0, 142.0),
                            (70.0, 142.0), (70.0, 130.0),
                        ))],
                    },
                }],
            }), encoding="utf-8")
            source_spec = Milestone9Spec(source_dir=Path("unused"))
            augmented = augment_dataset_with_overture_buildings(dataset, projection, source_spec, path)

        self.assertEqual(len(augmented.building_polygons), 1)
        self.assertEqual(augmented.building_polygons[0].tags.get("source"), "overturemaps")
        library = ProceduralBuildingLibrary(world_name=spec.name, maximum_variants=16)
        library.prepare(augmented, projection, spec.point_building_footprint)
        plans, _ = plan_building_placements(augmented, projection, raster, spec, library)
        self.assertEqual(len([plan for plan in plans if plan.osm_key.startswith("overture/")]), 1)
        self.assertFalse(any(plan.synthetic_infill for plan in plans))

    def test_overture_building_classes_drive_barn_warehouse_and_shed_families(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        residential = self._polygon(
            projection, "way/residential", {"landuse": "residential"},
            (10.0, 10.0, 290.0, 290.0),
        )
        dataset = OsmDataset(
            source_generator="overture-building-classes", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(residential,), roads=(),
        )

        def geojson_ring(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
            return [[lon, lat] for lat, lon in (projection.to_latlon(point) for point in points)]

        definitions = (
            ("barn-a", "barn", (30.0, 30.0, 42.0, 72.0)),
            ("warehouse-a", "warehouse", (80.0, 30.0, 104.0, 90.0)),
            ("shed-a", "shed", (140.0, 30.0, 146.0, 38.0)),
        )
        features = []
        for source_id, building_class, (x0, z0, x1, z1) in definitions:
            features.append({
                "type": "Feature",
                "id": source_id,
                "properties": {
                    "id": source_id,
                    "class": building_class,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geojson_ring((
                        (x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0),
                    ))],
                },
            })

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overture-building-classes.geojson"
            path.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": features,
            }), encoding="utf-8")
            augmented = augment_dataset_with_overture_buildings(
                dataset, projection, Milestone9Spec(source_dir=Path("unused")), path
            )

        self.assertEqual(
            {feature.osm_key: feature.tags.get("building") for feature in augmented.building_polygons},
            {
                "overture/barn-a": "barn",
                "overture/warehouse-a": "warehouse",
                "overture/shed-a": "shed",
            },
        )
        library = ProceduralBuildingLibrary(world_name="cwr_overture_classes", maximum_variants=16)
        library.prepare(augmented, projection, 12.0)
        families = {}
        for feature in augmented.building_polygons:
            polygon = feature.polygons[0]
            points = [projection.to_world(point) for point in polygon.outer[:-1]]
            families[feature.osm_key] = library.plan_polygon(feature.tags, points).requested.family
        self.assertEqual(
            families,
            {
                "overture/barn-a": "agricultural",
                "overture/warehouse-a": "industrial",
                "overture/shed-a": "outbuilding",
            },
        )

    def test_overture_buildings_do_not_cover_existing_osm_building_areas(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        village = OsmPointFeature(
            "node/village", {"place": "village", "name": "Mapped Village"},
            projection.to_latlon((150.0, 150.0)),
        )
        house = self._polygon(
            projection, "way/house", {"building": "house"}, (145.0, 145.0, 157.0, 160.0)
        )
        dataset = OsmDataset(
            source_generator="mapped-overture", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(house,), places=(village,),
        )

        def geojson_ring(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
            return [[lon, lat] for lat, lon in (projection.to_latlon(point) for point in points)]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overture-buildings.geojson"
            path.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "id": "duplicate-house",
                    "properties": {"id": "duplicate-house"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geojson_ring((
                            (146.0, 146.0), (158.0, 146.0), (158.0, 160.0),
                            (146.0, 160.0), (146.0, 146.0),
                        ))],
                    },
                }],
            }), encoding="utf-8")
            augmented = augment_dataset_with_overture_buildings(dataset, projection, Milestone9Spec(source_dir=Path("unused")), path)

        self.assertEqual(augmented.building_polygons, dataset.building_polygons)

    def test_overture_buildings_can_fill_empty_road_endings(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        road = OsmLineFeature(
            "way/dead-end", {"highway": "residential"},
            tuple(projection.to_latlon(point) for point in ((40.0, 150.0), (140.0, 150.0))),
        )
        dataset = OsmDataset(
            source_generator="road-ending-overture", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
        )

        def geojson_ring(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
            return [[lon, lat] for lat, lon in (projection.to_latlon(point) for point in points)]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overture-buildings.geojson"
            path.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "id": "dead-end-house",
                    "properties": {"id": "dead-end-house"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geojson_ring((
                            (150.0, 160.0), (164.0, 160.0), (164.0, 174.0),
                            (150.0, 174.0), (150.0, 160.0),
                        ))],
                    },
                }],
            }), encoding="utf-8")
            augmented = augment_dataset_with_overture_buildings(dataset, projection, Milestone9Spec(source_dir=Path("unused")), path)

        self.assertEqual(len(augmented.building_polygons), 1)
        self.assertEqual(augmented.building_polygons[0].osm_key, "overture/dead-end-house")

    def test_missing_osm_infill_defaults_are_sparser_and_near_roads(self) -> None:
        from cwr_worldgen.osm import plan_building_placements

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 500.0)
        residential = self._polygon(
            projection, "way/residential", {"landuse": "residential"}, (20.0, 20.0, 480.0, 480.0)
        )
        road = OsmLineFeature(
            "way/road", {"highway": "residential"},
            tuple(projection.to_latlon(p) for p in ((40.0, 250.0), (460.0, 250.0))),
        )
        dataset = OsmDataset(
            source_generator="infill-density", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(residential,), roads=(road,),
        )
        raster = rasterize_osm(dataset, projection, cells=20, include_minor_roads=True, supersample=2)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_infill_density", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=20, cell_size=25.0,
            max_road_objects=0, max_buildings=500, max_forest_objects=0,
            residential_infill_enabled=True, residential_infill_minimum_area=1000.0,
            strict_assets=False,
        )
        denser = replace(
            spec, maximum_residential_infill_buildings=3000, residential_infill_spacing=48.0
        )

        def synthetic_plans(current: _Milestone9PlayabilitySpec):
            library = ProceduralBuildingLibrary(
                world_name=f"{current.name}_{current.residential_infill_spacing}",
                maximum_variants=16,
            )
            library.prepare(dataset, projection, current.point_building_footprint)
            plans, _ = plan_building_placements(dataset, projection, raster, current, library)
            return [plan for plan in plans if plan.synthetic_infill]

        sparse_plans = synthetic_plans(spec)
        dense_count = len(synthetic_plans(denser))
        self.assertEqual(spec.maximum_residential_infill_buildings, 1500)
        self.assertEqual(spec.residential_infill_spacing, 68.0)
        self.assertEqual(spec.residential_infill_road_clearance, 0.5)
        self.assertGreater(len(sparse_plans), 0)
        self.assertLess(len(sparse_plans), dense_count)
        self.assertTrue(all(abs(plan.z - 250.0) <= 25.0 for plan in sparse_plans))

    def test_empty_village_and_hamlet_places_get_residential_infill(self) -> None:
        from cwr_worldgen.osm import plan_building_placements

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        road = OsmLineFeature(
            "way/village-road", {"highway": "residential"},
            tuple(projection.to_latlon(p) for p in ((40.0, 150.0), (260.0, 150.0))),
        )
        places = (
            OsmPointFeature(
                "node/village", {"place": "village", "name": "Test Village"},
                projection.to_latlon((95.0, 150.0)),
            ),
            OsmPointFeature(
                "node/hamlet", {"place": "hamlet", "name": "Test Hamlet"},
                projection.to_latlon((225.0, 150.0)),
            ),
        )
        dataset = OsmDataset(
            source_generator="settlement-infill", element_count=3,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,),
            places=places,
        )
        raster = rasterize_osm(dataset, projection, cells=12, include_minor_roads=True, supersample=2)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_settlement_infill", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=12, cell_size=25.0,
            max_road_objects=0, max_buildings=100, max_forest_objects=0,
            residential_infill_enabled=True, maximum_residential_infill_buildings=20,
            residential_infill_spacing=55.0, residential_infill_minimum_area=1000.0,
            strict_assets=False,
        )
        library = ProceduralBuildingLibrary(world_name=spec.name, maximum_variants=16)
        library.prepare(dataset, projection, spec.point_building_footprint)
        first, _ = plan_building_placements(dataset, projection, raster, spec, library)
        second, _ = plan_building_placements(dataset, projection, raster, spec, library)
        self.assertEqual(first, second)
        infill = [plan for plan in first if plan.synthetic_infill]
        self.assertGreater(len(infill), 0)
        self.assertLessEqual(len(infill), 20)
        self.assertTrue(any(plan.osm_key.startswith("infill/node/village/") for plan in infill))
        self.assertTrue(any(plan.osm_key.startswith("infill/node/hamlet/") for plan in infill))
        self.assertEqual({plan.building_family for plan in infill}, {"residential"})

    def test_small_settlement_infill_only_runs_when_place_has_no_mapped_buildings(self) -> None:
        from cwr_worldgen.osm import plan_building_placements

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        village = OsmPointFeature(
            "node/village", {"place": "village", "name": "Mapped Village"},
            projection.to_latlon((150.0, 150.0)),
        )
        house = self._polygon(
            projection, "way/house", {"building": "house"}, (145.0, 145.0, 157.0, 160.0)
        )
        dataset = OsmDataset(
            source_generator="mapped-settlement", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(house,), places=(village,),
        )
        raster = rasterize_osm(dataset, projection, cells=12, include_minor_roads=True, supersample=2)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_mapped_settlement", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=12, cell_size=25.0,
            max_road_objects=0, max_buildings=100, max_forest_objects=0,
            residential_infill_enabled=True, maximum_residential_infill_buildings=20,
            residential_infill_spacing=55.0, residential_infill_minimum_area=1000.0,
            strict_assets=False,
        )
        library = ProceduralBuildingLibrary(world_name=spec.name, maximum_variants=16)
        library.prepare(dataset, projection, spec.point_building_footprint)
        plans, _ = plan_building_placements(dataset, projection, raster, spec, library)
        self.assertEqual(len([plan for plan in plans if not plan.synthetic_infill]), 1)
        self.assertFalse(any(plan.synthetic_infill for plan in plans))

    def test_osm_dry_land_below_sea_is_lifted_without_raising_mapped_water(self) -> None:
        from cwr_worldgen.terrain_solver import solve_terrain_constraints

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        farmland = self._polygon(
            projection, "way/farmland", {"landuse": "farmland"}, (25.0, 25.0, 50.0, 50.0)
        )
        water = self._polygon(
            projection, "way/water", {"natural": "water"}, (50.0, 50.0, 75.0, 75.0)
        )
        utility = OsmPointFeature(
            "node/water-tower", {"man_made": "water_tower", "utility": "water_tower"},
            projection.to_latlon((87.5, 87.5)),
        )
        water_mask = [False] * 16
        water_mask[10] = True
        farmland_mask = [False] * 16
        farmland_mask[5] = True
        raster = OsmRaster(
            cells=4, water=tuple(water_mask), forest=(False,) * 16,
            farmland=tuple(farmland_mask), urban=(False,) * 16, roads=(False,) * 16,
            buildings=(False,) * 16, high_resolution=4, coastline_seed_count=0,
        )
        dataset = OsmDataset(
            source_generator="dry-land-floor", element_count=3,
            coastlines=(), water=(water,), forests=(), farmland=(farmland,),
            urban=(), roads=(), utility_points=(utility,),
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_dry_land_floor", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=4, cell_size=25.0, sea_level=0.0,
            water_depth=5.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=0, solver_iterations=0, strict_assets=False,
        )
        report = solve_terrain_constraints(
            (-2.0,) * 16, dataset, projection, raster, spec,
        )
        self.assertGreaterEqual(report.osm_land_floor_cells, 2)
        self.assertGreaterEqual(report.elevations[5], spec.sea_level + 7.00)
        self.assertGreaterEqual(report.elevations[15], spec.sea_level + 7.00)
        self.assertLess(report.elevations[10], spec.sea_level)

    def test_stock_nogova_bridge_stays_near_road_level_over_water(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/bridge-water", {"highway": "secondary", "bridge": "yes", "lanes": "2"},
            tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="bridge-water", element_count=1, coastlines=(), water=(), forests=(), farmland=(),
            urban=(), roads=(bridge,),
        )
        water = [False] * 64
        # Bridge centre crosses water; banks remain outside the wet cells.
        for row in range(3, 5):
            for col in range(2, 6):
                water[row * 8 + col] = True
        raster = OsmRaster(
            cells=8, water=tuple(water), forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        elevations = [-3.0] * 64
        # Dry approach cells are near sea level rather than three metres underwater.
        for row in range(8):
            elevations[row * 8 + 0] = 0.0
            elevations[row * 8 + 7] = 0.0
        spec = _Milestone9PlayabilitySpec(
            name="cwr_bridge_water", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
            cells=8, cell_size=25.0, sea_level=0.0, max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=False,
            bridge_water_clearance=1.25, bridge_deck_clearance=0.22, bridge_module_length=12.0,
            forest_undergrowth_enabled=False, forest_border_enabled=False, ditch_grass_enabled=False,
            barriers_enabled=False, rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(dataset, projection, raster, tuple(elevations), spec, include_roads=False)
        bridge_objects = [obj for obj in result.objects if obj.model_path == NOGOVA_BRIDGE_MODEL]
        self.assertGreater(len(bridge_objects), 0)
        self.assertAlmostEqual(
            min(obj.y for obj in bridge_objects),
            NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES,
            places=6,
        )
        self.assertLess(max(obj.y for obj in bridge_objects), 0.25)
        self.assertLess(max(abs(obj.pitch_degrees) for obj in bridge_objects), 1.0)

    def test_explicit_bridge_does_not_raise_ground_beneath_span(self) -> None:
        from cwr_worldgen.terrain_solver import solve_terrain_constraints

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/125776130",
            {"highway": "secondary", "bridge": "yes", "ref": "D 957"},
            tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="bridge-no-terrain-fill", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(True,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        original = tuple(2.0 + (index % 8) * 0.1 for index in range(64))
        spec = _Milestone9PlayabilitySpec(
            name="bridge_no_fill", heightmap_path=Path("unused.png"), bbox=(0.0, 0.0, 1.0, 1.0),
            cells=8, cell_size=25.0, sea_level=0.0, solver_iterations=0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0, strict_assets=False,
        )
        report = solve_terrain_constraints(original, dataset, projection, raster, spec)
        baseline_dataset = replace(dataset, roads=())
        baseline = solve_terrain_constraints(original, baseline_dataset, projection, raster, spec)
        self.assertNotIn("bridge-support", report.category_adjustments)
        self.assertEqual(report.elevations, baseline.elevations)

    def test_procedural_bridge_uses_flat_raised_underfill(self) -> None:
        from cwr_worldgen.terrain_solver import solve_terrain_constraints

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/procedural-flat-underfill",
            {"highway": "secondary", "bridge": "yes"},
            tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="bridge-flat-underfill", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(True,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        original = tuple(8.0 if 2 <= (index % 8) <= 5 else 12.0 for index in range(64))
        spec = _Milestone9PlayabilitySpec(
            name="bridge_flat_fill", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
            cells=8, cell_size=25.0, sea_level=0.0, solver_iterations=0,
            procedural_bridges=True, max_road_objects=0, max_buildings=0,
            max_forest_objects=0, strict_assets=False,
        )
        report = solve_terrain_constraints(original, dataset, projection, raster, spec)
        self.assertIn("bridge-underfill", report.category_adjustments)
        centre_row = report.elevations[4 * 8:5 * 8]
        bridge_values = centre_row[1:7]
        self.assertLess(max(bridge_values) - min(bridge_values), 0.06)
        self.assertGreater(min(bridge_values), 10.9)

    def test_stock_nogova_bridge_only_clamps_just_above_water_without_mask(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/bridge-missing-water-mask",
            {"highway": "secondary", "bridge": "yes", "lanes": "2"},
            tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="bridge-missing-water-mask", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_bridge_missing_water_mask", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0, sea_level=0.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=False,
            bridge_water_clearance=2.50, bridge_deck_clearance=0.75,
            bridge_module_length=12.0, forest_undergrowth_enabled=False,
            forest_border_enabled=False, ditch_grass_enabled=False,
            barriers_enabled=False, rural_vegetation_enabled=False,
            wetland_reeds_enabled=False, rocky_forest_fallback_enabled=False,
            steep_hill_bushes_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (-4.0,) * 64, spec, include_roads=False
        )
        bridge_objects = [
            obj for obj in result.objects if obj.model_path == NOGOVA_BRIDGE_MODEL
        ]
        self.assertGreater(len(bridge_objects), 0)
        self.assertAlmostEqual(
            min(obj.y for obj in bridge_objects),
            NOGOVA_BRIDGE_MINIMUM_WATER_DECK_METRES,
            places=6,
        )
        self.assertLess(max(obj.y for obj in bridge_objects), 0.25)

    def test_positive_layer_road_over_water_gets_bridge_modules(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        road = OsmLineFeature(
            "way/layer-water", {"highway": "secondary", "layer": "1", "lanes": "2"},
            tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="layer-water", element_count=1, coastlines=(), water=(), forests=(), farmland=(),
            urban=(), roads=(road,),
        )
        water = [False] * 64
        for row in range(3, 5):
            for col in range(2, 6):
                water[row * 8 + col] = True
        raster = OsmRaster(
            cells=8, water=tuple(water), forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_layer_bridge", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
            cells=8, cell_size=25.0, sea_level=0.0, max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=False,
            bridge_water_clearance=1.25, bridge_deck_clearance=0.22, bridge_module_length=12.0,
            forest_undergrowth_enabled=False, forest_border_enabled=False, ditch_grass_enabled=False,
            barriers_enabled=False, rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(dataset, projection, raster, (0.0,) * 64, spec, include_roads=False)
        bridge_objects = [obj for obj in result.objects if obj.model_path == NOGOVA_BRIDGE_MODEL]
        self.assertGreater(len(bridge_objects), 0)

    def test_procedural_bridges_generate_world_local_fitted_modules(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/125776130",
            {"highway": "secondary", "bridge": "yes", "ref": "D 957", "lanes": "2"},
            tuple(projection.to_latlon(point) for point in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="procedural-bridge", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(bridge,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_proc_bridge", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=True, bridge_module_length=18.0,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (0.0,) * 64, spec, include_roads=False
        )
        bridge_objects = [
            obj for obj in result.objects
            if obj.model_path.casefold().startswith(r"cwr_proc_bridge\i\br_")
        ]
        self.assertEqual(result.bridge_segments, 1)
        self.assertEqual(len(bridge_objects), 1)
        self.assertEqual(len(bridge_objects), result.bridge_objects)
        self.assertFalse(any(obj.model_path == NOGOVA_BRIDGE_MODEL for obj in bridge_objects))
        paths = {obj.model_path.casefold() for obj in bridge_objects}
        self.assertTrue(all("br_single_" in path for path in paths))
        self.assertTrue(all("_w" in path and "_l" in path for path in paths))
        expected_origin = spec.bridge_deck_clearance - GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES
        self.assertTrue(all(abs(obj.y - expected_origin) < 1e-6 for obj in bridge_objects))
        self.assertTrue(all(abs(obj.pitch_degrees) < 1e-9 for obj in bridge_objects))
        self.assertTrue(all(abs(obj.pitch_degrees) < 1e-9 for obj in bridge_objects))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralInfrastructureLibrary(spec.name)
            library.register_models(obj.model_path for obj in bridge_objects)
            assets = library.write_assets(root, root / "infrastructure.json")
            self.assertEqual(assets.generated_variants, 1)
            bridge_texture = root / "i" / "b.paa"
            self.assertTrue(bridge_texture.is_file())
            info = inspect_paa(bridge_texture)
            self.assertEqual((info.width, info.height), (256, 256))
            for relative in assets.model_files:
                summary = inspect_mlod(root / relative)
                self.assertTrue(any(math.isclose(value, 3.0e15, rel_tol=1e-6) for value in summary.resolutions))
                all_properties = {prop for lod in summary.named_properties for prop in lod}
                # The stock road pieces underneath the generated bridge own the
                # road network. The visible bridge must remain a static roadway
                # structure rather than a second vertically stacked class=road.
                self.assertNotIn(("class", "road"), all_properties)
                self.assertNotIn(("map", "road"), all_properties)
                self.assertIn(("autocenter", "0"), all_properties)
                geometry_properties = summary.named_properties[1]
                self.assertIn(("canbeoccluded", "0"), geometry_properties)
                self.assertIn(("canocclude", "0"), geometry_properties)

    def test_explicit_bridge_over_ditch_stays_an_ordinary_road(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        road = OsmLineFeature(
            "way/ditch-bridge",
            {"highway": "residential", "bridge": "yes"},
            tuple(projection.to_latlon(point) for point in ((20.0, 100.0), (180.0, 100.0))),
        )
        ditch = OsmLineFeature(
            "way/ditch",
            {"waterway": "ditch"},
            tuple(projection.to_latlon(point) for point in ((100.0, 20.0), (100.0, 180.0))),
        )
        dataset = OsmDataset(
            source_generator="ditch-bridge", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(road,), watercourses=(ditch,),
        )
        self.assertTrue(road_bridge_crosses_ditch_only(road, dataset, projection))

        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="ditch_bridge", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
            max_road_objects=1000, max_buildings=0, max_forest_objects=0,
            bridges_enabled=True, procedural_bridges=False,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False,
            strict_assets=False,
        )
        nonroads = generate_world_objects(
            dataset, projection, raster, (2.0,) * 64, spec, include_roads=False
        )
        self.assertEqual(nonroads.bridge_segments, 0)
        self.assertEqual(nonroads.bridge_objects, 0)

        roads = fit_road_objects(dataset, projection, (2.0,) * 64, spec)
        self.assertGreater(len(roads.objects), 0)
        self.assertFalse(any(obj.model_path == NOGOVA_BRIDGE_MODEL for obj in roads.objects))

    def test_procedural_bridge_uses_bank_approach_level_and_stays_visible(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/125776130",
            {"highway": "secondary", "bridge": "yes", "ref": "D 957"},
            tuple(projection.to_latlon(point) for point in ((55.0, 100.0), (145.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="d957-bank-probe", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,),
        )
        water = [False] * 64
        for row in range(3, 5):
            for col in range(2, 6):
                water[row * 8 + col] = True
        raster = OsmRaster(
            cells=8, water=tuple(water), forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        # Low centre cells mimic a coarse DEM river channel while the outer bank
        # cells remain at road level. The bridge must not inherit the channel
        # elevation or disappear below the visible approaches.
        elevations = []
        for z in range(8):
            for x in range(8):
                elevations.append(-2.0 if 2 <= x <= 5 else 3.0)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_d957_visible", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0, sea_level=0.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=True, bridge_module_length=18.0,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, tuple(elevations), spec, include_roads=False
        )
        bridge_objects = [obj for obj in result.objects if r"\i\br_" in obj.model_path.casefold()]
        self.assertTrue(bridge_objects)
        self.assertEqual(result.bridge_segments, 1)
        self.assertEqual(result.bridge_rejections, 0)
        self.assertTrue(all(
            obj.y + GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES >= 3.0 + spec.bridge_deck_clearance - 1e-9
            for obj in bridge_objects
        ))
        self.assertLess(max(obj.y for obj in bridge_objects) - min(obj.y for obj in bridge_objects), 1e-9)
        self.assertTrue(all(abs(obj.pitch_degrees) < 1e-9 for obj in bridge_objects))

    def test_procedural_bridge_uses_road_level_before_beach_downslope(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 250.0)
        bridge = OsmLineFeature(
            "way/beach-approach", {"highway": "secondary", "bridge": "yes"},
            tuple(projection.to_latlon(point) for point in ((95.0, 125.0), (155.0, 125.0))),
        )
        dataset = OsmDataset(
            source_generator="beach-bridge", element_count=1, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(bridge,),
        )
        raster = OsmRaster(
            cells=10, water=(False,) * 100, forest=(False,) * 100, farmland=(False,) * 100,
            urban=(False,) * 100, roads=(False,) * 100, buildings=(False,) * 100,
            high_resolution=10, coastline_seed_count=0,
        )
        elevations = []
        for z in range(10):
            for x in range(10):
                world_x = (x + 0.5) * 25.0
                # Flat road at 6 m inland, rolling down toward the beach/bridge.
                if world_x < 65.0 or world_x > 185.0:
                    value = 6.0
                elif world_x < 95.0:
                    value = 6.0 - (world_x - 65.0) / 30.0 * 4.0
                elif world_x > 155.0:
                    value = 2.0 + (world_x - 155.0) / 30.0 * 4.0
                else:
                    value = 2.0
                elevations.append(value)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_beach_bridge", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
            cells=10, cell_size=25.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=0, procedural_bridges=True, bridge_module_length=15.0,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, rural_vegetation_enabled=False,
            wetland_reeds_enabled=False, rocky_forest_fallback_enabled=False,
            steep_hill_bushes_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(dataset, projection, raster, tuple(elevations), spec, include_roads=False)
        bridge_objects = [obj for obj in result.objects if r"\i\br_" in obj.model_path.casefold()]
        self.assertTrue(bridge_objects)
        roadway_levels = [obj.y + GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES for obj in bridge_objects]
        self.assertLess(max(roadway_levels) - min(roadway_levels), 1e-9)
        self.assertGreaterEqual(roadway_levels[0], 5.5)
        # The tagged OSM bridge starts at x=95/155, but the road begins its beach
        # descent near x=65/185. The one generated bridge model must be long
        # enough that its *ends* reach those stable approaches even though there
        # is now only one object centre in the WRP.
        self.assertEqual(len(bridge_objects), 1)
        obj = bridge_objects[0]
        match = re.search(r"_l(\d+)\.p3d$", obj.model_path.casefold())
        self.assertIsNotNone(match)
        model_length = int(match.group(1)) / 10.0
        half = model_length * 0.5
        angle = math.radians(obj.heading_degrees)
        end1_x = obj.x - math.sin(angle) * half
        end2_x = obj.x + math.sin(angle) * half
        self.assertLess(min(end1_x, end2_x), 75.0)
        self.assertGreater(max(end1_x, end2_x), 165.0)

    def test_procedural_bridge_reaches_road_crest_beyond_old_short_probe(self) -> None:
        from cwr_worldgen.osm import _extend_procedural_bridge_to_approach_plateaus

        cells = 16
        cell_size = 25.0
        world_size = cells * cell_size
        points = ((160.0, 200.0), (240.0, 200.0))
        elevations = []
        for _z in range(cells):
            for x in range(cells):
                world_x = (x + 0.5) * cell_size
                if world_x <= 65.0 or world_x >= 335.0:
                    value = 8.0
                elif world_x < 160.0:
                    value = 8.0 - (world_x - 65.0) / 95.0 * 6.0
                elif world_x > 240.0:
                    value = 2.0 + (world_x - 240.0) / 95.0 * 6.0
                else:
                    value = 2.0
                elevations.append(value)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_long_beach_bridge", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=cells, cell_size=cell_size,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=True, bridge_module_length=30.0, strict_assets=False,
        )
        extended = _extend_procedural_bridge_to_approach_plateaus(
            points, tuple(elevations), spec, 30.0
        )
        self.assertLess(extended[0][0], 90.0)
        self.assertGreater(extended[-1][0], 310.0)

    def test_procedural_bridge_follows_connected_curved_road_to_plateau(self) -> None:
        from cwr_worldgen.osm import _extend_procedural_bridge_to_approach_plateaus

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        bridge = OsmLineFeature(
            "way/bridge", {"highway": "secondary", "bridge": "yes", "ref": "D 957"},
            tuple(projection.to_latlon(p) for p in ((115.0, 150.0), (185.0, 150.0))),
        )
        west = OsmLineFeature(
            "way/west", {"highway": "secondary", "ref": "D 957"},
            tuple(projection.to_latlon(p) for p in ((60.0, 205.0), (80.0, 185.0), (100.0, 165.0), (115.0, 150.0))),
        )
        east = OsmLineFeature(
            "way/east", {"highway": "secondary", "ref": "D 957"},
            tuple(projection.to_latlon(p) for p in ((185.0, 150.0), (205.0, 165.0), (225.0, 185.0), (245.0, 205.0))),
        )
        dataset = OsmDataset(
            source_generator="connected-bridge-approach", element_count=3,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(bridge, west, east),
        )
        cells = 12
        cell_size = 25.0
        elevations = []
        for row in range(cells):
            for col in range(cells):
                x = (col + 0.5) * cell_size
                z = (row + 0.5) * cell_size
                # Low shoreline around the literal bridge, upper road plateau
                # near the curved connected approaches.
                distance = min(math.dist((x, z), (115.0, 150.0)), math.dist((x, z), (185.0, 150.0)))
                elevations.append(2.0 + min(5.0, distance / 14.0))
        spec = _Milestone9PlayabilitySpec(
            name="cwr_connected_bridge", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=cells, cell_size=cell_size,
            procedural_bridges=True, bridge_module_length=15.0, strict_assets=False,
        )
        base = tuple(projection.to_world(point) for point in bridge.points)
        extended = _extend_procedural_bridge_to_approach_plateaus(
            base, tuple(elevations), spec, 15.0,
            feature=bridge, dataset=dataset, projection=projection,
        )
        # Straight tangent extrapolation would keep z=150. Following the road
        # reaches the curved plateau instead, matching the actual approach.
        self.assertGreater(extended[0][1], 170.0)
        self.assertGreater(extended[-1][1], 170.0)
        self.assertLess(extended[0][0], 100.0)
        self.assertGreater(extended[-1][0], 200.0)

    def test_forest_proxy_blocks_are_kept_away_from_world_border(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 300.0)
        forest = self._polygon(
            projection, "way/edge-forest", {"landuse": "forest"},
            (0.0, 0.0, 300.0, 300.0),
        )
        dataset = OsmDataset(
            source_generator="edge-forest", element_count=1, coastlines=(), water=(),
            forests=(forest,), farmland=(), urban=(), roads=(),
        )
        raster = OsmRaster(
            cells=12, water=(False,) * 144, forest=(True,) * 144,
            farmland=(False,) * 144, urban=(False,) * 144, roads=(False,) * 144,
            buildings=(False,) * 144, high_resolution=12, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_edge_forest", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=12, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=100,
            forest_tree_spacing=50.0, forest_border_enabled=True,
            forest_undergrowth_enabled=False, forest_single_tree_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (3.0,) * 144, spec, include_roads=False
        )
        forest_models = [
            obj for obj in result.objects
            if obj.model_path == spec.forest_tree_model or r"\f\b_" in obj.model_path.casefold()
        ]
        self.assertTrue(forest_models)
        self.assertTrue(all(16.0 <= obj.x <= 284.0 and 16.0 <= obj.z <= 284.0 for obj in forest_models))
        primary = [obj for obj in forest_models if obj.model_path == spec.forest_tree_model]
        self.assertTrue(primary)
        self.assertTrue(all(29.0 <= obj.x <= 271.0 and 29.0 <= obj.z <= 271.0 for obj in primary))

    def test_stock_nogova_bridge_centres_short_span_and_overlaps_long_span(self) -> None:
        short_points = ((20.0, 50.0), (30.0, 50.0))
        long_points = tuple((float(x), 150.0) for x in range(20, 111, 10)) + ((115.0, 150.0),)
        short_chunks = _bridge_module_chunks(short_points)
        long_chunks = _bridge_module_chunks(long_points)
        exact_chunks = _bridge_module_chunks(
            ((20.0, 100.0), (50.00000000001, 100.0), (80.00000000002, 100.0))
        )
        self.assertEqual(len(short_chunks), 1)
        self.assertAlmostEqual(short_chunks[0][0], 25.0)
        self.assertEqual(len(long_chunks), 4)
        self.assertEqual(len(exact_chunks), 2)
        self.assertTrue(all(
            later[0] - earlier[0] <= NOGOVA_BRIDGE_MODULE_LENGTH_METRES
            for earlier, later in zip(long_chunks, long_chunks[1:])
        ))

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        dataset = OsmDataset(
            source_generator="stock-nogova-span-fitting",
            element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(),
            roads=(
                OsmLineFeature(
                    "way/short-bridge", {"highway": "secondary", "bridge": "yes"},
                    tuple(projection.to_latlon(point) for point in short_points),
                ),
                OsmLineFeature(
                    "way/long-bridge", {"highway": "secondary", "bridge": "yes"},
                    tuple(projection.to_latlon(point) for point in long_points),
                ),
            ),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_stock_nogova_bridges", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=False,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False,
            rural_vegetation_enabled=False, wetland_reeds_enabled=False,
            rocky_forest_fallback_enabled=False, steep_hill_bushes_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (0.0,) * 64, spec, include_roads=False
        )
        bridge_objects = [
            obj for obj in result.objects if obj.model_path == NOGOVA_BRIDGE_MODEL
        ]
        self.assertEqual(result.bridge_segments, 2)
        self.assertEqual(result.bridge_objects, 7)
        self.assertEqual(len(bridge_objects), 7)
        self.assertEqual(len([obj for obj in bridge_objects if abs(obj.z - 50.0) < 0.01]), 2)
        self.assertEqual(len([obj for obj in bridge_objects if abs(obj.z - 150.0) < 0.01]), 5)

        capped_result = generate_world_objects(
            OsmDataset(
                source_generator="stock-nogova-object-cap",
                element_count=1,
                coastlines=(), water=(), forests=(), farmland=(), urban=(),
                roads=(dataset.roads[1],),
            ),
            projection,
            raster,
            (0.0,) * 64,
            replace(spec, maximum_bridge_objects=3),
            include_roads=False,
        )
        self.assertEqual(capped_result.bridge_segments, 1)
        self.assertEqual(capped_result.bridge_objects, 0)
        self.assertEqual(capped_result.bridge_rejections, 1)

    def test_stock_nogova_bridge_is_flat_at_highest_road_centreline(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        bridge = OsmLineFeature(
            "way/bridge-raised-centre",
            {"highway": "secondary", "bridge": "yes", "lanes": "2"},
            tuple(projection.to_latlon(p) for p in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="bridge-raised-centre", element_count=1,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64,
            farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64,
            buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0,
        )
        elevations = [0.0] * 64
        for row in range(3, 6):
            for col in range(2, 6):
                elevations[row * 8 + col] = 6.0
        spec = _Milestone9PlayabilitySpec(
            name="cwr_bridge_footprint", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=8, cell_size=25.0, sea_level=0.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=False,
            bridge_deck_clearance=0.75, bridge_water_clearance=2.50,
            bridge_module_length=12.0, forest_undergrowth_enabled=False,
            forest_border_enabled=False, ditch_grass_enabled=False,
            barriers_enabled=False, rural_vegetation_enabled=False,
            wetland_reeds_enabled=False, rocky_forest_fallback_enabled=False,
            steep_hill_bushes_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, tuple(elevations), spec, include_roads=False
        )
        bridge_objects = [
            obj for obj in result.objects if obj.model_path == NOGOVA_BRIDGE_MODEL
        ]
        self.assertGreater(len(bridge_objects), 0)
        self.assertTrue(all(
            abs(obj.y - (6.0 + NOGOVA_BRIDGE_APPROACH_OFFSET_METRES)) < 1e-6
            for obj in bridge_objects
        ))
        self.assertEqual(len({round(obj.y, 6) for obj in bridge_objects}), 1)
        self.assertLess(max(abs(obj.pitch_degrees) for obj in bridge_objects), 1e-9)

    def test_barriers_bridges_and_rural_clusters_are_deterministic(self) -> None:
        from cwr_worldgen.procedural_infrastructure import ProceduralInfrastructureLibrary
        projection = BboxProjection.create((0.0,0.0,1.0,1.0), 200.0)
        ll = projection.to_latlon
        bridge = OsmLineFeature("way/bridge", {"highway":"secondary","bridge":"yes","lanes":"2"}, tuple(ll(p) for p in ((20.0,100.0),(180.0,100.0))))
        hedge = OsmLineFeature("way/hedge", {"barrier":"hedge"}, tuple(ll(p) for p in ((20.0,40.0),(180.0,40.0))))
        tree_row = OsmLineFeature("way/trees", {"natural":"tree_row"}, tuple(ll(p) for p in ((20.0,160.0),(180.0,160.0))))
        rural = (
            self._polygon(projection,"way/orchard",{"landuse":"orchard"},(20,110,75,150)),
            self._polygon(projection,"way/vineyard",{"landuse":"vineyard"},(80,110,135,150)),
            self._polygon(projection,"way/scrub",{"natural":"scrub"},(140,110,190,150)),
            self._polygon(projection,"way/rock",{"natural":"bare_rock"},(20,50,75,90)),
        )
        dataset = OsmDataset(source_generator="m14", element_count=7, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(bridge,), barriers=(hedge,), tree_rows=(tree_row,), rural_vegetation=rural)
        raster = OsmRaster(cells=8, water=(False,)*64, forest=(False,)*64, farmland=(False,)*64, urban=(False,)*64, roads=(False,)*64, buildings=(False,)*64, high_resolution=8, coastline_seed_count=0)
        spec = _Milestone9PlayabilitySpec(name="cwr_m14_features", heightmap_path=Path("unused.png"), bbox=(0,0,1,1), cells=8, cell_size=25.0, max_road_objects=0, max_buildings=0, max_forest_objects=0,
            procedural_bridges=False, forest_undergrowth_enabled=False, forest_border_enabled=False, ditch_grass_enabled=False, strict_assets=False)
        elevations = tuple(5.0 + (x * 0.1) for _z in range(8) for x in range(8))
        first = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        second = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        self.assertEqual(first.objects, second.objects)
        self.assertGreater(first.hedge_objects, 0)
        self.assertGreater(first.bridge_objects, 0)
        bridge_paths = [
            obj.model_path
            for obj in first.objects
            if obj.model_path == NOGOVA_BRIDGE_MODEL
        ]
        self.assertEqual(len(bridge_paths), first.bridge_objects)
        self.assertGreater(first.tree_row_objects, 0)
        self.assertGreater(first.orchard_objects, 0)
        self.assertGreater(first.vineyard_objects, 0)
        self.assertGreater(first.scrub_objects, 0)
        self.assertGreater(first.rural_rock_objects, 0)
        scrub_paths = [obj.model_path for obj in first.objects if "ditch_grass" in obj.model_path]
        scrub_patch_paths = [obj.model_path for obj in first.objects if "scrub_patch" in obj.model_path]
        self.assertGreater(len(scrub_paths), 0)
        self.assertGreater(len(scrub_patch_paths), 0)
        self.assertEqual(len(scrub_paths) + len(scrub_patch_paths), first.scrub_objects)
        self.assertGreater(first.scrub_objects, first.orchard_objects)
        hedge_paths = [obj.model_path for obj in first.objects if obj.model_path in STOCK_HEDGE_MODELS]
        self.assertEqual(len(hedge_paths), first.hedge_objects)
        self.assertTrue(set(hedge_paths).issubset(set(STOCK_HEDGE_MODELS)))
        infra_paths = [obj.model_path for obj in first.objects if "\\i\\" in obj.model_path]
        self.assertFalse(any("bar_hedge" in path.casefold() for path in infra_paths))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralInfrastructureLibrary(spec.name)
            library.register_models(infra_paths)
            assets = library.write_assets(root, root / "infrastructure.json")
            # This stock-Nogova branch plus stock in-game stones no longer needs
            # world-local bridge/rock infrastructure models.
            self.assertEqual(assets.generated_variants, 0)
            self.assertFalse(any("/br_" in path for path in assets.model_files))

    def test_hedge_grounding_uses_full_widened_footprint(self) -> None:
        from cwr_worldgen.osm import _hedge_anchor_height, _infrastructure_anchor

        cells = 8
        cell_size = 25.0
        elevations = [0.0] * (cells * cells)
        elevations[4 * cells + 4] = 12.0
        line_y, _pitch = _infrastructure_anchor(
            elevations, cells, cell_size, 25.0, 100.0, 175.0, 100.0
        )
        hedge_y = _hedge_anchor_height(
            elevations, cells, cell_size, 25.0, 100.0, 175.0, 100.0
        )
        stock_hedge_y = _hedge_anchor_height(
            elevations, cells, cell_size, 25.0, 100.0, 175.0, 100.0,
            model_path=r"data3d\Krovi_long.p3d",
        )
        self.assertGreater(hedge_y, line_y + 2.0)
        self.assertGreaterEqual(hedge_y, 3.0)
        self.assertAlmostEqual(stock_hedge_y - hedge_y, 1.0, places=6)

    def test_hedges_rotate_with_stock_model_offset_and_nudge_off_roads(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        ll = projection.to_latlon
        road = OsmLineFeature("way/road", {"highway": "secondary"}, tuple(ll(p) for p in ((20.0, 100.0), (180.0, 100.0))))
        hedge = OsmLineFeature("way/hedge", {"barrier": "hedge"}, tuple(ll(p) for p in ((20.0, 100.0), (180.0, 100.0))))
        dataset = OsmDataset(
            source_generator="hedge-offset",
            element_count=2,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(road,),
            barriers=(hedge,),
        )
        raster = OsmRaster(cells=8, water=(False,) * 64, forest=(False,) * 64, farmland=(False,) * 64, urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64, high_resolution=8, coastline_seed_count=0)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_hedge_offset",
            heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1),
            cells=8,
            cell_size=25.0,
            max_road_objects=0,
            max_buildings=0,
            max_forest_objects=0,
            forest_undergrowth_enabled=False,
            forest_border_enabled=False,
            ditch_grass_enabled=False,
            rural_vegetation_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(dataset, projection, raster, (2.0,) * 64, spec, include_roads=False)
        hedges = [obj for obj in result.objects if obj.model_path in STOCK_HEDGE_MODELS]
        self.assertTrue(hedges)
        self.assertTrue(all(obj.heading_degrees == 180.0 for obj in hedges))
        self.assertTrue(all(abs(obj.z - 100.0) > 0.5 for obj in hedges))

    def test_walls_stay_on_mapped_boundary_and_leave_road_entrance(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        ll = projection.to_latlon
        road = OsmLineFeature(
            "way/road", {"highway": "secondary"},
            tuple(ll(p) for p in ((100.0, 20.0), (100.0, 180.0))),
        )
        wall = OsmLineFeature(
            "way/wall", {"barrier": "wall"},
            tuple(ll(p) for p in ((20.0, 100.0), (180.0, 100.0))),
        )
        dataset = OsmDataset(
            source_generator="wall-gate", element_count=2, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(road,), barriers=(wall,),
        )
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(False,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_wall_gate", heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1),
            cells=8, cell_size=25.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=0, forest_undergrowth_enabled=False,
            forest_border_enabled=False, ditch_grass_enabled=False,
            rural_vegetation_enabled=False, strict_assets=False,
        )
        result = generate_world_objects(
            dataset, projection, raster, (2.0,) * 64, spec, include_roads=False
        )
        walls = [obj for obj in result.objects if obj.model_path in STOCK_WALL_MODELS]
        self.assertTrue(walls)
        self.assertEqual(len(walls), result.wall_objects)
        self.assertTrue(all(obj.heading_degrees == 180.0 for obj in walls))
        self.assertTrue(all(abs(obj.z - 100.0) < 1e-6 for obj in walls))
        self.assertTrue(any(obj.x < 90.0 for obj in walls))
        self.assertTrue(any(obj.x > 110.0 for obj in walls))
        self.assertFalse(any(95.0 <= obj.x <= 105.0 for obj in walls))
        left_positions = sorted(obj.x for obj in walls if obj.x < 95.0)
        right_positions = sorted(obj.x for obj in walls if obj.x > 105.0)
        for positions in (left_positions, right_positions):
            if len(positions) > 1:
                self.assertLessEqual(
                    max(right - left for left, right in zip(positions, positions[1:])),
                    2.45,
                )
        self.assertTrue(
            {obj.model_path.casefold() for obj in walls}
            <= {r"o\hous\zidka01.p3d", r"o\hous\zidka02.p3d"}
        )
        self.assertFalse(any(
            obj.model_path.casefold().endswith(("zidka03.p3d", "zidka04.p3d"))
            for obj in result.objects
        ))

    def test_bus_stop_signs_are_nudged_off_road_centrelines(self) -> None:
        from cwr_worldgen.osm import OsmPointFeature
        from cwr_worldgen.semantic_features import ProceduralSiteLibrary, generate_semantic_objects

        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        ll = projection.to_latlon
        road = OsmLineFeature("way/road", {"highway": "secondary"}, tuple(ll(p) for p in ((20.0, 100.0), (180.0, 100.0))))
        bus = OsmPointFeature("node/bus", {"landmark": "bus_stop"}, ll((100.0, 100.0)))
        dataset = OsmDataset(
            source_generator="bus-offset",
            element_count=2,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(road,),
            landmarks=(bus,),
        )
        spec = _Milestone9PlayabilitySpec(
            heightmap_path=Path("unused.png"),
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=8,
            cell_size=25.0,
            strict_assets=False,
            bus_stops_enabled=True,
        )
        site_library = ProceduralSiteLibrary("bus_offset")
        site_library.prepare(dataset, projection)
        elevations = tuple(
            float(row * 12 + column * 3)
            for row in range(spec.cells)
            for column in range(spec.cells)
        )
        result = generate_semantic_objects(
            dataset, projection, elevations, spec, site_library, starting_object_id=1
        )
        self.assertEqual(result.bus_stop_objects, 1)
        bus_object = next(obj for obj in result.objects if obj.model_path == r"o\misc\aut_z_st.p3d")
        self.assertGreater(abs(bus_object.z - 100.0), 0.5)
        from cwr_worldgen.osm import _square_elevation_extrema
        _minimum, support = _square_elevation_extrema(
            elevations, spec.cells, spec.cell_size,
            bus_object.x, bus_object.z, spec.bus_stop_footprint,
        )
        self.assertAlmostEqual(
            bus_object.y, support + spec.bus_stop_ground_clearance, places=6
        )

    def test_large_buildings_are_nudged_off_road_masks(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)

        def polygon(key: str, bounds: tuple[float, float, float, float]) -> OsmPolygonFeature:
            x0, z0, x1, z1 = bounds
            ring = tuple(
                projection.to_latlon(point)
                for point in ((x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0))
            )
            return OsmPolygonFeature(key, {"building": "house"}, (GeoPolygon(ring),))

        road = OsmLineFeature(
            "way/road",
            {"highway": "secondary"},
            tuple(projection.to_latlon(point) for point in ((20.0, 100.0), (180.0, 100.0))),
        )
        large = polygon("way/large", (60.0, 90.0, 140.0, 110.0))
        small = polygon("way/small", (20.0, 96.0, 28.0, 104.0))
        road_mask = [False] * 64
        for x in range(8):
            road_mask[4 * 8 + x] = True
        raster = OsmRaster(
            cells=8,
            water=(False,) * 64,
            forest=(False,) * 64,
            farmland=(False,) * 64,
            urban=(False,) * 64,
            roads=tuple(road_mask),
            buildings=(False,) * 64,
            high_resolution=8,
            coastline_seed_count=0,
        )
        dataset = OsmDataset(
            source_generator="building-offset",
            element_count=3,
            coastlines=(),
            water=(),
            forests=(),
            farmland=(),
            urban=(),
            roads=(road,),
            building_polygons=(large, small),
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_building_offset",
            heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1),
            cells=8,
            cell_size=25.0,
            max_road_objects=0,
            max_buildings=10,
            max_forest_objects=0,
            building_minimum_area=1.0,
            forest_undergrowth_enabled=False,
            forest_border_enabled=False,
            ditch_grass_enabled=False,
            barriers_enabled=False,
            bridges_enabled=False,
            rural_vegetation_enabled=False,
            strict_assets=False,
        )
        result = generate_world_objects(dataset, projection, raster, (5.0,) * 64, spec, include_roads=False)
        self.assertEqual(result.building_objects, 2)
        large_object = next(obj for obj in result.objects if abs(obj.x - 100.0) < 1.0)
        small_object = next(obj for obj in result.objects if abs(obj.x - 24.0) < 1.0)
        self.assertGreater(abs(large_object.z - 100.0), 0.5)
        self.assertAlmostEqual(small_object.z, 100.0)

    def test_extreme_forest_patches_receive_stone_and_scattered_rocks(self) -> None:
        projection = BboxProjection.create((0.0,0.0,1.0,1.0), 100.0)
        dataset = OsmDataset(source_generator="rocky", element_count=0, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=())
        raster = OsmRaster(cells=4, water=(False,)*16, forest=(True,)*16, farmland=(False,)*16, urban=(False,)*16, roads=(False,)*16, buildings=(False,)*16, high_resolution=4, coastline_seed_count=0)
        spec = _Milestone9PlayabilitySpec(
            name="cwr_rocky_forest", heightmap_path=Path("unused.png"),
            bbox=(0,0,1,1), cells=4, cell_size=25.0,
            forest_tree_spacing=50.0, max_road_objects=0, max_buildings=0,
            max_forest_objects=4, forest_maximum_block_relief=0.01,
            forest_everon_steep_maximum_relief=0.01,
            forest_cluster_maximum_relief=0.01,
            forest_severe_hill_fallback=False,
            maximum_rocky_forest_objects=12,
            rocky_forest_rocks_per_patch=3, rocky_forest_spread=18.0,
            rocky_forest_maximum_relief=100.0,
            rocky_forest_maximum_burial=100.0,
            rocky_forest_maximum_float=100.0,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            steep_hill_bushes_enabled=False, ditch_grass_enabled=False, strict_assets=False,
        )
        elevations = tuple(float(x * 8) for _z in range(4) for x in range(4))
        result = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        repeated = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        self.assertGreater(result.forest_objects, 0)
        self.assertGreater(result.forest_hillside_tree_objects, 0)
        self.assertGreater(result.rocky_forest_objects, 0)
        self.assertEqual(result.objects, repeated.objects)
        stock_stones = {path.casefold() for path in STOCK_STONE_MODELS}
        self.assertTrue(any(obj.model_path.casefold() in stock_stones for obj in result.objects))
        self.assertFalse(any(r"\i\rock_" in obj.model_path.casefold() for obj in result.objects))
        self.assertGreater(len({(round(obj.x, 3), round(obj.z, 3)) for obj in result.objects}), 4)

        base = (MATERIAL_INDEX["f"],) * 16
        updated, _report, counts = _placement_driven_surface_overlay(
            base, None, result, raster, spec, slopes=(18.0,) * 16
        )
        self.assertEqual(counts["rocky_forest_cells"], 0)
        self.assertEqual(updated, base)
        self.assertEqual(
            surface_texture_wire_paths("cwr_rocky_forest", "everon")[MATERIAL_INDEX["k"]],
            r"o\lom2.paa",
        )

    def test_uncovered_steep_forest_cells_expose_stone_without_a_tree_object(self) -> None:
        raster = OsmRaster(
            cells=4,
            water=(False,) * 16,
            forest=(True,) * 16,
            farmland=(False,) * 16,
            urban=(False,) * 16,
            roads=(False,) * 16,
            buildings=(False,) * 16,
            high_resolution=4,
            coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_stone_gap", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=4, cell_size=25.0,
            strict_assets=False,
        )
        generated = ObjectGenerationResult(
            objects=(),
            road_objects=0, building_objects=0, forest_objects=0,
            road_objects_truncated=False, building_objects_truncated=False, forest_objects_truncated=False,
        )
        base = (MATERIAL_INDEX["f"],) * 16
        slopes = (26.0,) * 16
        updated, _report, counts = _placement_driven_surface_overlay(
            base, None, generated, raster, spec, slopes=slopes
        )
        self.assertEqual(updated, base)
        self.assertEqual(counts["rocky_forest_cells"], 0)

        very_steep, _report, steep_counts = _placement_driven_surface_overlay(
            base, None, generated, raster, spec, slopes=(55.0,) * 16
        )
        self.assertEqual(very_steep, (MATERIAL_INDEX["k"],) * 16)
        self.assertEqual(steep_counts["rocky_forest_cells"], 16)

    def test_rocky_surface_overlay_can_extend_onto_open_hillside_cells(self) -> None:
        raster = OsmRaster(
            cells=4,
            water=(False,) * 16,
            forest=(True, True, False, False) * 4,
            farmland=(False,) * 16,
            urban=(False,) * 16,
            roads=(False,) * 16,
            buildings=(False,) * 16,
            high_resolution=4,
            coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_stone_edge", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=4, cell_size=25.0,
            strict_assets=False,
        )
        rock_object = WorldObject(1, r"cwr_stone_edge\i\rock_group_0.p3d", 49.0, 0.0, 49.0, 0.0)
        generated = ObjectGenerationResult(
            objects=(rock_object,),
            road_objects=0, building_objects=0, forest_objects=0,
            road_objects_truncated=False, building_objects_truncated=False, forest_objects_truncated=False,
        )
        base = (MATERIAL_INDEX["g"],) * 16
        updated, _report, counts = _placement_driven_surface_overlay(
            base, None, generated, raster, spec, slopes=(55.0,) * 16
        )
        open_indices = {2, 6, 10, 14, 3, 7, 11, 15}
        self.assertTrue(any(updated[index] == MATERIAL_INDEX["k"] for index in open_indices))
        self.assertGreater(counts["rocky_forest_cells"], 0)

    def test_synthetic_single_tree_limit_scales_with_physical_area(self) -> None:
        self.assertEqual(_scaled_synthetic_tree_limit(1000, 3_200.0), 250)
        self.assertEqual(_scaled_synthetic_tree_limit(1000, 6_400.0), 1000)
        self.assertEqual(_scaled_synthetic_tree_limit(1000, 12_800.0), 4000)

    def test_geographic_single_tree_lattice_matches_in_overlapping_worlds(self) -> None:
        centre = (59.45, 17.0)
        small_projection = BboxProjection.create(
            square_bbox(*centre, 6_400.0), 6_400.0
        )
        large_projection = BboxProjection.create(
            square_bbox(*centre, 12_800.0), 12_800.0
        )
        spacing = 45.0
        small_cells = {
            (column, row): (latitude, longitude)
            for column, row, latitude, longitude, _x, _z
            in _geographic_forest_single_tree_cells(small_projection, spacing)
        }
        large_cells = {
            (column, row): (latitude, longitude)
            for column, row, latitude, longitude, _x, _z
            in _geographic_forest_single_tree_cells(large_projection, spacing)
        }
        self.assertTrue(small_cells.keys() <= large_cells.keys())
        self.assertEqual(
            {key: large_cells[key] for key in small_cells},
            small_cells,
        )
        self.assertAlmostEqual(len(large_cells) / len(small_cells), 4.0, delta=0.08)

        small_ranked = sorted(
            small_cells,
            key=lambda key: (_forest_single_tree_rank("same-seed", *key), key[1], key[0]),
        )[:1000]
        large_ranked = set(sorted(
            large_cells,
            key=lambda key: (_forest_single_tree_rank("same-seed", *key), key[1], key[0]),
        )[:4000])
        self.assertTrue(set(small_ranked) <= large_ranked)

        identity = next(iter(small_cells))
        latitude, longitude = small_cells[identity]
        small_candidates = _forest_single_tree_candidates(
            "same-seed", small_projection, *identity, latitude, longitude, spacing
        )
        large_candidates = _forest_single_tree_candidates(
            "same-seed", large_projection, *identity, latitude, longitude, spacing
        )
        for small_candidate, large_candidate in zip(small_candidates, large_candidates):
            self.assertAlmostEqual(small_candidate[2], large_candidate[2])
            small_latlon = small_projection.to_latlon(small_candidate[:2])
            large_latlon = large_projection.to_latlon(large_candidate[:2])
            self.assertAlmostEqual(small_latlon[0], large_latlon[0], places=10)
            self.assertAlmostEqual(small_latlon[1], large_latlon[1], places=10)

    def test_everon_forests_gain_extra_single_trees(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.002, 0.002), 200.0)
        dataset = OsmDataset(source_generator="forest-singles", element_count=0, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=())
        raster = OsmRaster(
            cells=8, water=(False,) * 64, forest=(True,) * 64, farmland=(False,) * 64,
            urban=(False,) * 64, roads=(False,) * 64, buildings=(False,) * 64,
            high_resolution=8, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_forest_singles", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 0.002, 0.002), cells=8, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=64,
            forest_maximum_block_relief=100.0, forest_everon_steep_maximum_relief=100.0,
            forest_cluster_maximum_relief=100.0, forest_undergrowth_enabled=False,
            forest_border_enabled=False, ditch_grass_enabled=False, strict_assets=False,
        )
        elevations = (10.0,) * 64
        result = generate_world_objects(dataset, projection, raster, elevations, spec, include_roads=False)
        extra_tree_model = spec.forest_single_tree_model.casefold()
        extra_tree_objects = [obj for obj in result.objects if obj.model_path.casefold() == extra_tree_model]
        self.assertGreater(len(extra_tree_objects), 0)
        self.assertGreater(result.forest_single_tree_objects, 0)

    def test_nonroad_placement_reports_granular_progress(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
        dataset = OsmDataset(source_generator="progress", element_count=0, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=())
        raster = OsmRaster(
            cells=4, water=(False,) * 16, forest=(True,) * 16, farmland=(False,) * 16,
            urban=(False,) * 16, roads=(False,) * 16, buildings=(False,) * 16,
            high_resolution=4, coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="cwr_progress", heightmap_path=Path("unused.png"),
            bbox=(0, 0, 1, 1), cells=4, cell_size=25.0,
            max_road_objects=0, max_buildings=0, max_forest_objects=4,
            forest_undergrowth_enabled=False, forest_border_enabled=False,
            ditch_grass_enabled=False, barriers_enabled=False, bridges_enabled=False,
            rural_vegetation_enabled=False, strict_assets=False,
        )
        events: list[tuple[int, str]] = []
        generate_world_objects(
            dataset, projection, raster, (5.0,) * 16, spec, include_roads=False,
            progress_callback=lambda percent, stage: events.append((percent, stage)),
        )
        percentages = [percent for percent, _stage in events]
        self.assertEqual(percentages, sorted(percentages))
        self.assertGreaterEqual(len(events), 9)
        self.assertIn((54, "Placing buildings"), events)
        self.assertIn((57, "Placing primary forest blocks"), events)
        self.assertIn((60, "Scattering individual forest trees"), events)
        self.assertIn((67, "Placing meadow grass, rural vegetation, wetland reeds and rocks"), events)

    def test_invalid_world_name_fails_before_source_validation(self) -> None:
        from unittest.mock import patch

        spec = Milestone9Spec(source_dir=Path("missing-source"), name="Bad World Name")
        with patch("cwr_worldgen.milestone9.validate_source_bundle") as validate_source:
            with self.assertRaisesRegex(ValueError, "world name must be"):
                build_milestone9(Path("unused-output"), spec)
        validate_source.assert_not_called()

    def test_building_front_rotates_toward_nearest_road(self) -> None:
        from cwr_worldgen.procedural_buildings import (
            ProceduralBuildingLibrary,
            _front_vector_for_heading,
        )
        library = ProceduralBuildingLibrary(world_name="cwr_facing")
        points = ((0.0,0.0),(12.0,0.0),(12.0,20.0),(0.0,20.0))
        projection = BboxProjection.create((0.0,0.0,1.0,1.0), 100.0)
        ring = tuple(projection.to_latlon(point) for point in (*points, points[0]))
        dataset = OsmDataset(source_generator="facing", element_count=1, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(), building_polygons=(OsmPolygonFeature("way/house", {"building":"house"}, (GeoPolygon(ring),)),))
        library.prepare(dataset, projection, 12.0)
        north = library.place_polygon({"building":"house"}, points, road_point=(6.0,30.0))
        south = library.place_polygon({"building":"house"}, points, road_point=(6.0,-10.0))
        self.assertAlmostEqual((north.heading_degrees - south.heading_degrees) % 360.0, 180.0)

        # Near-square house footprints can rotate freely without changing their
        # practical occupied footprint, so their actual door normal can point
        # directly at a side road rather than only choosing front vs back.
        square = ((40.0,40.0),(52.0,40.0),(52.0,52.0),(40.0,52.0))
        square_ring = tuple(projection.to_latlon(point) for point in (*square, square[0]))
        square_dataset = OsmDataset(source_generator="square-facing", element_count=1, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(), building_polygons=(OsmPolygonFeature("way/square", {"building":"house"}, (GeoPolygon(square_ring),)),))
        square_library = ProceduralBuildingLibrary(world_name="cwr_square_facing")
        square_library.prepare(square_dataset, projection, 12.0)
        east = square_library.place_polygon(
            {"building":"house"}, square, road_point=(70.0,46.0)
        )
        front_x, front_z = _front_vector_for_heading(east.heading_degrees)
        self.assertGreater(front_x, 0.999)
        self.assertAlmostEqual(front_z, 0.0, places=6)


    def test_steep_forest_cells_keep_everon_rock_material(self) -> None:
        from cwr_worldgen.osm import overlay_materials
        raster = OsmRaster(cells=2, water=(False,)*4, forest=(True,)*4, farmland=(False,)*4, urban=(False,)*4, roads=(False,)*4, buildings=(False,)*4, high_resolution=2, coastline_seed_count=0)
        # Base material index 3 is the rock/mountain tile; grass is index 2.
        result = overlay_materials((3, 2, 2, 3), raster)
        self.assertEqual(result, (3, 4, 4, 3))

    def test_default_hill_thresholds_use_performance_polygon_ladder(self) -> None:
        spec = Milestone9Spec(source_dir=Path("unused"))
        self.assertEqual(spec.forest_maximum_block_relief, 8.0)
        self.assertEqual(spec.forest_everon_steep_maximum_relief, 18.0)
        self.assertEqual(spec.forest_block_maximum_burial, 8.0)
        self.assertEqual(spec.forest_block_maximum_ground_sink, 0.0)
        self.assertEqual(spec.forest_everon_steep_maximum_burial, 18.0)
        self.assertEqual(spec.forest_everon_steep_maximum_ground_sink, 0.0)
        self.assertTrue(spec.forest_severe_hill_fallback)
        self.assertEqual(spec.forest_severe_hill_relief, 5.0)
        self.assertEqual(spec.forest_severe_hill_trees_per_block, 10)
        self.assertEqual(spec.forest_polygon_sink_fraction, 0.5)
        self.assertEqual(spec.forest_single_tree_maximum_float, 0.5)
        self.assertEqual(spec.bridge_module_length, 30.0)
        self.assertFalse(spec.procedural_bridges)
        self.assertEqual(spec.forest_cluster_maximum_relief, 48.0)
        self.assertEqual(spec.rocky_forest_rocks_per_patch, 3)
        self.assertEqual(spec.rocky_forest_spread, 18.0)
        with self.assertRaises(ValueError):
            replace(spec, rocky_forest_rocks_per_patch=0).validate()
        with self.assertRaises(ValueError):
            replace(spec, forest_polygon_sink_fraction=1.01).validate()


    def test_road_fitting_reports_granular_progress_without_changing_output(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)
        road = OsmLineFeature(
            "way/progress",
            {"highway": "primary", "surface": "asphalt"},
            tuple(projection.to_latlon((100.0 + index * 40.0, 500.0)) for index in range(18)),
        )
        dataset = OsmDataset(source_generator="road-progress", element_count=1, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(road,))
        spec = _Milestone9PlayabilitySpec(heightmap_path=Path("unused.png"), bbox=(0, 0, 1, 1), cells=40, cell_size=25.0, strict_assets=False)
        elevations = (0.0,) * (spec.cells * spec.cells)
        events: list[tuple[int, str]] = []
        plain = fit_road_objects(dataset, projection, elevations, spec)
        reported = fit_road_objects(dataset, projection, elevations, spec, progress_callback=lambda percent, stage: events.append((percent, stage)))
        self.assertEqual(plain, reported)
        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[-1][0], 100)
        self.assertTrue(any("Fitting stock road lines" in stage for _percent, stage in events))

    def test_overview_building_mask_and_fast_paa_are_deterministic(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
        dataset = OsmDataset(source_generator="overview-mask", element_count=0, coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=())
        cells = 8
        building_mask = [False] * (cells * cells)
        building_mask[4 * cells + 4] = True
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = render_overview_map(
                root / "overview.png",
                (MATERIAL_INDEX["g"],) * (cells * cells),
                (0.0,) * (cells * cells),
                (0.0,) * (cells * cells),
                dataset, projection, cells, 128, building_mask=building_mask,
            )
            first = root / "first.paa"
            second = root / "second.paa"
            write_rgb_dxt1_paa(first, image)
            write_rgb_dxt1_paa(second, image)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual((inspect_paa(first).width, inspect_paa(first).height), (128, 128))
            self.assertEqual(image.getpixel((72, 56)), (213, 205, 182))

    def test_building_source_reference_colours_osm_overture_and_generated_buildings(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)

        def polygon(min_x: float, min_z: float, max_x: float, max_z: float) -> GeoPolygon:
            ring = tuple(
                projection.to_latlon(point)
                for point in (
                    (min_x, min_z), (max_x, min_z), (max_x, max_z),
                    (min_x, max_z), (min_x, min_z),
                )
            )
            return GeoPolygon(ring)

        dataset = OsmDataset(
            source_generator="building-source-reference", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(
                OsmPolygonFeature("way/osm-house", {"building": "house"}, (polygon(8.0, 8.0, 16.0, 16.0),)),
                OsmPolygonFeature("overture/house", {"building": "house", "source": "overturemaps"}, (polygon(38.0, 8.0, 46.0, 16.0),)),
            ),
        )
        generated = BuildingPlacementPlan(
            osm_key="infill/area/0-0", geometry_index=0, geometry_kind="synthetic",
            x=72.0, z=12.0, heading_degrees=0.0, model_path=r"generated\house.p3d",
            support_polygon=((68.0, 8.0), (76.0, 8.0), (76.0, 16.0), (68.0, 16.0)),
            synthetic_infill=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "building-source-reference.png"
            image = render_building_source_reference(path, dataset, projection, (generated,), 128)
            self.assertTrue(path.is_file())
            self.assertEqual(image.getpixel((15, 113)), (74, 156, 255))
            self.assertEqual(image.getpixel((54, 113)), (255, 178, 58))
            self.assertEqual(image.getpixel((92, 113)), (116, 224, 120))
