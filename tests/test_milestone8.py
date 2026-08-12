# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import json
import tempfile
import unittest

from cwr_worldgen.milestone8 import Milestone8Spec, build_milestone8
from cwr_worldgen.osm import BboxProjection, GeoPolygon, OsmDataset, OsmPointFeature, OsmPolygonFeature
from cwr_worldgen.paa import inspect_paa, write_solid_dxt1_paa
from cwr_worldgen.pbo import read_pbo
from cwr_worldgen.procedural_buildings import (
    BuildingVariantKey,
    ProceduralBuildingLibrary,
    inspect_mlod,
    write_building_mlod,
)
from cwr_worldgen import procedural_buildings as building_models
from cwr_worldgen.source_pipeline import SourceFetchSpec, fetch_sources
from cwr_worldgen.terrain import NOGOVA_GROUND_TEXTURES
from cwr_worldgen.wrp import inspect_rvw4, quantize_elevations


class ProceduralBuildingTests(unittest.TestCase):
    def test_rural_floor_caps_and_oversized_footprints_use_barn_family(self) -> None:
        library = ProceduralBuildingLibrary(world_name="smart_buildings")
        rural_house = library.key_for({"building": "house", "building:levels": "5"}, 11.0, 17.0)
        school = library.key_for({"building": "school", "building:levels": "3"}, 18.0, 30.0)
        shop = library.key_for({"building": "retail", "building:levels": "4"}, 12.0, 18.0)
        barn = library.key_for({"building": "yes", "building:levels": "3"}, 12.0, 42.0)
        warehouse = library.key_for({"building": "warehouse", "building:levels": "3"}, 24.0, 60.0)

        self.assertEqual((rural_house.family, rural_house.height_m), ("residential", 6.0))
        self.assertEqual((school.family, school.height_m), ("school", 3.0))
        self.assertEqual((shop.family, shop.height_m), ("shop", 3.0))
        self.assertEqual((barn.family, barn.roof_style, barn.height_m), ("agricultural", "gabled", 6.0))
        self.assertEqual(warehouse.family, "industrial")

    def test_tiny_generic_buildings_become_windowless_outbuildings_with_large_gate(self) -> None:
        library = ProceduralBuildingLibrary(world_name="small_outbuildings")
        library.region_identifier = "sweden"
        tiny = library.key_for({"building": "yes"}, 6.0, 8.0)
        garage = library.key_for({"building": "garage"}, 7.0, 9.0)
        small_house = library.key_for({"building": "house"}, 6.0, 8.0)

        self.assertEqual((tiny.family, tiny.height_m, tiny.regional_style), ("outbuilding", 3.0, "sweden_red"))
        self.assertEqual(garage.family, "outbuilding")
        self.assertFalse(tiny.interiors)
        self.assertEqual(small_house.family, "residential")

        side = building_models._wall_texture_image("outbuilding", 128, "sweden_red", 0)
        front = building_models._front_texture_image("outbuilding", 128, "sweden_red", 0)
        self.assertNotEqual(side.tobytes(), front.tobytes())
        # The vehicle gate occupies a broad central area, far wider than the
        # ordinary pedestrian-door artwork used by houses.
        changed = 0
        total = 0
        for y in range(36, 124):
            for x in range(24, 104):
                total += 1
                if side.getpixel((x, y)) != front.getpixel((x, y)):
                    changed += 1
        self.assertGreater(changed / total, 0.45)

    def test_single_house_inside_isolated_dwelling_polygon_becomes_one_storey_cabin(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        cabin_ring = tuple(projection.to_latlon(point) for point in (
            (490.0, 490.0), (496.0, 490.0), (496.0, 498.0),
            (490.0, 498.0), (490.0, 490.0),
        ))
        area_ring = tuple(projection.to_latlon(point) for point in (
            (470.0, 470.0), (525.0, 470.0), (525.0, 525.0),
            (470.0, 525.0), (470.0, 470.0),
        ))
        building = OsmPolygonFeature(
            "way/lone-home", {"building": "yes"}, (GeoPolygon(cabin_ring),),
        )
        place_area = OsmPolygonFeature(
            "way/lone-place", {"place": "isolated_dwelling", "name": "Lone Cabin"},
            (GeoPolygon(area_ring),),
        )
        dataset = OsmDataset(
            source_generator="single-isolated-dwelling", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(building,), place_areas=(place_area,),
        )
        library = ProceduralBuildingLibrary(world_name="single_cabin")
        library.prepare(dataset, projection, 12.0)
        placement = library.plan_polygon(
            building.tags, [projection.to_world(point) for point in cabin_ring[:-1]],
        )
        self.assertEqual(placement.requested.family, "residential")
        self.assertEqual(placement.selected.family, "residential")
        self.assertEqual(placement.requested.height_m, 3.0)
        self.assertEqual(placement.selected.height_m, 3.0)

        garage = library.key_for(
            {"building": "garage"}, 6.0, 8.0,
            settlement_context="isolated_dwelling_single",
        )
        self.assertEqual(garage.family, "outbuilding")

    def test_exact_isolated_dwelling_polygon_outranks_nearby_hamlet(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        building_ring = tuple(projection.to_latlon(point) for point in (
            (500.0, 500.0), (506.0, 500.0), (506.0, 507.0),
            (500.0, 507.0), (500.0, 500.0),
        ))
        area_ring = tuple(projection.to_latlon(point) for point in (
            (480.0, 480.0), (530.0, 480.0), (530.0, 530.0),
            (480.0, 530.0), (480.0, 480.0),
        ))
        building = OsmPolygonFeature(
            "way/788104420", {"building": "yes"}, (GeoPolygon(building_ring),),
        )
        place_area = OsmPolygonFeature(
            "way/isolated", {"place": "isolated_dwelling"}, (GeoPolygon(area_ring),),
        )
        # This radius-based settlement hint was incorrectly outranking the exact
        # isolated polygon in 0.9.205.
        hamlet = OsmPointFeature(
            "node/nearby-hamlet", {"place": "hamlet", "name": "Nearby"},
            projection.to_latlon((510.0, 510.0)),
        )
        dataset = OsmDataset(
            source_generator="isolated-versus-hamlet", element_count=3,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(building,), place_areas=(place_area,), places=(hamlet,),
        )
        library = ProceduralBuildingLibrary(world_name="exact_isolated")
        library.prepare(dataset, projection, 12.0)
        placement = library.plan_polygon(
            building.tags, [projection.to_world(point) for point in building_ring[:-1]],
        )
        self.assertEqual(placement.requested.family, "residential")
        self.assertEqual(placement.selected.family, "residential")
        self.assertEqual(placement.requested.height_m, 3.0)
        self.assertEqual(placement.selected.height_m, 3.0)

    def test_way_788104416_inside_unnamed_isolated_area_becomes_cabin(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        building_ring = tuple(projection.to_latlon(point) for point in (
            (500.0, 500.0), (506.0, 500.0), (506.0, 507.0),
            (500.0, 507.0), (500.0, 500.0),
        ))
        area_ring = tuple(projection.to_latlon(point) for point in (
            (480.0, 480.0), (530.0, 480.0), (530.0, 530.0),
            (480.0, 530.0), (480.0, 480.0),
        ))
        building = OsmPolygonFeature(
            "way/788104416", {"building": "shed"}, (GeoPolygon(building_ring),),
        )
        place_area = OsmPolygonFeature(
            "way/1295713271", {"place": "isolated_dwelling"}, (GeoPolygon(area_ring),),
        )
        dataset = OsmDataset(
            source_generator="exact-user-isolated-dwelling", element_count=2,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(building,), place_areas=(place_area,),
        )
        library = ProceduralBuildingLibrary(world_name="way788104416")
        library.prepare(dataset, projection, 12.0)
        placement = library.plan_polygon(
            building.tags, [projection.to_world(point) for point in building_ring[:-1]],
        )
        self.assertEqual(placement.requested.family, "residential")
        self.assertEqual(placement.selected.family, "residential")
        self.assertEqual(placement.requested.height_m, 3.0)
        self.assertEqual(placement.selected.height_m, 3.0)

    def test_isolated_dwelling_polygon_does_not_capture_nearby_other_property(self) -> None:
        projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)
        cabin_ring = tuple(projection.to_latlon(point) for point in (
            (560.0, 490.0), (567.0, 490.0), (567.0, 498.0),
            (560.0, 498.0), (560.0, 490.0),
        ))
        shed_ring = tuple(projection.to_latlon(point) for point in (
            (545.0, 505.0), (550.0, 505.0), (550.0, 510.0),
            (545.0, 510.0), (545.0, 505.0),
        ))
        neighbour_ring = tuple(projection.to_latlon(point) for point in (
            (615.0, 490.0), (622.0, 490.0), (622.0, 498.0),
            (615.0, 498.0), (615.0, 490.0),
        ))
        area_ring = tuple(projection.to_latlon(point) for point in (
            (530.0, 470.0), (590.0, 470.0), (590.0, 525.0),
            (530.0, 525.0), (530.0, 470.0),
        ))
        cabin = OsmPolygonFeature("way/788104420", {"building": "yes"}, (GeoPolygon(cabin_ring),))
        shed = OsmPolygonFeature("way/explicit-shed", {"building": "shed"}, (GeoPolygon(shed_ring),))
        neighbour = OsmPolygonFeature("way/neighbour-home", {"building": "yes"}, (GeoPolygon(neighbour_ring),))
        place_area = OsmPolygonFeature(
            "way/isolated-area", {"place": "isolated_dwelling", "name": "Remote Home"},
            (GeoPolygon(area_ring),),
        )
        dataset = OsmDataset(
            source_generator="isolated-area-polygon", element_count=4,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(cabin, shed, neighbour), place_areas=(place_area,),
        )
        library = ProceduralBuildingLibrary(world_name="far_cabin")
        library.prepare(dataset, projection, 12.0)
        cabin_placement = library.plan_polygon(
            cabin.tags, [projection.to_world(point) for point in cabin_ring[:-1]],
        )
        neighbour_placement = library.plan_polygon(
            neighbour.tags, [projection.to_world(point) for point in neighbour_ring[:-1]],
        )
        self.assertEqual(cabin_placement.requested.family, "residential")
        self.assertEqual(cabin_placement.requested.height_m, 3.0)
        self.assertEqual(neighbour_placement.requested.family, "outbuilding")

    def test_swedish_outbuilding_palette_fits_p3d_texture_path_budget(self) -> None:
        # 0.9.190 added a ninth building family. With ten texture variants the
        # Sweden-red outbuilding palette crosses 36^2, so it needs a third
        # base36 digit. A maximum-length world name must still fit the 31-byte
        # MLOD texture field.
        world_name = "abcdefghijklmnopqrst"
        self.assertEqual(len(world_name), 20)
        bbox = (0.0, 0.0, 0.01, 0.01)
        projection = BboxProjection.create(bbox, 1000.0)
        dataset = OsmDataset(
            source_generator="outbuilding-texture-code", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = ProceduralBuildingLibrary(world_name=world_name)
            library.prepare(dataset, projection, 12.0)
            library.region_identifier = "sweden"
            placement = None
            for index in range(128):
                candidate = library.plan_point(
                    {"building": "garage"}, 6.0, 0.0,
                    x=10.0 + index * 7.0, z=20.0 + index * 11.0,
                )
                if candidate.selected.texture_variant >= 6:
                    placement = candidate
                    break
            self.assertIsNotNone(placement)
            library.register_placement(placement)
            result = library.write_assets(root, root / "buildings.json")
            self.assertTrue(any(rel.startswith("d/w1") for rel in result.texture_files))
            for asset in result.model_assets:
                summary = inspect_mlod(root / asset.relative_path)
                for texture_path in summary.texture_paths:
                    self.assertLessEqual(len(texture_path.encode("ascii")), 31)

    def test_writes_mlod_with_visual_geometry_and_land_contact_lods(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "house.p3d"
            write_building_mlod(
                path,
                BuildingVariantKey("residential", "gabled", 10.0, 16.0, 9.0),
                wall_texture=r"testworld\d\w0.paa",
                roof_texture=r"testworld\d\r1.paa",
            )
            summary = inspect_mlod(path)
            self.assertEqual(summary.version_major, 1)
            self.assertEqual(summary.version_minor, 1)
            self.assertEqual(summary.lod_count, 3)
            self.assertEqual(summary.resolutions[0], 1.0)
            self.assertGreater(summary.resolutions[1], 1.0e12)
            self.assertGreater(summary.resolutions[2], 1.0e15)
            self.assertGreater(summary.face_count, 0)
            self.assertEqual(summary.face_counts, (20, 6, 0))
            self.assertEqual(summary.selection_names[0], ())
            self.assertEqual(summary.selection_names[1], ("component01",))
            self.assertEqual(summary.selection_names[2], ())
            self.assertEqual(summary.mass_point_counts, (0, 8, 0))
            self.assertEqual(summary.named_properties[0], ())
            self.assertEqual(summary.named_properties[1], (("map", "house"),))
            self.assertEqual(summary.named_properties[2], ())
            self.assertEqual(
                summary.texture_paths,
                (r"testworld\d\r1.paa", r"testworld\d\w0.paa"),
            )



    def test_church_tower_mesh_does_not_overlap_nave_roof_and_windows_stay_on_one_level(self) -> None:
        key = BuildingVariantKey("church", "gabled", 26.0, 46.0, 12.0)
        wall = r"testworld\d\church.paa"
        plain = r"testworld\d\church_plain.paa"
        roof = r"testworld\d\roof.paa"
        front = r"testworld\d\church_front.paa"
        lod = building_models._visual_lod(
            key, wall, roof, 35.0, front_texture=front, plain_wall_texture=plain
        )

        half_length = key.length_m * 0.5
        tower_half = min(4.0, max(1.5, key.width_m * 0.22))
        tower_depth = min(6.0, max(3.0, key.length_m * 0.24))
        tower_back = min(half_length - 0.02, -half_length + tower_depth)

        # No roof polygon may cross through the central tower volume in the
        # front portion of the nave. The tower and church are one P3D, but their
        # surfaces now meet instead of interpenetrating.
        for face in (face for face in lod.faces if face.texture == roof):
            vertices = [lod.points[index] for index, _normal, _u, _v in face.vertices]
            min_z = min(point[2] for point in vertices)
            min_x = min(point[0] for point in vertices)
            max_x = max(point[0] for point in vertices)
            min_y = min(point[1] for point in vertices)
            # Ignore the tower spire itself; inspect only nave-roof polygons.
            if min_y <= key.height_m + 1e-6 and min_z < tower_back - 1e-6:
                self.assertFalse(min_x < -tower_half + 1e-6 and max_x > tower_half - 1e-6)

        # The nave keeps one ground-level window row. The integrated tower may
        # additionally use one compact upper belfry window band, but every
        # painted face maps the facade atlas only once vertically.
        upper_tower_window_faces = 0
        for face in lod.faces:
            if face.texture not in {wall, front}:
                continue
            v_values = [vertex[3] for vertex in face.vertices]
            self.assertLessEqual(max(v_values) - min(v_values), 1.000001)
            y_values = [lod.points[vertex[0]][1] for vertex in face.vertices]
            if face.texture == wall and min(y_values) > 3.0 + 1e-6:
                upper_tower_window_faces += 1
        self.assertGreaterEqual(upper_tower_window_faces, 4)

        self.assertIn(plain, {face.texture for face in lod.faces})

        # The nave is one standard 3 m storey shorter than the semantic church
        # height, while the integrated tower remains tall. Points 8/9 are the
        # authored front/back nave gable peaks before the tower is appended.
        self.assertAlmostEqual(lod.points[8][1], key.height_m - 3.0, places=6)
        self.assertAlmostEqual(lod.points[9][1], key.height_m - 3.0, places=6)
        self.assertGreater(max(point[1] for point in lod.points), key.height_m)

    def test_church_geometry_uses_authored_origin_without_autocenter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "church.p3d"
            write_building_mlod(
                path,
                BuildingVariantKey("church", "gabled", 26.0, 46.0, 12.0),
                wall_texture=r"testworld\d\church.paa",
                roof_texture=r"testworld\d\roof.paa",
                foundation_texture=r"testworld\d\foundation.paa",
                foundation_depth=1.75,
            )
            summary = inspect_mlod(path)
            properties = dict(summary.named_properties[1])
            self.assertEqual(properties["map"], "building")
            self.assertEqual(properties["autocenter"], "0")
            geometry = building_models._geometry_lod(
                BuildingVariantKey("church", "gabled", 26.0, 46.0, 12.0)
            )
            self.assertAlmostEqual(max(point[1] for point in geometry.points), 9.0, places=6)

    def test_urban_and_industrial_models_use_building_map_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for family in ("urban", "industrial"):
                path = root / f"{family}.p3d"
                write_building_mlod(
                    path,
                    BuildingVariantKey(family, "flat", 20.0, 30.0, 12.0),
                    wall_texture=rf"testworld\d\{family}.paa",
                    roof_texture=r"testworld\d\roof.paa",
                )
                summary = inspect_mlod(path)
                self.assertEqual(dict(summary.named_properties[1])["map"], "building")

    def test_visual_shell_contains_opposite_winding_for_every_surface(self) -> None:
        key = BuildingVariantKey("residential", "gabled", 10.0, 16.0, 9.0)
        lod = building_models._visual_lod(key, r"test\wall.paa", r"test\roof.paa", 35.0)
        self.assertEqual(len(lod.faces), 20)
        for index in range(0, len(lod.faces), 2):
            front = lod.faces[index]
            back = lod.faces[index + 1]
            self.assertEqual(front.texture, back.texture)
            self.assertEqual(front.flags, back.flags)
            self.assertEqual(front.vertices, tuple(reversed(back.vertices)))

    def test_generated_foundation_skirt_extends_below_origin(self) -> None:
        key = BuildingVariantKey("residential", "gabled", 10.0, 16.0, 9.0)
        foundation = r"test\foundation.paa"
        lod = building_models._visual_lod(
            key, r"test\wall.paa", r"test\roof.paa", 35.0,
            foundation_texture=foundation, foundation_depth=0.75,
        )
        foundation_faces = [face for face in lod.faces if face.texture == foundation]
        self.assertEqual(len(foundation_faces), 8)
        used_points = {point for face in foundation_faces for point, *_rest in face.vertices}
        heights = [lod.points[index][1] for index in used_points]
        self.assertAlmostEqual(min(heights), -0.75, places=6)
        self.assertAlmostEqual(max(heights), 0.0, places=6)

    def test_entrance_texture_is_confined_to_ground_floor(self) -> None:
        key = BuildingVariantKey("urban", "flat", 20.0, 30.0, 12.0)
        front_texture = r"test\front.paa"
        lod = building_models._visual_lod(
            key, r"test\wall.paa", r"test\roof.paa", 35.0, front_texture
        )
        entrance_faces = [face for face in lod.faces if face.texture == front_texture]
        self.assertEqual(len(entrance_faces), 2)  # exterior and reverse winding
        for face in entrance_faces:
            heights = [lod.points[point_index][1] for point_index, _normal, _u, _v in face.vertices]
            self.assertEqual(min(heights), 0.0)
            self.assertLessEqual(max(heights), 3.0)
        self.assertTrue(
            any(
                face.texture == r"test\wall.paa"
                and min(lod.points[point_index][1] for point_index, *_rest in face.vertices) >= 3.0
                for face in lod.faces
            )
        )

    def test_large_geometry_is_split_into_named_collision_components(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "warehouse.p3d"
            write_building_mlod(
                path,
                BuildingVariantKey("industrial", "flat", 80.0, 160.0, 12.0),
                wall_texture=r"testworld\d\w2.paa",
                roof_texture=r"testworld\d\r0.paa",
            )
            summary = inspect_mlod(path)
            self.assertEqual(summary.point_counts[1], 64)
            self.assertEqual(summary.face_counts[1], 48)
            self.assertEqual(
                summary.selection_names[1],
                tuple(f"component{index:02d}" for index in range(1, 9)),
            )
            self.assertEqual(summary.mass_point_counts[1], 64)

    def test_grounding_uses_selected_model_footprint_after_variant_reuse(self) -> None:
        from cwr_worldgen.model import PlayabilitySpec
        from cwr_worldgen.osm import OsmRaster, generate_world_objects

        bbox = (0.0, 0.0, 1.0, 1.0)
        projection = BboxProjection.create(bbox, 100.0)

        def polygon(key: str, x: float, z: float, width: float, length: float) -> OsmPolygonFeature:
            ring = (
                projection.to_latlon((x - width / 2.0, z - length / 2.0)),
                projection.to_latlon((x + width / 2.0, z - length / 2.0)),
                projection.to_latlon((x + width / 2.0, z + length / 2.0)),
                projection.to_latlon((x - width / 2.0, z + length / 2.0)),
                projection.to_latlon((x - width / 2.0, z - length / 2.0)),
            )
            return OsmPolygonFeature(key, {"building": "house"}, (GeoPolygon(ring),))

        buildings = (
            polygon("building/large-1", 20.0, 20.0, 40.0, 40.0),
            polygon("building/large-2", 80.0, 80.0, 40.0, 40.0),
            polygon("building/small", 50.0, 50.0, 4.0, 4.0),
        )
        dataset = OsmDataset(
            source_generator="test", element_count=3, coastlines=(), water=(), forests=(),
            farmland=(), urban=(), roads=(), building_polygons=buildings,
        )
        library = ProceduralBuildingLibrary(world_name="grounding_test", maximum_variants=1)
        library.prepare(dataset, projection, 12.0)
        spec = PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=bbox, cells=10, cell_size=10.0,
            building_minimum_area=1.0, max_buildings=10, max_forest_objects=0,
        )
        raster = OsmRaster(
            cells=10, water=(False,) * 100, forest=(False,) * 100, farmland=(False,) * 100,
            urban=(False,) * 100, roads=(False,) * 100, buildings=(False,) * 100,
            high_resolution=40, coastline_seed_count=0,
        )
        elevations = [0.0] * 100
        elevations[5 * 10 + 6] = 10.0  # x=65m, inside reused 40m model but outside 4m OSM outline
        result = generate_world_objects(
            dataset, projection, raster, elevations, spec, include_roads=False,
            building_asset_library=library,
        )
        small = min(result.objects, key=lambda obj: abs(obj.x - 50.0) + abs(obj.z - 50.0))
        self.assertGreaterEqual(small.y, 10.09)

    def test_final_selected_footprint_is_graded_before_building_placement(self) -> None:
        from cwr_worldgen.milestone8 import _Milestone8PlayabilitySpec
        from cwr_worldgen.osm import (
            OsmRaster, _polygon_elevation_extrema, generate_world_objects,
            plan_building_placements,
        )
        from cwr_worldgen.terrain_solver import solve_terrain_constraints

        bbox = (0.0, 0.0, 1.0, 1.0)
        projection = BboxProjection.create(bbox, 100.0)

        def polygon(key: str, x: float, z: float, width: float, length: float) -> OsmPolygonFeature:
            ring = tuple(
                projection.to_latlon(point)
                for point in (
                    (x - width / 2.0, z - length / 2.0),
                    (x + width / 2.0, z - length / 2.0),
                    (x + width / 2.0, z + length / 2.0),
                    (x - width / 2.0, z + length / 2.0),
                    (x - width / 2.0, z - length / 2.0),
                )
            )
            return OsmPolygonFeature(key, {"building": "house"}, (GeoPolygon(ring),))

        buildings = (
            polygon("building/large-1", 20.0, 20.0, 40.0, 40.0),
            polygon("building/large-2", 80.0, 80.0, 40.0, 40.0),
            polygon("building/small", 50.0, 50.0, 4.0, 4.0),
        )
        dataset = OsmDataset(
            source_generator="final-footprint", element_count=3, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(), building_polygons=buildings,
        )
        raster = OsmRaster(
            cells=10, water=(False,) * 100, forest=(False,) * 100,
            farmland=(False,) * 100, urban=(False,) * 100, roads=(False,) * 100,
            buildings=(False,) * 100, high_resolution=40, coastline_seed_count=0,
        )
        library = ProceduralBuildingLibrary(
            world_name="final_footprint", maximum_variants=1,
            foundation_depth=0.5, maximum_foundation_depth=2.5,
        )
        library.prepare(dataset, projection, 12.0)
        spec = _Milestone8PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=bbox, cells=10, cell_size=10.0,
            building_minimum_area=1.0, max_buildings=10, max_forest_objects=0,
            strict_assets=False,
        )
        plans, truncated = plan_building_placements(dataset, projection, raster, spec, library)
        self.assertFalse(truncated)
        small_plan = next(plan for plan in plans if plan.osm_key == "building/small")
        self.assertEqual(small_plan.procedural_placement.selected.width_m, 40.0)

        original = [0.0] * 100
        original[5 * 10 + 6] = 10.0  # inside the selected model, outside the OSM outline
        grading = solve_terrain_constraints(
            original, dataset, projection, raster, spec,
            building_placement_plans=plans,
        )
        minimum, maximum = _polygon_elevation_extrema(
            grading.elevations, spec.cells, spec.cell_size, small_plan.support_polygon
        )
        self.assertLessEqual(maximum - minimum, spec.building_maximum_pad_relief + 1e-6)

        result = generate_world_objects(
            dataset, projection, raster, grading.elevations, spec, include_roads=False,
            building_asset_library=library, building_placement_plans=plans,
        )
        self.assertEqual(result.building_foundation_rejections, 0)
        small = min(result.objects, key=lambda obj: abs(obj.x - 50.0) + abs(obj.z - 50.0))
        self.assertAlmostEqual(small.y, maximum + spec.building_ground_clearance, places=6)

    def test_large_church_uses_interpolation_safe_hillside_terrace(self) -> None:
        from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
        from cwr_worldgen.osm import (
            OsmRaster, _polygon_elevation_extrema, generate_world_objects,
            plan_building_placements,
        )
        from cwr_worldgen.terrain_solver import solve_terrain_constraints

        bbox = (0.0, 0.0, 1.0, 1.0)
        cells = 16
        cell_size = 25.0
        projection = BboxProjection.create(bbox, cells * cell_size)
        ring = tuple(
            projection.to_latlon(point)
            for point in (
                (160.0, 150.0), (240.0, 150.0), (240.0, 250.0),
                (160.0, 250.0), (160.0, 150.0),
            )
        )
        church = OsmPolygonFeature(
            "way/152410159",
            {"building": "church", "amenity": "place_of_worship"},
            (GeoPolygon(ring),),
        )
        dataset = OsmDataset(
            source_generator="steep-church", element_count=1, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(), building_polygons=(church,),
        )
        raster = OsmRaster(
            cells=cells, water=(False,) * (cells * cells),
            forest=(False,) * (cells * cells), farmland=(False,) * (cells * cells),
            urban=(False,) * (cells * cells), roads=(False,) * (cells * cells),
            buildings=(False,) * (cells * cells), high_resolution=cells,
            coastline_seed_count=0,
        )
        spec = _Milestone9PlayabilitySpec(
            name="steep_church", heightmap_path=Path("unused.png"), bbox=bbox,
            cells=cells, cell_size=cell_size, building_minimum_area=1.0,
            max_buildings=4, max_forest_objects=0, strict_assets=False,
        )
        library = ProceduralBuildingLibrary(
            world_name=spec.name, maximum_foundation_depth=spec.building_foundation_maximum_depth
        )
        library.prepare(dataset, projection, 12.0)
        plans, _ = plan_building_placements(dataset, projection, raster, spec, library)
        original = [float(x * 2) for _z in range(cells) for x in range(cells)]
        grading = solve_terrain_constraints(
            original, dataset, projection, raster, spec, building_placement_plans=plans
        )
        final_elevations = quantize_elevations(grading.elevations, spec.height_scale)
        from cwr_worldgen.osm import _oriented_rectangle
        selected = plans[0].procedural_placement.selected
        church_support = _oriented_rectangle(
            plans[0].x, plans[0].z, selected.width_m, selected.length_m,
            plans[0].heading_degrees, margin=max(0.75, min(3.00, spec.cell_size * 0.04)),
        )
        minimum, maximum = _polygon_elevation_extrema(
            final_elevations, cells, cell_size, church_support
        )
        self.assertLessEqual(maximum - minimum, spec.building_maximum_pad_relief + spec.height_scale + 1e-6)
        result = generate_world_objects(
            dataset, projection, raster, final_elevations, spec, include_roads=False,
            building_asset_library=library, building_placement_plans=plans,
        )
        self.assertEqual(result.building_foundation_rejections, 0)
        self.assertEqual(result.building_objects, 1)
        church_object = result.objects[0]
        self.assertAlmostEqual(
            church_object.y,
            maximum + spec.building_ground_clearance,
            places=6,
        )
        self.assertEqual(library.church_plinth_height, 0.0)
        church_key = next(key for key in library._usage if key.family == "church")
        visual = building_models._visual_lod(
            church_key,
            r"test\wall.paa",
            r"test\roof.paa",
            35.0,
            foundation_texture=r"test\foundation.paa",
            foundation_depth=church_key.foundation_depth_m,
            church_plinth_height=library.church_plinth_height,
        )
        self.assertAlmostEqual(min(point[1] for point in visual.points[:10]), 0.0, places=6)
        self.assertGreaterEqual(church_key.foundation_depth_m, spec.building_foundation_depth)

    def test_foundation_depth_is_calculated_and_quantized_per_building(self) -> None:
        from cwr_worldgen.milestone8 import _Milestone8PlayabilitySpec
        from cwr_worldgen.osm import OsmRaster, generate_world_objects, plan_building_placements

        bbox = (0.0, 0.0, 1.0, 1.0)
        projection = BboxProjection.create(bbox, 40.0)
        ring = tuple(
            projection.to_latlon(point)
            for point in ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0), (10.0, 10.0))
        )
        dataset = OsmDataset(
            source_generator="dynamic-foundation", element_count=1, coastlines=(), water=(),
            forests=(), farmland=(), urban=(), roads=(),
            building_polygons=(OsmPolygonFeature("building/one", {"building": "house"}, (GeoPolygon(ring),)),),
        )
        raster = OsmRaster(
            cells=4, water=(False,) * 16, forest=(False,) * 16, farmland=(False,) * 16,
            urban=(False,) * 16, roads=(False,) * 16, buildings=(False,) * 16,
            high_resolution=16, coastline_seed_count=0,
        )
        library = ProceduralBuildingLibrary(
            world_name="dynamic_foundation", foundation_depth=0.5,
            maximum_foundation_depth=2.5, foundation_depth_quantum=0.25,
        )
        library.prepare(dataset, projection, 12.0)
        spec = _Milestone8PlayabilitySpec(
            heightmap_path=Path("unused.png"), bbox=bbox, cells=4, cell_size=10.0,
            building_minimum_area=1.0, max_buildings=1, max_forest_objects=0,
            strict_assets=False,
        )
        plans, _ = plan_building_placements(dataset, projection, raster, spec, library)
        elevations = [0.0] * 16
        elevations[2 * 4 + 2] = 1.0
        result = generate_world_objects(
            dataset, projection, raster, elevations, spec, include_roads=False,
            building_asset_library=library, building_placement_plans=plans,
        )
        self.assertEqual(result.building_foundation_rejections, 0)
        self.assertEqual(result.maximum_building_foundation_depth, 1.5)
        self.assertEqual({key.foundation_depth_m for key in library._usage}, {1.5})

    def test_reuses_quantized_variants_and_honours_variant_cap(self) -> None:
        bbox = (0.0, 0.0, 0.01, 0.01)
        projection = BboxProjection.create(bbox, 1000.0)

        def polygon(key: str, building: str, x: float, z: float, width: float, length: float, levels: str) -> OsmPolygonFeature:
            ring = (
                projection.to_latlon((x, z)),
                projection.to_latlon((x + width, z)),
                projection.to_latlon((x + width, z + length)),
                projection.to_latlon((x, z + length)),
                projection.to_latlon((x, z)),
            )
            return OsmPolygonFeature(
                key,
                {"building": building, "building:levels": levels},
                (GeoPolygon(ring),),
            )

        buildings = (
            polygon("building/1", "house", 100, 100, 10.2, 15.7, "2"),
            polygon("building/2", "house", 200, 100, 10.4, 15.6, "2"),
            polygon("building/3", "apartments", 300, 100, 18.0, 28.0, "5"),
        )
        dataset = OsmDataset(
            source_generator="test",
            element_count=3,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
            building_polygons=buildings,
        )
        library = ProceduralBuildingLibrary(world_name="reuse_test", maximum_variants=1)
        library.prepare(dataset, projection, 12.0)
        for feature in buildings:
            points = [projection.to_world(point) for point in feature.polygons[0].outer[:-1]]
            library.place_polygon(feature.tags, points)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = library.write_assets(root, root / "catalogue.json")
            self.assertEqual(result.placements, 3)
            self.assertEqual(result.generated_variants, 3)
            self.assertEqual(result.reused_placements, 0)
            self.assertEqual(result.reuse_ratio, 0.0)
            self.assertEqual(len(result.model_assets), 3)
            geometry_signatures = {
                (
                    asset.key.family, asset.key.roof_style, asset.key.width_m,
                    asset.key.length_m, asset.key.height_m,
                    asset.key.foundation_depth_m, asset.key.regional_style,
                )
                for asset in result.model_assets
            }
            self.assertEqual(len(geometry_signatures), 1)
            self.assertEqual(len({asset.key.texture_variant for asset in result.model_assets}), 3)
            for asset in result.model_assets:
                model = root / asset.relative_path
                self.assertTrue(model.is_file())
                self.assertEqual(inspect_mlod(model).lod_count, 3)

            walls = [relative for relative in result.texture_files if relative.startswith("d/w")]
            fronts = [relative for relative in result.texture_files if relative.startswith("d/e")]
            roofs = [relative for relative in result.texture_files if relative.startswith("d/r")]
            self.assertEqual(len(walls), 10)
            self.assertEqual(len(fronts), 10)
            self.assertEqual(len(roofs), 10)
            self.assertEqual(len({(root / relative).read_bytes() for relative in walls}), 10)
            self.assertEqual(len({(root / relative).read_bytes() for relative in fronts}), 10)
            self.assertEqual(len({(root / relative).read_bytes() for relative in roofs}), 10)
            wall = root / walls[0]
            roof = root / roofs[0]
            from cwr_worldgen.procedural_buildings import BUILDING_ASSET_TEXTURE_SIZE
            wall_info = inspect_paa(wall)
            roof_info = inspect_paa(roof)
            self.assertEqual((wall_info.width, wall_info.height), (BUILDING_ASSET_TEXTURE_SIZE,) * 2)
            self.assertEqual((roof_info.width, roof_info.height), (BUILDING_ASSET_TEXTURE_SIZE,) * 2)
            self.assertGreater(wall_info.mipmap_count, 1)
            self.assertGreater(roof_info.mipmap_count, 1)
            solid = root / "solid.paa"
            write_solid_dxt1_paa(solid, colour=(184, 169, 143))
            self.assertNotEqual(wall.read_bytes(), solid.read_bytes())

    def test_high_quality_building_textures_are_opt_in(self) -> None:
        bbox = (0.0, 0.0, 0.01, 0.01)
        projection = BboxProjection.create(bbox, 1000.0)
        dataset = OsmDataset(
            source_generator="hq-building-texture-test", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            low = ProceduralBuildingLibrary(world_name="lowtex")
            low.prepare(dataset, projection, 12.0)
            low.register_placement(low.plan_point({"building": "house"}, 12.0, 0.0, x=10.0, z=10.0))
            low_result = low.write_assets(root / "low", root / "low.json")
            high = ProceduralBuildingLibrary(world_name="hightex", high_quality_textures=True)
            high.prepare(dataset, projection, 12.0)
            high.register_placement(high.plan_point({"building": "house"}, 12.0, 0.0, x=10.0, z=10.0))
            high_result = high.write_assets(root / "high", root / "high.json")
            low_wall = next(root / "low" / rel for rel in low_result.texture_files if rel.startswith("d/w"))
            high_wall = next(root / "high" / rel for rel in high_result.texture_files if rel.startswith("d/w"))
            self.assertEqual(inspect_paa(low_wall).width, 128)
            self.assertEqual(inspect_paa(high_wall).width, 256)

    def test_building_positions_select_all_ten_texture_variants_deterministically(self) -> None:
        bbox = (0.0, 0.0, 0.01, 0.01)
        projection = BboxProjection.create(bbox, 1000.0)
        dataset = OsmDataset(
            source_generator="texture-variants", element_count=0,
            coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        )
        first = ProceduralBuildingLibrary(world_name="texture_variants")
        second = ProceduralBuildingLibrary(world_name="texture_variants")
        first.prepare(dataset, projection, 12.0)
        second.prepare(dataset, projection, 12.0)
        variants_first = [
            first.plan_point(
                {"building": "house"}, 12.0, 0.0,
                x=50.0 + index * 7.0, z=100.0 + index * 11.0,
            ).selected.texture_variant
            for index in range(200)
        ]
        variants_second = [
            second.plan_point(
                {"building": "house"}, 12.0, 0.0,
                x=50.0 + index * 7.0, z=100.0 + index * 11.0,
            ).selected.texture_variant
            for index in range(200)
        ]
        self.assertEqual(variants_first, variants_second)
        self.assertEqual(set(variants_first), set(range(10)))

    def test_ofp_style_textures_are_low_palette_and_dark_glazed(self) -> None:
        from cwr_worldgen.procedural_buildings import (
            PAINTED_WINDOW_MINIMUM_SILL_M,
            _front_texture_image,
            _roof_texture_image,
            _wall_texture_image,
        )

        residential = _wall_texture_image("residential")
        agricultural = _wall_texture_image("agricultural")
        roof = _roof_texture_image("gabled")
        self.assertEqual(residential.size, (128, 128))
        self.assertLess(len(set(residential.get_flattened_data())), 600)
        self.assertLess(len(set(agricultural.get_flattened_data())), 500)
        self.assertLess(len(set(roof.get_flattened_data())), 450)
        # Old CWA windows are intentionally dark rather than glossy blue panels.
        self.assertLess(sum(residential.getpixel((50, 50))) / 3.0, 100.0)

        # Closed-building artwork keeps a full metre of facade below every
        # painted window, while its painted entrance still reaches the local
        # ground threshold instead of floating above the enlarged footing.
        townhouse = _wall_texture_image("townhouse")
        glazed_rows = [
            y
            for y in range(townhouse.height)
            for x in range(townhouse.width)
            if townhouse.getpixel((x, y)) == (54, 67, 69)
        ]
        self.assertTrue(glazed_rows)
        maximum_window_row = round(
            townhouse.height
            * (1.0 - PAINTED_WINDOW_MINIMUM_SILL_M / 3.0)
        )
        self.assertLess(max(glazed_rows), maximum_window_row)

        # A wall atlas repeats once per storey, so its lower metre must remain
        # ordinary facade material. The real stone foundation is model geometry
        # and must not become a grey belt between every floor.
        swedish = _wall_texture_image("residential", regional_style="sweden_red")
        lower_sample = swedish.getpixel((64, 116))
        self.assertGreater(lower_sample[0], lower_sample[1])
        self.assertGreater(lower_sample[0], lower_sample[2])
        self.assertNotEqual(lower_sample, (82, 79, 70))

        front = _front_texture_image("townhouse")
        self.assertEqual(front.getpixel((64, 40)), (66, 58, 47))
        self.assertNotEqual(front.getpixel((64, 116)), townhouse.getpixel((64, 116)))


class Milestone8BuildTests(unittest.TestCase):
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
            "generator": "milestone8-test",
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
                {"type": "way", "id": 6, "tags": {"building": "house", "building:levels": "2", "roof:shape": "gabled"}, "geometry": [
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

    def test_build_emits_embedded_reusable_building_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            spec = Milestone8Spec(
                source_dir=source,
                name="cwr_m8_test",
                display_name="CWR M8 Test",
                solver_iterations=8,
                world_edge_blend_cells=2,
                max_forest_objects=0,
            )
            first = build_milestone8(root / "one", spec)
            second = build_milestone8(root / "two", spec)
            self.assertEqual(first.wrp_path.read_bytes(), second.wrp_path.read_bytes())
            self.assertEqual(first.pbo_path.read_bytes(), second.pbo_path.read_bytes())
            self.assertEqual(first.pbo_path.parent.parent.name, "@CWR-Milestone8")
            self.assertIsNotNone(first.building_catalogue_path)
            catalogue = json.loads(first.building_catalogue_path.read_text(encoding="utf-8"))
            self.assertEqual(catalogue["placements"], 1)
            self.assertEqual(catalogue["generated_variants"], 1)
            model_path = catalogue["models"][0]["model_path"]
            self.assertTrue(model_path.startswith(r"cwr_m8_test\g\b_"))

            entries = {entry.name for entry in read_pbo(first.pbo_path)}
            self.assertTrue(any(name.startswith(r"g\b_") and name.endswith(".p3d") for name in entries))
            self.assertIn(r"g\buildings.json", entries)
            self.assertTrue(any(name.startswith(r"d\w") and name.endswith(".paa") for name in entries))
            self.assertTrue(any(name.startswith(r"d\r") and name.endswith(".paa") for name in entries))

            wrp = inspect_rvw4(first.wrp_path, height_scale=0.05)
            self.assertIn(model_path, wrp.object_models)
            self.assertEqual(
                set(wrp.texture_paths),
                {rf"{spec.name}\data\d.paa", *NOGOVA_GROUND_TEXTURES.values()},
            )
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["ground_texture_profile"], "nogova")
            self.assertEqual(
                {material["texture_path"] for material in manifest["materials"]},
                set(NOGOVA_GROUND_TEXTURES.values()),
            )
            report = first.report_path.read_text(encoding="utf-8")
            self.assertIn("Procedural building P3Ds emitted", report)
            self.assertIn("Building asset reuse bounded", report)


    def test_strict_assets_reuses_milestone7_roads_without_revalidating_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            assets = root / "assets"
            for relative in set(NOGOVA_GROUND_TEXTURES.values()):
                path = assets / Path(relative.replace("\\", "/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"legacy-ground-texture")

            spec = Milestone8Spec(
                source_dir=source,
                name="cwr_m8_roads",
                display_name="CWR M8 Roads",
                solver_iterations=4,
                world_edge_blend_cells=2,
                max_forest_objects=16,
                asset_roots=(assets,),
                strict_assets=True,
            )
            result = build_milestone8(root / "build", spec)
            wrp = inspect_rvw4(result.wrp_path, height_scale=0.05)
            self.assertIn(r"o\road\sil25.p3d", {path.casefold() for path in wrp.object_models})
            self.assertIn(r"o\road\ces25.p3d", {path.casefold() for path in wrp.object_models})

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            strict = manifest["playability"]["strict_asset_validation"]
            self.assertTrue(strict["verified"])
            self.assertEqual(
                set(strict["trusted_legacy_assets"]),
                {
                    r"o\road\sil25.p3d",
                    r"o\road\ces25.p3d",
                    r"data3d\les_su_ctver_pruhozi.p3d",
                },
            )
            full = manifest["playability"]["asset_catalogue"]
            self.assertEqual(
                set(full["missing_models"]),
                {
                    r"o\road\sil25.p3d",
                    r"o\road\ces25.p3d",
                    r"data3d\les_su_ctver_pruhozi.p3d",
                },
            )


if __name__ == "__main__":
    unittest.main()
