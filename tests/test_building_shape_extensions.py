from __future__ import annotations

from dataclasses import replace
import json
import math
import tempfile
from pathlib import Path

import pytest
from shapely.geometry import Point

from cwr_worldgen.model import OsmSpec
from cwr_worldgen.normalization import (
    NormalizationSpec, load_normalized_dataset, normalize_source_bundle,
)
from cwr_worldgen.source_pipeline import FrozenSourceBundle
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmPointFeature,
    OsmPolygonFeature,
    OsmRaster,
    generate_world_objects,
    parse_overpass_json,
    plan_building_placements,
)
from cwr_worldgen.procedural_buildings import (
    BuildingVariantKey,
    ProceduralBuildingLibrary,
    _door_dimensions,
    _front_vector_for_heading,
    _gabled_profile,
    _interior_window_openings,
    _interior_storey_count,
    _land_contact_lod,
    _polygon_native_land_contact_lod,
    _polygon_native_edge_openings,
    _polygon_native_roof_mesh,
    _polygon_native_shape,
    _triangulate_polygon_coordinates,
    _polygon_native_visual_lod,
    _visual_lod,
    decompose_footprint_rectangles,
    inspect_mlod,
    write_building_mlod,
)


def _empty_dataset() -> OsmDataset:
    return OsmDataset(
        source_generator="test", element_count=0,
        coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
    )


def test_overpass_parser_preserves_mapped_building_entrance_nodes() -> None:
    document = {
        "generator": "test",
        "elements": [
            {"type": "node", "id": 7, "lat": 59.0, "lon": 16.0,
             "tags": {"entrance": "main"}},
            {"type": "node", "id": 8, "lat": 59.1, "lon": 16.1,
             "tags": {"entrance": "no"}},
        ],
    }
    dataset = parse_overpass_json(json.dumps(document).encode("utf-8"))
    assert len(dataset.building_entrances) == 1
    assert dataset.building_entrances[0].osm_key == "node/7"
    assert dataset.building_entrances[0].tags["entrance"] == "main"

def test_normalization_preserves_building_entrances() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "source"
        root.mkdir()
        raw = root / "raw-overpass.json"
        raw.write_text(json.dumps({
            "generator": "test",
            "elements": [
                {
                    "type": "way", "id": 1, "tags": {"building": "house"},
                    "geometry": [
                        {"lat": 0.2, "lon": 0.2}, {"lat": 0.2, "lon": 0.4},
                        {"lat": 0.4, "lon": 0.4}, {"lat": 0.4, "lon": 0.2},
                        {"lat": 0.2, "lon": 0.2},
                    ],
                },
                {
                    "type": "node", "id": 2, "lat": 0.2, "lon": 0.3,
                    "tags": {"entrance": "main"},
                },
            ],
        }), encoding="utf-8")
        dummy = root / "placeholder"
        dummy.write_text("x", encoding="ascii")
        source = FrozenSourceBundle(
            root=root, manifest_path=dummy, checksum_path=dummy, heightmap_path=dummy,
            osm_json_path=raw, overpass_query_path=dummy, osm_attribution_path=dummy,
            dem_attribution_path=dummy, reference_map_path=None,
            overture_buildings_geojson_path=None, bbox=(0.0, 0.0, 1.0, 1.0),
            cells=64, cell_size=25.0, heightmap_grid="game-cell-centres",
            fingerprint="a" * 64,
        )
        normalized = normalize_source_bundle(
            NormalizationSpec(source_dir=root, output_dir=root / "normalized"),
            validated_source=source,
        )
        dataset = load_normalized_dataset(normalized, use_cache=False)
        assert normalized.counts["building-entrances"] == 1
        assert len(dataset.building_entrances) == 1
        assert dataset.building_entrances[0].tags["entrance"] == "main"


def test_mapped_entrance_overrides_nearest_road_frontage() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="entrance_facing")
    library.prepare(_empty_dataset(), projection, 12.0)
    square = ((40.0, 40.0), (52.0, 40.0), (52.0, 52.0), (40.0, 52.0))

    placement = library.plan_polygon(
        {"building": "house"}, square,
        road_point=(46.0, 20.0),        # road is south
        entrance_point=(60.0, 46.0),    # mapped entrance is east
    )
    front_x, front_z = _front_vector_for_heading(placement.heading_degrees)
    assert front_x > 0.999
    assert abs(front_z) < 1.0e-6


@pytest.mark.parametrize("tag, expected", [
    ("hipped", "hipped"),
    ("half-hipped", "hipped"),
    ("pyramidal", "pyramidal"),
    ("dome", "dome"),
    ("onion", "onion"),
])
def test_requested_roof_shapes_are_preserved(tag: str, expected: str) -> None:
    library = ProceduralBuildingLibrary(world_name="roof_shapes")
    key = library.key_for({"building": "house", "roof:shape": tag}, 12.0, 16.0)
    assert key.roof_style == expected


@pytest.mark.parametrize("roof_style", ["hipped", "pyramidal", "dome", "onion"])
def test_new_roofs_generate_real_visual_geometry(roof_style: str) -> None:
    key = BuildingVariantKey("residential", roof_style, 12.0, 16.0, 6.0)
    lod = _visual_lod(key, r"test\wall.paa", r"test\roof.paa", 35.0)
    roof_faces = [face for face in lod.faces if face.texture == r"test\roof.paa"]
    assert roof_faces
    assert max(point[1] for point in lod.points) >= 6.0
    # Dome/onion require substantially more than the legacy box/gable point set.
    if roof_style in {"dome", "onion"}:
        assert len(lod.points) > 40


def test_l_shape_decomposes_into_two_adjoining_rectangles() -> None:
    l_shape = (
        (0.0, 0.0), (12.0, 0.0), (12.0, 5.0),
        (5.0, 5.0), (5.0, 15.0), (0.0, 15.0),
    )
    parts = decompose_footprint_rectangles(l_shape)
    assert len(parts) == 2
    areas = sorted(
        abs(sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(part, part[1:] + part[:1])
        )) * 0.5
        for part in parts
    )
    assert areas == pytest.approx([50.0, 60.0])


def test_l_shape_hipped_roof_stays_one_polygon_native_building() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    points = ((20.0, 20.0), (50.0, 20.0), (50.0, 30.0),
              (30.0, 30.0), (30.0, 55.0), (20.0, 55.0))
    ring = tuple(projection.to_latlon(point) for point in (*points, points[0]))
    entrance = projection.to_latlon((25.0, 55.0))
    dataset = OsmDataset(
        source_generator="l-shape", element_count=2,
        coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        building_polygons=(
            OsmPolygonFeature(
                "way/l", {"building": "house", "roof:shape": "hipped"},
                (GeoPolygon(ring),),
            ),
        ),
        building_entrances=(
            OsmPointFeature("node/entrance", {"entrance": "main"}, entrance),
        ),
    )
    library = ProceduralBuildingLibrary(world_name="l_shape")
    library.prepare(dataset, projection, 12.0)
    raster = OsmRaster(
        cells=4, water=(False,) * 16, forest=(False,) * 16,
        farmland=(False,) * 16, urban=(False,) * 16, roads=(False,) * 16,
        buildings=(False,) * 16, high_resolution=4, coastline_seed_count=0,
    )
    spec = OsmSpec(
        heightmap_path=Path("unused"), bbox=(0.0, 0.0, 1.0, 1.0),
        cells=4, cell_size=25.0, max_buildings=100,
        max_forest_objects=0, max_road_objects=0,
    )
    plans, truncated = plan_building_placements(dataset, projection, raster, spec, library)
    assert not truncated
    assert len(plans) == 1
    plan = plans[0]
    assert plan.geometry_kind == "polygon"
    assert plan.procedural_placement is not None
    assert plan.procedural_placement.selected.roof_style == "hipped"
    assert len(plan.procedural_placement.selected.footprint_vertices) == 6
    assert not plan.procedural_placement.selected.footprint_holes

    # Deliberately sloped terrain: the whole L remains one rigid semantic
    # building and is grounded once rather than producing wing seams.
    elevations = tuple(float((index // 4) * 2) for index in range(16))
    result = generate_world_objects(
        dataset, projection, raster, elevations, spec,
        include_roads=False, building_asset_library=library,
        building_placement_plans=plans,
    )
    buildings = result.objects[:result.building_objects]
    assert len(buildings) == 1


def test_polygon_native_courtyard_is_not_filled_by_roof_or_collision() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="courtyard_native")
    library.prepare(_empty_dataset(), projection, 12.0)
    outer = ((10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0))
    hole = ((22.0, 22.0), (38.0, 22.0), (38.0, 38.0), (22.0, 38.0))
    placement = library.plan_polygon(
        {"building": "apartments", "roof:shape": "hipped"},
        outer,
        holes=(hole,),
        entrance_point=(10.0, 31.0),
    )
    key = placement.selected
    assert key.footprint_vertices
    assert len(key.footprint_holes) == 1
    assert key.roof_style == "hipped"
    assert _polygon_native_shape(key).area == pytest.approx(1600.0 - 256.0)

    detail = _polygon_native_visual_lod(
        key, "wall.paa", "roof.paa", front_texture="front.paa"
    )
    roof_faces = [face for face in detail.faces if face.texture == "roof.paa"]
    assert roof_faces
    # No roof triangle centroid may land inside the courtyard ring.
    courtyard = _polygon_native_shape(replace(key, footprint_vertices=key.footprint_holes[0], footprint_holes=()))
    for face in roof_faces[::2]:  # visual shell is double sided
        coords = [detail.points[vertex[0]] for vertex in face.vertices]
        cx = sum(point[0] for point in coords) / len(coords)
        cz = sum(point[2] for point in coords) / len(coords)
        assert not courtyard.contains(Point(cx, cz))


def test_polygon_native_mapped_entrance_keeps_lateral_position() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="native_entrance_position")
    library.prepare(_empty_dataset(), projection, 12.0)
    l_shape = (
        (10.0, 10.0), (40.0, 10.0), (40.0, 20.0),
        (25.0, 20.0), (25.0, 40.0), (10.0, 40.0),
    )
    placement = library.plan_polygon(
        {"building": "house", "roof:shape": "gabled"},
        l_shape,
        entrance_point=(10.0, 17.0),
    )
    key = placement.selected
    assert key.entrance_edge >= 0
    assert key.entrance_fraction != pytest.approx(0.5)

    detail = _polygon_native_visual_lod(
        key, "wall.paa", "roof.paa", front_texture="front.paa"
    )
    front_faces = [face for face in detail.faces if face.texture == "front.paa"]
    assert len(front_faces) == 2
    face = front_faces[0]
    centre_x = sum(detail.points[vertex[0]][0] for vertex in face.vertices) / 4.0
    centre_z = sum(detail.points[vertex[0]][2] for vertex in face.vertices) / 4.0
    edge_start = key.footprint_vertices[key.entrance_edge]
    edge_end = key.footprint_vertices[(key.entrance_edge + 1) % len(key.footprint_vertices)]
    expected_x = edge_start[0] + (edge_end[0] - edge_start[0]) * key.entrance_fraction
    expected_z = edge_start[1] + (edge_end[1] - edge_start[1]) * key.entrance_fraction
    assert math.hypot(centre_x - expected_x, centre_z - expected_z) <= 0.2


def test_polygon_native_house_gets_real_enterable_lods(tmp_path: Path) -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(
        world_name="native_interior", generate_interiors=True
    )
    library.prepare(_empty_dataset(), projection, 12.0)
    l_shape = (
        (10.0, 10.0), (36.0, 10.0), (36.0, 18.0),
        (24.0, 18.0), (24.0, 36.0), (10.0, 36.0),
    )
    placement = library.plan_polygon(
        {"building": "house", "roof:shape": "gabled"},
        l_shape,
        entrance_point=(10.0, 16.0),
    )
    key = placement.selected
    assert key.footprint_vertices
    assert key.interiors
    assert not key.second_storey

    target = tmp_path / "native_enterable.p3d"
    write_building_mlod(
        target,
        key,
        wall_texture="open.paa",
        roof_texture="roof.paa",
        front_texture="front.paa",
        foundation_texture="floor.paa",
        foundation_depth=0.5,
        interior_texture="inside.paa",
        window_trim_texture="trim.paa",
        plain_wall_texture="plain.paa",
        door_texture="door.paa",
        distance_wall_texture="wall.paa",
    )
    summary = inspect_mlod(target)
    assert summary.lod_count >= 7
    assert "door1" in summary.selection_names[0]
    geometry_index = min(
        range(summary.lod_count),
        key=lambda index: abs(summary.resolutions[index] - 1.0e13),
    )
    assert "door1" in summary.selection_names[geometry_index]
    assert any("door1_axis" in names for names in summary.selection_names)
    assert any("In1" in names for names in summary.selection_names)
    assert "inside.paa" in summary.texture_paths
    assert "trim.paa" in summary.texture_paths


def test_domestic_door_dimensions_are_human_scale() -> None:
    key = BuildingVariantKey("residential", "gabled", 12.0, 16.0, 6.0, interiors=True)
    half_width, height, _pivot = _door_dimensions(key)
    assert 0.96 <= half_width * 2.0 <= 1.08
    assert 2.0 <= height <= 2.12


def test_sectioned_l_roof_has_multiple_high_ridge_regions() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="sectioned_roof")
    library.prepare(_empty_dataset(), projection, 12.0)
    l_shape = (
        (10.0, 10.0), (42.0, 10.0), (42.0, 20.0),
        (27.0, 20.0), (27.0, 42.0), (10.0, 42.0),
    )
    key = library.plan_polygon(
        {"building": "house", "roof:shape": "gabled"}, l_shape
    ).selected
    assert key.footprint_vertices
    eave, triangles, height_at = _polygon_native_roof_mesh(key, 35.0)
    assert triangles
    # Sectioned roofs should have high points in more than one coordinate band,
    # unlike the old single global ridge that forced every wing onto one line.
    high_points = {
        (round(x, 1), round(z, 1))
        for triangle in triangles
        for x, z in triangle
        if height_at((x, z)) >= eave + 0.8
    }
    assert len({point[0] for point in high_points}) >= 2
    assert len({point[1] for point in high_points}) >= 2


def test_polygon_native_pitched_roof_adds_under_cap_to_prevent_visible_holes() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="roof_cap")
    library.prepare(_empty_dataset(), projection, 12.0)
    l_shape = (
        (10.0, 10.0), (42.0, 10.0), (42.0, 20.0),
        (27.0, 20.0), (27.0, 42.0), (10.0, 42.0),
    )
    key = library.plan_polygon(
        {"building": "barn", "roof:shape": "hipped"}, l_shape
    ).selected
    detail = _polygon_native_visual_lod(
        key, "wall.paa", "roof.paa", plain_wall_texture="plain.paa"
    )
    cap_faces = [face for face in detail.faces if face.texture == "plain.paa"]
    assert cap_faces


def test_polygon_native_foundation_has_visible_reveal() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="native_foundation")
    library.prepare(_empty_dataset(), projection, 12.0)
    key = library.plan_polygon(
        {"building": "house", "roof:shape": "gabled"},
        ((10.0, 10.0), (34.0, 10.0), (34.0, 20.0), (22.0, 20.0), (22.0, 34.0), (10.0, 34.0)),
    ).selected
    detail = _polygon_native_visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        foundation_texture="foundation.paa",
        foundation_depth=0.75,
        plain_wall_texture="plain.paa",
    )
    foundation_points = {
        detail.points[vertex[0]]
        for face in detail.faces if face.texture == "foundation.paa"
        for vertex in face.vertices
    }
    assert foundation_points
    assert max(point[1] for point in foundation_points) > 0.0
    assert min(point[1] for point in foundation_points) < 0.0


def test_polygon_native_short_edge_uses_plain_wall_texture() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="short_edge_plain")
    library.prepare(_empty_dataset(), projection, 12.0)
    shape = (
        (10.0, 10.0), (28.0, 10.0), (28.0, 12.4),
        (20.8, 12.4), (20.8, 30.0), (10.0, 30.0),
    )
    key = library.plan_polygon(
        {"building": "house", "roof:shape": "gabled"}, shape
    ).selected
    detail = _polygon_native_visual_lod(
        key, "wall.paa", "roof.paa", plain_wall_texture="plain.paa"
    )
    assert any(face.texture == "plain.paa" for face in detail.faces)


def test_land_contact_lods_include_midpoints_and_centres() -> None:
    rectangular = _land_contact_lod(
        BuildingVariantKey("residential", "gabled", 12.0, 16.0, 6.0)
    )
    assert len(rectangular.points) >= 9

    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="native_landcontact")
    library.prepare(_empty_dataset(), projection, 12.0)
    key = library.plan_polygon(
        {"building": "house"},
        ((10.0, 10.0), (34.0, 10.0), (34.0, 24.0), (22.0, 24.0), (22.0, 34.0), (10.0, 34.0)),
    ).selected
    land = _polygon_native_land_contact_lod(key)
    assert len(land.points) > len(key.footprint_vertices)


def test_closed_gabled_residential_uses_plain_gable_texture() -> None:
    key = BuildingVariantKey("residential", "gabled", 10.0, 16.0, 6.0)
    detail = _visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        plain_wall_texture="plain.paa",
    )
    assert any(face.texture == "plain.paa" and len(face.vertices) == 3 for face in detail.faces)


def test_closed_flat_upper_band_switches_to_plain_when_too_shallow() -> None:
    key = BuildingVariantKey("residential", "flat", 9.0, 14.0, 4.1)
    detail = _visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        front_texture="front.paa",
        plain_wall_texture="plain.paa",
    )
    assert any(face.texture == "plain.paa" for face in detail.faces)


def test_closed_gabled_upper_band_switches_to_plain_when_too_narrow() -> None:
    key = BuildingVariantKey("residential", "gabled", 7.0, 11.0, 6.0)
    detail = _visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        front_texture="front.paa",
        plain_wall_texture="plain.paa",
    )
    plain_quads = [face for face in detail.faces if face.texture == "plain.paa" and len(face.vertices) == 4]
    assert plain_quads


def test_closed_two_storey_residential_uses_two_nonwrapping_window_bands() -> None:
    key = BuildingVariantKey(
        "residential", "gabled", 12.0, 18.0, 6.0, facade_storeys=2
    )
    detail = _visual_lod(
        key,
        "windowed.paa",
        "roof.paa",
        35.0,
        front_texture="front.paa",
        plain_wall_texture="plain.paa",
    )
    window_faces = [
        face for face in detail.faces
        if face.texture == "windowed.paa" and len(face.vertices) == 4
    ]
    assert window_faces
    vertical_bands = {
        tuple(round(value, 3) for value in (
            min(detail.points[v[0]][1] for v in face.vertices),
            max(detail.points[v[0]][1] for v in face.vertices),
        ))
        for face in window_faces[::2]
    }
    assert len(vertical_bands) == 2
    for face in window_faces:
        v_values = [vertex[3] for vertex in face.vertices]
        assert max(v_values) - min(v_values) < 1.0
        assert min(v_values) > 0.0
        assert max(v_values) < 1.0


def test_generic_townhouse_defaults_to_two_storeys_not_three() -> None:
    library = ProceduralBuildingLibrary(world_name="townhouse_storeys")
    key = library.key_for({"building": "terrace"}, 10.0, 18.0)
    assert key.family == "townhouse"
    assert key.height_m == pytest.approx(6.0)
    assert key.facade_storeys == 2

    explicit = library.key_for(
        {"building": "terrace", "building:levels": "3"}, 10.0, 18.0
    )
    assert explicit.height_m == pytest.approx(9.0)
    assert explicit.facade_storeys == 3


def test_closed_polygon_native_windows_are_storey_banded_and_nonwrapping() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    library = ProceduralBuildingLibrary(world_name="native_closed_storeys")
    library.prepare(_empty_dataset(), projection, 12.0)
    shape = (
        (10.0, 10.0), (36.0, 10.0), (36.0, 20.0),
        (24.0, 20.0), (24.0, 36.0), (10.0, 36.0),
    )
    key = library.plan_polygon(
        {"building": "house", "roof:shape": "gabled"}, shape
    ).selected
    assert key.facade_storeys == 2
    detail = _polygon_native_visual_lod(
        key,
        "windowed.paa",
        "roof.paa",
        front_texture="front.paa",
        plain_wall_texture="plain.paa",
        door_texture="door.paa",
    )
    window_faces = [
        face for face in detail.faces
        if face.texture == "windowed.paa" and len(face.vertices) == 4
    ]
    assert window_faces
    for face in window_faces:
        v_values = [vertex[3] for vertex in face.vertices]
        assert min(v_values) > 0.0
        assert max(v_values) < 1.0
        assert max(v_values) - min(v_values) < 1.0


def test_explicit_one_storey_interior_never_randomly_gets_upper_floor() -> None:
    library = ProceduralBuildingLibrary(
        world_name="one_storey_authoritative", generate_interiors=True
    )
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)
    library.prepare(_empty_dataset(), projection, 12.0)
    tags = {
        "building": "house",
        "building:levels": "1",
        # Deliberately tall enough that the old random selector could otherwise
        # decide a second interior physically fits.
        "height": "6",
    }
    for index in range(32):
        placement = library.plan_point(
            tags, 10.0, 0.0, x=100.0 + index * 7.0, z=200.0 + index * 11.0
        )
        assert placement.selected.interiors
        assert placement.selected.facade_storeys == 1
        assert not placement.selected.second_storey


def test_interior_storeys_are_capped_by_visible_facade_storeys() -> None:
    key = BuildingVariantKey(
        "residential",
        "gabled",
        10.0,
        16.0,
        6.0,
        interiors=True,
        second_storey=True,
        facade_storeys=1,
    )
    assert _interior_storey_count(key) == 1


def test_inferred_two_storey_interior_gets_real_upper_floor_windows() -> None:
    library = ProceduralBuildingLibrary(
        world_name="inferred_two_storey", generate_interiors=True
    )
    key = library.key_for({"building": "house"}, 10.0, 16.0)
    assert key.interiors
    assert key.facade_storeys == 2
    assert key.second_storey
    eave_height, _rise, _slope = _gabled_profile(key, 35.0)
    openings = _interior_window_openings(
        key, -key.width_m * 0.5, key.width_m * 0.5, eave_height
    )
    assert any(opening[2] > 2.5 for opening in openings)


def test_inferred_two_storey_facade_keeps_upper_windows_without_stairs() -> None:
    key = BuildingVariantKey(
        "residential",
        "gabled",
        10.0,
        16.0,
        6.0,
        interiors=True,
        second_storey=False,
        facade_storeys=2,
    )
    eave_height, _rise, _slope = _gabled_profile(key, 35.0)
    assert _interior_storey_count(key, wall_top=eave_height) == 1
    openings = _interior_window_openings(
        key, -key.width_m * 0.5, key.width_m * 0.5, eave_height
    )
    assert any(opening[2] >= 3.5 for opening in openings)


def test_one_storey_authoritative_facade_does_not_get_upper_windows() -> None:
    key = BuildingVariantKey(
        "residential",
        "gabled",
        10.0,
        16.0,
        6.0,
        interiors=True,
        second_storey=False,
        facade_storeys=1,
    )
    eave_height, _rise, _slope = _gabled_profile(key, 35.0)
    openings = _interior_window_openings(
        key, -key.width_m * 0.5, key.width_m * 0.5, eave_height
    )
    assert openings
    assert all(opening[2] < 3.0 for opening in openings)


def test_polygon_native_two_storey_facade_has_upper_windows_without_stairs() -> None:
    key = BuildingVariantKey(
        "residential",
        "gabled",
        10.0,
        16.0,
        6.0,
        interiors=True,
        second_storey=False,
        facade_storeys=2,
        footprint_vertices=((0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (5.0, 8.0), (5.0, 16.0), (0.0, 16.0)),
        entrance_edge=0,
    )
    eave_height, _triangles, _height_at = _polygon_native_roof_mesh(key, 35.0)
    openings = _polygon_native_edge_openings(
        key, 1, 8.0, eave_height
    )
    assert any(opening[2] >= 3.5 for opening in openings)


def test_isolated_dwelling_closed_model_keeps_door_and_ground_windows() -> None:
    library = ProceduralBuildingLibrary(world_name="isolated_facade")
    key = library.key_for(
        {"building": "yes"},
        6.0,
        8.0,
        settlement_context="isolated_dwelling_single",
    )
    assert key.isolated_dwelling
    assert key.facade_storeys == 1
    detail = _visual_lod(
        key,
        "windowed.paa",
        "roof.paa",
        35.0,
        front_texture="front.paa",
        plain_wall_texture="plain.paa",
    )
    assert any(face.texture == "front.paa" for face in detail.faces)
    assert any(face.texture == "windowed.paa" for face in detail.faces)


def test_interior_distance_style_uses_real_plain_upper_facade() -> None:
    key = BuildingVariantKey(
        "residential",
        "gabled",
        10.0,
        16.0,
        6.0,
        interiors=False,
        facade_storeys=2,
    )
    distance = _visual_lod(
        key,
        "windowed.paa",
        "roof.paa",
        35.0,
        front_texture="front.paa",
        plain_wall_texture="plain.paa",
    )
    assert any(face.texture == "plain.paa" for face in distance.faces)
    window_faces = [
        face for face in distance.faces
        if face.texture == "windowed.paa" and len(face.vertices) == 4
    ]
    assert window_faces
    for face in window_faces:
        v_values = [vertex[3] for vertex in face.vertices]
        assert min(v_values) > 0.0
        assert max(v_values) < 1.0


def test_deeply_concave_valid_polygon_gets_complete_roof_triangulation() -> None:
    # This valid footprint is intentionally awkward for ordinary Delaunay
    # triangulation: the legacy "triangulate then discard crossing triangles"
    # path leaves uncovered notches and used to return an empty mesh.
    outer = (
        (5.30, 0.72), (19.61, 3.16), (17.19, 3.17), (4.10, 0.80),
        (1.66, 7.81), (2.92, 14.70), (2.62, 14.57), (2.26, 18.72),
        (1.02, 9.54), (-4.19, 16.41), (-10.33, 9.55), (-6.59, 3.51),
        (-11.63, 4.78), (-17.31, 4.49), (-17.23, 0.45), (-10.72, -2.85),
        (-11.82, -4.38), (-1.64, -2.04), (3.33, -5.43), (13.32, -9.49),
        (8.77, -3.53), (4.81, -1.72),
    )
    triangles = _triangulate_polygon_coordinates(outer)
    assert len(triangles) == len(outer) - 2
    source_area = abs(sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(outer, outer[1:] + outer[:1])
    )) * 0.5
    triangle_area = sum(abs(sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(triangle, triangle[1:] + triangle[:1])
    )) * 0.5 for triangle in triangles)
    assert triangle_area == pytest.approx(source_area)


def test_concave_polygon_with_courtyard_gets_complete_roof_triangulation() -> None:
    outer = (
        (0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (12.0, 20.0),
        (12.0, 8.0), (8.0, 8.0), (8.0, 20.0), (0.0, 20.0),
    )
    hole = ((2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0))
    triangles = _triangulate_polygon_coordinates(outer, (hole,))
    assert triangles
    triangle_area = sum(abs(sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(triangle, triangle[1:] + triangle[:1])
    )) * 0.5 for triangle in triangles)
    assert triangle_area == pytest.approx(336.0)
