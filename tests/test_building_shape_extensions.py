from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

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
    _front_vector_for_heading,
    _visual_lod,
    decompose_footprint_rectangles,
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


def test_l_shape_plans_multiple_wings_with_common_ground_height() -> None:
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
    assert len(plans) == 2
    assert all(plan.geometry_kind == "polygon_part" for plan in plans)
    assert all(plan.procedural_placement.selected.roof_style == "hipped" for plan in plans)
    assert sorted(
        (plan.procedural_placement.selected.width_m, plan.procedural_placement.selected.length_m)
        for plan in plans
    ) == [(10.0, 24.0), (10.0, 30.0)]

    # Deliberately sloped terrain: both adjoining wings must still receive the
    # same final origin height so their walls meet cleanly at the seam.
    elevations = tuple(float((index // 4) * 2) for index in range(16))
    result = generate_world_objects(
        dataset, projection, raster, elevations, spec,
        include_roads=False, building_asset_library=library,
        building_placement_plans=plans,
    )
    buildings = result.objects[:result.building_objects]
    assert len(buildings) == 2
    assert buildings[0].y == pytest.approx(buildings[1].y)
