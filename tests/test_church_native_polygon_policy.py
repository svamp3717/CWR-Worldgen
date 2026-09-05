from __future__ import annotations

from shapely.geometry import Point

from cwr_worldgen.cache import cache_key as raw_cache_key
from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import procedural_buildings as buildings
from cwr_worldgen.church_native_polygon_policy import (
    _BUILDING_MODEL_CACHE_V57,
    _church_tower_base,
    install_church_native_polygon_policy,
)
from cwr_worldgen.church_native_tower_cache_policy import (
    install_church_native_tower_cache_policy,
)


def _library() -> buildings.ProceduralBuildingLibrary:
    library = buildings.ProceduralBuildingLibrary(
        world_name="test_church_native",
        maximum_variants=16,
        maximum_polygon_variants=16,
    )
    # plan_polygon's native path does not require capped rectangle preparation;
    # it only needs the normal geographic/style defaults supplied by key_for.
    return library


def _irregular_church_placement():
    install_church_native_polygon_policy()
    library = _library()
    # Strongly non-rectangular church outline. The old church-only exclusion
    # replaced this with a 40 x 100 m minimum-rotated rectangle.
    points = (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 12.0),
        (72.0, 12.0),
        (72.0, 24.0),
        (56.0, 24.0),
        (56.0, 40.0),
        (0.0, 40.0),
    )
    return library.plan_polygon(
        {
            "building": "church",
            "amenity": "place_of_worship",
            "religion": "christian",
        },
        points,
        road_point=(0.0, -20.0),
        allow_native_polygon=True,
    )


def test_irregular_church_uses_polygon_native_model() -> None:
    placement = _irregular_church_placement()

    assert placement.requested.family == "church"
    assert placement.selected.family == "church"
    assert placement.selected.footprint_vertices
    assert len(placement.selected.footprint_vertices) >= 6
    assert placement.selected.roof_style in {"flat", "gabled", "hipped", "pyramidal"}


def test_polygon_native_church_keeps_tower_and_spire_silhouette() -> None:
    placement = _irregular_church_placement()
    key = placement.selected
    visual = buildings._polygon_native_visual_lod(
        key,
        r"testworld\d\church_wall.paa",
        r"testworld\d\church_roof.paa",
        front_texture=r"testworld\d\church_front.paa",
        plain_wall_texture=r"testworld\d\church_plain.paa",
    )

    maximum_y = max(point[1] for point in visual.points)
    # The ordinary native shell stops at the nave roof. Christian church models
    # must retain the familiar tower/spire that rises well above the nave.
    assert maximum_y >= max(18.0, key.height_m + 8.0) + 4.5
    assert any(face.texture.endswith("church_roof.paa") for face in visual.faces)
    assert any(face.texture.endswith("church_plain.paa") for face in visual.faces)


def test_native_church_tower_stays_inside_mapped_footprint() -> None:
    placement = _irregular_church_placement()
    key = placement.selected
    shape = buildings._polygon_native_shape(key)
    base = _church_tower_base(key)

    assert base is not None
    corners, _half, _depth = base
    assert all(shape.buffer(0.021).covers(Point(x, z)) for x, z in corners)


def test_church_native_policy_bumps_nonroad_cache_revision() -> None:
    install_church_native_polygon_policy()
    assert clearance._CACHE_REVISION == "final-road-building-clearance-v5-church-native-towers"


def test_church_tower_cache_promotes_legacy_model_namespace() -> None:
    install_church_native_polygon_policy()
    install_church_native_tower_cache_policy()
    payload = {
        "world_name": "test_church_native",
        "variant": {"family": "church", "footprint_vertices": [[0.0, 0.0]]},
    }
    assert buildings.cache_key(
        "procedural-building-model-v49-robust-polygon-roof-triangulation",
        payload,
    ) == raw_cache_key(_BUILDING_MODEL_CACHE_V57, payload)


def test_church_tower_cache_does_not_force_nonchurch_models_to_v57() -> None:
    install_church_native_polygon_policy()
    install_church_native_tower_cache_policy()
    payload = {
        "world_name": "test_church_native",
        "variant": {"family": "residential"},
    }
    assert buildings.cache_key(
        "procedural-building-model-v49-robust-polygon-roof-triangulation",
        payload,
    ) != raw_cache_key(_BUILDING_MODEL_CACHE_V57, payload)
