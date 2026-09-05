from __future__ import annotations

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import procedural_buildings as buildings
from cwr_worldgen.church_native_polygon_policy import (
    install_church_native_polygon_policy,
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


def test_irregular_church_uses_polygon_native_model() -> None:
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
    placement = library.plan_polygon(
        {
            "building": "church",
            "amenity": "place_of_worship",
            "religion": "christian",
        },
        points,
        allow_native_polygon=True,
    )

    assert placement.requested.family == "church"
    assert placement.selected.family == "church"
    assert placement.selected.footprint_vertices
    assert len(placement.selected.footprint_vertices) >= 6
    assert placement.selected.roof_style in {"flat", "gabled", "hipped", "pyramidal"}


def test_church_native_policy_bumps_nonroad_cache_revision() -> None:
    install_church_native_polygon_policy()
    assert clearance._CACHE_REVISION == "final-road-building-clearance-v4-church-native-polygons"
