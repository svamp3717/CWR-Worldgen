from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import osm
from cwr_worldgen import procedural_buildings as buildings
from cwr_worldgen.church_native_polygon_policy import (
    _mapped_church_support_polygon,
    install_church_native_polygon_policy,
)


def _library() -> buildings.ProceduralBuildingLibrary:
    library = buildings.ProceduralBuildingLibrary(
        world_name="test_legacy_church",
        maximum_variants=16,
        maximum_polygon_variants=16,
    )
    # The legacy rectangle fallback calls _selected(), which normally becomes
    # prepared during the real build's prepare() phase. Empty mapping means an
    # isolated unit test may use its exact requested church key.
    library._prepared = True
    return library


def _legacy_irregular_church_placement():
    install_church_native_polygon_policy()
    library = _library()
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


def test_irregular_church_uses_legacy_rectangular_renderer() -> None:
    placement = _legacy_irregular_church_placement()

    assert placement.requested.family == "church"
    assert placement.selected.family == "church"
    # This is the deliberate old church path. Exact OSM geometry is now used
    # only for road/collision policy, not as a replacement church renderer.
    assert placement.selected.footprint_vertices == ()
    assert placement.selected.footprint_holes == ()


def test_legacy_church_renderer_keeps_end_tower_and_spire() -> None:
    placement = _legacy_irregular_church_placement()
    key = placement.selected
    visual = buildings._visual_lod(
        key,
        r"testworld\d\church_wall.paa",
        r"testworld\d\church_roof.paa",
        front_texture=r"testworld\d\church_front.paa",
        plain_wall_texture=r"testworld\d\church_plain.paa",
    )

    maximum_y = max(point[1] for point in visual.points)
    # The mature rectangular renderer integrates its church tower at the front
    # end of the nave and extends the spire substantially above the nave roof.
    assert maximum_y >= max(18.0, key.height_m + 8.0) + 4.5
    assert any(face.texture.endswith("church_roof.paa") for face in visual.faces)
    assert any(face.texture.endswith("church_plain.paa") for face in visual.faces)


def test_mapped_church_road_support_uses_source_outline_and_follows_nudge() -> None:
    plan = osm.BuildingPlacementPlan(
        osm_key="way/123",
        geometry_index=0,
        geometry_kind="polygon",
        x=6.0,
        z=3.0,
        heading_degrees=0.0,
        model_path=r"testworld\g\church.p3d",
        # Deliberately oversized fitted rectangle, similar to the cathedral bug.
        support_polygon=((-4.0, -7.0), (16.0, -7.0), (16.0, 13.0), (-4.0, 13.0)),
        building_family="church",
    )
    feature = SimpleNamespace(
        osm_key="way/123",
        tags={
            "building": "church",
            "amenity": "place_of_worship",
            "religion": "christian",
        },
        polygons=(SimpleNamespace(
            outer=((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0), (0.0, 0.0)),
        ),),
    )
    dataset = SimpleNamespace(building_polygons=(feature,))
    projection = SimpleNamespace(to_world=lambda point: point)

    support = _mapped_church_support_polygon(plan, dataset, projection)

    # Source centroid is (5,2), while the current model origin is (6,3). The
    # support therefore receives the same +1,+1 rigid translation as the model.
    assert support == ((1.0, 1.0), (11.0, 1.0), (11.0, 5.0), (1.0, 5.0))


def test_nonchurch_does_not_receive_church_source_support_override() -> None:
    plan = osm.BuildingPlacementPlan(
        osm_key="way/124",
        geometry_index=0,
        geometry_kind="polygon",
        x=5.0,
        z=2.0,
        heading_degrees=0.0,
        model_path=r"testworld\g\house.p3d",
        support_polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)),
        building_family="residential",
    )
    dataset = SimpleNamespace(building_polygons=())
    projection = SimpleNamespace(to_world=lambda point: point)
    assert _mapped_church_support_polygon(plan, dataset, projection) is None


def test_legacy_church_policy_bumps_nonroad_cache_revision() -> None:
    install_church_native_polygon_policy()
    assert clearance._CACHE_REVISION == (
        "final-road-building-clearance-v6-legacy-church-source-road-footprints"
    )
