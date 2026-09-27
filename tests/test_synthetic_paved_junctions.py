from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import paved_junction_policy as paved
from cwr_worldgen import playability
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


_BBOX = (0.0, 0.0, 1.0, 1.0)


def _projection() -> BboxProjection:
    return BboxProjection.create(_BBOX, 1000.0)


def _road(
    projection: BboxProjection,
    key: str,
    points: tuple[tuple[float, float], ...],
    **tags: str,
) -> OsmLineFeature:
    values = {"highway": "residential"}
    values.update(tags)
    return OsmLineFeature(
        key,
        values,
        tuple(projection.to_latlon(point) for point in points),
    )


def _dataset(*roads: OsmLineFeature) -> OsmDataset:
    return OsmDataset(
        source_generator="synthetic-paved-junction-test",
        element_count=len(roads),
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=tuple(roads),
    )


def _light_spec():
    return SimpleNamespace(include_minor_roads=True)


def _junction_spec() -> _Milestone9PlayabilitySpec:
    return _Milestone9PlayabilitySpec(
        name="synthetic_junction",
        heightmap_path=Path("unused.png"),
        bbox=_BBOX,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )


def _has_point(
    points: tuple[tuple[float, float], ...],
    expected: tuple[float, float],
    *,
    tolerance: float = 0.05,
) -> bool:
    return any(
        abs(point[0] - expected[0]) <= tolerance
        and abs(point[1] - expected[1]) <= tolerance
        for point in points
    )


def test_unrepresentable_short_paved_corner_is_collapsed() -> None:
    pieces = playability.road_model_variants(
        r"o\road\sil25.p3d",
        24.5,
    )
    source = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))

    repaired = playability._representable_road_run(
        source,
        pieces,
    )

    assert repaired == (source[0], source[-1])


def test_impossible_paved_hairpin_collapses_instead_of_self_intersecting() -> None:
    pieces = playability.road_model_variants(
        r"o\road\sil25.p3d",
        24.5,
    )
    source = ((0.0, 0.0), (4.0, 0.0), (0.10, 0.0))

    repaired = playability._representable_road_run(
        source,
        pieces,
    )

    assert repaired == (source[0], source[-1])


def test_junction_guard_preserves_historical_paved_rounding() -> None:
    pieces = playability.road_model_variants(
        r"o\road\sil25.p3d",
        24.5,
    )
    source = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))

    historical = playability._rounded_road_run(source)
    repaired = playability._representable_road_run(
        source,
        pieces,
        preserve_start_metres=87.0,
    )

    assert repaired == historical


def test_moderately_sharp_paved_turn_gets_bounded_representable_fillet() -> None:
    pieces = playability.road_model_variants(
        r"o\road\sil25.p3d",
        24.5,
    )
    source = ((0.0, 0.0), (50.0, 0.0), (41.3176, 49.2404))

    repaired = playability._representable_road_run(
        source,
        pieces,
    )

    assert repaired[0] == source[0]
    assert repaired[-1] == source[-1]
    assert len(repaired) > 5
    turns = tuple(
        playability._turn_degrees(
            repaired[index - 1],
            repaired[index],
            repaired[index + 1],
        )
        for index in range(1, len(repaired) - 1)
    )
    assert max(turns, default=0.0) <= 30.0


def test_very_sharp_paved_turn_is_simplified_instead_of_long_fillet() -> None:
    pieces = playability.road_model_variants(
        r"o\road\sil25.p3d",
        24.5,
    )
    source = ((0.0, 0.0), (50.0, 0.0), (25.0, 43.3013))

    repaired = playability._representable_road_run(
        source,
        pieces,
    )

    assert repaired == (source[0], source[-1])


def test_dirt_run_keeps_existing_rounding_behavior() -> None:
    pieces = playability.road_model_variants(
        r"o\road\ces25.p3d",
        24.5,
    )
    source = ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0))

    repaired = playability._representable_road_run(
        source,
        pieces,
    )

    assert repaired == playability._rounded_road_run(source)


def test_mixed_paved_dirt_node_keeps_only_paved_cap_incidents() -> None:
    values = (
        ((0.0, 1.0), False, r"o\road\sil25.p3d", "paved/north", "paved"),
        ((0.0, -1.0), False, r"o\road\sil25.p3d", "paved/south", "paved"),
        ((1.0, 0.0), True, r"o\road\ces25.p3d", "dirt/east", "dirt"),
    )

    actual = playability._junction_cap_incidents(values)

    assert actual == values[:2]


def test_pure_dirt_node_keeps_dirt_junction_topology() -> None:
    values = (
        ((0.0, 1.0), True, r"o\road\ces25.p3d", "dirt/north", "dirt-a"),
        ((0.0, -1.0), True, r"o\road\ces25.p3d", "dirt/south", "dirt-a"),
        ((1.0, 0.0), True, r"o\road\ces25.p3d", "dirt/east", "dirt-b"),
    )

    assert playability._junction_cap_incidents(values) == values


def test_dirt_road_surface_is_always_below_paved_surface() -> None:
    paved_offset = playability._road_vertical_offset(
        {"highway": "residential", "surface": "asphalt"}
    )
    dirt_offset = playability._road_vertical_offset(
        {"highway": "track", "surface": "dirt"}
    )
    gravel_offset = playability._road_vertical_offset(
        {"highway": "track", "surface": "gravel"}
    )

    assert dirt_offset < paved_offset
    assert gravel_offset < paved_offset
    assert playability._STOCK_DIRT_VERTICAL_OFFSET_METRES < (
        playability._STOCK_PAVED_JUNCTION_VERTICAL_OFFSET_METRES
    )


def test_geometric_paved_crossing_is_promoted_to_x_junction() -> None:
    projection = _projection()
    centre = (500.0, 500.0)
    dataset = _dataset(
        _road(
            projection,
            "way/east-west",
            ((200.0, 500.0), (800.0, 500.0)),
        ),
        _road(
            projection,
            "way/north-south",
            ((500.0, 200.0), (500.0, 800.0)),
        ),
    )

    projected = playability._paved_junction_augmented_polylines(
        dataset,
        projection,
        _light_spec(),
    )

    assert _has_point(projected[0], centre)
    assert _has_point(projected[1], centre)

    plans = paved._plans(dataset, projection, _junction_spec())
    key = playability._road_node_key(centre)
    assert key in plans
    assert len(plans[key].arms) == 4


def test_short_paved_endpoint_miss_is_snapped_to_t_junction() -> None:
    projection = _projection()
    centre = (500.0, 500.0)
    dataset = _dataset(
        _road(
            projection,
            "way/main",
            ((200.0, 500.0), (800.0, 500.0)),
        ),
        _road(
            projection,
            "way/branch",
            ((500.0, 200.0), (500.0, 498.5)),
        ),
    )

    projected = playability._paved_junction_augmented_polylines(
        dataset,
        projection,
        _light_spec(),
    )

    assert _has_point(projected[0], centre)
    assert _has_point(projected[1], centre)

    plans = paved._plans(dataset, projection, _junction_spec())
    key = playability._road_node_key(centre)
    assert key in plans
    assert len(plans[key].arms) == 3


def test_nearby_paved_endpoints_snap_to_one_shared_node() -> None:
    projection = _projection()
    shared = (500.0, 500.0)
    dataset = _dataset(
        _road(
            projection,
            "way/west",
            ((200.0, 500.0), (499.0, 500.0)),
        ),
        _road(
            projection,
            "way/north",
            ((501.0, 500.0), (501.0, 800.0)),
        ),
    )

    projected = playability._paved_junction_augmented_polylines(
        dataset,
        projection,
        _light_spec(),
    )

    assert _has_point(projected[0], shared)
    assert _has_point(projected[1], shared)
    assert playability._road_node_key(projected[0][-1]) == (
        playability._road_node_key(projected[1][0])
    )


def test_generated_fallback_supports_four_way_paved_junctions() -> None:
    plan = paved._generated_plan(
        (100.0, 100.0),
        (
            ((0.0, 1.0), "sil"),
            ((1.0, 0.0), "sil"),
            ((0.0, -1.0), "sil"),
            ((-1.0, 0.0), "sil"),
        ),
        world_name="synthetic_junction",
    )

    assert plan is not None
    assert len(plan.arms) == 4
    assert playability.is_generated_paved_junction_model(plan.model_path)


def test_bridge_crossing_is_not_promoted_to_at_grade_junction() -> None:
    projection = _projection()
    centre = (500.0, 500.0)
    dataset = _dataset(
        _road(
            projection,
            "way/main",
            ((200.0, 500.0), (800.0, 500.0)),
        ),
        _road(
            projection,
            "way/bridge",
            ((500.0, 200.0), (500.0, 800.0)),
            bridge="yes",
            layer="1",
        ),
    )

    projected = playability._paved_junction_augmented_polylines(
        dataset,
        projection,
        _light_spec(),
    )

    assert not _has_point(projected[0][1:-1], centre)
    assert not _has_point(projected[1][1:-1], centre)


def test_dirt_crossing_stays_out_of_synthetic_paved_junctions() -> None:
    projection = _projection()
    centre = (500.0, 500.0)
    dataset = _dataset(
        _road(
            projection,
            "way/paved",
            ((200.0, 500.0), (800.0, 500.0)),
        ),
        _road(
            projection,
            "way/dirt",
            ((500.0, 200.0), (500.0, 800.0)),
            surface="dirt",
        ),
    )

    projected = playability._paved_junction_augmented_polylines(
        dataset,
        projection,
        _light_spec(),
    )

    assert not _has_point(projected[0][1:-1], centre)
    assert not _has_point(projected[1][1:-1], centre)
