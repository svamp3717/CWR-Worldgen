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
