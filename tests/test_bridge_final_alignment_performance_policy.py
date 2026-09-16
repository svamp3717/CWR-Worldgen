import math

from cwr_worldgen import bridge_final_alignment_performance_policy as perf
from cwr_worldgen import bridge_final_alignment_policy as historical
from cwr_worldgen.model import WorldObject


def _road(
    object_id: int,
    *,
    x: float = 0.0,
    y: float = 10.0,
    z: float = 0.0,
    heading: float = 0.0,
    pitch: float = 0.0,
    model: str = r"o\road\sil6.p3d",
) -> WorldObject:
    return WorldObject(object_id, model, x, y, z, heading, pitch)


def _result_signature(result):
    if result is None:
        return None
    obj, near, gap = result
    return int(obj.object_id), tuple(float(value) for value in near), float(gap)


def test_indexed_candidate_matches_historical_connected_priority() -> None:
    # The isolated road is closer to the bridge, but the farther candidate is
    # connected to an outward continuation. The historical sort intentionally
    # prefers that connected chain before comparing the bridge gap.
    isolated = _road(1, x=1.0, z=6.0)
    connected = _road(2, z=8.0)
    continuation = _road(3, z=14.0)
    roads = (isolated, connected, continuation)
    bridge_point = (0.0, 0.0)
    bridge_heading = 0.0

    expected = historical._approach_candidate(
        bridge_point, bridge_heading, roads
    )
    actual = perf._indexed_approach_candidate(
        bridge_point,
        bridge_heading,
        perf._RoadCentreIndex.build(roads),
    )

    assert _result_signature(actual) == _result_signature(expected)
    assert actual is not None
    assert actual[0].object_id == connected.object_id


def test_indexed_neighbour_preserves_exact_three_dimensional_join() -> None:
    candidate = _road(1, z=8.0, y=10.0)
    # X/Z endpoints touch exactly, but the second slab is two metres higher. The
    # original predicate uses 3D math.dist and therefore must not count this as a
    # continuing road chain.
    high_continuation = _road(2, z=14.0, y=12.0)
    roads = (candidate, high_continuation)
    index = perf._RoadCentreIndex.build(roads)

    expected = historical._road_has_outward_neighbour(
        candidate, (0.0, 0.0), roads
    )
    actual = perf._indexed_road_has_outward_neighbour(
        candidate, (0.0, 0.0), index
    )

    assert expected is False
    assert actual is expected


def test_indexed_neighbour_matches_join_across_spatial_bucket_boundary() -> None:
    # Put the shared endpoint exactly around the 32 m bucket boundary so the
    # broad phase must inspect adjacent centre buckets without changing the exact
    # 0.75 m acceptance rule.
    candidate = _road(1, z=29.0)
    continuation = _road(2, z=35.0)
    roads = (candidate, continuation)
    index = perf._RoadCentreIndex.build(roads)

    expected = historical._road_has_outward_neighbour(
        candidate, (0.0, 20.0), roads
    )
    actual = perf._indexed_road_has_outward_neighbour(
        candidate, (0.0, 20.0), index
    )

    assert expected is True
    assert actual is expected


def test_indexed_candidate_does_not_scan_distant_road_endpoints(monkeypatch) -> None:
    local = _road(1, z=8.0)
    continuation = _road(2, z=14.0)
    distant = tuple(
        _road(index + 10, x=10_000.0 + index * 40.0, z=10_000.0)
        for index in range(2_000)
    )
    roads = (local, continuation, *distant)
    index = perf._RoadCentreIndex.build(roads)

    calls = 0
    original = historical._road_endpoints

    def counted(obj):
        nonlocal calls
        calls += 1
        return original(obj)

    monkeypatch.setattr(historical, "_road_endpoints", counted)
    result = perf._indexed_approach_candidate((0.0, 0.0), 0.0, index)

    assert result is not None
    assert result[0].object_id == local.object_id
    # Only the tiny local bucket neighborhood should reach exact endpoint work.
    assert calls < 20


def test_runtime_uses_indexed_bridge_approach_filler() -> None:
    assert historical._add_bridge_approach_fillers is perf._fast_add_bridge_approach_fillers
