from types import SimpleNamespace

from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen.road_quality_policy import _Context, _Junction
from cwr_worldgen import road_audit_performance_policy as audit_policy


def _report(*objects: WorldObject) -> RoadFitReport:
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=4,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )


def _context(junction: _Junction):
    return _Context(
        [0.0],
        SimpleNamespace(
            road_segment_length=25.0,
            road_connection_tolerance=0.35,
        ),
        {(0, 0): junction},
    )


def test_four_way_audit_reuses_each_candidate_distance_across_arms(monkeypatch) -> None:
    junction = _Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=3.0,
        half_width=3.0,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)),
    )
    vertical = WorldObject(
        1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, heading_degrees=0.0
    )
    horizontal = WorldObject(
        2, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, heading_degrees=90.0
    )

    calls = 0
    original = audit_policy._audit_axis_distance

    def counted(point_x, point_z, axis):
        nonlocal calls
        calls += 1
        return original(point_x, point_z, axis)

    monkeypatch.setattr(audit_policy, "_audit_axis_distance", counted)
    audited = audit_policy.audit_road_junctions(
        _report(vertical, horizontal),
        _context(junction),
    )

    assert audited.failed_connections == 0
    # The legacy arm-by-arm loop measured each local axis for both opposite
    # directions. The shared-candidate pass measures each axis only once.
    assert calls == 2


def test_audit_reports_index_and_junction_progress() -> None:
    junction = _Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=3.0,
        half_width=3.0,
        directions=((0.0, 1.0),),
    )
    road = WorldObject(
        1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, heading_degrees=0.0
    )
    events: list[tuple[int, str]] = []

    audited = audit_policy.audit_road_junctions(
        _report(road),
        _context(junction),
        progress_callback=lambda value, message: events.append((value, message)),
    )

    assert audited.failed_connections == 0
    assert events
    assert all(value == 99 for value, _message in events)
    assert any("indexing road pieces" in message for _value, message in events)
    assert any(
        "Auditing road junctions (1/1, 100%" in message
        for _value, message in events
    )


def test_far_road_axes_are_not_materialized_for_a_local_junction(monkeypatch) -> None:
    junction = _Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=3.0,
        half_width=3.0,
        directions=((0.0, 1.0),),
    )
    local = WorldObject(
        1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, heading_degrees=0.0
    )
    far = tuple(
        WorldObject(
            index + 2,
            r"o\road\sil25.p3d",
            1000.0 + (index % 100) * 25.0,
            0.0,
            1000.0 + (index // 100) * 25.0,
            heading_degrees=float(index % 360),
        )
        for index in range(2000)
    )

    constructed = 0
    original_axis = audit_policy._p._model_axis

    def counted_axis(*args, **kwargs):
        nonlocal constructed
        constructed += 1
        return original_axis(*args, **kwargs)

    monkeypatch.setattr(audit_policy._p, "_model_axis", counted_axis)
    audited = audit_policy.audit_road_junctions(
        _report(local, *far),
        _context(junction),
    )

    assert audited.failed_connections == 0
    assert constructed < 20
