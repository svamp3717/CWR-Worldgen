from types import SimpleNamespace

import cwr_worldgen.playability as playability
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen.road_quality_policy import _Context, _Junction, _audit


def test_junction_audit_uses_nearby_spatial_candidates(monkeypatch) -> None:
    junction = _Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=3.0,
        half_width=3.0,
        directions=((0.0, 1.0),),
    )
    local = WorldObject(
        1, r"o\road\sil25.p3d", 0.0, 0.0, 15.0, heading_degrees=0.0
    )
    far = tuple(
        WorldObject(
            index + 2,
            r"o\road\sil25.p3d",
            1000.0 + (index % 100) * 25.0,
            0.0,
            1000.0 + (index // 100) * 25.0,
            heading_degrees=0.0,
        )
        for index in range(2000)
    )
    report = RoadFitReport(
        objects=(local, *far),
        chain_count=1,
        connection_count=1,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )
    context = _Context(
        [0.0],
        SimpleNamespace(road_segment_length=25.0, road_connection_tolerance=0.35),
        {(0, 0): junction},
    )

    calls = 0
    original = playability._point_segment_distance

    def counted(point, start, end):
        nonlocal calls
        calls += 1
        return original(point, start, end)

    monkeypatch.setattr(playability, "_point_segment_distance", counted)
    audited = _audit(report, context)

    assert audited.failed_connections == 0
    assert audited.maximum_connection_gap <= 0.35
    # The old implementation called this once for every emitted road object.
    # The spatial audit should only touch axes in buckets near this junction.
    assert calls < 20
