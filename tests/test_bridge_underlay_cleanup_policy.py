from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_underlay_cleanup_policy as cleanup
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport


def _report(*objects: WorldObject) -> RoadFitReport:
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )


def test_tinybjorsund_submerged_road_chain_is_removed_beneath_bridge() -> None:
    # Regression geometry from wg_a_tinybjorsund.pbo. The generated bridge is
    # centred at about (738.45, 765.27), heading 56.11 degrees and spans roughly
    # 321 m. The ordinary sil road chain was still emitted underneath it.
    span = cleanup._BridgeSpan(
        points=((605.08, 675.70), (871.81, 854.84)),
        road_width=7.0,
    )
    underlay = (
        WorldObject(1070, r"o\road\sil25.p3d", 689.4, -3.02, 732.5, 56.11),
        WorldObject(1071, r"o\road\sil25.p3d", 709.8, -4.93, 746.2, 56.11),
        WorldObject(1072, r"o\road\sil25.p3d", 730.1, -4.97, 759.8, 56.11),
        WorldObject(1073, r"o\road\sil25.p3d", 750.4, -4.67, 773.5, 56.11),
        WorldObject(1074, r"o\road\sil12.p3d", 765.5, -3.97, 783.6, 56.11),
        WorldObject(1075, r"o\road\sil12.p3d", 775.3, -3.05, 790.2, 56.11),
        WorldObject(1076, r"o\road\sil12.p3d", 785.0, -1.89, 796.7, 56.11),
        WorldObject(1077, r"o\road\sil12.p3d", 794.8, -0.74, 803.3, 56.11),
    )
    crossing = WorldObject(
        2000, r"o\road\sil25.p3d", 738.4, 0.0, 765.3, 146.11
    )
    bridge = WorldObject(
        3319,
        r"wg_a_tinybjorsund\i\br_single_w084_l3213.p3d",
        738.4467,
        8.1179,
        765.2676,
        56.1145,
    )

    cleaned, removed = cleanup._remove_bridge_underlays(
        _report(*underlay, crossing, bridge), (span,)
    )

    assert removed == len(underlay)
    assert {obj.object_id for obj in cleaned.objects} == {2000, 3319}


def test_cleanup_keeps_parallel_road_outside_narrow_bridge_corridor() -> None:
    span = cleanup._BridgeSpan(points=((0.0, 0.0), (100.0, 0.0)), road_width=6.0)
    underlay = WorldObject(1, r"o\road\sil25.p3d", 50.0, 0.0, 0.0, 90.0)
    parallel = WorldObject(2, r"o\road\sil25.p3d", 50.0, 0.0, 4.0, 90.0)

    cleaned, removed = cleanup._remove_bridge_underlays(
        _report(underlay, parallel), (span,)
    )

    assert removed == 1
    assert tuple(obj.object_id for obj in cleaned.objects) == (2,)


def test_procedural_bridge_candidates_are_cleaned_not_only_stock_mode() -> None:
    feature = SimpleNamespace(tags={"highway": "primary", "bridge": "yes"})
    dataset = SimpleNamespace(roads=(feature,))
    spec = SimpleNamespace(
        bridges_enabled=True,
        maximum_bridge_objects=1000,
        advisory_object_limits=True,
        procedural_bridges=True,
        bridge_module_length=30.0,
        cells=4,
        cell_size=50.0,
        sea_level=0.0,
    )
    points = ((10.0, 50.0), (190.0, 50.0))

    with patch.object(cleanup._osm, "projected_road_polylines", return_value=(points,)), patch.object(
        cleanup._osm, "road_bridge_crosses_ditch_only", return_value=False
    ), patch.object(cleanup, "_feature_needs_bridge", return_value=True):
        spans = cleanup._bridge_spans(dataset, None, [0.0] * 16, spec)

    assert len(spans) == 1
    assert spans[0].points == points
