"""Regression tests for the indexed generated-junction seam stitch pass."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import paved_junction_fallback_policy as fallback
from cwr_worldgen import paved_junction_policy as paved
from cwr_worldgen import procedural_infrastructure as infrastructure


_RADIUS = (
    infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
    + infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
)


def _road(object_id: int, model: str, x: float, z: float, heading: float = 0.0):
    return SimpleNamespace(
        object_id=object_id,
        model_path=model,
        x=x,
        y=0.0,
        z=z,
        heading_degrees=heading,
        pitch_degrees=0.0,
    )


def _plan(point: tuple[float, float], *, angle: float = 0.0):
    connector = paved._Connector(
        "sil", (point[0], point[1] + _RADIUS), (0.0, 1.0)
    )
    return paved._Plan(
        r"seam_world\i\paved_j3_m091_b091_a265.p3d",
        point,
        (0.0, 1.0),
        (paved._Arm("sil", paved._direction(angle), connector),),
    )


def _spec():
    return SimpleNamespace(
        name="seam_world",
        road_segment_length=25.0,
        cells=16,
        cell_size=25.0,
        custom_road_shapes=False,
    )


@dataclass(frozen=True)
class _Report:
    objects: tuple
    junction_cap_objects: int = 0


def _report(objects, cap_count: int = 0):
    return _Report(tuple(objects), cap_count)


def test_indexed_stitch_matches_unpruned_original_order_and_seam_geometry() -> None:
    first = _plan((0.0, 0.0), angle=10.7)
    second = _plan((1000.0, 1000.0))
    plans = {(0, 0): first, (100, 100): second}
    first_cap = _road(1, first.model_path, 0.0, 0.0)
    second_cap = _road(2, second.model_path, 1000.0, 1000.0)
    first_approach = _road(
        3,
        r"seam_world\i\paved_w091_l0081.p3d",
        math.sin(math.radians(10.7)) * 4.05,
        _RADIUS + math.cos(math.radians(10.7)) * 4.05,
        10.7,
    )
    # The second connector is already exactly aligned and must be left alone.
    second_approach = _road(
        4, r"seam_world\i\paved_w091_l0062.p3d",
        1000.0, 1000.0 + _RADIUS + 3.1,
    )
    far_roads = (
        _road(i + 10, r"o\road\sil25.p3d", 20_000 + i * 2.0, 20_000)
        for i in range(2400)
    )
    report = _report(
        (first_cap, second_cap, first_approach, second_approach, *far_roads),
        cap_count=2,
    )
    spec = _spec()
    progress = []
    indexed = fallback._stitch_generated_hub_approaches(
        report, plans, (0.0,) * 256, spec,
        progress_callback=lambda _percent, message: progress.append(message),
    )
    with patch.object(
        fallback, "_stitch_candidate_indices",
        lambda _index, _plan: tuple(range(2, len(report.objects))),
    ):
        full_scan = fallback._stitch_generated_hub_approaches(
            report, plans, (0.0,) * 256, spec,
        )

    assert indexed.objects == full_scan.objects
    assert indexed.objects[0] is first_cap
    assert indexed.objects[1] is second_cap
    assert any("Indexing generated paved-junction seams" in text for text in progress)
    assert any("Stitching generated paved junctions" in text for text in progress)
    assert any("Generated paved-junction stitching complete" in text for text in progress)


def test_long_axis_crossing_hub_is_not_lost_by_midpoint_indexing() -> None:
    # A 500 m road has its centre hundreds of metres away but its segment
    # crosses the hub seam. Midpoint-only buckets would silently miss it.
    plan = _plan((0.0, 0.0))
    long_piece = _road(
        10, r"seam_world\i\paved_w091_l5000.p3d", 0.0, 252.0,
    )
    report = _report((long_piece,))
    index = fallback._build_stitch_road_index(report, _spec())
    assert fallback._stitch_candidate_indices(index, plan) == (0,)


def test_oversized_axis_fallback_stays_searchable() -> None:
    plan = _plan((0.0, 0.0))
    piece = _road(
        10, r"seam_world\i\paved_w091_l5000.p3d", 0.0, 252.0,
    )
    with patch.object(fallback, "_STITCH_MAX_BUCKETS_PER_ROAD", 1):
        index = fallback._build_stitch_road_index(
            _report((piece,)), _spec(),
        )
    assert index.oversized == (0,)
    assert fallback._stitch_candidate_indices(index, plan) == (0,)


def test_large_map_queries_only_visit_nearby_roads() -> None:
    plan = _plan((0.0, 0.0))
    nearby = _road(2, r"o\road\sil6.p3d", 0.0, _RADIUS + 3.1)
    distant = [
        _road(i + 100, r"o\road\sil25.p3d", 50_000 + 20 * i, 40_000)
        for i in range(2500)
    ]
    report = _report((nearby, *distant))
    spec = _spec()
    real_axis = fallback._generated_paved_axis
    with patch.object(
        fallback, "_generated_paved_axis", wraps=real_axis,
    ) as axis:
        index = fallback._build_stitch_road_index(report, spec)
        nearby_indices = fallback._stitch_candidate_indices(index, plan)
        for i in nearby_indices:
            fallback._generated_paved_axis(report.objects[i], spec)

    assert nearby_indices == (0,)
    # Each road is indexed once, with only nearby roads revisited per plan.
    assert axis.call_count == len(report.objects) + len(nearby_indices)
