"""Real final-pass gravel-vs-paved road crossing regressions.

The stock asf texture was visibly drawn *over* sil on flat crossings even
after a vertical-offset change, because final road dedup considered asf to
be paved and therefore never clipped it at the asphalt footprint.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen.final_road_dedup_policy import (
    _is_paved_family,
    _surface_priority,
    _road_axis,
    _surface_polygon,
    _clip_convex_polygon,
    _polygon_area,
    deduplicate_final_road_objects,
)
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen.procedural_infrastructure import (
    GENERATED_GRAVEL_VISUAL_TOP_METRES,
    custom_road_model_path, custom_road_model_signature,
)


def _report(objects, *, caps=0):
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
        junction_cap_objects=caps,
    )


def _road(oid, path, x, z, *, heading=0.0, y=0.0):
    return WorldObject(oid, path, x, y, z, heading, 0.0)


def _spec(*, custom=False):
    return SimpleNamespace(
        road_segment_length=25.0, custom_road_shapes=custom,
        name="gravel_crossing",
    )


def _gravel(objects):
    return tuple(
        obj for obj in objects
        if obj.model_path.replace("/", "\\").casefold().split("\\")[-1]
        in {"asf25.p3d", "asf12.p3d", "asf6.p3d"}
    )


def _assert_no_gravel_footprint_over_paved(objects, spec):
    gravel = []
    paved = []
    for index, obj in enumerate(objects):
        axis = _road_axis(obj, index, spec)
        if axis is None:
            continue
        if axis.family in {"asf", "gravel"}:
            gravel.append(_surface_polygon(axis))
        if axis.family in {"sil", "kos", "paved"}:
            paved.append(_surface_polygon(axis))
    for lower in gravel:
        for upper in paved:
            assert _polygon_area(
                _clip_convex_polygon(lower, upper)
            ) < 1.0e-5, "stock gravel extends into the visible paved footprint"


def test_stock_asf_is_gravel_not_paved_in_final_pass():
    assert not _is_paved_family("asf")
    assert _is_paved_family("sil")
    assert _surface_priority("ces") < _surface_priority("asf") < _surface_priority("sil")


def test_perpendicular_stock_gravel_is_trimmed_below_stock_paved():
    # Same geometry as the in-game screenshot: 25-m stock asf crossing sil.
    original = _road(1, r"o\road\asf25.p3d", 100, 100, heading=90.0, y=0.035)
    paved = _road(2, r"o\road\sil25.p3d", 100, 100, heading=0.0, y=0.095)
    result = deduplicate_final_road_objects(_report((original, paved)), _spec())
    assert any(obj.object_id == 2 for obj in result.objects)
    assert original not in result.objects
    gravel = _gravel(result.objects)
    assert gravel, "the gravel approaches must remain on both sides"
    assert len(gravel) >= 2
    assert all(obj.pitch_degrees == 0.0 for obj in gravel)
    # CWA can show stock P3D shoulders through pavement even when the object's
    # centre origin is lower. Reject gravel geometry under the paved footprint.
    _assert_no_gravel_footprint_over_paved(result.objects, _spec())
    # Stock 6/12/25 parts cannot end at an arbitrary asphalt boundary.
    # Exact-length generated gravel connectors fill the remainder, using the
    # generated P3D's 25 mm local visual lift rather than copying donor Y.
    fillers = [
        obj for obj in result.objects
        if (sig := custom_road_model_signature(obj.model_path)) is not None
        and sig[0] == "gravel"
    ]
    assert len(fillers) >= 2
    assert all(
        obj.y + GENERATED_GRAVEL_VISUAL_TOP_METRES == pytest.approx(original.y)
        for obj in fillers
    )
    assert all(
        0.5 <= custom_road_model_signature(obj.model_path)[2] < 6.25
        for obj in fillers
    )
    assert all(obj.model_path.casefold().endswith(
        ("asf6.p3d", "asf12.p3d", "asf25.p3d")
    ) for obj in gravel)
    # None of the objects crossing the paved centre may be the original asf25.
    assert all(obj.object_id != original.object_id or obj != original for obj in gravel)


def test_generated_connectors_reach_the_paved_boundary_without_cutoff():
    """The stock pieces alone leave a ~1.7 m approach gap on each side."""
    stock = _road(1, r"o\road\asf25.p3d", 100, 100, heading=90.0, y=0.035)
    asphalt = _road(2, r"o\road\sil25.p3d", 100, 100, y=0.095)
    result = deduplicate_final_road_objects(_report((stock, asphalt)), _spec())
    edge_left, edge_right = 100 - 4.55, 100 + 4.55
    gravel_ranges = []
    for index, obj in enumerate(result.objects):
        axis = _road_axis(obj, index, _spec())
        if axis is not None and axis.family in {"asf", "gravel"}:
            gravel_ranges.append((min(axis.start[0], axis.end[0]),
                                  max(axis.start[0], axis.end[0])))
    assert gravel_ranges
    left_edge = max(hi for lo, hi in gravel_ranges if hi <= edge_left + 1.0e-5)
    right_edge = min(lo for lo, hi in gravel_ranges if lo >= edge_right - 1.0e-5)
    # Named generated 0.1m models leave < 0.1m nominal gaps, covered by
    # their deliberately lowered 0.9m visual overlap tips. Stock donor-only
    # truncation previously left > 1m missing at the junction.
    assert 0.0 <= edge_left - left_edge <= 0.11
    assert 0.0 <= right_edge - edge_right <= 0.11
    _assert_no_gravel_footprint_over_paved(result.objects, _spec())


def test_two_stock_gravel_approaches_survive_paved_crossing():
    reports = (
        _road(1, r"o\road\asf25.p3d", 90, 100, heading=90.0, y=0.035),
        _road(2, r"o\road\asf25.p3d", 110, 100, heading=90.0, y=0.035),
        _road(3, r"o\road\sil25.p3d", 100, 100, heading=0.0, y=0.095),
    )
    actual = deduplicate_final_road_objects(_report(reports), _spec())
    assert any(obj.object_id == 3 for obj in actual.objects)
    assert any(obj.x < 100 for obj in _gravel(actual.objects))
    assert any(obj.x > 100 for obj in _gravel(actual.objects))
    _assert_no_gravel_footprint_over_paved(actual.objects, _spec())


def test_paved_overpass_does_not_remove_stock_gravel_underneath():
    gravel = _road(1, r"o\road\asf25.p3d", 100, 100, heading=90.0, y=0.0)
    bridge = _road(2, r"o\road\sil25.p3d", 100, 100, y=5.0)
    result = deduplicate_final_road_objects(_report((gravel, bridge)), _spec())
    assert result.objects == (gravel, bridge)


def test_stock_gravel_junction_blocks_dirt_but_never_counts_as_paved():
    # Gravel-only asf junction is not a *paved* blocker, but is correctly
    # above ces dirt; its footprint prevents the dirt texture covering asf.
    asf = _road(1, r"o\road\kr_new_asf_asf_t.p3d", 100, 100, y=0.055)
    dirt = _road(2, r"o\road\ces25.p3d", 100, 100, heading=90.0)
    result = deduplicate_final_road_objects(_report((asf, dirt), caps=1), _spec())
    assert asf in result.objects
    assert dirt not in result.objects


def test_stock_gravel_junction_cap_cannot_paint_over_paved():
    cap = _road(1, r"o\road\kr_new_asf_asf_t.p3d", 100, 100, y=0.055)
    asphalt = _road(2, r"o\road\sil25.p3d", 100, 100, y=0.095)
    result = deduplicate_final_road_objects(
        _report((cap, asphalt), caps=1), _spec(),
    )
    assert tuple(obj.object_id for obj in result.objects) == (2,)
    assert result.junction_cap_objects == 0


def test_stock_gravel_junction_cap_under_real_paved_overpass_is_preserved():
    cap = _road(1, r"o\road\kr_new_asf_asf_x.p3d", 100, 100, y=0.055)
    elevated = _road(2, r"o\road\sil25.p3d", 100, 100, y=4.0)
    result = deduplicate_final_road_objects(
        _report((cap, elevated), caps=1), _spec(),
    )
    assert result.objects == (cap, elevated)
    assert result.junction_cap_objects == 1


def test_stock_dirt_is_split_below_gravel_even_without_paved_roads():
    lower = _road(1, r"o\road\ces25.p3d", 100, 100, heading=90.0, y=0.0)
    upper = _road(2, r"o\road\asf25.p3d", 100, 100, heading=0.0, y=0.035)
    result = deduplicate_final_road_objects(_report((lower, upper)), _spec())
    assert upper in result.objects
    assert lower not in result.objects
    repaired = [
        obj for obj in result.objects if obj.model_path.casefold().endswith(
            (r"\ces6.p3d", r"\ces12.p3d")
        )
    ]
    assert len(repaired) >= 2
    assert any(abs(obj.pitch_degrees) > 0.01 for obj in repaired)


def test_unified_generated_gravel_is_retiled_as_gravel_not_dirt():
    spec = _spec(custom=True)
    lower = custom_road_model_path(spec.name, "gravel", 4.6, 25.0)
    original = _road(1, lower, 100, 89, heading=0.0, y=0.035)
    higher = _road(2, r"o\road\sil25.p3d", 100, 100, heading=90.0, y=0.095)
    result = deduplicate_final_road_objects(_report((original, higher)), spec)
    generated = [
        obj for obj in result.objects
        if (
            (signature := custom_road_model_signature(obj.model_path))
            is not None and signature[0] == "gravel"
        )
    ]
    assert generated
    assert all(obj.model_path.casefold() != original.model_path.casefold()
               or obj != original for obj in generated)
    assert all(abs(obj.pitch_degrees) <= 0.01 for obj in generated)
    _assert_no_gravel_footprint_over_paved(result.objects, spec)
