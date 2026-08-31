# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_emitted_seam_policy as _emitted
from cwr_worldgen import stock_road_emitted_seam_refinement_policy as _refinement
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_visual_finish_policy as _finish


def _object(object_id, model, x, z, heading, *, pitch=0.0, y=0.0):
    return SimpleNamespace(
        object_id=int(object_id),
        model_path=model,
        x=float(x),
        y=float(y),
        z=float(z),
        heading_degrees=float(heading),
        pitch_degrees=float(pitch),
    )


def _straight_from_start(object_id, start, heading, *, pitch=0.0):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    horizontal = length * math.cos(math.radians(float(pitch)))
    half = horizontal * 0.5
    angle = math.radians(float(heading))
    direction = (math.sin(angle), math.cos(angle))
    centre = (
        float(start[0]) + direction[0] * half,
        float(start[1]) + direction[1] * half,
    )
    return _object(
        object_id,
        r"o\road\sil6.p3d",
        centre[0],
        centre[1],
        heading,
        pitch=pitch,
    )


def test_final_pass_sees_pitch_projected_straight_gap():
    first_pitch = 8.0
    first_length = 6.25 * math.cos(math.radians(first_pitch))
    first = _object(
        1,
        r"o\road\sil6.p3d",
        0.0,
        -first_length * 0.5,
        0.0,
        pitch=first_pitch,
    )
    second = _straight_from_start(2, (0.18, 0.0), 8.0, pitch=5.0)
    report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)

    plans = _emitted._emitted_seam_cover_plans(report)

    assert len(plans) == 1
    assert all(plan.model_path.casefold() == r"o\road\sil6.p3d" for plan in plans)
    assert all(math.dist(plan.centre, (0.09, 0.0)) < 0.01 for plan in plans)
    assert math.isclose(plans[0].tangent_axis_degrees, 4.0, abs_tol=1.0e-6)
    assert math.isclose(plans[0].turn_degrees, 8.0, abs_tol=1.0e-6)


def test_final_curve_gap_uses_miter_bisector():
    model = r"o\road\sil10 50.p3d"
    geometry = _geometry.stock_curve_connectors(model)
    assert geometry is not None
    curve = _object(1, model, 0.0, 0.0, 0.0)
    curve_axis = _p._model_axis(curve, geometry.chord_length_metres)
    seam = curve_axis[1]

    straight_heading = 15.0
    direction = (
        math.sin(math.radians(straight_heading)),
        math.cos(math.radians(straight_heading)),
    )
    straight_start = (
        seam[0] + direction[0] * 1.0,
        seam[1] + direction[1] * 1.0,
    )
    straight = _straight_from_start(2, straight_start, straight_heading)
    report = SimpleNamespace(objects=(curve, straight), junction_cap_objects=0)

    plans = _emitted._emitted_seam_cover_plans(report)

    assert len(plans) == 1
    expected = _finish._average_axis_heading(
        _geometry.STOCK_CURVE_ANGLE_DEGREES,
        straight_heading,
    )
    assert math.isclose(plans[0].tangent_axis_degrees, expected, abs_tol=1.0e-6)


def test_existing_paved_underlay_prevents_duplicate_cover():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, (0.10, 0.0), 8.0)
    existing = _object(3, r"o\road\sil6.p3d", 0.05, 0.0, 4.0, y=-0.01)
    report = SimpleNamespace(
        objects=(first, second, existing),
        junction_cap_objects=0,
    )

    assert _emitted._emitted_seam_cover_plans(report) == ()


def test_buried_turn_gets_only_borderless_final_wedge():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, (0.0, 0.0), 12.0)
    existing = _object(3, r"o\road\sil6.p3d", 0.0, 0.0, 6.0, y=0.0)
    report = _p.RoadFitReport(
        objects=(first, second, existing),
        chain_count=1,
        connection_count=1,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )
    spec = SimpleNamespace(
        name="wg_test",
        cells=2,
        cell_size=25.0,
        max_road_objects=100,
        advisory_object_limits=False,
    )

    fixed = _emitted._apply_emitted_seam_covers(report, [0.0] * 4, spec)

    assert len(fixed.objects) == 4
    helper = fixed.objects[-1]
    assert "\\paved_wedge_q" in helper.model_path.replace("/", "\\").casefold()
    assert "paved_miter" not in helper.model_path.casefold()
    assert "paved_fill" not in helper.model_path.casefold()
    assert fixed.short_piece_objects == 1


def test_aligned_physical_gap_gets_underlay_even_without_tangent_error():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, (0.0, 0.18), 0.0)
    report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)

    plans = _refinement._refined_emitted_seam_cover_plans(report)

    assert len(plans) == 1
    assert math.isclose(plans[0].tangent_axis_degrees, 0.0, abs_tol=1.0e-9)
    assert math.dist(plans[0].centre, (0.0, 0.09)) < 1.0e-9


def test_large_straight_miter_uses_one_angle_matched_underlay():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, (0.12, 0.0), 12.0)
    report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)

    plans = _refinement._refined_emitted_seam_cover_plans(report)

    assert len(plans) == 1
    assert math.isclose(plans[0].tangent_axis_degrees, 6.0, abs_tol=1.0e-9)
    assert math.isclose(plans[0].turn_degrees, 12.0, abs_tol=1.0e-9)


def test_coincident_straight_miter_uses_one_bisecting_underlay():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, (0.0, 0.0), 12.0)
    report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)

    plans = _refinement._refined_emitted_seam_cover_plans(report)

    assert len(plans) == 1
    assert math.isclose(plans[0].tangent_axis_degrees, 6.0, abs_tol=1.0e-9)
    assert math.isclose(plans[0].turn_degrees, 12.0, abs_tol=1.0e-9)
    assert math.dist(plans[0].centre, (0.0, 0.0)) < 1.0e-9
    assert plans[0].outer_miter_apex is not None


def test_paved_turn_seam_uses_wedge_not_overlap_helper():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, (0.0, 0.0), 12.0)
    report = _p.RoadFitReport(
        objects=(first, second),
        chain_count=1,
        connection_count=1,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )
    spec = SimpleNamespace(
        name="wg_test",
        cells=2,
        cell_size=25.0,
        max_road_objects=100,
        advisory_object_limits=False,
    )

    fixed = _emitted._apply_emitted_seam_covers(report, [0.0] * 4, spec)

    assert len(fixed.objects) == 3
    helper = fixed.objects[-1]
    assert "\\paved_wedge_q" in helper.model_path.replace("/", "\\").casefold()
    assert all("paved_miter" not in str(obj.model_path).casefold() for obj in fixed.objects)
    assert all("paved_fill" not in str(obj.model_path).casefold() for obj in fixed.objects)
    assert fixed.short_piece_objects == 1


def test_lundby34_compiled_grass_wedges_still_generate_diagnostic_plans():
    cases = (
        (
            (
                _object(
                    7740,
                    r"o\road\sil6.p3d",
                    4666.94384765625,
                    4628.9404296875,
                    340.2384706417884,
                    pitch=1.4365560450584551,
                    y=23.183277130126953,
                ),
                _object(
                    7741,
                    r"o\road\sil6.p3d",
                    4663.43310546875,
                    4633.814453125,
                    308.2403992415925,
                    pitch=0.09258072651614063,
                    y=23.26667022705078,
                ),
            ),
            1,
        ),
        (
            (
                _object(
                    7708,
                    r"o\road\sil6.p3d",
                    334.6238708496094,
                    6365.3916015625,
                    353.3420158342863,
                    y=16.684999465942383,
                ),
                _object(
                    7709,
                    r"o\road\sil10 25.p3d",
                    333.6912536621094,
                    6370.19189453125,
                    154.21315842768192,
                    y=16.684999465942383,
                ),
            ),
            1,
        ),
        (
            (
                _object(
                    8738,
                    r"o\road\sil10 50.p3d",
                    3218.660888671875,
                    4948.8486328125,
                    348.3063195744053,
                    pitch=-0.035445535762980804,
                    y=10.746047973632812,
                ),
                _object(
                    8739,
                    r"o\road\sil10 100.p3d",
                    3222.28125,
                    4936.3740234375,
                    333.41577386280625,
                    pitch=-0.13692661349488416,
                    y=10.768783569335938,
                ),
            ),
            1,
        ),
    )

    for objects, expected_plans in cases:
        report = SimpleNamespace(objects=objects, junction_cap_objects=0)
        plans = _emitted._emitted_seam_cover_plans(report)

        assert len(plans) == expected_plans
        assert all(plan.model_path.casefold() == r"o\road\sil6.p3d" for plan in plans)


def test_unambiguous_legacy_paved_cap_endpoint_can_be_sealed():
    cap = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    approach = _straight_from_start(2, (0.0, 0.15), 0.0)
    report = SimpleNamespace(objects=(cap, approach), junction_cap_objects=1)

    plans = _refinement._refined_emitted_seam_cover_plans(report)

    assert len(plans) == 1
    assert math.dist(plans[0].centre, (0.0, 0.075)) < 1.0e-9


def test_near_coincident_overlapping_pieces_are_not_treated_as_seam():
    cap = _object(1, r"o\road\sil6.p3d", 0.0, 0.0, 0.0)
    overlapping = _object(2, r"o\road\sil6.p3d", 0.18, 0.02, 0.4)
    report = SimpleNamespace(objects=(cap, overlapping), junction_cap_objects=1)

    assert _refinement._refined_emitted_seam_cover_plans(report) == ()
