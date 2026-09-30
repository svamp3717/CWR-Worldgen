import math
from types import SimpleNamespace

from cwr_worldgen import playability
from cwr_worldgen.final_road_dedup_policy import deduplicate_final_road_objects
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport
from cwr_worldgen.procedural_infrastructure import (
    custom_road_model_path,
    custom_road_model_signature,
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


def _spec():
    return SimpleNamespace(road_segment_length=25.0)


def _road(object_id, model, x, z, *, heading=0.0, y=0.0):
    return WorldObject(object_id, model, x, y, z, heading, 0.0)


def test_exact_duplicate_straights_keep_one_deterministically():
    report = _report((
        _road(20, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(10, r"o\road\sil25.p3d", 100.0, 100.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (10,)


def test_short_intentional_chain_seam_overlap_is_preserved():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 124.8),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_adjacent_curves_with_only_a_small_end_overlap_are_preserved():
    report = _report((
        _road(1, r"o\road\sil10 100.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil10 100.p3d", 100.0, 117.2),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_perpendicular_crossing_is_preserved():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0, heading=0.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=90.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_parallel_divided_road_is_preserved():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 105.0, 100.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_grade_separated_roads_are_preserved():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0, y=5.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, y=8.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_paved_surface_wins_over_coincident_dirt_piece():
    report = _report((
        _road(1, r"o\road\ces25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (2,)


def test_exact_duplicate_stock_curves_keep_one_deterministically():
    report = _report((
        _road(20, r"o\road\sil10 25.p3d", 100.0, 100.0),
        _road(10, r"o\road\sil10 25.p3d", 100.0, 100.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (10,)


def test_curve_fully_covered_by_long_stock_slab_is_removed():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil10 50.p3d", 100.0, 100.0, heading=180.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1,)


def test_generated_paved_micro_ribbon_covered_by_stock_road_is_removed():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(
            2,
            r"wg_test\i\paved_w091_l0018.p3d",
            100.0,
            100.0,
        ),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1,)


def test_generated_paved_curve_covered_by_stock_road_is_removed():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(
            2,
            r"wg_test\i\paved_w091_l0059_l25.p3d",
            100.0,
            100.0,
        ),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1,)


def test_shallow_angle_paved_piece_buried_by_longer_paved_piece_is_removed():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil6.p3d", 100.0, 100.0, heading=9.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1,)


def test_shallow_angle_paved_partial_overlap_is_preserved():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil12.p3d", 100.0, 111.0, heading=9.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_paved_second_pass_does_not_delete_longer_candidate_under_short_wide_piece():
    report = _report((
        _road(1, r"o\road\sil6.p3d", 100.0, 100.0),
        _road(2, r"o\road\asf25.p3d", 100.0, 100.0, heading=9.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_dirt_curves_remain_out_of_scope_for_paved_second_pass():
    report = _report((
        _road(1, r"o\road\ces10 25.p3d", 100.0, 100.0),
        _road(2, r"o\road\ces10 25.p3d", 100.0, 100.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_shallow_angle_dirt_straights_remain_out_of_paved_second_pass():
    report = _report((
        _road(1, r"o\road\ces25.p3d", 100.0, 100.0),
        _road(2, r"o\road\ces6.p3d", 100.0, 100.0, heading=9.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_junction_models_are_still_not_axis_deduplicated():
    report = _report((
        _road(3, r"o\road\kr_new_sil_sil_t.p3d", 120.0, 120.0),
        _road(4, r"o\road\kr_new_sil_sil_t.p3d", 120.0, 120.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


def test_redundant_paved_junction_cap_can_be_removed_and_prefix_shrinks():
    report = _report((
        _road(1, r"o\road\sil6.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (2,)
    assert result.junction_cap_objects == 0


def test_dirt_junction_cap_prefix_remains_protected():
    report = _report((
        _road(1, r"o\road\ces6.p3d", 100.0, 100.0),
        _road(2, r"o\road\ces6.p3d", 100.0, 100.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2)
    assert result.junction_cap_objects == 1


def test_short_paved_overlap_is_tightened_only_near_a_paved_junction():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 120.0, 100.0),
        _road(3, r"o\road\sil6.p3d", 120.0, 100.0, heading=20.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2)
    assert result.junction_cap_objects == 1


def test_same_short_paved_overlap_away_from_junction_is_preserved():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 200.0, 200.0),
        _road(3, r"o\road\sil6.p3d", 200.0, 200.0, heading=20.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2, 3)
    assert result.junction_cap_objects == 1


def test_partial_short_paved_duplicate_at_junction_is_removed():
    # terrtest58 has this shape near 1100,1379: two nominally equal sil6 slabs
    # share about 56% of their length. The second is a chain piece laid over the
    # cap, so the junction-local pass should remove it while preserving the cap.
    report = _report((
        _road(1, r"o\road\sil6.p3d", 100.0, 100.0, heading=90.0),
        _road(2, r"o\road\sil6.p3d", 102.64, 100.0, heading=270.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1,)
    assert result.junction_cap_objects == 1


def test_offset_paved_surface_overlap_is_removed_near_junction():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 120.0, 100.0),
        _road(3, r"o\road\sil12.p3d", 122.0, 100.0, heading=7.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2)


def test_same_offset_surface_overlap_away_from_junction_is_preserved():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 200.0, 200.0),
        _road(3, r"o\road\sil12.p3d", 202.0, 200.0, heading=7.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2, 3)


def test_heavily_covered_paved_surface_is_removed_away_from_junction():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 200.0, 200.0),
        _road(3, r"o\road\sil6.p3d", 200.0, 200.0, heading=15.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2)


def test_surface_overlap_rule_does_not_remove_perpendicular_paved_crossing():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 120.0, 100.0),
        _road(3, r"o\road\sil6.p3d", 120.0, 100.0, heading=90.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2, 3)


def test_surface_overlap_rule_stays_paved_only():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\ces25.p3d", 120.0, 100.0),
        _road(3, r"o\road\ces12.p3d", 122.0, 100.0, heading=7.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2, 3)


def test_near_collinear_paved_piece_just_over_two_metres_offset_is_removed():
    report = _report((
        _road(1, r"o\road\sil25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil6.p3d", 102.15, 100.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1,)


def test_generated_paved_curve_can_replace_substantially_covered_stock_cap():
    report = _report((
        _road(1, r"o\road\sil6.p3d", 100.0, 100.0),
        _road(
            2,
            r"wg_test\i\paved_w091_l0059_r25.p3d",
            100.0,
            102.35,
        ),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (2,)
    assert result.junction_cap_objects == 0


def test_generated_paved_straight_does_not_use_curve_cap_exception():
    report = _report((
        _road(1, r"o\road\sil6.p3d", 100.0, 100.0),
        _road(
            2,
            r"wg_test\i\paved_w091_l0059.p3d",
            100.0,
            102.35,
        ),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2)
    assert result.junction_cap_objects == 1


def test_perpendicular_dirt_crossing_keeps_lowered_terminal_pieces():
    report = _report((
        _road(1, r"o\road\ces25.p3d", 100.0, 100.0, heading=90.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=0.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    dirt = tuple(
        obj for obj in result.objects
        if obj.model_path.casefold().endswith(r"\ces6.p3d")
    )
    underlays = tuple(obj for obj in dirt if abs(obj.pitch_degrees) > 0.01)
    assert len(dirt) == 4
    assert len(underlays) == 2
    assert any(obj.object_id == 1 for obj in dirt)
    assert tuple(
        obj.object_id for obj in result.objects
        if obj.model_path.casefold().endswith(r"\sil25.p3d")
    ) == (2,)
    assert all(obj.y < 0.0 for obj in underlays)


def test_dirt_t_approach_finishes_with_piece_diving_under_paved():
    report = _report((
        _road(1, r"o\road\ces25.p3d", 100.0, 89.0, heading=0.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=90.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    dirt = tuple(
        obj for obj in result.objects
        if obj.model_path.casefold().endswith(
            (r"\ces25.p3d", r"\ces12.p3d", r"\ces6.p3d")
        )
    )
    underlays = tuple(obj for obj in dirt if abs(obj.pitch_degrees) > 0.01)

    assert dirt
    assert len(underlays) == 1
    terminal = underlays[0]
    axis = playability._model_axis(terminal, 6.25)
    sine_pitch = math.sin(math.radians(terminal.pitch_degrees))
    backward_y = terminal.y - 3.125 * sine_pitch
    forward_y = terminal.y + 3.125 * sine_pitch
    assert axis[0][1] < 95.0
    assert axis[1][1] > 95.0
    assert math.isclose(backward_y, 0.0, abs_tol=1.0e-6)
    assert forward_y <= -0.079


def test_short_dirt_piece_between_two_paved_strips_can_extend_under_them():
    original = _road(1, r"o\road\ces6.p3d", 100.0, 100.0, heading=0.0)
    report = _report((
        original,
        _road(
            2,
            r"wg_test\i\paved_w020_l0250.p3d",
            100.0,
            98.0,
            heading=90.0,
        ),
        _road(
            3,
            r"wg_test\i\paved_w020_l0250.p3d",
            100.0,
            102.0,
            heading=90.0,
        ),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    dirt = tuple(
        obj for obj in result.objects
        if obj.model_path.casefold().endswith(r"\ces6.p3d")
    )
    assert len(dirt) == 2
    assert all(abs(obj.pitch_degrees) > 0.01 for obj in dirt)

    original_axis = playability._model_axis(original, 6.25)
    terminal_axes = tuple(
        playability._model_axis(obj, 6.25)
        for obj in dirt
    )
    assert any(axis[0][1] < original_axis[0][1] for axis in terminal_axes)
    assert any(axis[1][1] > original_axis[1][1] for axis in terminal_axes)


def test_short_dirt_piece_fully_consumed_by_paved_crossing_is_removed():
    report = _report((
        _road(1, r"o\road\ces6.p3d", 100.0, 100.0, heading=90.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=0.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (2,)


def test_dirt_curve_touching_paved_surface_is_removed_wholesale():
    report = _report((
        _road(1, r"o\road\ces10 50.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (2,)


def test_paved_junction_footprint_keeps_dirt_terminals_below_hub():
    report = _report((
        _road(1, r"o\road\kr_new_sil_sil_t.p3d", 100.0, 100.0),
        _road(2, r"o\road\ces25.p3d", 100.0, 100.0, heading=90.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert result.objects[0].object_id == 1
    dirt = tuple(
        obj for obj in result.objects
        if obj.model_path.casefold().endswith(r"\ces6.p3d")
    )
    assert len(dirt) == 2
    assert all(abs(obj.pitch_degrees) > 0.01 for obj in dirt)
    assert all(obj.y < 0.0 for obj in dirt)
    assert result.junction_cap_objects == 1


def test_dirt_under_high_paved_overpass_is_preserved():
    report = _report((
        _road(1, r"o\road\ces25.p3d", 100.0, 100.0, heading=90.0, y=0.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=0.0, y=4.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (1, 2)


def test_badly_grounded_dirt_above_paved_is_still_trimmed():
    report = _report((
        _road(1, r"o\road\ces6.p3d", 100.0, 100.0, heading=90.0, y=2.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=0.0, y=0.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert tuple(obj.object_id for obj in result.objects) == (2,)


def test_unified_dirt_overlap_repairs_remain_generated_ribbons():
    spec = SimpleNamespace(
        name="unified",
        road_segment_length=25.0,
        custom_road_shapes=True,
    )
    dirt_model = custom_road_model_path(
        "unified",
        "dirt",
        3.5,
        25.0,
    )
    report = _report((
        _road(1, dirt_model, 100.0, 89.0, heading=0.0, y=0.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=90.0, y=0.0),
    ))

    result = deduplicate_final_road_objects(report, spec)

    dirt = tuple(
        obj
        for obj in result.objects
        if (
            (signature := custom_road_model_signature(obj.model_path))
            is not None
            and signature[0] == "dirt"
        )
    )
    assert dirt
    assert all(
        not obj.model_path.casefold().endswith(
            (r"\ces25.p3d", r"\ces12.p3d", r"\ces6.p3d")
        )
        for obj in dirt
    )
    assert any(abs(obj.pitch_degrees) > 0.01 for obj in dirt)


def test_extreme_dirt_underlay_never_exceeds_rvw4_pitch_limit():
    report = _report((
        _road(1, r"o\road\ces25.p3d", 100.0, 89.0, heading=0.0, y=10.0),
        _road(2, r"o\road\sil25.p3d", 100.0, 100.0, heading=90.0, y=0.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    dirt = tuple(
        obj for obj in result.objects
        if obj.model_path.casefold().endswith(
            (r"\ces25.p3d", r"\ces12.p3d", r"\ces6.p3d")
        )
    )
    underlays = tuple(obj for obj in dirt if abs(obj.pitch_degrees) > 0.01)

    assert underlays
    assert all(abs(obj.pitch_degrees) <= 88.0 for obj in underlays)
    assert all(-89.0 < obj.pitch_degrees < 89.0 for obj in underlays)


def test_progress_reports_bounded_spatial_comparisons():
    events = []
    roads = tuple(
        _road(index + 1, r"o\road\sil25.p3d", float(index * 100), 100.0)
        for index in range(100)
    )

    result = deduplicate_final_road_objects(
        _report(roads),
        _spec(),
        progress_callback=lambda percent, text: events.append((percent, text)),
    )

    assert len(result.objects) == 100
    assert events
    assert events[-1][0] == 99
    assert "100/100" in events[-1][1]
    assert "0 removed" in events[-1][1]
    assert "nearby comparisons" in events[-1][1]
