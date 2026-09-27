from types import SimpleNamespace

from cwr_worldgen.final_road_dedup_policy import deduplicate_final_road_objects
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport


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
