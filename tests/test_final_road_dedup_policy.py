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


def test_curves_and_junction_models_are_not_axis_deduplicated():
    report = _report((
        _road(1, r"o\road\sil10 25.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil10 25.p3d", 100.0, 100.0),
        _road(3, r"o\road\kr_new_sil_sil_t.p3d", 120.0, 120.0),
        _road(4, r"o\road\kr_new_sil_sil_t.p3d", 120.0, 120.0),
    ))

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 4


def test_junction_cap_prefix_is_protected_from_deduplication():
    report = _report((
        _road(1, r"o\road\sil6.p3d", 100.0, 100.0),
        _road(2, r"o\road\sil6.p3d", 100.0, 100.0),
    ), caps=1)

    result = deduplicate_final_road_objects(report, _spec())

    assert len(result.objects) == 2


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
