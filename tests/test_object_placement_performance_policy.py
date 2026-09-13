from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import object_placement_performance_policy as perf
from cwr_worldgen import osm
from cwr_worldgen import paved_junction_performance_policy as paved_perf
from cwr_worldgen import playability
from cwr_worldgen.procedural_forests import FOREST_UNDERGROWTH_VARIANTS


def test_object_placement_performance_policy_is_installed() -> None:
    assert paved_perf._approach_choice_to_target is perf._fast_approach_choice_to_target
    assert paved_perf._plan_application is perf._fast_plan_application
    assert playability._PolylineMeasure.chord_endpoint is perf._fast_chord_endpoint
    assert playability._PolylineMeasure.maximum_chord_deviation is perf._fast_maximum_chord_deviation
    assert playability._road_piece_sequence is perf._fast_road_piece_sequence
    assert playability._nearest_polyline_heading is perf._fast_nearest_polyline_heading
    assert osm.IndexedRoadCorridors.intersects_rectangle is perf._fast_corridor_intersects_rectangle
    assert osm._roadside_vegetation_candidates is perf._fast_roadside_vegetation_candidates
    assert osm._place_cluster_at is perf._fast_place_cluster_at


def test_paved_approach_prefilter_matches_previous_solver() -> None:
    connector = SimpleNamespace(point=(100.0, 100.0), direction=(0.0, 1.0))
    arm = SimpleNamespace(connector=connector, source_direction=(0.0, 1.0))
    plan = SimpleNamespace(point=(100.0, 100.0))

    for point, continuation in (
        ((100.0, 131.6), (0.0, 1.0)),
        ((112.0, 132.0), (0.25, 0.9682458365518543)),
        ((88.0, 132.0), (-0.25, 0.9682458365518543)),
        ((125.0, 125.0), (0.7071067811865476, 0.7071067811865476)),
    ):
        target = SimpleNamespace(
            object_id=7,
            point=point,
            continuation=continuation,
        )
        expected = perf._ORIGINAL_APPROACH_CHOICE(plan, arm, target, 0.35)
        actual = perf._fast_approach_choice_to_target(plan, arm, target, 0.35)
        assert actual == expected


def test_paved_assignment_pruning_matches_cartesian_product() -> None:
    targets = [SimpleNamespace(object_id=value) for value in (10, 11, 12, 13)]
    choices = [object() for _ in range(8)]
    option_sets = (
        ((1.0, targets[0], choices[0]), (2.0, targets[1], choices[1])),
        ((1.1, targets[0], choices[2]), (2.1, targets[2], choices[3])),
        ((1.2, targets[1], choices[4]), (2.2, targets[3], choices[5])),
        ((1.3, targets[2], choices[6]), (2.3, targets[3], choices[7])),
    )
    plan = SimpleNamespace(arms=(object(), object(), object(), object()))

    def options(_state, _plan, arm, _spec):
        return option_sets[plan.arms.index(arm)]

    with patch.object(paved_perf, "_arm_options", side_effect=options):
        expected = perf._ORIGINAL_PLAN_APPLICATION(None, plan, None)
        actual = perf._fast_plan_application(None, plan, None)
    assert actual == expected


def test_stock_road_measure_bisection_matches_previous_methods() -> None:
    measure = playability._PolylineMeasure.create(
        (
            (0.0, 0.0),
            (30.0, 0.0),
            (55.0, 12.0),
            (80.0, 20.0),
            (105.0, 20.0),
            (135.0, 5.0),
        )
    )
    for start in (0.0, 5.0, 25.0, 45.0, 80.0):
        for chord in (6.0, 12.0, 25.0):
            maximum = min(measure.total, start + 42.0)
            expected = perf._ORIGINAL_CHORD_ENDPOINT(measure, start, chord, maximum)
            actual = perf._fast_chord_endpoint(measure, start, chord, maximum)
            assert actual == expected
            if expected is not None:
                end_distance, end_x, end_z, _heading = expected
                start_x, start_z, _ = measure.point(start)
                expected_deviation = perf._ORIGINAL_MAXIMUM_CHORD_DEVIATION(
                    measure,
                    start,
                    end_distance,
                    (start_x, start_z),
                    (end_x, end_z),
                )
                actual_deviation = perf._fast_maximum_chord_deviation(
                    measure,
                    start,
                    end_distance,
                    (start_x, start_z),
                    (end_x, end_z),
                )
                assert actual_deviation == expected_deviation


def test_stock_piece_sequence_cache_preserves_selection() -> None:
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    for remaining in (2.0, 5.5, 6.0, 11.0, 18.0, 25.0, 51.0):
        assert perf._fast_road_piece_sequence(remaining, pieces) == perf._ORIGINAL_ROAD_PIECE_SEQUENCE(
            remaining, pieces
        )


def test_nearest_heading_index_matches_full_scan_for_points_on_run() -> None:
    points = (
        (0.0, 0.0),
        (30.0, 0.0),
        (50.0, 15.0),
        (75.0, 15.0),
        (95.0, 35.0),
    )
    for point in ((15.0, 0.0), (40.0, 7.5), (62.0, 15.0), (85.0, 25.0)):
        assert perf._fast_nearest_polyline_heading(points, point) == perf._ORIGINAL_NEAREST_POLYLINE_HEADING(
            points, point
        )


def _corridor_index() -> osm.IndexedRoadCorridors:
    corridors = (
        ((10.0, 10.0), (190.0, 10.0), 5.0),
        ((90.0, -20.0), (90.0, 180.0), 4.0),
        ((150.0, 50.0), (230.0, 130.0), 6.0),
    )
    bucket_size = 100.0
    buckets = {}
    for index, (start, end, radius) in enumerate(corridors):
        x0 = int((min(start[0], end[0]) - radius) // bucket_size)
        x1 = int((max(start[0], end[0]) + radius) // bucket_size)
        z0 = int((min(start[1], end[1]) - radius) // bucket_size)
        z1 = int((max(start[1], end[1]) + radius) // bucket_size)
        for bz in range(z0, z1 + 1):
            for bx in range(x0, x1 + 1):
                buckets.setdefault((bx, bz), []).append(index)
    return osm.IndexedRoadCorridors(
        corridors,
        bucket_size,
        {key: tuple(values) for key, values in buckets.items()},
    )


def test_forest_corridor_epoch_dedup_matches_set_based_query() -> None:
    index = _corridor_index()
    rectangles = (
        (0.0, 0.0, 20.0, 20.0),
        (40.0, 40.0, 80.0, 80.0),
        (80.0, 70.0, 110.0, 115.0),
        (120.0, 80.0, 180.0, 140.0),
        (210.0, 0.0, 250.0, 30.0),
    )
    for rectangle in rectangles:
        assert perf._fast_corridor_intersects_rectangle(index, *rectangle) == perf._ORIGINAL_CORRIDOR_INTERSECTS_RECTANGLE(
            index, *rectangle
        )


def test_lazy_forest_candidate_heap_preserves_stable_priority_order() -> None:
    kwargs = dict(
        seed="test-world",
        column=17,
        row=23,
        x=500.0,
        z=600.0,
        block_size=50.0,
        label="tree",
        minimum_spacing=3.75,
        candidate_count=64,
    )
    expected = tuple(perf._ORIGINAL_ROADSIDE_VEGETATION_CANDIDATES(**kwargs))
    actual = tuple(perf._fast_roadside_vegetation_candidates(**kwargs))
    assert actual == expected


def test_single_pass_cluster_grounding_matches_previous_result() -> None:
    cells = 8
    cell_size = 25.0
    size = cells * cells
    elevations = tuple(
        5.0 + (index // cells) * 0.03 + (index % cells) * 0.02
        for index in range(size)
    )
    raster = SimpleNamespace(
        forest=(True,) * size,
        water=(False,) * size,
        roads=(False,) * size,
        buildings=(False,) * size,
    )
    spec = SimpleNamespace(
        world_size=cells * cell_size,
        cells=cells,
        cell_size=cell_size,
        name="test_world",
        forest_cluster_footprint_margin=0.75,
        forest_low_anchor=False,
        forest_cluster_tree_maximum_float=10.0,
        forest_cluster_bush_maximum_float=10.0,
    )
    variant = FOREST_UNDERGROWTH_VARIANTS[0]
    kwargs = dict(
        variant=variant,
        elevations=elevations,
        raster=raster,
        road_corridors=(),
        spec=spec,
        x=100.0,
        z=100.0,
        heading=35.0,
        require_forest=True,
        minimum_forest_fraction=0.80,
        maximum_relief=20.0,
        maximum_burial=0.8,
        maximum_float=10.0,
        clearance=0.03,
    )
    assert perf._fast_place_cluster_at(**kwargs) == perf._ORIGINAL_PLACE_CLUSTER_AT(**kwargs)
