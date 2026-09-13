from types import SimpleNamespace

from cwr_worldgen import playability
from cwr_worldgen import road_chain_parallel_policy as perf
from cwr_worldgen import road_quality_parallel_compat_policy as quality_perf
from cwr_worldgen import road_quality_policy as road_quality


def _measure():
    return playability._PolylineMeasure.create((
        (0.0, 0.0),
        (18.0, 0.0),
        (36.0, 5.0),
        (54.0, 14.0),
        (76.0, 14.0),
        (102.0, 4.0),
    ))


def test_batched_chord_endpoints_match_scalar_queries() -> None:
    measure = _measure()
    for start in (0.0, 4.0, 17.0, 31.0, 60.0):
        maximum = min(measure.total, start + 45.0)
        lengths = (6.0, 12.0, 25.0)
        actual = perf._batch_chord_endpoints(measure, start, lengths, maximum)
        for length in lengths:
            expected = measure.chord_endpoint(start, length, maximum)
            assert actual.get(length) == expected


def test_batched_stock_chain_matches_previous_chain_without_quality_context() -> None:
    measure = _measure()
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    kwargs = dict(
        start_distance=0.0,
        preferred_end_distance=measure.total - 2.0,
        minimum_end_distance=measure.total - 5.0,
        maximum_end_distance=measure.total + 3.0,
    )
    expected = road_quality._ORIGINAL_CHAIN(measure, pieces, **kwargs)
    actual = perf._batched_stock_piece_chain(measure, pieces, **kwargs)
    assert actual == expected


def test_batched_quality_chain_matches_existing_quality_scorer() -> None:
    measure = _measure()
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    cells = 8
    elevations = tuple(
        2.0 + (index // cells) * 0.10 + (index % cells) * 0.04
        for index in range(cells * cells)
    )
    spec = SimpleNamespace(
        cells=cells,
        cell_size=25.0,
        road_connection_tolerance=0.35,
    )
    context = road_quality._Context(elevations, spec, {})
    kwargs = dict(
        start_distance=0.0,
        preferred_end_distance=measure.total - 2.0,
        minimum_end_distance=measure.total - 5.0,
        maximum_end_distance=measure.total + 3.0,
    )
    token = road_quality._CONTEXT.set(context)
    try:
        expected = road_quality._quality_chain(measure, pieces, **kwargs)
        actual = quality_perf._batched_quality_chain(measure, pieces, **kwargs)
    finally:
        road_quality._CONTEXT.reset(token)
    assert actual == expected


def test_run_job_preserves_quality_aware_chain_plan() -> None:
    run = ((0.0, 0.0), (30.0, 0.0), (55.0, 8.0), (85.0, 8.0))
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    job = perf._RunJob(
        order=0,
        feature_index=0,
        run_index=0,
        run=run,
        variants=pieces,
        start_trim=2.0,
        end_trim=2.0,
        start_cover=3.2,
        end_cover=3.2,
        cap_surface_mismatch=False,
        world_size=6400.0,
    )
    plan = quality_perf._quality_aware_plan_run(job)
    measure = playability._PolylineMeasure.create(run)
    expected = perf._batched_stock_piece_chain(
        measure,
        pieces,
        start_distance=2.0,
        preferred_end_distance=measure.total - 2.0,
        minimum_end_distance=measure.total - 3.2,
        maximum_end_distance=measure.total + 0.70,
    )
    assert plan.fitted_pieces == expected
    assert plan.order == 0
    assert plan.skipped_short_runs == 0


def test_parallel_policy_is_bridge_base_and_quality_chain_is_live() -> None:
    # Source-water bridge policy wraps the optimized base fitter. The chain
    # function itself remains quality-aware so serial and worker paths use the
    # same terrain-bulge/junction score.
    from cwr_worldgen import bridge_source_water_policy

    assert (
        bridge_source_water_policy._ORIGINAL_STOCK_FIT
        is perf._fit_stock_piece_road_objects_parallel
    )
    assert playability._stock_piece_chain is quality_perf._batched_quality_chain
    assert perf._plan_run is quality_perf._quality_aware_plan_run
    assert perf._execute_run_jobs is quality_perf._quality_aware_execute_run_jobs
