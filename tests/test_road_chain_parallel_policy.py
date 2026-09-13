from types import SimpleNamespace

from cwr_worldgen import playability
from cwr_worldgen import road_chain_parallel_policy as perf


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


def test_batched_stock_chain_matches_previous_chain() -> None:
    measure = _measure()
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    kwargs = dict(
        start_distance=0.0,
        preferred_end_distance=measure.total - 2.0,
        minimum_end_distance=measure.total - 5.0,
        maximum_end_distance=measure.total + 3.0,
    )
    expected = perf._ORIGINAL_STOCK_CHAIN(measure, pieces, **kwargs)
    actual = perf._batched_stock_piece_chain(measure, pieces, **kwargs)
    assert actual == expected


def test_run_job_preserves_stock_chain_plan() -> None:
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
    plan = perf._plan_run(job)
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


def test_parallel_policy_is_bridge_base_not_final_wrapper() -> None:
    # The bridge source-water policy installs after this policy and wraps the
    # optimized base fitter. The live function may therefore be a bridge wrapper,
    # but that wrapper must have captured our implementation as its original.
    from cwr_worldgen import bridge_source_water_policy

    assert bridge_source_water_policy._ORIGINAL_STOCK_FIT is perf._fit_stock_piece_road_objects_parallel
    assert playability._stock_piece_chain is perf._batched_stock_piece_chain
