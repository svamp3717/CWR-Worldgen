from types import SimpleNamespace
import math

from cwr_worldgen import playability
from cwr_worldgen import road_chain_parallel_policy as parallel
from cwr_worldgen import road_quality_parallel_compat_policy as quality_perf
from cwr_worldgen import road_quality_policy as quality


def _measure():
    return playability._PolylineMeasure.create((
        (0.0, 0.0),
        (18.0, 0.0),
        (36.0, 5.0),
        (54.0, 14.0),
        (76.0, 14.0),
        (102.0, 4.0),
    ))


def _quality_context():
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
    return quality._Context(elevations, spec, {})


def test_batched_tail_error_matches_scalar_lookahead() -> None:
    measure = _measure()
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    preferred = measure.total - 2.0
    maximum = measure.total + 3.0

    for current in (0.0, 17.0, 48.0, 70.0):
        for depth in (0, 1, 2):
            expected = quality._tail_error(
                measure, pieces, current, preferred, maximum, depth
            )
            actual = quality_perf._batched_tail_error(
                measure, pieces, current, preferred, maximum, depth, {}
            )
            assert actual == expected


def test_batched_terrain_bulges_match_scalar_quality_samples() -> None:
    context = _quality_context()
    start = (18.0, 3.0)
    candidates = (
        ((42.0, 9.0), 25),
        ((31.0, 17.0), 12),
        ((24.0, 5.0), 6),
    )
    expected = tuple(
        quality._terrain_bulge(context, start, end, nominal)
        for end, nominal in candidates
    )
    actual = quality_perf._batched_terrain_bulges(
        context, start, candidates, {}
    )
    assert actual == expected


def test_run_plan_cache_reuses_unchanged_pass_and_invalidates_changed_endpoint(
    monkeypatch,
) -> None:
    quality_perf._clear_run_plan_cache()
    context = _quality_context()
    run = ((0.0, 0.0), (30.0, 0.0), (55.0, 8.0), (85.0, 8.0))
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    job = parallel._RunJob(
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

    calls = 0
    base_plan = quality_perf._quality_aware_plan_run

    def counted_plan(current_job):
        nonlocal calls
        calls += 1
        return base_plan(current_job)

    monkeypatch.setattr(quality_perf, "_quality_aware_plan_run", counted_plan)
    monkeypatch.setattr(parallel, "_worker_count", lambda _count: 1)

    token = quality._CONTEXT.set(context)
    try:
        first = quality_perf._quality_aware_execute_run_jobs((job,))
        second = quality_perf._quality_aware_execute_run_jobs((job,))
    finally:
        quality._CONTEXT.reset(token)

    assert calls == 1
    assert second == first

    start_key = playability._road_node_key(run[0])
    changed_context = quality._Context(
        context.elevations,
        context.spec,
        {
            start_key: quality._Junction(
                point=run[0],
                axis=(0.0, 1.0),
                half_length=32.0,
                half_width=32.0,
                directions=((0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)),
            )
        },
    )
    token = quality._CONTEXT.set(changed_context)
    try:
        quality_perf._quality_aware_execute_run_jobs((job,))
    finally:
        quality._CONTEXT.reset(token)
        quality_perf._clear_run_plan_cache()

    assert calls == 2


def test_parallel_quality_chain_matches_serial_stock_joint_selection() -> None:
    cells = 8
    context = quality._Context(
        tuple(0.0 for _ in range(cells * cells)),
        SimpleNamespace(
            cells=cells,
            cell_size=25.0,
            road_connection_tolerance=0.35,
        ),
        {},
    )
    angle = math.radians(18.0)
    measure = playability._PolylineMeasure.create((
        (0.0, 0.0),
        (0.0, 12.0),
        (math.sin(angle) * 12.0, 12.0 + math.cos(angle) * 12.0),
        (math.sin(angle) * 24.0, 12.0 + math.cos(angle) * 24.0),
    ))
    pieces = playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
    kwargs = dict(
        start_distance=0.0,
        preferred_end_distance=measure.total,
        minimum_end_distance=0.0,
        maximum_end_distance=measure.total,
    )

    token = quality._CONTEXT.set(context)
    try:
        serial = quality._quality_chain(measure, pieces, **kwargs)
        batched = quality_perf._batched_quality_chain(measure, pieces, **kwargs)
    finally:
        quality._CONTEXT.reset(token)

    assert batched == serial
    assert all(
        quality._is_stock_paved_piece(piece)
        for piece, _start, _end in batched
    )
