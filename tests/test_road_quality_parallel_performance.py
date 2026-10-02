from concurrent.futures import Future, ProcessPoolExecutor
from types import SimpleNamespace

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


def test_spawned_worker_receives_mod_road_context(monkeypatch) -> None:
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

    variants = {r"bas_o\_road\bas_asf25.p3d": frozenset({
        r"bas_o\_road\bas_asf25.p3d",
    })}
    dimensions = {
        r"bas_o\_road\bas_asf25.p3d": (5.2, 25.0),
    }
    measurement_errors = {
        r"bas_o\_road\bad.p3d": "synthetic failure",
    }
    effective = {
        r"bas_o\_road\legacy10 25.p3d": r"bas_o\_road\bas_asf25.p3d",
    }
    observed: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, *, max_workers, initializer, initargs):
            observed["max_workers"] = max_workers
            observed["initargs"] = initargs
            self.initializer = initializer
            self.initargs = initargs

        def __enter__(self):
            # Simulate a fresh Windows spawn where ContextVars start empty.
            playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(None)
            playability._ROAD_MODEL_DIMENSIONS.set(None)
            playability._ROAD_MODEL_MEASUREMENT_ERRORS.set(None)
            playability._ROAD_MODEL_EFFECTIVE_DONORS.set(None)
            self.initializer(*self.initargs)
            observed["worker_context"] = quality_perf._road_worker_context()
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(parallel, "_worker_count", lambda _count: 2)
    monkeypatch.setattr(quality_perf, "ProcessPoolExecutor", FakeExecutor)

    quality_token = quality._CONTEXT.set(context)
    variants_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(variants)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    errors_token = playability._ROAD_MODEL_MEASUREMENT_ERRORS.set(measurement_errors)
    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(effective)
    try:
        plans = quality_perf._quality_aware_execute_run_jobs((job,))
    finally:
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)
        playability._ROAD_MODEL_MEASUREMENT_ERRORS.reset(errors_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(variants_token)
        quality._CONTEXT.reset(quality_token)
        quality_perf._clear_run_plan_cache()

    assert len(plans) == 1
    assert observed["max_workers"] == 2
    assert observed["worker_context"] == (
        variants,
        dimensions,
        measurement_errors,
        effective,
    )


def test_real_process_worker_receives_mod_road_context() -> None:
    variants = {r"bas_o\_road\bas_asf25.p3d": frozenset({
        r"bas_o\_road\bas_asf25.p3d",
    })}
    dimensions = {r"bas_o\_road\bas_asf25.p3d": (5.2, 25.0)}
    measurement_errors = {r"bas_o\_road\bad.p3d": "synthetic failure"}
    effective = {
        r"bas_o\_road\legacy10 25.p3d": r"bas_o\_road\bas_asf25.p3d",
    }
    road_context = (variants, dimensions, measurement_errors, effective)

    with ProcessPoolExecutor(
        max_workers=1,
        initializer=quality_perf._install_worker_quality_context,
        initargs=(None, road_context),
    ) as executor:
        observed = executor.submit(quality_perf._road_worker_context).result()

    assert observed == road_context
