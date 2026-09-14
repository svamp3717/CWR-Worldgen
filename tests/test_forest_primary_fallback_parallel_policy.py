from types import SimpleNamespace

import numpy as np

from cwr_worldgen import forest_primary_fallback_parallel_policy as fallback


def test_forest_probe_counts_preserve_two_of_five_everon_candidates() -> None:
    spec = SimpleNamespace(cells=4, cell_size=10.0, world_size=40.0)
    raster = SimpleNamespace(forest=np.ones(16, dtype=np.bool_))
    primary = SimpleNamespace(
        possible_primary=np.ones((2, 2), dtype=np.bool_),
        spacing=20.0,
    )

    counts = fallback._forest_probe_counts(raster, spec, primary)

    assert counts.tolist() == [5, 5, 5, 5]


def test_cached_primary_helpers_return_worker_values_without_touching_base(monkeypatch) -> None:
    elevations = object()
    context = fallback._FallbackContext(elevations=elevations)
    context.active = True
    context.samples[(10.0, 20.0)] = 3.25
    context.squares[(10.0, 20.0, 50.0)] = (1.0, 2.0, 3.0)
    context.triangles[(10.0, 20.0)] = (4.0, 5.0, 6.0)
    context.road_hits[(10.0, 20.0, 50.0)] = True
    context.gradients[(10.0, 20.0)] = (0.25, -0.5)
    context.oriented[(10.0, 20.0, 20.0, 35.0, 90.0)] = (7.0, 8.0, 9.0)

    def unexpected(*args, **kwargs):
        raise AssertionError("base helper should not run for a cached primary value")

    monkeypatch.setattr(fallback, "_BASE_SAMPLE", unexpected)
    monkeypatch.setattr(fallback, "_BASE_SQUARE", unexpected)
    monkeypatch.setattr(fallback, "_BASE_TRIANGLE", unexpected)
    monkeypatch.setattr(fallback, "_BASE_ROAD_TEST", unexpected)
    monkeypatch.setattr(fallback, "_BASE_GRADIENT", unexpected)
    monkeypatch.setattr(fallback, "_BASE_ORIENTED", unexpected)

    token = fallback._CONTEXT.set(context)
    try:
        assert fallback._cached_sample(elevations, 1, 1.0, 10.0, 20.0) == 3.25
        assert fallback._cached_square(elevations, 1, 1.0, 10.0, 20.0, 50.0) == (1.0, 2.0, 3.0)
        assert fallback._cached_triangle(elevations, 1, 1.0, 10.0, 20.0) == (4.0, 5.0, 6.0)
        assert fallback._cached_road(object(), 10.0, 20.0, block_size=50.0)
        assert fallback._cached_gradient(elevations, 1, 1.0, 10.0, 20.0) == (0.25, -0.5)
        assert fallback._cached_oriented(
            elevations, 1, 1.0, 10.0, 20.0, 20.0, 35.0, 90.0
        ) == (7.0, 8.0, 9.0)
    finally:
        fallback._CONTEXT.reset(token)


def test_broad_context_suppresses_only_overlapping_regular_primary_pool(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        fallback,
        "_BASE_PRIMARY_BUILD_CONTEXT",
        lambda *args, **kwargs: sentinel,
    )

    assert fallback._primary_build_context_bridge() is sentinel

    context = fallback._FallbackContext(elevations=())
    token = fallback._CONTEXT.set(context)
    try:
        assert fallback._primary_build_context_bridge() is None
    finally:
        fallback._CONTEXT.reset(token)
