from types import SimpleNamespace

import numpy as np

from cwr_worldgen import playability
from cwr_worldgen import road_chain_parallel_policy
from cwr_worldgen import road_finish_parallel_policy as perf


def _terrain(cells: int = 6) -> tuple[float, ...]:
    return tuple(
        2.0 + (index // cells) * 0.55 + (index % cells) * 0.17
        for index in range(cells * cells)
    )


def test_vector_endpoint_sampling_matches_scalar_bilinear_sampling() -> None:
    cells = 6
    cell_size = 25.0
    elevations = _terrain(cells)
    points = (
        (0.0, 0.0),
        (12.5, 8.0),
        (25.0, 25.0),
        (61.25, 77.75),
        (124.9, 110.5),
    )
    actual = perf._vector_sample_points(elevations, cells, cell_size, points)
    for point in points:
        expected = perf._BASE_SAMPLE(
            elevations, cells, cell_size, point[0], point[1]
        )
        assert np.isclose(actual[point], expected, rtol=0.0, atol=1.0e-12)


def test_chain_finalize_cache_preserves_stock_object_transform_and_axis() -> None:
    cells = 2
    cell_size = 25.0
    elevations = (3.0, 4.0, 3.0, 4.0)
    spec = SimpleNamespace(cells=cells, cell_size=cell_size)
    run = ((0.0, 0.0), (25.0, 0.0), (50.0, 5.0))
    piece = next(
        value
        for value in playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
        if value.nominal_length == 25
    )
    start = (0.0, 0.0)
    end = (25.0, 0.0)
    job = perf._FinalizeJob(
        0,
        run,
        ((piece.model_path, piece.length_metres, piece.nominal_length, start, end, 3.0, 4.0),),
    )
    result = perf._finalize_job(job)
    context = perf._Context(elevations, cells, cell_size)
    context.samples.update({start: 3.0, end: 4.0})
    perf._store_result(context, result)

    expected_model = perf._BASE_CURVED_MODEL(piece.model_path, run, start, end)
    expected = perf._BASE_ROAD_OBJECT(
        17,
        expected_model,
        start,
        end,
        elevations,
        spec,
        vertical_offset=0.035,
    )
    token = perf._CONTEXT.set(context)
    try:
        actual_model = perf._cached_curved_model(piece.model_path, run, start, end)
        actual = perf._cached_road_object(
            17,
            actual_model,
            start,
            end,
            elevations,
            spec,
            vertical_offset=0.035,
        )
        actual_axis = perf._cached_model_axis(actual, piece.length_metres)
    finally:
        perf._CONTEXT.reset(token)

    expected_axis = perf._BASE_MODEL_AXIS(expected, piece.length_metres)
    assert actual_model == expected_model
    assert actual.model_path == expected.model_path
    assert actual.object_id == expected.object_id
    assert np.isclose(actual.x, expected.x, rtol=0.0, atol=1.0e-12)
    assert np.isclose(actual.y, expected.y, rtol=0.0, atol=1.0e-12)
    assert np.isclose(actual.z, expected.z, rtol=0.0, atol=1.0e-12)
    assert np.isclose(actual.heading_degrees, expected.heading_degrees, rtol=0.0, atol=1.0e-12)
    assert np.isclose(actual.pitch_degrees, expected.pitch_degrees, rtol=0.0, atol=1.0e-12)
    assert actual_axis == expected_axis


def test_road_finish_parallel_policy_is_live() -> None:
    assert road_chain_parallel_policy._execute_run_jobs is perf._execute
    assert playability._fit_stock_piece_road_objects is perf._fit
    assert playability._sample_elevation is perf._cached_sample
    assert playability._curved_gravel_model_for_run is perf._cached_curved_model
    assert playability._road_object_on_slope is perf._cached_road_object
    assert playability._model_axis is perf._cached_model_axis
