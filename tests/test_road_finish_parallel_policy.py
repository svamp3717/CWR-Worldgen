from types import SimpleNamespace

import numpy as np

from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
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


def test_vector_prepare_preserves_stock_object_transform_and_axis() -> None:
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
    plan = SimpleNamespace(
        order=0,
        run=run,
        fitted_pieces=((piece, start, end),),
    )
    context = perf._Context(elevations, cells, cell_size)
    assert perf._vector_prepare((plan,), context) == 1

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
    assert np.allclose(actual_axis, expected_axis, rtol=0.0, atol=1.0e-12)


def test_cached_generated_paved_transform_matches_stock_surface_plane() -> None:
    elevations = (0.0,) * 4
    spec = SimpleNamespace(cells=2, cell_size=25.0)
    start = (0.0, 0.0)
    end = (0.0, 6.25)
    requested_surface_height = 0.035
    models = (
        infrastructure.paved_fallback_model_path(
            "height_world", 9.10, 6.25, 0.0
        ),
        infrastructure.paved_junction_model_path(
            "height_world", 9.10, 9.10, 260.0
        ),
    )

    context = perf._Context(elevations, 2, 25.0)
    context.geometry[perf._geometry_key(start, end)] = (
        0.0,
        0.0,
        3.125,
        0.0,
        0.0,
    )
    token = perf._CONTEXT.set(context)
    try:
        for index, model in enumerate(models, start=1):
            obj = perf._cached_road_object(
                index,
                model,
                start,
                end,
                elevations,
                spec,
                vertical_offset=requested_surface_height,
            )
            assert np.isclose(
                obj.y + infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES,
                requested_surface_height,
                rtol=0.0,
                atol=1.0e-12,
            )
    finally:
        perf._CONTEXT.reset(token)


def test_vector_chain_diagnostics_match_scalar_axis_formula() -> None:
    cells = 4
    elevations = (2.0,) * (cells * cells)
    piece = next(
        value
        for value in playability.road_model_variants(r"data3d\sil25.p3d", 25.0)
        if value.nominal_length == 25
    )
    first = (piece, (0.0, 0.0), (25.0, 0.0))
    second = (piece, (24.0, 0.0), (49.0, 0.0))
    plan = SimpleNamespace(
        order=0,
        run=((0.0, 0.0), (49.0, 0.0)),
        fitted_pieces=(first, second),
    )
    context = perf._Context(elevations, cells, 25.0)
    perf._vector_prepare((plan,), context)

    objects = []
    for object_id, (_piece, start, end) in enumerate(plan.fitted_pieces, start=1):
        objects.append(
            perf._BASE_ROAD_OBJECT(
                object_id,
                piece.model_path,
                start,
                end,
                elevations,
                SimpleNamespace(cells=cells, cell_size=25.0),
                vertical_offset=0.035,
            )
        )
    previous_axis = perf._BASE_MODEL_AXIS(objects[0], piece.length_metres)
    current_axis = perf._BASE_MODEL_AXIS(objects[1], piece.length_metres)
    offset = (
        current_axis[0][0] - previous_axis[1][0],
        current_axis[0][1] - previous_axis[1][1],
    )
    expected_gap = float(np.hypot(*offset))
    angle = np.radians(objects[0].heading_degrees)
    direction = (np.sin(angle), np.cos(angle))
    lateral = abs(direction[0] * offset[1] - direction[1] * offset[0])
    longitudinal = direction[0] * offset[0] + direction[1] * offset[1]
    expected_overlap = -longitudinal if lateral <= 0.10 and longitudinal < 0.0 else 0.0

    assert np.isclose(context.maximum_chain_gap, expected_gap, rtol=0.0, atol=1.0e-12)
    assert np.isclose(context.maximum_model_overlap, expected_overlap, rtol=0.0, atol=1.0e-12)


def test_diagnostic_zip_only_skips_worldobject_chain_adjacency() -> None:
    first = playability.WorldObject(1, "a", 0.0, 0.0, 0.0, 0.0)
    second = playability.WorldObject(2, "b", 0.0, 0.0, 1.0, 0.0)
    left = [(first, 25.0), (second, 25.0)]
    right = left[1:]
    context = perf._Context((0.0,), 1, 1.0, prepared=True)
    token = perf._CONTEXT.set(context)
    try:
        assert list(perf._diagnostic_zip(left, right)) == []
        assert list(perf._diagnostic_zip([1, 2], [3])) == [(1, 3)]
    finally:
        perf._CONTEXT.reset(token)


def test_road_finish_vector_policy_is_live() -> None:
    assert road_chain_parallel_policy._execute_run_jobs is perf._execute
    assert road_chain_parallel_policy.zip is perf._diagnostic_zip
    assert playability._fit_stock_piece_road_objects is perf._fit
    assert playability._sample_elevation is perf._BASE_SAMPLE
    assert playability._curved_gravel_model_for_run is perf._cached_curved_model
    assert playability._road_object_on_slope is perf._cached_road_object
    assert playability._model_axis is perf._cached_model_axis
