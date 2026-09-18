from unittest.mock import patch

from shapely.geometry import LineString, Point

from cwr_worldgen import road_constraint_pathology_policy as pathology
from cwr_worldgen import road_constraint_performance_policy as perf
from cwr_worldgen import terrain_solver as terrain


def test_analytic_complex_corridor_matches_full_line_distance_and_projection() -> None:
    line = LineString(
        [
            (0.0, 0.0),
            (1200.0, 900.0),
            (2400.0, 100.0),
            (3600.0, 1400.0),
        ]
    )
    radius = 40.0
    cells = 200
    cell_size = 25.0

    indices, distances, projections, local_instances = pathology._analytic_corridor_values(
        line,
        radius,
        cells,
        cell_size,
    )

    assert indices.size
    assert local_instances >= indices.size
    assert indices.size == distances.size == projections.size

    bounds = (
        line.bounds[0] - radius,
        line.bounds[1] - radius,
        line.bounds[2] + radius,
        line.bounds[3] + radius,
    )
    expected = tuple(
        index
        for index in perf._ORIGINAL_CANDIDATE_CELLS(bounds, cells, cell_size)
        if line.distance(Point(terrain._cell_center(index, cells, cell_size)))
        <= radius + 1.0e-9
    )
    assert tuple(int(index) for index in indices) == expected

    for index, distance, projection in zip(indices, distances, projections):
        point = Point(terrain._cell_center(int(index), cells, cell_size))
        assert abs(float(distance) - line.distance(point)) < 1.0e-8
        assert abs(float(projection) - line.project(point)) < 1.0e-8


def test_pathological_corridor_preloads_distance_and_projection_batches() -> None:
    line = LineString([(0.0, 0.0), (5000.0, 5000.0)])
    fast_line = perf._FastLine(line, {"highway": "primary"})
    radius = 24.0
    cells = 256
    cell_size = 25.0
    corridor = fast_line.buffer(radius, cap_style=2, join_style=2)

    indices = tuple(
        pathology._pathological_candidate_cells(
            corridor.bounds,
            cells,
            cell_size,
        )
    )
    assert indices
    batch = fast_line._batch
    assert batch is not None
    assert batch.distances is not None
    assert batch.projections is not None

    index = indices[len(indices) // 2]
    point = Point(terrain._cell_center(index, cells, cell_size))
    distance_values = batch.distances
    projection_values = batch.projections
    assert abs(fast_line.distance(point) - line.distance(point)) < 1.0e-8
    assert abs(fast_line.project(point) - line.project(point)) < 1.0e-8
    assert batch.distances is distance_values
    assert batch.projections is projection_values


def test_watchdog_rearms_on_stalled_road_checkpoints() -> None:
    stages: list[str] = []

    def fake_solve(*args, **kwargs):
        progress = kwargs["progress_callback"]
        progress(35, "Applying road terrain constraints 45,708/56,256")
        progress(35, "Applying road terrain constraints 49,224/56,256")
        progress(44, "Applying watercourse constraints 1/1")
        return "done"

    with patch.object(pathology, "_ORIGINAL_SOLVE_TERRAIN", side_effect=fake_solve), patch.object(
        pathology, "_arm_watchdog"
    ) as arm, patch.object(pathology, "_cancel_watchdog") as cancel:
        result = pathology._solve_with_road_watchdog(
            progress_callback=lambda _percent, stage: stages.append(stage)
        )

    assert result == "done"
    assert arm.call_count == 2
    assert cancel.call_count >= 2
    assert stages[-1] == "Applying watercourse constraints 1/1"
