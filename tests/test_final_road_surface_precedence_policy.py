from types import SimpleNamespace
import math

from cwr_worldgen import final_road_surface_precedence_policy as precedence
from cwr_worldgen import playability


def _spec():
    return SimpleNamespace(
        cells=32,
        cell_size=10.0,
        road_segment_length=25.0,
    )


def _report(*objects):
    return playability.RoadFitReport(
        objects=tuple(objects),
        chain_count=2,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )


def _endpoint_height(obj, length, endpoint_index):
    delta = math.sin(math.radians(obj.pitch_degrees)) * length * 0.5
    return obj.y - delta if endpoint_index == 0 else obj.y + delta


def test_final_paved_surface_lowers_only_touching_dirt_endpoint() -> None:
    spec = _spec()
    elevations = (0.0,) * (spec.cells * spec.cells)
    paved = playability._road_object_on_slope(
        1,
        r"o\road\sil6.p3d",
        (96.875, 100.0),
        (103.125, 100.0),
        elevations,
        spec,
        vertical_offset=playability._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )
    dirt = playability._road_object_on_slope(
        2,
        r"o\road\ces6.p3d",
        (100.0, 93.75),
        (100.0, 100.0),
        elevations,
        spec,
        vertical_offset=playability._STOCK_DIRT_VERTICAL_OFFSET_METRES,
    )

    result = precedence.enforce_final_paved_over_dirt(
        _report(paved, dirt),
        elevations,
        spec,
    )
    rebuilt = next(obj for obj in result.objects if obj.object_id == 2)
    length = playability.stock_road_piece_length_metres(
        rebuilt.model_path, 6, spec.road_segment_length
    )
    start, end = playability._model_axis(rebuilt, length)
    # The endpoint nearest the paved road is deeply buried; the far endpoint
    # remains on the normal dirt plane so the previous dirt slab still joins.
    endpoint_heights = (
        _endpoint_height(rebuilt, length, 0),
        _endpoint_height(rebuilt, length, 1),
    )
    paved_point = (100.0, 100.0)
    near_index = 0 if math.dist(start, paved_point) < math.dist(end, paved_point) else 1
    far_index = 1 - near_index
    assert math.isclose(
        endpoint_heights[near_index],
        playability._MIXED_DIRT_UNDERLAY_VERTICAL_OFFSET_METRES,
        abs_tol=1.0e-6,
    )
    assert math.isclose(
        endpoint_heights[far_index],
        playability._STOCK_DIRT_VERTICAL_OFFSET_METRES,
        abs_tol=1.0e-6,
    )


def test_grade_separated_paved_road_does_not_sink_ground_dirt() -> None:
    spec = _spec()
    elevations = (0.0,) * (spec.cells * spec.cells)
    bridge_like_paved = playability.WorldObject(
        1,
        r"o\road\sil6.p3d",
        100.0,
        10.0,
        100.0,
        90.0,
        0.0,
    )
    dirt = playability._road_object_on_slope(
        2,
        r"o\road\ces6.p3d",
        (100.0, 93.75),
        (100.0, 100.0),
        elevations,
        spec,
        vertical_offset=playability._STOCK_DIRT_VERTICAL_OFFSET_METRES,
    )
    result = precedence.enforce_final_paved_over_dirt(
        _report(bridge_like_paved, dirt),
        elevations,
        spec,
    )
    assert next(obj for obj in result.objects if obj.object_id == 2) == dirt
