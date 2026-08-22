from __future__ import annotations

import struct
from pathlib import Path

from cwr_worldgen.cache import streaming_hash
from cwr_worldgen.generator import _assemble_world_objects
from cwr_worldgen.model import WorldObject
from cwr_worldgen.osm import (
    BboxProjection, CompactOrientedRectangle, ObjectGenerationResult,
    _compact_support_polygon, _oriented_rectangle,
)
from cwr_worldgen.procedural_buildings import (
    _polygon_with_footprint,
    _simple_rectangle_footprint,
)
from cwr_worldgen.wrp import _height_grid_bytes, inspect_rvw4, quantize_height, write_rvw4


def _object(object_id: int, model: str) -> WorldObject:
    return WorldObject(object_id, model, float(object_id), 1.25, float(object_id * 2), 90.0)


def test_simple_rectangle_fast_path_matches_shapely_dimensions_and_centroid() -> None:
    points = ((10.0, 20.0), (22.0, 20.0), (22.0, 38.0), (10.0, 38.0))
    fast = _simple_rectangle_footprint(points)
    assert fast is not None
    footprint, centre_x, centre_z = fast
    polygon, reference = _polygon_with_footprint(points)
    assert footprint.width_m == reference.width_m
    assert footprint.length_m == reference.length_m
    assert centre_x == float(polygon.centroid.x)
    assert centre_z == float(polygon.centroid.y)

    # Anything genuinely irregular remains on the mature Shapely/native path.
    assert _simple_rectangle_footprint(((0.0, 0.0), (12.0, 0.0), (10.0, 8.0), (0.0, 8.0))) is None


def test_vectorized_height_bytes_match_scalar_rvw4_quantization() -> None:
    values = tuple(index * 0.0125 for index in range(-80, 81))
    encoded = _height_grid_bytes(values, 0.05)
    unpacked = struct.unpack(f"<{len(values)}h", encoded)
    assert unpacked == tuple(quantize_height(value, 0.05) for value in values)


def test_large_world_ordering_can_skip_clone_and_writer_renumbers(tmp_path: Path) -> None:
    road = _object(1, r"o\road\sil25.p3d")
    buildings = (_object(2, r"world\g\house_a.p3d"), _object(3, r"world\g\house_b.p3d"))
    forests = (_object(4, r"data3d\les ctverec pruchozi_T1.p3d"),)
    rural = (_object(5, r"data3d\str borovice.p3d"),)
    semantic = _object(6, r"world\s\site.p3d")
    nonroads = ObjectGenerationResult(
        objects=buildings + forests + rural,
        road_objects=0,
        building_objects=2,
        forest_objects=1,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
        tree_row_objects=1,
    )

    ordered = _assemble_world_objects((road,), nonroads, (semantic,), renumber=False)
    assert ordered == (road, forests[0], rural[0], buildings[0], buildings[1], semantic)
    # The fast path reuses the original frozen objects instead of cloning them.
    assert ordered[1] is forests[0]
    assert ordered[3] is buildings[0]
    assert [obj.object_id for obj in ordered] == [1, 4, 5, 2, 3, 6]

    wrp = tmp_path / "ordered.wrp"
    write_rvw4(
        wrp,
        16,
        16,
        (0.0,) * 256,
        (0,) * 256,
        (r"world\data\g.paa",),
        ordered,
        height_scale=0.05,
        renumber_object_ids=True,
    )
    summary = inspect_rvw4(wrp, height_scale=0.05)
    assert summary.object_ids == (1, 2, 3, 4, 5, 6)
    assert summary.object_models == tuple(obj.model_path for obj in ordered)


def test_streaming_hash_consumes_large_style_iterables_once_and_deterministically() -> None:
    class OneShot:
        def __init__(self, values):
            self.values = values
            self.used = False

        def __iter__(self):
            assert not self.used, "streaming hash attempted a preliminary/counting pass"
            self.used = True
            yield from self.values

    first = OneShot(range(10_000))
    second = OneShot(range(10_000))
    assert streaming_hash("one-shot-v1", first) == streaming_hash("one-shot-v1", second)
    assert first.used and second.used


def test_compact_support_rectangle_preserves_polygon_coordinates() -> None:
    original = _oriented_rectangle(123.5, 456.25, 11.0, 17.5, 37.0)
    compact = _compact_support_polygon(original, 123.5, 456.25, 37.0)
    assert isinstance(compact, CompactOrientedRectangle)
    materialized = tuple(compact)
    for actual, expected in zip(materialized, original):
        assert abs(actual[0] - expected[0]) < 1.0e-12
        assert abs(actual[1] - expected[1]) < 1.0e-12
    assert abs(compact[-1][0] - original[-1][0]) < 1.0e-12
    assert len(compact[1:3]) == 2


def test_bbox_projection_hot_path_is_affine_and_round_trips() -> None:
    projection = BboxProjection.create((55.0, 12.0, 55.25, 12.5), 20_000.0)
    point = (55.1, 12.2)
    x, z = projection.to_world(point)
    assert abs(x - 8_000.0) < 1.0e-9
    assert abs(z - 8_000.0) < 1.0e-9
    latitude, longitude = projection.to_latlon((x, z))
    assert abs(latitude - point[0]) < 1.0e-12
    assert abs(longitude - point[1]) < 1.0e-12
