from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_render_policy as bridge_render
from cwr_worldgen.model import WorldObject


@dataclass(frozen=True)
class _Result:
    objects: tuple[WorldObject, ...]
    bridge_objects: int
    model_usage: tuple[tuple[str, int], ...]


def _spec(*, module_length: float = 30.0):
    return SimpleNamespace(
        procedural_bridges=True,
        bridge_module_length=module_length,
    )


def _tinybjorsund_bridge() -> WorldObject:
    return WorldObject(
        3319,
        r"wg_a_tinybjorsund\i\br_single_w084_l3213.p3d",
        738.4467,
        8.1179,
        765.2676,
        56.1145,
    )


def _axis_offset(obj: WorldObject, origin: WorldObject) -> float:
    heading = math.radians(origin.heading_degrees)
    dx = obj.x - origin.x
    dz = obj.z - origin.z
    return dx * math.sin(heading) + dz * math.cos(heading)


def test_tinybjorsund_bridge_uses_contiguous_stock_cwa_modules() -> None:
    original = _tinybjorsund_bridge()
    modules, next_id = bridge_render._split_bridge_object(
        original, _spec(), next_object_id=4000
    )

    assert len(modules) == 11
    assert next_id == 4010
    assert all(
        obj.model_path.casefold() == bridge_render._STOCK_BRIDGE_MODEL.casefold()
        for obj in modules
    )
    assert all(abs(obj.y - original.y) < 1e-9 for obj in modules)
    assert all(abs(obj.pitch_degrees - original.pitch_degrees) < 1e-9 for obj in modules)

    offsets = tuple(_axis_offset(obj, original) for obj in modules)
    expected_step = 321.3 / 11.0
    for first, second in zip(offsets, offsets[1:]):
        assert abs((second - first) - expected_step) < 1e-6
        assert second - first <= bridge_render._STOCK_BRIDGE_LENGTH_METRES + 1e-9

    half_stock = bridge_render._STOCK_BRIDGE_LENGTH_METRES * 0.5
    assert offsets[0] - half_stock <= -321.3 * 0.5 + 1e-6
    assert offsets[-1] + half_stock >= 321.3 * 0.5 - 1e-6


def test_stock_bridge_rewrite_updates_count_and_removes_generated_asset_usage() -> None:
    bridge = _tinybjorsund_bridge()
    prop = WorldObject(3320, r"o\misc\lavicka_1.p3d", 10.0, 0.0, 10.0, 0.0)
    result = _Result(
        objects=(bridge, prop),
        bridge_objects=1,
        model_usage=((bridge.model_path, 1), (prop.model_path, 1)),
    )

    rewritten = bridge_render._modularize_result(result, _spec())
    assert rewritten.bridge_objects == 11
    assert len(rewritten.objects) == 12
    assert rewritten.objects[-1] == prop

    usage = Counter(dict(rewritten.model_usage))
    assert bridge.model_path not in usage
    assert usage[prop.model_path] == 1
    assert usage[bridge_render._STOCK_BRIDGE_MODEL] == 11
    assert not any("\\i\\br_" in path.casefold() for path in usage)


def test_cached_nonroad_result_is_rewritten_without_cache_clear() -> None:
    bridge = _tinybjorsund_bridge()
    cached = _Result(
        objects=(bridge,),
        bridge_objects=1,
        model_usage=((bridge.model_path, 1),),
    )
    cached_tuple = (cached, object(), True, "cache-key", "cache-path")

    with patch.object(
        bridge_render, "_ORIGINAL_LOAD_NONROAD_OBJECTS", return_value=cached_tuple
    ):
        value = bridge_render._load_nonroad_objects(
            None,
            None,
            None,
            (),
            _spec(),
            starting_object_id=1,
        )

    assert value[2:] == cached_tuple[2:]
    assert value[0].bridge_objects == 11
    assert len(value[0].objects) == 11
    assert all(
        obj.model_path.casefold() == bridge_render._STOCK_BRIDGE_MODEL.casefold()
        for obj in value[0].objects
    )


def test_short_procedural_bridge_becomes_one_stock_module() -> None:
    short = WorldObject(
        1,
        r"world\i\br_single_w070_l250.p3d",
        100.0,
        4.0,
        100.0,
        90.0,
    )
    pieces, next_id = bridge_render._split_bridge_object(
        short, _spec(), next_object_id=2
    )

    assert len(pieces) == 1
    assert pieces[0].object_id == short.object_id
    assert pieces[0].model_path.casefold() == bridge_render._STOCK_BRIDGE_MODEL.casefold()
    assert (pieces[0].x, pieces[0].y, pieces[0].z) == (short.x, short.y, short.z)
    assert next_id == 2


def test_pitched_stock_modules_follow_the_solved_deck_line_not_terrain() -> None:
    bridge = WorldObject(
        10,
        r"world\i\br_single_w084_l600.p3d",
        200.0,
        12.0,
        300.0,
        0.0,
        5.0,
    )
    pieces, _next_id = bridge_render._split_bridge_object(
        bridge, _spec(), next_object_id=20
    )

    assert len(pieces) == 2
    expected_delta = math.tan(math.radians(5.0)) * 15.0
    assert abs(pieces[0].y - (bridge.y - expected_delta)) < 1e-6
    assert abs(pieces[1].y - (bridge.y + expected_delta)) < 1e-6
    assert all(obj.pitch_degrees == 5.0 for obj in pieces)
