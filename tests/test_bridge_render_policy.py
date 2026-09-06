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


def _module_length_dm(model_path: str) -> int:
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
    return int(filename.rsplit("_l", 1)[1][:-4])


def _axis_endpoint(obj: WorldObject, *, forward: bool) -> tuple[float, float]:
    length = _module_length_dm(obj.model_path) / 10.0
    sign = 1.0 if forward else -1.0
    heading = math.radians(obj.heading_degrees)
    return (
        obj.x + math.sin(heading) * length * 0.5 * sign,
        obj.z + math.cos(heading) * length * 0.5 * sign,
    )


def test_tinybjorsund_long_bridge_is_split_into_cwa_safe_modules() -> None:
    original = _tinybjorsund_bridge()
    modules, next_id = bridge_render._split_bridge_object(
        original, _spec(), next_object_id=4000
    )

    lengths = tuple(_module_length_dm(obj.model_path) for obj in modules)
    assert len(modules) == 11
    assert sum(lengths) == 3213
    assert max(lengths) <= 300
    assert modules[0].model_path.endswith(r"\br_start_w084_l293.p3d")
    assert modules[-1].model_path.endswith(r"\br_end_w084_l292.p3d")
    assert all(
        r"\br_middle_w084_" in obj.model_path
        for obj in modules[1:-1]
    )
    assert next_id == 4010

    total_length = 321.3
    heading = math.radians(original.heading_degrees)
    expected_start = (
        original.x - math.sin(heading) * total_length * 0.5,
        original.z - math.cos(heading) * total_length * 0.5,
    )
    expected_end = (
        original.x + math.sin(heading) * total_length * 0.5,
        original.z + math.cos(heading) * total_length * 0.5,
    )
    assert math.dist(_axis_endpoint(modules[0], forward=False), expected_start) < 1e-6
    assert math.dist(_axis_endpoint(modules[-1], forward=True), expected_end) < 1e-6
    for first, second in zip(modules, modules[1:]):
        assert math.dist(
            _axis_endpoint(first, forward=True),
            _axis_endpoint(second, forward=False),
        ) < 1e-6


def test_modularization_updates_bridge_count_and_asset_usage() -> None:
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
    assert sum(
        count for path, count in usage.items()
        if "\\i\\br_" in path.casefold()
    ) == 11
    assert any("br_start_w084_l293.p3d" in path for path in usage)
    assert any("br_end_w084_l292.p3d" in path for path in usage)


def test_cached_nonroad_result_is_modularized_without_cache_clear() -> None:
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


def test_short_procedural_bridge_stays_single() -> None:
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
    assert pieces == (short,)
    assert next_id == 2
