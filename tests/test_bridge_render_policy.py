from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_render_policy as bridge_render
from cwr_worldgen.model import WorldObject


@dataclass(frozen=True)
class _Spec:
    procedural_bridges: bool = True
    bridge_module_length: float = 30.0
    marker: str = "kept"
    cells: int = 64
    cell_size: float = 10.0
    sea_level: float = 0.0

    @property
    def world_size(self) -> float:
        return self.cells * self.cell_size


@dataclass(frozen=True)
class _Result:
    objects: tuple[WorldObject, ...]


def _dry_raster(spec: _Spec):
    return SimpleNamespace(water=(False,) * (spec.cells * spec.cells))


def test_stock_bridge_spec_preserves_user_spec_and_overrides_only_bridge_mode() -> None:
    original = _Spec()
    rewritten = bridge_render._stock_bridge_spec(original)

    assert original.procedural_bridges is True
    assert rewritten is not original
    assert rewritten.procedural_bridges is False
    assert rewritten.bridge_module_length == original.bridge_module_length
    assert rewritten.marker == original.marker


def test_already_stock_bridge_spec_is_reused() -> None:
    original = _Spec(procedural_bridges=False)
    assert bridge_render._stock_bridge_spec(original) is original


def test_non_dataclass_compatibility_proxy_reads_all_other_fields_from_base() -> None:
    original = SimpleNamespace(
        procedural_bridges=True,
        bridge_module_length=24.0,
        cells=256,
        cell_size=25.0,
        world_size=6400.0,
        sea_level=0.0,
    )
    rewritten = bridge_render._stock_bridge_spec(original)

    assert rewritten.procedural_bridges is False
    assert rewritten.bridge_module_length == 24.0
    assert rewritten.cells == 256
    assert rewritten.sea_level == 0.0
    assert original.procedural_bridges is True


def test_direct_generation_calls_core_with_stock_bridge_mode() -> None:
    original = _Spec()
    sentinel = object()
    observed = {}

    def core(dataset, projection, raster, elevations, spec, *args, **kwargs):
        observed["spec"] = spec
        observed["args"] = args
        observed["kwargs"] = kwargs
        return sentinel

    with patch.object(bridge_render, "_ORIGINAL_GENERATE_WORLD_OBJECTS", core):
        result = bridge_render._generate_world_objects(
            "dataset",
            "projection",
            "raster",
            (1.0, 2.0),
            original,
            include_roads=False,
            starting_object_id=7,
        )

    assert result is sentinel
    assert observed["spec"].procedural_bridges is False
    assert observed["spec"].marker == "kept"
    assert observed["kwargs"] == {"include_roads": False, "starting_object_id": 7}
    assert original.procedural_bridges is True


def test_nonroad_cache_receives_stock_bridge_spec_positionally() -> None:
    original = _Spec()
    sentinel = (object(), object(), False, "new-cache-key", "cache-path")
    observed = {}

    def loader(dataset, projection, raster, elevations, spec, *args, **kwargs):
        observed["spec"] = spec
        observed["kwargs"] = kwargs
        return sentinel

    with patch.object(bridge_render, "_ORIGINAL_LOAD_NONROAD_OBJECTS", loader):
        result = bridge_render._load_nonroad_objects(
            "dataset",
            "projection",
            "raster",
            (1.0, 2.0),
            original,
            starting_object_id=10,
            road_fingerprint="roads",
        )

    assert result is sentinel
    assert observed["spec"].procedural_bridges is False
    assert observed["spec"].marker == "kept"
    assert observed["kwargs"]["road_fingerprint"] == "roads"
    assert original.procedural_bridges is True


def test_nonroad_cache_receives_stock_bridge_spec_when_named() -> None:
    original = _Spec()
    sentinel = (object(), object(), True, "stock-cache-key", "cache-path")
    observed = {}

    def loader(*args, **kwargs):
        observed.update(kwargs)
        return sentinel

    with patch.object(bridge_render, "_ORIGINAL_LOAD_NONROAD_OBJECTS", loader):
        result = bridge_render._load_nonroad_objects(
            dataset="dataset",
            projection="projection",
            raster="raster",
            elevations=(1.0, 2.0),
            spec=original,
            starting_object_id=10,
            road_fingerprint="roads",
        )

    assert result is sentinel
    assert observed["spec"].procedural_bridges is False
    assert observed["spec"].marker == "kept"
    assert original.procedural_bridges is True


def test_core_stock_branch_uses_verified_nogova_bridge_asset() -> None:
    assert bridge_render._osm.NOGOVA_BRIDGE_MODEL.casefold() == r"o\hous\most_stred30.p3d".casefold()
    assert bridge_render._osm.NOGOVA_BRIDGE_MODULE_LENGTH_METRES == 30.0


def test_stock_bridge_chain_is_anchored_to_both_bank_elevations() -> None:
    spec = _Spec(procedural_bridges=False)
    raster = _dry_raster(spec)
    modules = tuple(
        WorldObject(
            100 + index,
            bridge_render._osm.NOGOVA_BRIDGE_MODEL,
            100.0,
            20.0,
            15.0 + index * 30.0,
            0.0,
            0.0,
        )
        for index in range(4)
    )
    result = _Result(modules)

    def terrain(_elevations, _cells, _cell_size, _x, z):
        if z <= 1.0:
            return 3.0
        if z >= 119.0:
            return 7.0
        return -5.0

    with patch.object(bridge_render._osm, "_sample_elevation", side_effect=terrain):
        anchored = bridge_render._anchor_stock_bridge_chains(
            result,
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )

    assert anchored is not result
    first = anchored.objects[0]
    last = anchored.objects[-1]
    expected_start = 3.0 + bridge_render._osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES
    expected_end = 7.0 + bridge_render._osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES

    first_half_rise = math.tan(math.radians(first.pitch_degrees)) * 15.0
    last_half_rise = math.tan(math.radians(last.pitch_degrees)) * 15.0
    assert abs((first.y - first_half_rise) - expected_start) < 1e-6
    assert abs((last.y + last_half_rise) - expected_end) < 1e-6

    # Adjacent 30 m modules must meet at exactly the same deck elevation.
    for left, right in zip(anchored.objects, anchored.objects[1:]):
        left_end = left.y + math.tan(math.radians(left.pitch_degrees)) * 15.0
        right_start = right.y - math.tan(math.radians(right.pitch_degrees)) * 15.0
        assert abs(left_end - right_start) < 1e-6


def test_middle_water_depth_does_not_drag_stock_bridge_modules_down() -> None:
    spec = _Spec(procedural_bridges=False)
    raster = _dry_raster(spec)
    modules = tuple(
        WorldObject(
            200 + index,
            bridge_render._osm.NOGOVA_BRIDGE_MODEL,
            200.0,
            30.0,
            15.0 + index * 30.0,
            0.0,
            0.0,
        )
        for index in range(3)
    )

    sampled_z = []

    def terrain(_elevations, _cells, _cell_size, _x, z):
        sampled_z.append(z)
        return 4.0 if z < 45.0 else 5.0

    with patch.object(bridge_render._osm, "_sample_elevation", side_effect=terrain):
        anchored = bridge_render._anchor_stock_bridge_chains(
            _Result(modules),
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )

    # Only the two approaches are sampled for vertical anchoring. The seabed or
    # terrain under middle modules is intentionally irrelevant to the deck line.
    assert len(sampled_z) == 2
    assert min(obj.y for obj in anchored.objects) > 4.0


def test_cached_stock_bridge_result_is_reanchored_without_cache_clear() -> None:
    spec = _Spec()
    raster = _dry_raster(spec)
    cached_result = _Result((
        WorldObject(1, bridge_render._osm.NOGOVA_BRIDGE_MODEL, 100.0, 50.0, 15.0, 0.0),
        WorldObject(2, bridge_render._osm.NOGOVA_BRIDGE_MODEL, 100.0, 50.0, 45.0, 0.0),
    ))
    cached = (cached_result, object(), True, "cache", "path")

    with (
        patch.object(bridge_render, "_ORIGINAL_LOAD_NONROAD_OBJECTS", return_value=cached),
        patch.object(bridge_render._osm, "_sample_elevation", return_value=6.0),
    ):
        value = bridge_render._load_nonroad_objects(
            "dataset",
            "projection",
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
            starting_object_id=1,
        )

    assert value[1:] == cached[1:]
    assert value[0] is not cached_result
    assert all(obj.y < 10.0 for obj in value[0].objects)
