from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_render_policy as bridge_render


@dataclass(frozen=True)
class _Spec:
    procedural_bridges: bool = True
    bridge_module_length: float = 30.0
    marker: str = "kept"


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
