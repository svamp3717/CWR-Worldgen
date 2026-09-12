from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_final_count_policy as policy
from cwr_worldgen import bridge_render_policy as bridge
from cwr_worldgen.model import WorldObject


@dataclass(frozen=True)
class _Result:
    objects: tuple[WorldObject, ...]


@dataclass(frozen=True)
class _CategorizedResult:
    objects: tuple[WorldObject, ...]
    bridge_objects: int
    model_usage: tuple[tuple[str, int], ...]


def _spec():
    return SimpleNamespace(
        cells=64,
        cell_size=50.0,
        sea_level=0.0,
        world_size=3200.0,
        procedural_bridges=False,
        bridge_module_length=bridge._STOCK_MODULE_SPACING_METRES,
    )


def _ten_module_component() -> _Result:
    step = float(bridge._STOCK_MODULE_SPACING_METRES)
    # Start at half a module so the nominal component corridor is 0..10*step.
    # IDs deliberately alternate between the first and second half of the
    # physical chain, matching the two-batch pattern seen in wg_stocktest5.pbo.
    positions = [index * step + step * 0.5 for index in range(10)]
    id_order = (100, 105, 101, 106, 102, 107, 103, 108, 104, 109)
    objects = tuple(
        WorldObject(
            id_order[index],
            bridge._osm.NOGOVA_BRIDGE_MODEL,
            x,
            0.0,
            200.0,
            90.0,
            0.0,
        )
        for index, x in enumerate(positions)
    )
    return _Result(objects)


def _terrain(_elevations, _cells, _cell_size, x, _z):
    # About 146 m of actual water inside a roughly 502 m emitted bridge, matching
    # the failure measured in wg_stocktest5.pbo. Three 50.190 m modules suffice.
    return -2.0 if 174.0 <= float(x) <= 320.0 else 6.0


def test_final_physical_component_count_follows_wet_plan() -> None:
    spec = _spec()
    result = _ten_module_component()
    elevations = (0.0,) * (spec.cells * spec.cells)

    with patch.object(bridge._osm, "_sample_elevation", side_effect=_terrain):
        reconciled = policy._reconcile_stock_bridge_components(
            result,
            elevations,
            spec,
        )

    stock = [obj for obj in reconciled.objects if bridge._is_stock_bridge(obj)]
    assert len(stock) == 3

    centres = sorted(bridge._visible_deck_point(obj, 0.0)[0] for obj in stock)
    assert centres[0] > 170.0
    assert centres[-1] < 325.0
    assert all(
        abs((right - left) - bridge._STOCK_MODULE_SPACING_METRES) < 1.0e-6
        for left, right in zip(centres, centres[1:])
    )


def test_trimming_bridge_objects_updates_parallel_category_metadata() -> None:
    spec = _spec()
    objects = _ten_module_component().objects
    model = bridge._osm.NOGOVA_BRIDGE_MODEL
    result = _CategorizedResult(
        objects=objects,
        bridge_objects=10,
        model_usage=((model, 10), (r"data3d\strom.p3d", 4)),
    )
    elevations = (0.0,) * (spec.cells * spec.cells)

    with patch.object(bridge._osm, "_sample_elevation", side_effect=_terrain):
        reconciled = policy._reconcile_stock_bridge_components(
            result,
            elevations,
            spec,
        )

    stock = [obj for obj in reconciled.objects if bridge._is_stock_bridge(obj)]
    assert len(stock) == 3
    assert reconciled.bridge_objects == 3
    usage = dict(reconciled.model_usage)
    assert usage[model] == 3
    assert usage[r"data3d\strom.p3d"] == 4
    assert reconciled.bridge_objects == len(stock)


def test_final_count_guard_runs_before_existing_seam_anchor() -> None:
    spec = _spec()
    result = _ten_module_component()
    elevations = (0.0,) * (spec.cells * spec.cells)
    raster = SimpleNamespace(water=(False,) * (spec.cells * spec.cells))

    assert policy._INSTALLED
    assert policy._ORIGINAL_ANCHOR is not None

    with patch.object(bridge._osm, "_sample_elevation", side_effect=_terrain):
        anchored = bridge._anchor_stock_bridge_chains(
            result,
            raster,
            elevations,
            spec,
        )

    stock = [obj for obj in anchored.objects if bridge._is_stock_bridge(obj)]
    assert len(stock) == 3
    component = bridge._bridge_components(stock)
    assert len(component) == 1
    ordered_state = bridge._ordered_component(stock, component[0])
    assert ordered_state is not None
    ordered, _axis, _midpoint = ordered_state
    horizontal_error, vertical_error = bridge._component_seam_errors(stock, ordered)
    assert horizontal_error < 1.0e-6
    assert vertical_error < 1.0e-6


def test_no_final_wet_plan_does_not_destroy_source_fallback_component() -> None:
    spec = _spec()
    result = _ten_module_component()
    elevations = (0.0,) * (spec.cells * spec.cells)

    with patch.object(bridge, "stock_bridge_span_plan", return_value=None):
        reconciled = policy._reconcile_stock_bridge_components(
            result,
            elevations,
            spec,
        )

    assert reconciled == result
