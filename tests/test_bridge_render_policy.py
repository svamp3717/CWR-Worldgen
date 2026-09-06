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


def _roadway_world_y(obj: WorldObject, local_z: float = 0.0) -> float:
    """Return central stock Roadway-face world Y for one local-z position."""

    pitch = math.radians(obj.pitch_degrees)
    return (
        obj.y
        + bridge_render._STOCK_ROADWAY_LOCAL_Y_METRES * math.cos(pitch)
        + local_z * math.sin(pitch)
    )


def _visible_deck_world_y(obj: WorldObject, local_z: float = 0.0) -> float:
    """Return stock bridge rendered road-surface world Y."""

    pitch = math.radians(obj.pitch_degrees)
    return (
        obj.y
        + bridge_render._STOCK_VISIBLE_DECK_LOCAL_Y_METRES * math.cos(pitch)
        + local_z * math.sin(pitch)
    )


def test_stock_bridge_spec_preserves_user_spec_and_overrides_only_bridge_mode() -> None:
    original = _Spec()
    rewritten = bridge_render._stock_bridge_spec(original)

    assert original.procedural_bridges is True
    assert rewritten is not original
    assert rewritten.procedural_bridges is False
    assert rewritten.bridge_module_length == bridge_render._STOCK_MODULE_SPACING_METRES
    assert rewritten.marker == original.marker


def test_already_stock_bridge_spec_is_reused() -> None:
    original = _Spec(procedural_bridges=False, bridge_module_length=50.0)
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
    assert rewritten.bridge_module_length == bridge_render._STOCK_MODULE_SPACING_METRES
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
    assert observed["spec"].bridge_module_length == bridge_render._STOCK_MODULE_SPACING_METRES
    assert observed["spec"].marker == "kept"
    assert observed["kwargs"] == {
        "include_roads": False,
        "starting_object_id": 7,
    }
    assert original.procedural_bridges is True


def test_nonroad_cache_receives_stock_bridge_spec_positionally() -> None:
    original = _Spec()
    sentinel = (object(), object(), False, "new-cache-key", "cache-path")
    observed = {}

    def loader(dataset, projection, raster, elevations, spec, *args, **kwargs):
        observed["spec"] = spec
        observed["kwargs"] = kwargs
        return sentinel

    with patch.object(
        bridge_render, "_ORIGINAL_LOAD_NONROAD_OBJECTS", loader
    ):
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
    assert observed["spec"].bridge_module_length == bridge_render._STOCK_MODULE_SPACING_METRES
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

    with patch.object(
        bridge_render, "_ORIGINAL_LOAD_NONROAD_OBJECTS", loader
    ):
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
    assert observed["spec"].bridge_module_length == bridge_render._STOCK_MODULE_SPACING_METRES
    assert observed["spec"].marker == "kept"
    assert original.procedural_bridges is True


def test_stock_asset_geometry_matches_measured_original_p3d() -> None:
    assert (
        bridge_render._osm.NOGOVA_BRIDGE_MODEL.casefold()
        == r"o\hous\most_stred30.p3d".casefold()
    )
    assert bridge_render._STOCK_MODULE_SPACING_METRES == 50.0
    assert abs(
        bridge_render._STOCK_ROADWAY_HALF_LENGTH_METRES
        - 25.095142364501953
    ) < 1e-12
    assert abs(
        bridge_render._STOCK_ROADWAY_LOCAL_Y_METRES
        - 12.982887268066406
    ) < 1e-12
    assert abs(
        bridge_render._STOCK_VISIBLE_DECK_LOCAL_Y_METRES
        - 13.049331665039062
    ) < 1e-12
    assert abs(
        bridge_render._STOCK_VISIBLE_DECK_LOCAL_Y_METRES
        - bridge_render._STOCK_ROADWAY_LOCAL_Y_METRES
        - 0.06644439697265625
    ) < 1e-12
    assert (
        bridge_render._osm.NOGOVA_BRIDGE_MODULE_LENGTH_METRES
        == bridge_render._STOCK_MODULE_SPACING_METRES
    )


def test_model_origin_is_lowered_by_stock_visible_deck_local_height() -> None:
    desired_visible_deck_y = 7.035
    origin_y = bridge_render._model_origin_y_for_visible_deck(
        desired_visible_deck_y, 0.0
    )
    assert abs(
        origin_y
        - (
            desired_visible_deck_y
            - bridge_render._STOCK_VISIBLE_DECK_LOCAL_Y_METRES
        )
    ) < 1e-12


def test_stock_bridge_chain_visible_deck_is_anchored_to_both_bank_elevations() -> None:
    spec = _Spec(procedural_bridges=False, bridge_module_length=50.0)
    raster = _dry_raster(spec)
    modules = tuple(
        WorldObject(
            100 + index,
            bridge_render._osm.NOGOVA_BRIDGE_MODEL,
            100.0,
            20.0,
            125.0 + index * 50.0,
            0.0,
            0.0,
        )
        for index in range(4)
    )
    result = _Result(modules)

    def terrain(_elevations, _cells, _cell_size, _x, z):
        if z <= 101.0:
            return 3.0
        if z >= 299.0:
            return 7.0
        return -5.0

    with patch.object(
        bridge_render._osm, "_sample_elevation", side_effect=terrain
    ):
        anchored = bridge_render._anchor_stock_bridge_chains(
            result,
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )

    assert anchored is not result
    first = anchored.objects[0]
    last = anchored.objects[-1]
    expected_start = (
        3.0 + bridge_render._osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES
    )
    expected_end = (
        7.0 + bridge_render._osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES
    )
    half = bridge_render._STOCK_ROADWAY_HALF_LENGTH_METRES

    assert abs(_visible_deck_world_y(first, -half) - expected_start) < 1e-6
    assert abs(_visible_deck_world_y(last, half) - expected_end) < 1e-6

    # The Roadway collision plane is intentionally about 6.6 cm below the
    # rendered asphalt surface inside the stock model.
    expected_collision_delta = (
        bridge_render._STOCK_VISIBLE_DECK_LOCAL_Y_METRES
        - bridge_render._STOCK_ROADWAY_LOCAL_Y_METRES
    )
    assert abs(
        _roadway_world_y(first, -half)
        - (expected_start - expected_collision_delta)
    ) < 1e-6

    # The WRP origin itself must be roughly 13 m below the rendered deck, not
    # directly on the bank as the original broken implementation did.
    assert first.y < expected_start - 12.0
    assert last.y < expected_end - 12.0

    # Adjacent stock modules must present continuous rendered deck endpoints.
    for left, right in zip(anchored.objects, anchored.objects[1:]):
        left_end = _visible_deck_world_y(left, half)
        right_start = _visible_deck_world_y(right, -half)
        # The real deck is ~50.190 m long while centres are spaced 50 m,
        # intentionally giving a small overlap instead of an open seam.
        assert abs(left_end - right_start) < 0.05


def test_middle_water_depth_does_not_drag_stock_bridge_modules_down() -> None:
    spec = _Spec(procedural_bridges=False, bridge_module_length=50.0)
    raster = _dry_raster(spec)
    modules = tuple(
        WorldObject(
            200 + index,
            bridge_render._osm.NOGOVA_BRIDGE_MODEL,
            200.0,
            30.0,
            125.0 + index * 50.0,
            0.0,
            0.0,
        )
        for index in range(3)
    )

    sampled_z = []

    def terrain(_elevations, _cells, _cell_size, _x, z):
        sampled_z.append(z)
        return 4.0 if z < 180.0 else 5.0

    with patch.object(
        bridge_render._osm, "_sample_elevation", side_effect=terrain
    ):
        anchored = bridge_render._anchor_stock_bridge_chains(
            _Result(modules),
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )

    # Only the two approaches are sampled for vertical anchoring. The seabed or
    # terrain under middle modules is intentionally irrelevant to the deck line.
    assert len(sampled_z) == 2
    assert min(
        _visible_deck_world_y(obj) for obj in anchored.objects
    ) > 4.0
    assert max(obj.y for obj in anchored.objects) < 0.0


def test_cached_stock_bridge_result_is_reanchored_without_cache_clear() -> None:
    spec = _Spec()
    raster = _dry_raster(spec)
    cached_result = _Result(
        (
            WorldObject(
                1,
                bridge_render._osm.NOGOVA_BRIDGE_MODEL,
                100.0,
                50.0,
                125.0,
                0.0,
            ),
            WorldObject(
                2,
                bridge_render._osm.NOGOVA_BRIDGE_MODEL,
                100.0,
                50.0,
                175.0,
                0.0,
            ),
        )
    )
    cached = (cached_result, object(), True, "cache", "path")

    with (
        patch.object(
            bridge_render,
            "_ORIGINAL_LOAD_NONROAD_OBJECTS",
            return_value=cached,
        ),
        patch.object(
            bridge_render._osm, "_sample_elevation", return_value=6.0
        ),
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
    expected_visible_deck_y = (
        6.0 + bridge_render._osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES
    )
    assert all(
        abs(_visible_deck_world_y(obj) - expected_visible_deck_y) < 1e-6
        for obj in value[0].objects
    )
    assert all(obj.y < -6.0 for obj in value[0].objects)
