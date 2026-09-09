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
    bridges_enabled: bool = True
    maximum_bridge_objects: int = 1000
    advisory_object_limits: bool = True

    @property
    def world_size(self) -> float:
        return self.cells * self.cell_size


@dataclass(frozen=True)
class _Result:
    objects: tuple[WorldObject, ...]


def _dry_raster(spec: _Spec):
    return SimpleNamespace(water=(False,) * (spec.cells * spec.cells))


def _deck_point(obj: WorldObject, local_z: float):
    return bridge_render._visible_deck_point(obj, local_z)


def test_stock_bridge_spec_preserves_user_spec_and_uses_measured_join_length() -> None:
    original = _Spec()
    rewritten = bridge_render._stock_bridge_spec(original)

    assert original.procedural_bridges is True
    assert rewritten is not original
    assert rewritten.procedural_bridges is False
    assert rewritten.bridge_module_length == bridge_render._STOCK_MODULE_SPACING_METRES
    assert rewritten.marker == original.marker


def test_already_stock_bridge_spec_is_reused() -> None:
    original = _Spec(
        procedural_bridges=False,
        bridge_module_length=bridge_render._STOCK_MODULE_SPACING_METRES,
    )
    assert bridge_render._stock_bridge_spec(original) is original


def test_non_dataclass_compatibility_proxy_reads_other_fields_from_base() -> None:
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


def test_stock_asset_geometry_uses_actual_roadway_join_length() -> None:
    assert (
        bridge_render._osm.NOGOVA_BRIDGE_MODEL.casefold()
        == r"o\hous\most_stred30.p3d".casefold()
    )
    assert abs(
        bridge_render._STOCK_ROADWAY_HALF_LENGTH_METRES
        - 25.095142364501953
    ) < 1e-12
    assert abs(
        bridge_render._STOCK_MODULE_SPACING_METRES
        - 50.190284729003906
    ) < 1e-12
    assert (
        bridge_render._STOCK_MODULE_SPACING_METRES
        == bridge_render._STOCK_ROADWAY_LENGTH_METRES
    )
    assert abs(
        bridge_render._STOCK_ROADWAY_LOCAL_Y_METRES
        - 12.982887268066406
    ) < 1e-12
    assert abs(
        bridge_render._STOCK_VISIBLE_DECK_LOCAL_Y_METRES
        - 13.049331665039062
    ) < 1e-12
    assert bridge_render._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES == 0.7


def test_model_origin_round_trips_visible_deck_centre_with_pitch() -> None:
    heading = 63.0
    pitch = 4.0
    deck = (120.0, 7.25, 340.0)
    x, y, z = bridge_render._model_origin_for_visible_deck_center(
        *deck, heading, pitch
    )
    obj = WorldObject(
        1,
        bridge_render._osm.NOGOVA_BRIDGE_MODEL,
        x,
        y,
        z,
        heading,
        pitch,
    )
    reconstructed = _deck_point(obj, 0.0)
    assert math.dist(
        (reconstructed[0], reconstructed[2]), (deck[0], deck[2])
    ) < 1e-9
    assert abs(reconstructed[1] - deck[1]) < 1e-9


def test_stock_bridge_span_plan_clips_long_dry_osm_approaches() -> None:
    spec = _Spec()
    points = ((20.0, 200.0), (620.0, 200.0))

    def terrain(_elevations, _cells, _cell_size, x, _z):
        return -2.0 if 220.0 <= x <= 420.0 else 3.0

    with patch.object(
        bridge_render._osm, "_sample_elevation", side_effect=terrain
    ):
        plan = bridge_render.stock_bridge_span_plan(
            points,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )

    assert plan is not None
    assert plan.module_count == 4
    assert abs(plan.length - 4 * bridge_render._STOCK_MODULE_SPACING_METRES) < 1e-9
    assert abs(math.dist(*plan.points) - plan.length) < 1e-6
    assert plan.points[0][0] > 200.0
    assert plan.points[1][0] < 440.0
    assert abs(plan.wet_length - 200.0) < 0.05


def test_stock_bridge_span_plan_returns_none_without_actual_in_game_water() -> None:
    spec = _Spec()
    with patch.object(
        bridge_render._osm, "_sample_elevation", return_value=3.0
    ):
        plan = bridge_render.stock_bridge_span_plan(
            ((20.0, 100.0), (500.0, 100.0)),
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )
    assert plan is None


def _chain(
    count: int,
    *,
    first_x: float = 100.0,
    spacing: float | None = None,
) -> _Result:
    step = (
        bridge_render._STOCK_MODULE_SPACING_METRES
        if spacing is None
        else float(spacing)
    )
    return _Result(
        tuple(
            WorldObject(
                100 + index,
                bridge_render._osm.NOGOVA_BRIDGE_MODEL,
                first_x + index * step,
                20.0,
                200.0,
                90.0,
                0.0,
            )
            for index in range(count)
        )
    )


def test_shared_3d_joints_make_every_internal_seam_exactly_level() -> None:
    spec = _Spec(procedural_bridges=False)
    raster = _dry_raster(spec)
    result = _chain(4)

    def terrain(_elevations, _cells, _cell_size, x, _z):
        if x < 90.0:
            return 3.0
        if x > 260.0:
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

    ordered_state = bridge_render._ordered_component(
        list(anchored.objects), tuple(range(4))
    )
    assert ordered_state is not None
    ordered, _axis, _midpoint = ordered_state
    horizontal_error, vertical_error = bridge_render._component_seam_errors(
        anchored.objects, ordered
    )
    assert horizontal_error < 1e-6
    assert vertical_error < 1e-6

    for left_index, right_index in zip(ordered, ordered[1:]):
        left_end = _deck_point(
            anchored.objects[left_index],
            bridge_render._STOCK_ROADWAY_HALF_LENGTH_METRES,
        )
        right_start = _deck_point(
            anchored.objects[right_index],
            -bridge_render._STOCK_ROADWAY_HALF_LENGTH_METRES,
        )
        assert math.dist(left_end, right_start) < 1e-6


def test_compressed_old_bridge_chain_is_repaired_to_real_stock_spacing() -> None:
    spec = _Spec(procedural_bridges=False)
    raster = _dry_raster(spec)
    # tinybridgetest produced about 48.02 m centre spacing, creating ~2.17 m
    # overlap between 50.190 m Roadway LODs.
    result = _chain(4, spacing=48.02)

    with patch.object(
        bridge_render._osm, "_sample_elevation", return_value=6.0
    ):
        anchored = bridge_render._anchor_stock_bridge_chains(
            result,
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )

    centres = [_deck_point(obj, 0.0) for obj in anchored.objects]
    distances = [
        math.dist(left, right)
        for left, right in zip(centres, centres[1:])
    ]
    assert all(
        abs(distance - bridge_render._STOCK_MODULE_SPACING_METRES) < 1e-6
        for distance in distances
    )


def test_reanchoring_is_idempotent_for_cached_stock_bridge_objects() -> None:
    spec = _Spec(procedural_bridges=False)
    raster = _dry_raster(spec)
    result = _chain(3)

    with patch.object(
        bridge_render._osm, "_sample_elevation", return_value=6.0
    ):
        first = bridge_render._anchor_stock_bridge_chains(
            result,
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )
        second = bridge_render._anchor_stock_bridge_chains(
            first,
            raster,
            (0.0,) * (spec.cells * spec.cells),
            spec,
        )

    for left, right in zip(first.objects, second.objects):
        assert abs(left.x - right.x) < 1e-9
        assert abs(left.y - right.y) < 1e-9
        assert abs(left.z - right.z) < 1e-9
        assert abs(left.heading_degrees - right.heading_degrees) < 1e-9
        assert abs(left.pitch_degrees - right.pitch_degrees) < 1e-9


def test_cached_nonroad_result_is_reanchored_without_manual_cache_clear() -> None:
    spec = _Spec()
    raster = _dry_raster(spec)
    cached_result = _chain(2)
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
    horizontal_error, vertical_error = bridge_render._component_seam_errors(
        value[0].objects, (0, 1)
    )
    assert horizontal_error < 1e-6
    assert vertical_error < 1e-6
