import math
from types import SimpleNamespace
from unittest.mock import patch

import cwr_worldgen.paved_junction_fallback_policy as fallback
import cwr_worldgen.paved_junction_policy as paved
import cwr_worldgen.playability as playability
import cwr_worldgen.road_quality_policy as road_quality


def _plan(model: str, point: tuple[float, float], axis=(0.0, 1.0)):
    return SimpleNamespace(
        model_path=model,
        point=point,
        axis=axis,
        connectors=(
            SimpleNamespace(direction=(0.0, 1.0)),
            SimpleNamespace(direction=(0.0, -1.0)),
            SimpleNamespace(direction=(1.0, 0.0)),
        ),
    )


def _object(object_id: int, model: str, x: float, z: float):
    return SimpleNamespace(
        object_id=object_id,
        model_path=model,
        x=x,
        z=z,
    )


def test_successful_plan_keys_preserves_boundary_match_across_buckets() -> None:
    model = r"o\road\kr_new_sil_sil_t.p3d"
    key = (1, 2)
    plans = {key: _plan(model, (0.0, 0.0))}
    report = SimpleNamespace(
        objects=(
            _object(1, model, fallback._SUCCESS_DISTANCE_METRES, 0.0),
            _object(2, r"o\road\other.p3d", 0.0, 0.0),
        )
    )

    assert fallback._successful_plan_keys(report, plans) == frozenset({key})


def test_successful_plan_keys_prunes_distant_same_model_positions() -> None:
    model = r"o\road\kr_new_sil_sil_t.p3d"
    key = (5, 6)
    plans = {key: _plan(model, (0.0, 0.0))}
    objects = [
        _object(index + 1, model, 1000.0 + index * 2.0, 1000.0)
        for index in range(5000)
    ]
    objects.append(_object(6000, model, 0.10, 0.10))
    report = SimpleNamespace(objects=tuple(objects))
    original_dist = math.dist

    with patch.object(fallback.math, "dist", wraps=original_dist) as distance:
        result = fallback._successful_plan_keys(report, plans)

    assert result == frozenset({key})
    assert distance.call_count <= 4


def test_failed_stock_plan_uses_ordinary_junction_geometry_on_refit() -> None:
    active_key = (100, 100)
    failed_key = (200, 200)
    active_plan = _plan(r"o\road\kr_new_sil_sil_t.p3d", (100.0, 100.0))
    failed_plan = _plan(r"o\road\kr_new_sil_sil_t.p3d", (200.0, 200.0))
    plans = {active_key: active_plan, failed_key: failed_plan}
    base = {
        active_key: road_quality._Junction(
            point=(100.0, 100.0),
            axis=(1.0, 0.0),
            half_length=4.0,
            half_width=3.0,
            directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
        ),
        failed_key: road_quality._Junction(
            point=(200.0, 200.0),
            axis=(1.0, 0.0),
            half_length=5.0,
            half_width=3.5,
            directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
        ),
    }

    token = paved._PLANS.set({active_key: active_plan})
    try:
        with patch.object(paved, "_ORIGINAL_GEOMETRY", lambda *_args: base), patch.object(
            paved, "_plans", lambda *_args: plans
        ):
            geometry = fallback._junction_geometry(None, None, None)
    finally:
        paved._PLANS.reset(token)

    assert geometry[active_key].half_length == paved._APPROACH_RESERVE
    assert geometry[active_key].half_width == paved._APPROACH_RESERVE
    assert geometry[failed_key] == base[failed_key]



def test_generated_fallback_stitches_terminal_piece_and_removes_intruder() -> None:
    centre = (100.0, 100.0)
    connector = paved._Connector("sil", (100.0, 106.25), (0.0, 1.0))
    plan = paved._Plan(
        r"o\road\kr_new_sil_sil_t.p3d",
        centre,
        (0.0, 1.0),
        (
            paved._Arm("sil", (0.0, 1.0), connector),
            paved._Arm("sil", (0.0, -1.0), paved._Connector(
                "sil", (100.0, 93.75), (0.0, -1.0)
            )),
            paved._Arm("sil", (1.0, 0.0), paved._Connector(
                "sil", (106.25, 100.0), (1.0, 0.0)
            )),
        ),
    )
    spec = SimpleNamespace(
        name="fallback_world",
        cells=64,
        cell_size=10.0,
        road_segment_length=25.0,
    )
    elevations = [0.0] * (spec.cells * spec.cells)

    hub = playability.WorldObject(
        1,
        r"fallback_world\i\paved_j3_w091_h000_090_180.p3d",
        centre[0],
        0.035,
        centre[1],
        0.0,
        0.0,
    )
    # Stale piece penetrates deep into the generated hub.
    intruder = playability._road_object_on_slope(
        2,
        r"o\road\sil6.p3d",
        (100.0, 101.0),
        (100.0, 107.25),
        elevations,
        spec,
        vertical_offset=playability._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )
    # The next outward piece starts a metre beyond the true 6.25 m connector.
    terminal = playability._road_object_on_slope(
        3,
        r"o\road\sil6.p3d",
        (100.0, 107.25),
        (100.0, 113.50),
        elevations,
        spec,
        vertical_offset=playability._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )
    report = playability.RoadFitReport(
        objects=(hub, intruder, terminal),
        chain_count=1,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
        junction_cap_objects=1,
    )

    result = fallback._stitch_generated_fallback_approaches(
        report,
        {(1, 1): plan},
        elevations,
        spec,
    )

    ids = {obj.object_id for obj in result.objects}
    assert 2 not in ids
    rebuilt = next(obj for obj in result.objects if obj.object_id == 3)
    assert r"\i\paved_w091_" in rebuilt.model_path.casefold()
    length = road_quality._piece_length(
        rebuilt.model_path,
        spec.road_segment_length,
    )
    axis = playability._model_axis(rebuilt, length)
    assert min(
        math.dist(connector.point, endpoint)
        for endpoint in axis
    ) <= 0.06


def test_failed_stock_junction_is_refit_as_ordinary_connected_roads() -> None:
    key = (100, 100)
    plan = _plan(r"o\road\kr_new_sil_sil_t.p3d", (100.0, 100.0))
    plans = {key: plan}
    trimmed_report = SimpleNamespace(
        objects=(
            SimpleNamespace(
                object_id=1,
                model_path=r"o\road\sil6.p3d",
                x=100.0,
                z=100.0,
            ),
        )
    )
    ordinary_report = SimpleNamespace(
        objects=(
            SimpleNamespace(
                object_id=10,
                model_path=r"o\road\sil12.p3d",
                x=100.0,
                z=95.0,
            ),
            SimpleNamespace(
                object_id=11,
                model_path=r"o\road\sil12.p3d",
                x=100.0,
                z=105.0,
            ),
        )
    )
    active_plan_sets = []

    def ordinary_fit(*_args, **_kwargs):
        active_plan_sets.append(dict(paved._PLANS.get() or {}))
        return ordinary_report

    spec = SimpleNamespace(stock_road_piece_fitting=True)
    with patch.object(paved, "_plans", lambda *_args: plans), patch.object(
        fallback, "_ORIGINAL_FIT", lambda *_args, **_kwargs: trimmed_report
    ), patch.object(paved, "_ORIGINAL_FIT", ordinary_fit), patch.object(
        paved,
        "_apply_plans",
        side_effect=AssertionError("no stock junction should be re-applied after all plans fail"),
    ):
        result = fallback._fit(
            None,
            None,
            (),
            spec,
            starting_id=1,
            progress_callback=None,
        )

    assert result is ordinary_report
    assert active_plan_sets == [{}]
