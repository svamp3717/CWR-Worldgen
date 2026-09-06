from types import SimpleNamespace
from unittest.mock import patch

import cwr_worldgen.paved_junction_fallback_policy as fallback
import cwr_worldgen.paved_junction_policy as paved
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
