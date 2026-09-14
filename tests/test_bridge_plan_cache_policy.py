from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from cwr_worldgen import generator
from cwr_worldgen import bridge_abutment_terrain_policy as abutment
from cwr_worldgen import bridge_plan_cache_policy as policy
from cwr_worldgen.bridge_render_policy import StockBridgeSpanPlan


@dataclass(frozen=True)
class _Spec:
    cells: int = 8
    cell_size: float = 50.0
    world_size: float = 400.0
    cache_refresh: bool = False


def _plan() -> StockBridgeSpanPlan:
    return StockBridgeSpanPlan(
        points=((75.0, 100.0), (225.0, 100.0)),
        module_count=3,
        wet_start=(100.0, 100.0),
        wet_end=(200.0, 100.0),
        wet_length=100.0,
    )


def test_bridge_plan_cache_policy_is_installed() -> None:
    assert getattr(
        generator._load_terrain_solution,
        "_cwr_bridge_plan_cache_policy",
        False,
    )


def test_bridge_plan_sidecar_round_trip(tmp_path) -> None:
    spec = _Spec()
    cache_path = tmp_path / "terrain.pickle"
    points = ((50.0, 100.0), (250.0, 100.0))
    key = abutment._plan_key(points, spec)
    plan = _plan()

    abutment._PLAN_CACHE.clear()
    try:
        abutment._PLAN_CACHE[key] = plan
        assert policy._write_plan_sidecar(cache_path, spec)

        abutment._PLAN_CACHE.clear()
        assert policy._restore_plan_sidecar(cache_path, spec)
        assert abutment._PLAN_CACHE == {key: plan}
    finally:
        abutment._PLAN_CACHE.clear()


def test_old_terrain_cache_refreshes_once_then_restores_plan_sidecar(
    tmp_path,
    monkeypatch,
) -> None:
    spec = _Spec()
    cache_path = tmp_path / "terrain.pickle"
    points = ((50.0, 100.0), (250.0, 100.0))
    key = abutment._plan_key(points, spec)
    plan = _plan()
    dataset = SimpleNamespace(
        roads=(SimpleNamespace(tags={"highway": "primary", "bridge": "yes"}),)
    )
    calls: list[bool] = []

    def fake_load(
        loaded,
        dataset,
        projection,
        raster,
        current_spec,
        dem_key,
        raster_key,
        dataset_identity,
        *,
        building_placement_plans=(),
        progress_callback=None,
    ):
        del (
            loaded,
            dataset,
            projection,
            raster,
            dem_key,
            raster_key,
            dataset_identity,
            building_placement_plans,
            progress_callback,
        )
        calls.append(bool(current_spec.cache_refresh))
        if current_spec.cache_refresh:
            abutment._PLAN_CACHE[key] = plan
            return "fresh-grading", (), False, "terrain-key", str(cache_path)
        return "cached-grading", (), True, "terrain-key", str(cache_path)

    monkeypatch.setattr(policy, "_ORIGINAL_LOAD_TERRAIN_SOLUTION", fake_load)
    abutment._PLAN_CACHE.clear()
    try:
        first = generator._load_terrain_solution(
            None,
            dataset,
            None,
            None,
            spec,
            "dem",
            "raster",
            "dataset",
        )
        assert first[0] == "fresh-grading"
        assert calls == [False, True]
        assert abutment._PLAN_CACHE == {key: plan}
        assert policy._sidecar_path(cache_path).is_file()

        # A new process would begin with no in-memory plan. The next terrain hit
        # must restore the exact pre-fill plan without another terrain solve.
        abutment._PLAN_CACHE.clear()
        calls.clear()
        second = generator._load_terrain_solution(
            None,
            dataset,
            None,
            None,
            spec,
            "dem",
            "raster",
            "dataset",
        )
        assert second[0] == "cached-grading"
        assert calls == [False]
        assert abutment._PLAN_CACHE == {key: plan}
    finally:
        abutment._PLAN_CACHE.clear()
