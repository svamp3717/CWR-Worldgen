from __future__ import annotations

from unittest.mock import patch

from cwr_worldgen import terrain_grid_performance_policy as grid_perf
from cwr_worldgen import terrain_postsolve_watchdog_policy as watchdog
from cwr_worldgen import terrain_solver as terrain


def test_runtime_uses_postsolve_watchdog_inside_grid_wrapper() -> None:
    assert terrain.solve_terrain_constraints is grid_perf._fast_solve_terrain_constraints
    assert grid_perf._ORIGINAL_SOLVE is watchdog._solve_with_postsolve_watchdog


def test_watchdog_arms_after_core_solver_ready_and_cancels_on_return() -> None:
    stages: list[str] = []

    def fake_solve(*args, **kwargs):
        progress = kwargs["progress_callback"]
        progress(100, "Terrain constraint solution ready")
        return "done"

    with patch.object(watchdog, "_ORIGINAL_SOLVE_TERRAIN", side_effect=fake_solve), patch.object(
        watchdog, "_arm_watchdog"
    ) as arm, patch.object(watchdog, "_cancel_watchdog") as cancel:
        result = watchdog._solve_with_postsolve_watchdog(
            progress_callback=lambda _percent, stage: stages.append(stage)
        )

    assert result == "done"
    assert stages == ["Terrain constraint solution ready"]
    assert arm.call_count == 1
    assert cancel.call_count >= 1
