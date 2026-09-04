from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from cwr_worldgen import parallel_assets
from cwr_worldgen.building_progress_policy import process_asset_tasks_with_progress


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(source_root) + (os.pathsep + existing if existing else "")
    env["CWR_WORLDGEN_ASSET_WORKERS"] = "2"
    return env


def test_serial_progress_reports_each_completed_asset() -> None:
    events: list[tuple[int, int]] = []
    result = process_asset_tasks_with_progress(
        abs,
        [-3, -2, -1],
        minimum_parallel_tasks=999,
        progress_callback=lambda completed, total: events.append((completed, total)),
    )
    assert result == [3, 2, 1]
    assert events == [(1, 3), (2, 3), (3, 3)]


def test_spawned_workers_stream_completion_and_preserve_result_order(tmp_path: Path) -> None:
    script = tmp_path / "stream_asset_progress.py"
    script.write_text(
        """
import time
from cwr_worldgen.building_progress_policy import process_asset_tasks_with_progress

def worker(value):
    time.sleep(0.03 * (1 + value % 3))
    return value * value

if __name__ == '__main__':
    events = []
    result = process_asset_tasks_with_progress(
        worker,
        list(range(8)),
        minimum_parallel_tasks=2,
        progress_callback=lambda completed, total: events.append((completed, total)),
    )
    assert result == [value * value for value in range(8)]
    assert events == [(index, 8) for index in range(1, 9)]
    print('streamed', len(events))
""".lstrip(),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "streamed 8"
    assert "resource_tracker" not in completed.stderr
    assert "leaked semaphore" not in completed.stderr.lower()


def test_modeler_texture_route_emits_human_readable_counter(monkeypatch) -> None:
    from cwr_worldgen import progress

    messages: list[tuple[int, str]] = []
    monkeypatch.setattr(
        progress,
        "report_progress",
        lambda percent, stage: messages.append((percent, stage)),
    )

    def texture_worker(value):
        return value + 1

    texture_worker.__module__ = "cwr_worldgen.building_asset_budget_policy"
    texture_worker.__name__ = "_write_modeler_texture_cache_task"

    result = parallel_assets.process_asset_tasks(
        texture_worker,
        [1, 2, 3],
        minimum_parallel_tasks=999,
    )
    assert result == [2, 3, 4]
    assert messages
    assert messages[-1] == (
        73,
        "Generating procedural building textures (3/3, 100%)",
    )


def test_building_p3d_route_emits_human_readable_counter(monkeypatch) -> None:
    from cwr_worldgen import progress

    messages: list[tuple[int, str]] = []
    monkeypatch.setattr(
        progress,
        "report_progress",
        lambda percent, stage: messages.append((percent, stage)),
    )

    def p3d_worker(value):
        return value * 2

    p3d_worker.__module__ = "cwr_worldgen.procedural_buildings"
    p3d_worker.__name__ = "_write_building_asset_task"

    result = parallel_assets.process_asset_tasks(
        p3d_worker,
        [1, 2],
        minimum_parallel_tasks=999,
    )
    assert result == [2, 4]
    assert messages[-1] == (
        74,
        "Generating procedural building P3Ds (2/2, 100%)",
    )
