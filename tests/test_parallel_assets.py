from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from cwr_worldgen.parallel_assets import asset_worker_count


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(source_root) + (os.pathsep + existing if existing else "")
    env["CWR_WORLDGEN_ASSET_WORKERS"] = "4"
    return env


def test_asset_worker_count_honours_bounded_override(monkeypatch) -> None:
    monkeypatch.setenv("CWR_WORLDGEN_ASSET_WORKERS", "99")
    assert asset_worker_count(3) == 3
    assert asset_worker_count(100) == 16
    monkeypatch.setenv("CWR_WORLDGEN_ASSET_WORKERS", "nonsense")
    assert asset_worker_count(100) == 1


def test_spawned_asset_workers_exit_cleanly_without_resource_tracker_warning(tmp_path: Path) -> None:
    script = tmp_path / "parallel_clean_exit.py"
    script.write_text(
        """
from cwr_worldgen.parallel_assets import process_asset_tasks

if __name__ == '__main__':
    values = list(range(-64, 0))
    result = process_asset_tasks(abs, values, minimum_parallel_tasks=2)
    assert result == [abs(value) for value in values]
    print(sum(result))
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
    assert completed.stdout.strip() == "2080"
    assert "resource_tracker" not in completed.stderr
    assert "leaked semaphore" not in completed.stderr.lower()


def test_parent_reaps_worker_that_stalls_during_interpreter_exit(tmp_path: Path) -> None:
    script = tmp_path / "parallel_stuck_exit.py"
    script.write_text(
        """
import atexit
import time
import cwr_worldgen.parallel_assets as parallel_assets

# Keep the regression test quick. Production uses a longer grace window.
parallel_assets._WORKER_EXIT_GRACE_SECONDS = 0.2
parallel_assets._WORKER_TERMINATE_GRACE_SECONDS = 0.2

def worker(value):
    atexit.register(time.sleep, 30.0)
    return value * value

if __name__ == '__main__':
    started = time.perf_counter()
    result = parallel_assets.process_asset_tasks(worker, list(range(8)), minimum_parallel_tasks=2)
    elapsed = time.perf_counter() - started
    assert result == [value * value for value in range(8)]
    assert elapsed < 8.0
    print('clean', round(elapsed, 3))
""".lstrip(),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("clean ")
    assert "resource_tracker" not in completed.stderr
    assert "leaked semaphore" not in completed.stderr.lower()
