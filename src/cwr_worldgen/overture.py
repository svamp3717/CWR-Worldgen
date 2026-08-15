# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


OVERTURE_CLI_MARKER = "--cwr-overture"


def overture_buildings_cache_path(cache_dir: Path, bbox: tuple[float, float, float, float]) -> Path:
    digest = sha256(json.dumps(tuple(round(float(value), 7) for value in bbox)).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"overture-buildings-{digest}.geojson"


def overture_command_prefix() -> list[str]:
    python_path = Path(sys.executable).resolve()
    # Prefer the directory containing the currently launched application/entry
    # point. In frozen/packaged builds this also preserves support for users who
    # intentionally place a standalone overturemaps executable beside Worldgen.
    launch_path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else python_path
    candidate_dirs = (
        launch_path.parent,
        python_path.parent,
        python_path.parent / "Scripts",
        python_path.parent.parent / "Scripts",
        python_path.parent.parent / "bin",
    )
    # Preserve search priority while avoiding duplicate filesystem probes.
    script_dirs = tuple(dict.fromkeys(candidate_dirs))
    executable_names = (
        "overturemaps.exe",
        "overturemaps.cmd",
        "overturemaps.bat",
        "overturemaps",
    )
    for script_dir in script_dirs:
        for name in executable_names:
            candidate = script_dir / name
            if candidate.is_file():
                return [str(candidate)]
    executable = shutil.which("overturemaps")
    if executable:
        return [executable]
    if bool(getattr(sys, "frozen", False)):
        # The packaged GUI contains the official overturemaps Python package.
        # Re-enter the Worldgen executable through a private marker so Overture
        # still runs in a child process and retains the existing timeout/fallback
        # behavior without requiring a second executable beside the app.
        return [sys.executable, OVERTURE_CLI_MARKER]
    return [sys.executable, "-m", "overturemaps"]


def _subprocess_window_kwargs() -> dict[str, object]:
    if os.name != "nt":
        return {}
    kwargs: dict[str, object] = {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if creationflags:
        kwargs["creationflags"] = creationflags
    startupinfo_type = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_type is not None:
        startupinfo = startupinfo_type()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo
    return kwargs


def fetch_overture_buildings_geojson(
    bbox: tuple[float, float, float, float],
    output: Path,
    *,
    refresh: bool = False,
    timeout: int = 300,
) -> Path | None:
    if output.is_file() and not refresh:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.geojson")
    temporary.unlink(missing_ok=True)
    south, west, north, east = bbox
    try:
        subprocess.run(
            overture_command_prefix() + [
                "download",
                "--bbox",
                f"{west:.7f},{south:.7f},{east:.7f},{north:.7f}",
                "-f",
                "geojson",
                "--type",
                "building",
                "--no-stac",
                "--connect_timeout",
                "10",
                "--request_timeout",
                "30",
                "-o",
                str(temporary),
            ],
            check=True,
            timeout=timeout,
            **_subprocess_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        temporary.unlink(missing_ok=True)
        return None
    if not temporary.is_file():
        return None
    os.replace(temporary, output)
    return output
