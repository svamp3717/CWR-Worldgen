# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import subprocess
import sys


OVERTURE_CLI_MARKER = "--cwr-overture"
_OVERTURE_RELEASE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.(\d+)$")
_OVERTURE_BUCKET = "overturemaps-us-west-2"
_OVERTURE_RELEASE_ROOT = f"{_OVERTURE_BUCKET}/release"


def overture_buildings_cache_path(cache_dir: Path, bbox: tuple[float, float, float, float]) -> Path:
    digest = sha256(json.dumps(tuple(round(float(value), 7) for value in bbox)).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"overture-buildings-{digest}.geojson"


def _release_sort_key(release: str) -> tuple[int, int, int, int]:
    match = _OVERTURE_RELEASE_RE.fullmatch(release)
    if match is None:
        raise ValueError(f"invalid Overture release name: {release}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def discover_latest_overture_release(
    *,
    connect_timeout: int = 10,
    request_timeout: int = 30,
) -> str:
    """Discover the newest retained Overture release directly from public S3.

    This intentionally avoids Overture's STAC release-discovery endpoint. The
    upstream CLI currently validates its --release option through STAC before
    the --no-stac download option can take effect, so a broken STAC catalog can
    otherwise prevent even an explicitly non-STAC download from starting.
    """
    from pyarrow import fs

    filesystem = fs.S3FileSystem(
        anonymous=True,
        region="us-west-2",
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )
    entries = filesystem.get_file_info(
        fs.FileSelector(_OVERTURE_RELEASE_ROOT, recursive=False, allow_not_found=False)
    )
    releases: list[str] = []
    for entry in entries:
        name = str(entry.path).rstrip("/").rsplit("/", 1)[-1]
        if _OVERTURE_RELEASE_RE.fullmatch(name):
            releases.append(name)
    if not releases:
        raise RuntimeError("no Overture data releases were found in the public S3 bucket")
    return max(releases, key=_release_sort_key)


def download_overture_buildings_direct(
    bbox: tuple[float, float, float, float],
    output: Path,
    *,
    connect_timeout: int = 10,
    request_timeout: int = 30,
) -> str:
    """Download building GeoJSON through Overture's Python API without STAC."""
    from overturemaps import record_batch_reader
    from overturemaps.writers import copy, get_writer

    south, west, north, east = bbox
    release = discover_latest_overture_release(
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )
    reader = record_batch_reader(
        "building",
        bbox=(west, south, east, north),
        release=release,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        stac=False,
    )
    if reader is None:
        raise RuntimeError(f"Overture returned no building reader for release {release}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with get_writer("geojson", str(output), schema=reader.schema) as writer:
        copy(reader, writer)
    if not output.is_file():
        raise RuntimeError("Overture download completed without creating the GeoJSON output")
    return release


def run_overture_worker(argv: list[str]) -> int:
    """Run the isolated Overture downloader used by source and frozen builds."""
    parser = argparse.ArgumentParser(prog="cwr-worldgen-overture-worker", add_help=False)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=30)
    options = parser.parse_args(argv)

    parts = [part.strip() for part in str(options.bbox).split(",")]
    if len(parts) != 4:
        raise ValueError("Overture worker bbox must be west,south,east,north")
    west, south, east, north = (float(part) for part in parts)
    download_overture_buildings_direct(
        (south, west, north, east),
        Path(options.output),
        connect_timeout=options.connect_timeout,
        request_timeout=options.request_timeout,
    )
    return 0


def overture_command_prefix() -> list[str]:
    """Return a child-process command that bypasses the upstream Click CLI."""
    executable = str(Path(sys.executable).resolve())
    if bool(getattr(sys, "frozen", False)):
        return [executable, OVERTURE_CLI_MARKER]
    return [executable, "-m", "cwr_worldgen.gui_entry", OVERTURE_CLI_MARKER]


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
            overture_command_prefix()
            + [
                "--bbox",
                f"{west:.7f},{south:.7f},{east:.7f},{north:.7f}",
                "--output",
                str(temporary),
                "--connect-timeout",
                "10",
                "--request-timeout",
                "30",
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
