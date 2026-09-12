# SPDX-License-Identifier: GPL-3.0-or-later
"""Load modifiable JSON beside frozen CWR Worldgen builds before package import.

PyInstaller normally hides package data inside its bundle.  The release packages
ship a sibling ``config`` directory containing the same runtime JSON hierarchy.
Before importing :mod:`cwr_worldgen`, the frozen entry point mirrors those files
into PyInstaller's extracted package-data directory. Existing JSON consumers can
therefore keep using normal package-relative paths while users edit ordinary
files beside the executable.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Mapping

CONFIG_DIR_ENV = "CWR_WORLDGEN_CONFIG_DIR"
JSON_DIRECTORIES = ("data", "house_styles", "country_styles")


def default_config_root(
    *,
    executable: str | Path | None = None,
    platform: str | None = None,
) -> Path:
    """Return the external config directory associated with a frozen executable."""

    exe = Path(executable or sys.executable).resolve()
    platform_name = platform or sys.platform
    if platform_name == "darwin":
        # PyInstaller .app executables live in Foo.app/Contents/MacOS/Foo. Keep
        # editable configuration beside the .app, not buried inside the bundle.
        macos_dir = exe.parent
        contents_dir = macos_dir.parent
        app_dir = contents_dir.parent
        if macos_dir.name == "MacOS" and contents_dir.name == "Contents" and app_dir.suffix == ".app":
            return app_dir.parent / "config"
    return exe.parent / "config"


def config_root(
    *,
    environ: Mapping[str, str] | None = None,
    executable: str | Path | None = None,
    platform: str | None = None,
) -> Path:
    """Return an explicit override or the normal frozen-build config directory."""

    environment = os.environ if environ is None else environ
    override = str(environment.get(CONFIG_DIR_ENV, "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    return default_config_root(executable=executable, platform=platform)


def overlay_external_json(
    *,
    bundle_root: str | Path | None = None,
    external_root: str | Path | None = None,
) -> int:
    """Make external JSON authoritative inside an extracted PyInstaller bundle.

    Each external runtime directory is treated as a complete replacement when it
    contains JSON. Bundled JSON in that directory is removed first, which means
    modders may edit, add, or deliberately remove catalogue files. If an external
    directory is absent or empty, the bundled defaults remain untouched.
    """

    if bundle_root is None:
        pyinstaller_root = getattr(sys, "_MEIPASS", None)
        if not pyinstaller_root:
            return 0
        bundle = Path(pyinstaller_root)
    else:
        bundle = Path(bundle_root)

    external = Path(external_root) if external_root is not None else config_root()
    package_root = bundle / "cwr_worldgen"
    copied = 0

    for relative in JSON_DIRECTORIES:
        source_dir = external / relative
        source_files = sorted(path for path in source_dir.glob("*.json") if path.is_file())
        if not source_files:
            continue

        target_dir = package_root / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for old in target_dir.glob("*.json"):
            old.unlink()
        for source in source_files:
            shutil.copy2(source, target_dir / source.name)
            copied += 1

    return copied
