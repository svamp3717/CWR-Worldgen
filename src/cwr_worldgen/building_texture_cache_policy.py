# SPDX-License-Identifier: GPL-3.0-or-later
"""Persist reusable procedural-building PAAs in the cross-world shared cache."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

from .shared_cache_policy import (
    BUILDING_TEXTURE_CACHE_DIRNAME,
    shared_building_texture_cache_dir,
)

_INSTALLED = False


def _shared_texture_path(cache_path: Path | None) -> Path | None:
    if cache_path is None:
        return None
    local = Path(cache_path)
    if local.suffix.casefold() != ".paa" or local.parent.name != "procedural-assets":
        return None
    shared = shared_building_texture_cache_dir(local.parent.parent)
    return None if shared is None else shared / "paa" / local.name


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def install_building_texture_cache_policy() -> None:
    """Add a persistent shared PAA layer without moving build-specific P3Ds."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import building_asset_budget_policy as budget
    from . import procedural_buildings as buildings

    original_texture_cache_tasks = budget._texture_cache_tasks
    original_restore_or_create_file = buildings.restore_or_create_file

    def shared_texture_cache_tasks(library, buildings_module):
        total, misses = original_texture_cache_tasks(library, buildings_module)
        if (
            not getattr(library, "cache_enabled", True)
            or getattr(library, "cache_refresh", False)
        ):
            return total, misses

        remaining = []
        for task in misses:
            shared = _shared_texture_path(task.cache_path)
            if shared is None or not shared.is_file():
                remaining.append(task)
                continue
            try:
                _atomic_copy(shared, task.cache_path)
            except OSError:
                remaining.append(task)
        return total, remaining

    def restore_or_create_building_texture(
        *, cache_path, destination, producer, enabled, refresh
    ):
        local = Path(cache_path) if cache_path is not None else None
        shared = _shared_texture_path(local)
        if (
            enabled
            and not refresh
            and local is not None
            and not local.is_file()
            and shared is not None
            and shared.is_file()
        ):
            try:
                _atomic_copy(shared, local)
            except OSError:
                pass

        hit = original_restore_or_create_file(
            cache_path=local,
            destination=destination,
            producer=producer,
            enabled=enabled,
            refresh=refresh,
        )

        if enabled and local is not None and local.is_file() and shared is not None:
            try:
                if refresh or not shared.is_file():
                    _atomic_copy(local, shared)
            except OSError:
                pass
        return hit

    budget._texture_cache_tasks = shared_texture_cache_tasks
    buildings.restore_or_create_file = restore_or_create_building_texture
    _INSTALLED = True
