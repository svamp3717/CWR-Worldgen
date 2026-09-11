# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-world cache locations for immutable/reusable generation artifacts."""
from __future__ import annotations

from pathlib import Path

from .build_cache_policy import BUILD_CACHE_DIRNAME


SHARED_CACHE_DIRNAME = ".cwr-worldgen-shared-cache"
BUILDING_TEXTURE_CACHE_DIRNAME = "building-textures"
ASSET_INDEX_CACHE_DIRNAME = "asset-indexes"
_INSTALLED = False


def shared_cache_root(cache_dir: str | Path | None) -> Path | None:
    """Return the cache root shared by sibling world build directories.

    Milestone builds route their generator cache to::

        <world>/.cwr-worldgen-build-cache/<revision>

    Reusable artifacts therefore live one level above ``<world>`` so different
    world folders under the same build parent can reuse them. Explicit/non-build
    cache directories retain a self-contained shared-cache child.
    """
    if cache_dir is None:
        return None
    resolved = Path(cache_dir).expanduser().resolve()
    if resolved.parent.name == BUILD_CACHE_DIRNAME:
        world_dir = resolved.parent.parent
        return world_dir.parent / SHARED_CACHE_DIRNAME
    return resolved / SHARED_CACHE_DIRNAME


def shared_building_texture_cache_dir(cache_dir: str | Path | None) -> Path | None:
    root = shared_cache_root(cache_dir)
    return None if root is None else root / BUILDING_TEXTURE_CACHE_DIRNAME


def shared_asset_index_cache_dir(cache_dir: str | Path | None) -> Path | None:
    root = shared_cache_root(cache_dir)
    return None if root is None else root / ASSET_INDEX_CACHE_DIRNAME


def install_shared_asset_index_cache_policy() -> None:
    """Route persistent PBO header indexes into the cross-world cache root."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import fast_asset_scan_policy as fast

    fast._persistent_index_root = shared_asset_index_cache_dir
    _INSTALLED = True
