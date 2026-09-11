# SPDX-License-Identifier: GPL-3.0-or-later
"""Route persistent runway textures into the cross-world shared cache."""
from __future__ import annotations

from pathlib import Path

from .shared_cache_policy import SHARED_CACHE_DIRNAME, shared_cache_root


RUNWAY_TEXTURE_CACHE_DIRNAME = "runway-textures"
_INSTALLED = False


def shared_runway_texture_cache_dir(cache_dir: str | Path | None) -> Path | None:
    """Return the runway-texture directory shared by sibling world builds."""
    root = shared_cache_root(cache_dir)
    return None if root is None else root / RUNWAY_TEXTURE_CACHE_DIRNAME


def install_shared_runway_cache_policy() -> None:
    """Replace the legacy top-level runway cache with a shared-cache child."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import runway_texture_cache_policy as runway_cache

    original_runway_cache_dir = runway_cache._runway_cache_dir

    def shared_runway_cache_dir(source_dir: Path, spec) -> Path:
        requested = getattr(spec, "cache_dir", None)
        shared = shared_runway_texture_cache_dir(requested)
        if shared is not None:
            return shared

        # Direct library callers may omit cache_dir. Preserve the historical
        # resolver's parent choice, but consolidate beneath the shared root.
        legacy = Path(original_runway_cache_dir(source_dir, spec))
        return legacy.parent / SHARED_CACHE_DIRNAME / RUNWAY_TEXTURE_CACHE_DIRNAME

    runway_cache._runway_cache_dir = shared_runway_cache_dir
    _INSTALLED = True
