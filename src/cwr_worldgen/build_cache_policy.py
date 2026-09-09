# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep expensive build-stage caches beside the selected build output.

The historical Milestone 9 cache directory lives below the frozen source bundle
and was passed unchanged into the core world generator.  That mixed durable
source/download state with disposable terrain solves, placements, procedural
assets, overview images and PBO blobs.  Keep the source cache for source-side
work, but route the core generator to a build-local cache instead.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
import json
import os
from pathlib import Path
import shutil
from typing import Any

from .cache import resolve_cache_dir


BUILD_CACHE_DIRNAME = ".cwr-worldgen-build-cache"
# Cache keys inside generator.py deliberately describe data contracts rather than
# every runtime monkey-patch. Activating the bridge policy chain changes terrain
# solving and non-road placement without changing those historical key strings,
# so put this runtime generation in a fresh cache namespace. Old cache data can
# remain on disk safely and cleanup still removes the common parent directory.
BUILD_CACHE_REVISION = "v2-active-bridge-policies"
_INSTALLED = False


def build_cache_dir(output_dir: str | Path) -> Path:
    """Return the versioned persistent cache owned by one selected build folder."""
    return (
        Path(output_dir).expanduser().resolve()
        / BUILD_CACHE_DIRNAME
        / BUILD_CACHE_REVISION
    )


def route_build_cache_spec(output_dir: str | Path, spec: Any) -> Any:
    """Return a dataclass spec whose core-generator cache is build-local."""
    if not hasattr(spec, "cache_dir"):
        return spec
    return replace(spec, cache_dir=build_cache_dir(output_dir))


def _remove_stash(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _prepare_output_preserving_build_cache(original, root, world_name, *, clean: bool) -> None:
    """Let a clean build reset its output without throwing away a valid cache.

    Post-build cleanup is the explicit operation that removes the build cache.
    A normal clean build should still be able to reuse key-validated cache data
    from the previous run, otherwise locating the cache under the build folder
    would make it self-destruct before its first lookup.
    """
    root = Path(root).expanduser().resolve()
    cache = root / BUILD_CACHE_DIRNAME
    if not clean or not cache.exists():
        original(root, world_name, clean=clean)
        return

    stash = root.parent / f".{root.name}.{os.getpid()}.cwr-worldgen-build-cache-preserve"
    if stash.exists() or stash.is_symlink():
        _remove_stash(stash)
    cache.replace(stash)
    try:
        original(root, world_name, clean=clean)
        cache.parent.mkdir(parents=True, exist_ok=True)
        stash.replace(cache)
    except Exception:
        # Do not turn an unrelated build failure into accidental cache loss.
        try:
            root.mkdir(parents=True, exist_ok=True)
            if stash.exists() and not cache.exists():
                stash.replace(cache)
        except OSError:
            pass
        raise
    finally:
        if stash.exists() or stash.is_symlink():
            _remove_stash(stash)


def _rewrite_cache_report(result: Any, output_dir: str | Path, spec: Any) -> None:
    path = getattr(result, "cache_report_path", None)
    if path is None:
        path = Path(getattr(result, "output_dir", output_dir)) / "cache-report.json"
    path = Path(path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        report = {}
    if not isinstance(report, dict):
        report = {}

    source_dir = Path(getattr(spec, "source_dir", ".")).expanduser()
    requested = getattr(spec, "cache_dir", None)
    source_cache = resolve_cache_dir(source_dir, requested)
    local_build_cache = build_cache_dir(output_dir)

    # ``directory`` historically described the cache used by the core generator,
    # so keep that meaning while exposing both locations explicitly.
    report["directory"] = str(local_build_cache)
    report["build_directory"] = str(local_build_cache)
    report["source_directory"] = str(source_cache)
    report["split_cache_layout"] = True
    report["build_cache_revision"] = BUILD_CACHE_REVISION
    try:
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass


def install_build_cache_policy() -> None:
    """Route Milestone 9 generator caches to the build folder exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import generator
    from . import milestone9

    original_prepare = generator.prepare_output_directory

    @wraps(original_prepare)
    def prepare_output_directory(root, world_name, *, clean: bool):
        return _prepare_output_preserving_build_cache(
            original_prepare, root, world_name, clean=clean
        )

    generator.prepare_output_directory = prepare_output_directory

    original_build4 = milestone9.build_milestone4

    @wraps(original_build4)
    def build_milestone4(output_dir, spec, *args, **kwargs):
        return original_build4(
            output_dir,
            route_build_cache_spec(output_dir, spec),
            *args,
            **kwargs,
        )

    milestone9.build_milestone4 = build_milestone4

    original_build9 = milestone9.build_milestone9

    @wraps(original_build9)
    def build_milestone9(output_dir, spec, *, clean: bool = True):
        result = original_build9(output_dir, spec, clean=clean)
        _rewrite_cache_report(result, output_dir, spec)
        return result

    milestone9.build_milestone9 = build_milestone9
    _INSTALLED = True
