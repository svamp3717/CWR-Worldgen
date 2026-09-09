# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep expensive build-stage caches beside the selected build output.

The historical Milestone 9 cache directory lives below the frozen source bundle
and was passed unchanged into the core world generator.  That mixed durable
source/download state with disposable terrain solves, placements, procedural
assets, overview images and PBO blobs.  Keep the source cache for source-side
work, but route the core generator to a build-local cache instead.

Source storage also gets conservative garbage collection before a build.  The
cleanup removes artifacts that current builds cannot consume, and bounds old
fingerprinted/source-download history instead of letting it grow forever.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from .cache import DEFAULT_CACHE_DIRNAME, LEGACY_CACHE_DIRNAME, resolve_cache_dir


BUILD_CACHE_DIRNAME = ".cwr-worldgen-build-cache"
# Cache keys inside generator.py deliberately describe data contracts rather than
# every runtime monkey-patch. Activating the bridge policy chain changes terrain
# solving and non-road placement without changing those historical key strings,
# so put this runtime generation in a fresh cache namespace. Old cache data can
# remain on disk safely and cleanup still removes the common parent directory.
BUILD_CACHE_REVISION = "v2-active-bridge-policies"
_SOURCE_CACHE_HISTORY_KEEP = 2
_OVERTURE_RELEASE_HISTORY_KEEP = 2
_OVERTURE_RELEASE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")
# These directories are produced only by the core generator.  Since that cache
# moved below the selected build folder, copies left in the source cache are
# unreachable by current Milestone 9 builds.
_LEGACY_SOURCE_BUILD_CACHE_DIRS = (
    "assets",
    "dem",
    "overview",
    "pbo",
    "placements",
    "procedural-assets",
    "surfaces",
)
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


def _prune_files(paths: Iterable[Path], *, keep: int) -> list[Path]:
    """Remove all but the newest ``keep`` regular files and return removals."""
    candidates: list[tuple[int, str, Path]] = []
    for path in paths:
        try:
            if path.is_file() and not path.is_symlink():
                candidates.append((path.stat().st_mtime_ns, path.name, path))
        except OSError:
            continue
    candidates.sort(reverse=True)
    removed: list[Path] = []
    for _mtime, _name, path in candidates[max(0, int(keep)) :]:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            pass
    return removed


def _prune_overture_releases(overture_root: Path) -> list[Path]:
    """Keep the two newest release-scoped Overture tile caches."""
    try:
        releases = [
            child
            for child in overture_root.iterdir()
            if child.is_dir() and not child.is_symlink() and _OVERTURE_RELEASE_RE.fullmatch(child.name)
        ]
    except OSError:
        return []
    releases.sort(key=lambda path: path.name, reverse=True)
    removed: list[Path] = []
    for path in releases[_OVERTURE_RELEASE_HISTORY_KEEP:]:
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError:
            pass
    return removed


def cleanup_source_storage(
    source_dir: str | Path,
    requested_cache: str | Path | None = None,
) -> tuple[Path, ...]:
    """Prune source-side files that current builds do not need to retain.

    Raw OSM, raw DEM, the processed source heightmap, normalized GeoJSON, the
    final reference mosaic, dem-stitcher tiles, current parsed/spatial caches and
    recent Overture data are deliberately preserved.
    """
    source_root = Path(source_dir).expanduser().resolve()
    requested = Path(requested_cache).expanduser() if requested_cache is not None else None
    source_cache = resolve_cache_dir(source_root, requested)
    removed: list[Path] = []

    # The individual OpenTopoMap tiles are only assembly inputs.  Once the final
    # cropped reference image exists they are not part of the frozen manifest and
    # a refresh creates a fresh staged source tree anyway.
    reference_tiles = source_root / "reference" / "tiles"
    if (source_root / "reference" / "opentopomap.png").is_file() and reference_tiles.exists():
        try:
            _remove_stash(reference_tiles)
            removed.append(reference_tiles)
        except OSError:
            pass

    # If both cache names exist, resolve_cache_dir always chooses the new name.
    # The legacy tree therefore has no reader left and can be removed wholesale.
    current_cache = source_root / DEFAULT_CACHE_DIRNAME
    legacy_cache = source_root / LEGACY_CACHE_DIRNAME
    if current_cache.exists() and legacy_cache.exists() and source_cache != legacy_cache:
        try:
            _remove_stash(legacy_cache)
            removed.append(legacy_cache)
        except OSError:
            pass

    if source_cache.exists():
        for name in _LEGACY_SOURCE_BUILD_CACHE_DIRS:
            path = source_cache / name
            if not path.exists() and not path.is_symlink():
                continue
            try:
                _remove_stash(path)
                removed.append(path)
            except OSError:
                pass

        # Core-generator OSM rasters historically shared the source-side spatial
        # directory.  Preserve the actual source spatial index while dropping the
        # old raster-* entries that now belong exclusively to the build cache.
        spatial = source_cache / "spatial"
        if spatial.is_dir():
            for path in spatial.glob("raster-*.pickle"):
                try:
                    path.unlink()
                    removed.append(path)
                except OSError:
                    pass

        # Fingerprinted source-stage products are useful across nearby rebuilds,
        # but keeping every historical fingerprint forever is not.  Two versions
        # retain a practical rollback/retry window without unbounded accumulation.
        for directory in (source_cache / "sources", source_cache / "overture-conflation"):
            if directory.is_dir():
                removed.extend(_prune_files(directory.glob("*.pickle"), keep=_SOURCE_CACHE_HISTORY_KEEP))
        removed.extend(_prune_files(source_cache.glob("overture-buildings-*.geojson"), keep=_SOURCE_CACHE_HISTORY_KEEP))

    # Source fetching intentionally keeps resumable Overture tiles outside each
    # frozen world. Release directories are immutable, so only the newest two are
    # worth carrying indefinitely for normal unpinned operation.
    removed.extend(
        _prune_overture_releases(source_root.parent / ".cwr-worldgen-cache" / "overture")
    )
    return tuple(removed)


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
        cleanup_source_storage(
            getattr(spec, "source_dir", "."),
            getattr(spec, "cache_dir", None),
        )
        result = original_build9(output_dir, spec, clean=clean)
        _rewrite_cache_report(result, output_dir, spec)
        return result

    milestone9.build_milestone9 = build_milestone9
    _INSTALLED = True
