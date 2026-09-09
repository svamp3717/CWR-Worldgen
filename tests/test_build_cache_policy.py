from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from cwr_worldgen import generator
from cwr_worldgen.build_cache_policy import (
    BUILD_CACHE_DIRNAME,
    BUILD_CACHE_REVISION,
    build_cache_dir,
    cleanup_source_storage,
    install_build_cache_policy,
    route_build_cache_spec,
)


@dataclass(frozen=True)
class _CacheSpec:
    cache_dir: Path | None = None
    cache_enabled: bool = True


def test_build_cache_path_is_owned_by_selected_build_folder(tmp_path: Path) -> None:
    output = tmp_path / "build" / "demo"
    assert build_cache_dir(output) == (
        output / BUILD_CACHE_DIRNAME / BUILD_CACHE_REVISION
    ).resolve()


def test_playability_cache_is_routed_away_from_source_cache(tmp_path: Path) -> None:
    source_cache = tmp_path / "source" / ".cwr-worldgen-source-cache"
    output = tmp_path / "build" / "demo"
    routed = route_build_cache_spec(output, _CacheSpec(cache_dir=source_cache))

    assert routed.cache_dir == (
        output / BUILD_CACHE_DIRNAME / BUILD_CACHE_REVISION
    ).resolve()
    assert routed.cache_dir != source_cache


def test_bridge_runtime_revision_does_not_reuse_legacy_build_cache_root(tmp_path: Path) -> None:
    output = tmp_path / "build" / "demo"
    legacy = (output / BUILD_CACHE_DIRNAME).resolve()
    assert build_cache_dir(output).parent == legacy
    assert build_cache_dir(output) != legacy


def test_source_storage_cleanup_prunes_only_disposable_history(tmp_path: Path) -> None:
    source = tmp_path / "source-data" / "demo"
    source.mkdir(parents=True)
    cache = source / ".cwr-worldgen-source-cache"
    cache.mkdir()

    # Final reference mosaic survives; its downloaded assembly tiles do not.
    reference = source / "reference"
    (reference / "tiles" / "opentopomap" / "12").mkdir(parents=True)
    (reference / "tiles" / "opentopomap" / "12" / "tile.png").write_bytes(b"tile")
    (reference / "opentopomap.png").write_bytes(b"mosaic")

    # A superseded cache name is unreachable whenever the current cache exists.
    legacy = source / ".cwr-cache"
    legacy.mkdir()
    (legacy / "old.bin").write_bytes(b"old")

    for name in ("assets", "dem", "overview", "pbo", "placements", "procedural-assets", "surfaces"):
        directory = cache / name
        directory.mkdir()
        (directory / "old.bin").write_bytes(b"old")

    # Keep the source-side spatial index, but remove the old core-generator raster.
    spatial = cache / "spatial"
    spatial.mkdir()
    (spatial / "index-current.pickle").write_bytes(b"index")
    (spatial / "raster-old.pickle").write_bytes(b"raster")

    def historical_files(directory: Path, stem: str, suffix: str) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index in range(4):
            path = directory / f"{stem}-{index}{suffix}"
            path.write_bytes(str(index).encode("ascii"))
            os.utime(path, ns=(index + 1, index + 1))
            paths.append(path)
        return paths

    parsed = historical_files(cache / "sources", "dataset", ".pickle")
    conflation = historical_files(cache / "overture-conflation", "merged", ".pickle")
    overture_outputs = historical_files(cache, "overture-buildings", ".geojson")

    overture_root = source.parent / ".cwr-worldgen-cache" / "overture"
    for release in ("2026-06-01.0", "2026-07-22.0", "2026-09-01.0"):
        directory = overture_root / release
        directory.mkdir(parents=True)
        (directory / "collections.parquet").write_bytes(b"index")

    dem_shared = source.parent / ".dem-stitcher-cache" / "glo_30" / "tile.tif"
    dem_shared.parent.mkdir(parents=True)
    dem_shared.write_bytes(b"dem")

    cleanup_source_storage(source)

    assert (reference / "opentopomap.png").is_file()
    assert not (reference / "tiles").exists()
    assert not legacy.exists()
    for name in ("assets", "dem", "overview", "pbo", "placements", "procedural-assets", "surfaces"):
        assert not (cache / name).exists()
    assert (spatial / "index-current.pickle").is_file()
    assert not (spatial / "raster-old.pickle").exists()
    assert [path.exists() for path in parsed] == [False, False, True, True]
    assert [path.exists() for path in conflation] == [False, False, True, True]
    assert [path.exists() for path in overture_outputs] == [False, False, True, True]
    assert not (overture_root / "2026-06-01.0").exists()
    assert (overture_root / "2026-07-22.0").is_dir()
    assert (overture_root / "2026-09-01.0").is_dir()
    assert dem_shared.is_file()


def test_clean_build_preserves_existing_build_cache(tmp_path: Path) -> None:
    install_build_cache_policy()
    output = tmp_path / "build" / "demo"
    cache = output / BUILD_CACHE_DIRNAME
    cache.mkdir(parents=True)
    marker = cache / "terrain" / "cached.pickle"
    marker.parent.mkdir()
    marker.write_bytes(b"cached")
    disposable = output / "source" / "old.txt"
    disposable.parent.mkdir()
    disposable.write_text("old", encoding="utf-8")

    generator.prepare_output_directory(output, "wg_demo", clean=True)

    assert marker.read_bytes() == b"cached"
    assert not disposable.exists()


def test_nonclean_build_does_not_need_cache_stashing(tmp_path: Path) -> None:
    install_build_cache_policy()
    output = tmp_path / "build" / "demo"
    cache = output / BUILD_CACHE_DIRNAME
    cache.mkdir(parents=True)
    marker = cache / "pbo" / "blob.bin"
    marker.parent.mkdir()
    marker.write_bytes(b"blob")

    generator.prepare_output_directory(output, "wg_demo", clean=False)

    assert marker.read_bytes() == b"blob"
