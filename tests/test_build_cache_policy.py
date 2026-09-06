from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cwr_worldgen import generator
from cwr_worldgen.build_cache_policy import (
    BUILD_CACHE_DIRNAME,
    build_cache_dir,
    install_build_cache_policy,
    route_build_cache_spec,
)


@dataclass(frozen=True)
class _CacheSpec:
    cache_dir: Path | None = None
    cache_enabled: bool = True


def test_build_cache_path_is_owned_by_selected_build_folder(tmp_path: Path) -> None:
    output = tmp_path / "build" / "demo"
    assert build_cache_dir(output) == (output / BUILD_CACHE_DIRNAME).resolve()


def test_playability_cache_is_routed_away_from_source_cache(tmp_path: Path) -> None:
    source_cache = tmp_path / "source" / ".cwr-worldgen-source-cache"
    output = tmp_path / "build" / "demo"
    routed = route_build_cache_spec(output, _CacheSpec(cache_dir=source_cache))

    assert routed.cache_dir == (output / BUILD_CACHE_DIRNAME).resolve()
    assert routed.cache_dir != source_cache


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
