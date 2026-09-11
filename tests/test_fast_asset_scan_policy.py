from __future__ import annotations

from pathlib import Path

from cwr_worldgen.fast_asset_scan_policy import (
    _PBO_INDEX_MEMORY,
    scan_assets_fast,
)
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.shared_cache_policy import (
    ASSET_INDEX_CACHE_DIRNAME,
    SHARED_CACHE_DIRNAME,
    shared_asset_index_cache_dir,
)


def _game_fixture(root: Path) -> Path:
    game = root / "CWA"
    dta = game / "Dta"
    dta.mkdir(parents=True)
    write_pbo(
        dta / "Data3D.pbo",
        (
            PboEntry("selected.p3d", b"data\\selected.paa\0"),
            PboEntry("irrelevant.p3d", b"data\\irrelevant.paa\0"),
        ),
    )
    write_pbo(
        dta / "Data.pbo",
        (
            PboEntry("selected.paa", b"selected-texture"),
            PboEntry("irrelevant.paa", b"irrelevant-texture"),
        ),
    )
    return game


def _build_cache(root: Path, world: str) -> Path:
    return root / world / ".cwr-worldgen-build-cache" / "v2-active-bridge-policies"


def test_normal_cwa_layout_uses_targeted_packages_and_shared_index_cache(
    tmp_path, monkeypatch
) -> None:
    game = _game_fixture(tmp_path)
    cache_dir = _build_cache(tmp_path, "WorldA")

    def full_scan_forbidden(*_args, **_kwargs):
        raise AssertionError("normal CWA asset lookup must not fall back to recursive scan")

    import cwr_worldgen.fast_asset_scan_policy as fast

    monkeypatch.setattr(fast, "_FULL_SCAN", full_scan_forbidden)
    result = scan_assets_fast(
        (game,),
        (r"data3d\selected.p3d",),
        cache_dir=cache_dir,
    )

    assert result.verified
    assert not result.missing_models
    assert not result.missing_dependencies
    assert {record.path for record in result.records} == {
        r"data3d\selected.p3d",
        r"data\selected.paa",
    }
    assert all("irrelevant" not in record.path for record in result.records)

    persistent = shared_asset_index_cache_dir(cache_dir)
    assert persistent == (
        tmp_path.resolve()
        / SHARED_CACHE_DIRNAME
        / ASSET_INDEX_CACHE_DIRNAME
    )
    assert persistent.is_dir()
    assert len(tuple((persistent / "pbo").glob("*.json"))) == 2


def test_different_world_reuses_persistent_pbo_header_indexes(
    tmp_path, monkeypatch
) -> None:
    game = _game_fixture(tmp_path)
    cache_a = _build_cache(tmp_path, "WorldA")
    cache_b = _build_cache(tmp_path, "WorldB")

    first = scan_assets_fast(
        (game,),
        (r"data3d\selected.p3d",),
        cache_dir=cache_a,
    )
    assert first.verified
    assert shared_asset_index_cache_dir(cache_a) == shared_asset_index_cache_dir(cache_b)
    _PBO_INDEX_MEMORY.clear()

    import cwr_worldgen.fast_asset_scan_policy as fast

    def parse_forbidden(_path):
        raise AssertionError("World B should reuse World A's PBO header index")

    def full_scan_forbidden(*_args, **_kwargs):
        raise AssertionError("cached targeted scan must not use recursive scanner")

    monkeypatch.setattr(fast, "_parse_pbo_index", parse_forbidden)
    monkeypatch.setattr(fast, "_FULL_SCAN", full_scan_forbidden)
    second = scan_assets_fast(
        (game,),
        (r"data3d\selected.p3d",),
        cache_dir=cache_b,
    )

    assert second.verified
    assert second.cache_hit
    assert second.catalogue_sha256 == first.catalogue_sha256
