from __future__ import annotations

from pathlib import Path
import sys

import zstandard

from cwr_worldgen.pbo import PboEntry, write_pbo

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from p3d_asset_browser_catalog import (
    filter_assets,
    model_source,
    scan_catalogue,
    texture_users,
)


def _fake_p3d(texture: str) -> bytes:
    # Asset catalogue dependency extraction is deliberately format-light and
    # searches embedded PAA/PAC references before full preview parsing.
    return b"ODOL" + b"\0" * 32 + texture.encode("ascii") + b"\0" * 32


def test_browser_catalogue_lists_models_and_textures_from_pbo(tmp_path: Path) -> None:
    pbo = tmp_path / "roads.pbo"
    write_pbo(
        pbo,
        (
            PboEntry("house.p3d", _fake_p3d(r"roads\wall.paa")),
            PboEntry("wall.paa", b"synthetic-paa"),
        ),
    )

    catalogue = scan_catalogue((pbo,), use_cache=False)

    assert [item.path for item in catalogue.models] == [r"roads\house.p3d"]
    assert [item.path for item in catalogue.textures] == [r"roads\wall.paa"]
    assert texture_users(r"roads\wall.paa", catalogue) == catalogue.models
    assert filter_assets(catalogue, "house", ("model",)) == catalogue.models
    assert filter_assets(catalogue, "wall", ("texture",)) == catalogue.textures


def test_browser_catalogue_scans_nested_pbo_inside_zstd_package(tmp_path: Path) -> None:
    inner = tmp_path / "inner.pbo"
    write_pbo(
        inner,
        (
            PboEntry("thing.p3d", _fake_p3d(r"inner\thing.paa")),
            PboEntry("thing.paa", b"synthetic-paa"),
        ),
    )
    outer = tmp_path / "outer.pbo"
    write_pbo(
        outer,
        (
            PboEntry(r"addons\inner.pbo", inner.read_bytes()),
            PboEntry(r"lib_models\direct.p3d", _fake_p3d(r"inner\thing.paa")),
        ),
    )
    wrapped = tmp_path / "package.pbo.zst"
    wrapped.write_bytes(
        zstandard.ZstdCompressor(level=1).compress(outer.read_bytes())
    )

    catalogue = scan_catalogue((wrapped,), use_cache=False)
    model_paths = {item.path for item in catalogue.models}
    texture_paths = {item.path for item in catalogue.textures}

    assert r"inner\thing.p3d" in model_paths
    assert r"lib_models\direct.p3d" in model_paths
    assert r"inner\thing.paa" in texture_paths

    nested_model = next(
        item for item in catalogue.models if item.path == r"inner\thing.p3d"
    )
    assert nested_model.source.endswith(r"!addons\inner.pbo")
    assert model_source(nested_model).endswith(
        r"!addons\inner.pbo!inner\thing.p3d"
    )

    users = {item.path for item in texture_users(r"inner\thing.paa", catalogue)}
    assert r"inner\thing.p3d" in users
    assert r"lib_models\direct.p3d" in users


def test_browser_search_matches_source_as_well_as_asset_path(tmp_path: Path) -> None:
    root = tmp_path / "SpecialMod"
    root.mkdir()
    (root / "tree.p3d").write_bytes(_fake_p3d(r"data\tree.paa"))
    (root / "tree.paa").write_bytes(b"synthetic-paa")

    catalogue = scan_catalogue((root,), use_cache=False)

    assert len(filter_assets(catalogue, "specialmod", ("model", "texture"))) == 2
    assert filter_assets(catalogue, "does-not-exist") == ()
