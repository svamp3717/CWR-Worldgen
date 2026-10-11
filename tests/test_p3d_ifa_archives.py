"""End-to-end Iron Front IFA archive support for browse_p3d_assets.

Iron Front .ifa files contain a PBO-compatible addon archive; changing an
extension must not cause the browser to treat the archive as a loose model.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import zstandard

from cwr_worldgen import assets
from cwr_worldgen.pbo import (
    PboEntry, is_pbo_path, pbo_stem, open_pbo_stream, write_pbo,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from p3d_asset_browser_catalog import model_source, scan_catalogue
from p3d_texture_sources import TextureResolver, sibling_namespace_pbos


MODEL = b"ODOL" + bytes(16) + b"lib_textures\\brick.paa\x00" + bytes(12)
TEXTURE = b"synthetic-texture-payload"


def _ifa(tmp_path: Path, name: str = "lib_buildings") -> Path:
    archive = tmp_path / (name + ".ifa")
    write_pbo(
        archive,
        (
            PboEntry("house.p3d", MODEL),
            PboEntry("brick.paa", TEXTURE),
        ),
    )
    return archive


@pytest.mark.parametrize("extension", [".ifa", ".IFA", ".ifa.zst", ".IFA.ZST"])
def test_ifa_is_pbo_compatible_archive(extension: str) -> None:
    filename = "LIB_Buildings" + extension
    assert is_pbo_path(filename)
    assert pbo_stem(filename) == "LIB_Buildings"


@pytest.mark.parametrize("wrapped", [False, True])
def test_browse_ifa_lists_and_reads_model_and_texture(
    tmp_path: Path, wrapped: bool,
) -> None:
    raw = _ifa(tmp_path)
    archive = raw
    if wrapped:
        archive = tmp_path / "lib_buildings.ifa.zst"
        archive.write_bytes(zstandard.ZstdCompressor(level=1).compress(raw.read_bytes()))
        raw.unlink()

    original = archive.read_bytes()
    with open_pbo_stream(archive) as stream:
        assert stream.read(4) != b"ODOL"  # archive header, not loose P3D

    catalogue = scan_catalogue((archive,), use_cache=False)
    models, textures = catalogue.models, catalogue.textures
    assert [item.path for item in models] == [r"lib_buildings\house.p3d"]
    assert [item.path for item in textures] == [r"lib_buildings\brick.paa"]
    assert model_source(models[0]) == f"{archive}!lib_buildings\\house.p3d"
    assert assets.read_asset_record_bytes(catalogue.record_for(models[0])) == MODEL
    assert assets.read_asset_record_bytes(catalogue.record_for(textures[0])) == TEXTURE

    resolver = TextureResolver((archive,))
    assert resolver.load_bytes(r"lib_buildings\brick.paa", model_source(models[0])) == TEXTURE
    assert archive.read_bytes() == original


def test_folder_scanning_discovers_ifa_and_pbo(tmp_path: Path) -> None:
    folder = tmp_path / "addons"
    folder.mkdir()
    ifa = _ifa(folder, "lib_buildings")
    pbo = folder / "lib_trees.pbo"
    write_pbo(pbo, (PboEntry("oak.p3d", MODEL),))
    old_bytes = ifa.read_bytes()

    catalogue = scan_catalogue((folder,), use_cache=False)
    assert {item.path for item in catalogue.models} == {
        r"lib_buildings\house.p3d",
        r"lib_trees\oak.p3d",
    }
    assert {item.path for item in catalogue.textures} == {r"lib_buildings\brick.paa"}
    assert ifa.read_bytes() == old_bytes


def test_folder_scanning_discovers_wrapped_ifa(tmp_path: Path) -> None:
    raw = _ifa(tmp_path)
    wrapped = tmp_path / "lib_buildings.ifa.zst"
    wrapped.write_bytes(zstandard.ZstdCompressor().compress(raw.read_bytes()))
    raw.unlink()

    catalogue = scan_catalogue((tmp_path,), use_cache=False)
    assert [item.path for item in catalogue.models] == [r"lib_buildings\house.p3d"]
    assert [item.path for item in catalogue.textures] == [r"lib_buildings\brick.paa"]


def test_nested_ifa_addon_is_read_as_archive(tmp_path: Path) -> None:
    nested = _ifa(tmp_path, "lib_buildings")
    outer = tmp_path / "iron_front_mod.pbo"
    write_pbo(outer, (PboEntry(r"addons\lib_buildings.ifa", nested.read_bytes()),))
    original = outer.read_bytes()

    catalogue = scan_catalogue((outer,), use_cache=False)
    assert [item.path for item in catalogue.models] == [r"lib_buildings\house.p3d"]
    assert [item.path for item in catalogue.textures] == [r"lib_buildings\brick.paa"]
    record = catalogue.record_for(catalogue.models[0])
    assert record.source.endswith(r"!addons\lib_buildings.ifa")
    assert assets.read_asset_record_bytes(record) == MODEL

    resolver = TextureResolver((outer,))
    assert resolver.load_bytes(
        r"lib_buildings\brick.paa", model_source(catalogue.models[0])
    ) == TEXTURE
    assert outer.read_bytes() == original


def test_ifa_model_can_resolve_texture_in_sibling_ifa(tmp_path: Path) -> None:
    model_archive = _ifa(tmp_path, "lib_buildings")
    textures = tmp_path / "lib_textures.ifa"
    write_pbo(textures, (PboEntry("brick.paa", TEXTURE),))
    original = textures.read_bytes()

    assert sibling_namespace_pbos(model_archive, r"lib_textures\brick.paa") == (textures,)
    resolver = TextureResolver((model_archive,))
    model_source_path = f"{model_archive}!lib_buildings\\house.p3d"
    assert resolver.load_bytes(r"lib_textures\brick.paa", model_source_path) == TEXTURE
    assert textures.resolve() in resolver.auto_sibling_pbos
    assert textures.read_bytes() == original


def test_ifa_and_pbo_archives_are_both_accepted_by_shared_predicate() -> None:
    for suffix in (".pbo", ".pbo.zst", ".ifa", ".ifa.zst"):
        assert is_pbo_path("LIB_Objects" + suffix)
    for suffix in (".p3d", ".paa", ".wrp", ".zip", ".ifa.bak"):
        assert not is_pbo_path("LIB_Objects" + suffix)
