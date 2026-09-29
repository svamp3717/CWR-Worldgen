from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import zstandard

from cwr_worldgen.pbo import PboEntry, write_pbo

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import p3d_asset_browser_app as browser_app
from p3d_asset_browser_catalog import (
    BrowserAsset,
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


class _FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeAxis:
    def clear(self):
        pass

    def imshow(self, *_args, **_kwargs):
        pass

    def set_title(self, value):
        self.title = value

    def axis(self, *_args, **_kwargs):
        pass


class _FakeCanvas:
    def draw_idle(self):
        pass


class _FakeFigure:
    def tight_layout(self, **_kwargs):
        pass


class _FakeClipboardRoot:
    def __init__(self):
        self.value = ""

    def clipboard_clear(self):
        self.value = ""

    def clipboard_append(self, value):
        self.value += value

    def update_idletasks(self):
        pass


def test_browser_skip_textures_disables_texture_loading(monkeypatch) -> None:
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.ax_preview = _FakeAxis()
    app.canvas = _FakeCanvas()
    app.figure = _FakeFigure()
    app.texture_resolver = object()
    app.skip_textures_var = _FakeVar(True)
    app.azim = 35.0
    app.elev = 25.0
    app.zoom = 1.0

    captured = {}

    def fake_render(*args, **kwargs):
        captured.update(kwargs)
        return (
            __import__("numpy").zeros((4, 4, 3), dtype="uint8"),
            0,
            0,
        )

    monkeypatch.setattr(browser_app, "render_textured_model", fake_render)
    model = SimpleNamespace(
        points=__import__("numpy").zeros((3, 3), dtype="float32"),
        faces=(object(),),
        source="package.pbo!a\\road\\curve.p3d",
        model_path=r"a\road\curve.p3d",
    )

    app._draw_model(model)

    assert captured["load_textures"] is False
    assert "textures skipped" in app.ax_preview.title


def test_browser_copy_path_uses_worldgen_asset_path() -> None:
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.root = _FakeClipboardRoot()
    app.status_var = _FakeVar("")
    app.current = BrowserAsset(
        kind="model",
        path=r"a\road\modroad25.p3d",
        source=r"C:\mods\roads.pbo.zst!addons\roads.pbo",
        size=123,
    )

    app.copy_current_path()

    assert app.root.value == r"a\road\modroad25.p3d"
    assert "a\\road\\modroad25.p3d" in app.status_var.get()
