from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
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


class _FakeScanRoot:
    def __init__(self):
        self.cursor = ""
        self.after_calls = []

    def configure(self, **kwargs):
        if "cursor" in kwargs:
            self.cursor = kwargs["cursor"]

    def update_idletasks(self):
        pass

    def after(self, delay, callback):
        self.after_calls.append(delay)
        callback()


class _FakeButton:
    def __init__(self):
        self.state = None

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)


class _FakeProgress:
    def __init__(self):
        self.running = False
        self.visible = False
        self.interval = None

    def pack(self, **_kwargs):
        self.visible = True

    def pack_forget(self):
        self.visible = False

    def start(self, interval=None):
        self.running = True
        self.interval = interval

    def stop(self):
        self.running = False


def test_select_pbo_replaces_previous_sources(monkeypatch, tmp_path: Path) -> None:
    old_pbo = tmp_path / "old.pbo"
    new_pbo = tmp_path / "new.pbo"
    old_pbo.write_bytes(b"old")
    new_pbo.write_bytes(b"new")

    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.root = object()
    app._scan_in_progress = False
    app.status_var = _FakeVar("")
    app.sources = [old_pbo]
    calls = []

    def fake_clear_sources():
        calls.append("clear")
        app.sources.clear()

    def fake_add_sources(paths):
        values = list(paths)
        calls.append(("add", values))
        app.sources.extend(values)

    app.clear_sources = fake_clear_sources
    app._add_sources = fake_add_sources
    monkeypatch.setattr(
        browser_app.filedialog,
        "askopenfilenames",
        lambda **_kwargs: (str(new_pbo),),
    )

    app.select_pbo()

    assert calls == ["clear", ("add", [new_pbo])]
    assert app.sources == [new_pbo]


def test_browser_scan_progress_runs_while_catalogue_is_built(monkeypatch, tmp_path: Path) -> None:
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.root = _FakeScanRoot()
    app.sources = [tmp_path / "roads.pbo"]
    app.catalogue = browser_app.BrowserCatalogue((), (), {})
    app.texture_resolver = object()
    app.current = None
    app.current_model = None
    app._scan_queue = __import__("queue").Queue()
    app._scan_generation = 0
    app._scan_in_progress = False
    app._source_buttons = [_FakeButton(), _FakeButton(), _FakeButton()]
    app.scan_progress = _FakeProgress()
    app.status_var = _FakeVar("")
    app.sources_var = _FakeVar("")
    app.info_var = _FakeVar("")
    app._refresh_lists = lambda: None
    app._clear_related = lambda: None
    app._show_empty_preview = lambda: None

    catalogue = browser_app.BrowserCatalogue(
        (app.sources[0],),
        (
            BrowserAsset("model", r"roads\road25.p3d", "roads.pbo", 123),
            BrowserAsset("texture", r"roads\road.paa", "roads.pbo", 456),
        ),
        {},
    )
    monkeypatch.setattr(browser_app, "scan_catalogue", lambda _sources: catalogue)
    monkeypatch.setattr(browser_app, "TextureResolver", lambda sources: tuple(sources))

    class ImmediateThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            assert app.scan_progress.running is True
            assert app.scan_progress.visible is True
            assert all(button.state == browser_app.tk.DISABLED for button in app._source_buttons)
            self.target()

    monkeypatch.setattr(browser_app.threading, "Thread", ImmediateThread)

    app._scan_sources()

    assert app._scan_in_progress is False
    assert app.scan_progress.running is False
    assert app.scan_progress.visible is False
    assert all(button.state == browser_app.tk.NORMAL for button in app._source_buttons)
    assert app.catalogue is catalogue
    assert "Indexed 1 model(s) and 1 texture(s)." == app.status_var.get()
    assert app.root.cursor == ""
    assert app.root.after_calls == [50]


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



def test_archive_picker_offers_ifa_and_replaces_previous_sources(
    monkeypatch, tmp_path: Path,
) -> None:
    old_pbo = tmp_path / "old.pbo"
    new_ifa = tmp_path / "new.ifa"
    old_pbo.touch()
    new_ifa.touch()
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.root = object()
    app._scan_in_progress = False
    app.status_var = _FakeVar("")
    app.sources = [old_pbo]
    app.clear_sources = lambda: app.sources.clear()
    app._add_sources = lambda paths: app.sources.extend(paths)

    selected = {}

    def fake_picker(**kwargs):
        selected.update(kwargs)
        return (str(new_ifa),)

    monkeypatch.setattr(browser_app.filedialog, "askopenfilenames", fake_picker)
    app.select_pbo()

    assert app.sources == [new_ifa]
    assert "IFA" in selected["title"]
    extensions = " ".join(pattern for _title, pattern in selected["filetypes"])
    assert "*.ifa" in extensions
    assert "*.ifa.zst" in extensions
    assert "*.pbo" in extensions



def test_add_multiple_texture_archives_without_rescanning_or_changing_models(
    tmp_path: Path, monkeypatch,
) -> None:
    model_pbo = tmp_path / "models.pbo"
    texture_pbo = tmp_path / "textures.pbo"
    texture_ifa = tmp_path / "textures.ifa.zst"
    for path in (model_pbo, texture_pbo, texture_ifa):
        path.touch()

    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.root = object()
    app._scan_in_progress = False
    app.sources = [model_pbo]
    app.texture_sources = []
    app.texture_sources_var = _FakeVar("")
    app.status_var = _FakeVar("")
    app.current_model = object()
    app.catalogue = object()
    original_catalogue = app.catalogue
    app.mipmap_cache = SimpleNamespace(clear=Mock())
    app._redraw_current_model = Mock()
    app._scan_sources = Mock()

    captured = {}

    def choose(**kwargs):
        captured.update(kwargs)
        return (str(texture_pbo), str(texture_ifa), str(texture_pbo))

    monkeypatch.setattr(browser_app.filedialog, "askopenfilenames", choose)
    app.select_texture_archives()

    assert app.texture_sources == [texture_pbo, texture_ifa]
    assert app.texture_resolver.inputs == (model_pbo, texture_pbo, texture_ifa)
    assert app.catalogue is original_catalogue
    app._scan_sources.assert_not_called()
    app._redraw_current_model.assert_called_once_with()
    app.mipmap_cache.clear.assert_called_once_with()
    assert "2 extra texture archive(s)" == app.texture_sources_var.get()
    assert "*.ifa.zst" in " ".join(pattern for _, pattern in captured["filetypes"])

    app._add_texture_archives((texture_ifa,))
    assert app.texture_sources == [texture_pbo, texture_ifa]
    app._scan_sources.assert_not_called()
    app._redraw_current_model.assert_called_once_with()

    app.clear_texture_archives()
    assert app.texture_sources == []
    assert app.texture_resolver.inputs == (model_pbo,)
    assert app.catalogue is original_catalogue
    assert app._redraw_current_model.call_count == 2
    app._scan_sources.assert_not_called()


def test_texture_archive_picker_rejects_unrelated_files_atomically(tmp_path: Path) -> None:
    model_pbo = tmp_path / "models.ifa"
    archive = tmp_path / "textures.pbo"
    invalid = tmp_path / "textures.zip"
    for path in (model_pbo, archive, invalid):
        path.touch()
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.sources = [model_pbo]
    app.texture_sources = []
    app.status_var = _FakeVar("")
    app._refresh_texture_resolver = lambda: (_ for _ in ()).throw(
        AssertionError("invalid files must not replace the resolver")
    )
    app._add_texture_archives((archive, invalid))
    assert app.texture_sources == []
    assert "not a pbo / ifa" in app.status_var.get().lower()



def test_replacing_model_pbos_preserves_selected_extra_texture_archives(
    tmp_path: Path, monkeypatch,
) -> None:
    old_model = tmp_path / "old.pbo"
    new_model = tmp_path / "new.ifa"
    external_texture = tmp_path / "external.pbo"
    for path in (old_model, new_model, external_texture):
        path.touch()
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.root = object()
    app._scan_in_progress = False
    app.status_var = _FakeVar("")
    app.texture_sources_var = _FakeVar("")
    app.sources = [old_model]
    app.texture_sources = [external_texture]
    app.clear_sources = lambda: (app.sources.clear(), app.texture_sources.clear())
    app._add_sources = lambda values: app.sources.extend(values)
    monkeypatch.setattr(
        browser_app.filedialog, "askopenfilenames",
        lambda **kwargs: (str(new_model),),
    )

    app.select_pbo()

    assert app.sources == [new_model]
    assert app.texture_sources == [external_texture]
    assert app.texture_sources_var.get() == "1 extra texture archive(s)"



class _RelatedAssetTree:
    def __init__(self):
        self.rows = {}

    def get_children(self):
        return tuple(self.rows)

    def delete(self, iid):
        self.rows.pop(iid)

    def insert(self, _parent, _position, *, iid, values):
        self.rows[iid] = values


def test_related_texture_status_checks_external_pbos_instead_of_catalogue_only(
    tmp_path: Path,
) -> None:
    model_pbo = tmp_path / "models.pbo"
    external_pbo = tmp_path / "textures.ifa"
    write_pbo(model_pbo, (
        PboEntry("model.p3d", _fake_p3d(r"textures\outside.paa")),
        PboEntry("local.paa", b"local-paa"),
    ))
    write_pbo(external_pbo, (PboEntry("outside.paa", b"external-paa"),))
    catalogue = scan_catalogue((model_pbo,), use_cache=False)

    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.catalogue = catalogue
    app.texture_resolver = browser_app.TextureResolver((model_pbo, external_pbo))
    app.current_model = SimpleNamespace(
        source=f"{model_pbo}!models\\model.p3d",
        texture_paths=(r"models\local.paa", r"textures\outside.paa"),
    )
    app.skip_textures_var = _FakeVar(False)
    app._related_assets = {}
    app.related_tree = _RelatedAssetTree()

    paths = (r"models\local.paa", r"textures\outside.paa", r"textures\absent.paa")
    app._set_related_textures(paths)
    assert [value[0] for value in app.related_tree.rows.values()] == [
        "Texture", "External", "Missing",
    ]
    assert "r0" in app._related_assets
    assert "r1" not in app._related_assets  # Not part of model catalogue.
    assert "r2" not in app._related_assets

    # The resolver may locate an archive entry even if its PAA cannot decode.
    app.texture_resolver.image_cache[r"textures\outside.paa"] = None
    app._set_related_textures(paths)
    assert app.related_tree.rows["r1"][0] == "Unreadable"

    # Skip textures is not evidence that a texture is missing or unreadable.
    app.skip_textures_var.set(True)
    app._set_related_textures(paths)
    assert [value[0] for value in app.related_tree.rows.values()] == [
        "Texture", "Unchecked", "Unchecked",
    ]


def test_extra_texture_archive_change_refreshes_related_status_without_model_scan(
    tmp_path: Path,
) -> None:
    model_pbo = tmp_path / "models.pbo"
    external_pbo = tmp_path / "textures.pbo"
    write_pbo(model_pbo, (
        PboEntry("model.p3d", _fake_p3d(r"textures\outside.paa")),
    ))
    write_pbo(external_pbo, (
        PboEntry("outside.paa", b"external-paa"),
    ))

    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.sources = [model_pbo]
    app.texture_sources = [external_pbo]
    app.catalogue = scan_catalogue((model_pbo,), use_cache=False)
    app.texture_resolver = browser_app.TextureResolver((model_pbo, external_pbo))
    app.current = BrowserAsset("model", r"models\model.p3d", str(model_pbo), 24)
    app.current_model = SimpleNamespace(
        source=f"{model_pbo}!models\\model.p3d",
        texture_paths=(r"textures\outside.paa",),
    )
    app._related_assets = {}
    app.related_tree = _RelatedAssetTree()
    app.skip_textures_var = _FakeVar(False)
    app.texture_sources_var = _FakeVar("")
    app.status_var = _FakeVar("")
    app.mipmap_cache = SimpleNamespace(clear=Mock())
    app._draw_model = Mock()
    app._scan_sources = Mock()

    app._redraw_current_model()
    assert app.related_tree.rows["r0"][0] == "External"
    app.clear_texture_archives()
    assert app.related_tree.rows["r0"][0] == "Missing"
    app._scan_sources.assert_not_called()
    assert app._draw_model.call_count == 2
    assert app.texture_resolver.inputs == (model_pbo,)



def test_explorer_selected_asset_details_and_timing_share_selectable_readonly_text() -> None:
    """The whole Selected asset readout can be mouse-selected and Ctrl+C copied."""
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.info_var = _FakeVar("MODEL\nhouse.p3d\nSource: models.pbo")
    app.load_timing_var = _FakeVar("P3D read: 3.0 ms\nTotal: 6.0 ms")
    text = Mock()
    app.selected_asset_text = text

    app._refresh_selected_asset_text()
    text.configure.assert_any_call(state=browser_app.tk.NORMAL)
    text.configure.assert_any_call(state=browser_app.tk.DISABLED)
    text.delete.assert_called_once_with("1.0", browser_app.tk.END)
    text.insert.assert_called_once_with(
        "1.0",
        "MODEL\nhouse.p3d\nSource: models.pbo"
        "\n\nP3D read: 3.0 ms\nTotal: 6.0 ms",
    )
    assert text.configure.call_args.kwargs == {"state": browser_app.tk.DISABLED}

    # Source and timing values are updated independently in the real browser.
    text.reset_mock()
    app.info_var.set("TEXTURE\nwall.paa")
    app.load_timing_var.set("Texture read/decode: 1.0 ms")
    app._refresh_selected_asset_text()
    text.insert.assert_called_once_with(
        "1.0", "TEXTURE\nwall.paa\n\nTexture read/decode: 1.0 ms"
    )


def test_explorer_selected_asset_copy_copies_only_highlighted_selection() -> None:
    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.root = _FakeClipboardRoot()
    text = Mock()
    text.get.return_value = "models.pbo!house.p3d"
    event = SimpleNamespace(widget=text)

    assert app._copy_selected_asset_text(event) == "break"
    text.get.assert_called_once_with("sel.first", "sel.last")
    assert app.root.value == "models.pbo!house.p3d"

    # Ctrl+C on an unselected (read-only) Text should do nothing.
    text.get.side_effect = browser_app.tk.TclError("text has no selection")
    assert app._copy_selected_asset_text(event) == "break"
    assert app.root.value == "models.pbo!house.p3d"


def test_explorer_selected_asset_ctrl_a_highlights_full_readout() -> None:
    widget = Mock()
    event = SimpleNamespace(widget=widget)
    assert browser_app.AssetBrowserApp._select_all_asset_text(event) == "break"
    widget.tag_add.assert_called_once_with("sel", "1.0", "end-1c")
    widget.mark_set.assert_called_once_with("insert", "1.0")
    widget.see.assert_called_once_with("insert")
