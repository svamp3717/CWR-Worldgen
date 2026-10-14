"""Performance-path regression coverage for P3D preview loading.

These tests compare the NumPy BCn batch decoder to the original per-block
reference, verify bounded caches, and ensure rendering matches the actual
Tk canvas resolution instead of always computing a huge 1400x980 image.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import p3d_asset_browser_app as app_module
import p3d_asset_browser_catalog as catalog
from p3d_texture_io import TextureResolver, _decode_dxt, _dxt1_block
from p3d_texture_render import MipmapCache


def _reference_dxt(data: bytes, width: int, height: int, kind: str) -> np.ndarray:
    """Original scalar implementation, independent of NumPy batch indexing."""
    block_bytes = 8 if kind == "DXT1" else 16
    bw, bh = (width + 3) // 4, (height + 3) // 4
    output = np.zeros((height, width, 4), dtype=np.uint8)
    for by in range(bh):
        for bx in range(bw):
            pos = (by * bw + bx) * block_bytes
            chunk = data[pos:pos + block_bytes]
            if kind == "DXT1":
                rgba = _dxt1_block(chunk, True)
            else:
                rgba = _dxt1_block(chunk[8:], False)
                if kind in ("DXT2", "DXT3"):
                    for i in range(16):
                        rgba[i // 4, i % 4, 3] = (
                            (chunk[i // 2] >> (4 * (i & 1))) & 15
                        ) * 17
                else:
                    first, second = chunk[:2]
                    alphas = [first, second]
                    if first > second:
                        alphas.extend(
                            ((7 - i) * first + i * second + 3) // 7
                            for i in range(1, 7)
                        )
                    else:
                        alphas.extend(
                            ((5 - i) * first + i * second + 2) // 5
                            for i in range(1, 5)
                        )
                        alphas.extend((0, 255))
                    bits = int.from_bytes(chunk[2:8], "little")
                    for i in range(16):
                        rgba[i // 4, i % 4, 3] = alphas[(bits >> (3 * i)) & 7]
            y0, x0 = by * 4, bx * 4
            h, w = min(4, height - y0), min(4, width - x0)
            output[y0:y0 + h, x0:x0 + w] = rgba[:h, :w]
    return output


@pytest.mark.parametrize("kind", ["DXT1", "DXT2", "DXT3", "DXT4", "DXT5"])
@pytest.mark.parametrize("width,height", [(4, 4), (7, 9), (16, 8), (3, 5)])
def test_vector_dxt_matches_reference_for_odd_dimensions_and_alpha(
    kind: str, width: int, height: int,
) -> None:
    n_blocks = ((width + 3) // 4) * ((height + 3) // 4)
    width_bytes = 8 if kind == "DXT1" else 16
    rng = np.random.default_rng(19321)
    payload = rng.integers(0, 256, n_blocks * width_bytes, dtype=np.uint8).tobytes()
    expected = _reference_dxt(payload, width, height, kind)
    result = _decode_dxt(payload, width, height, kind)
    assert result.shape == (height, width, 4)
    np.testing.assert_array_equal(result, expected)


def test_large_dxt_texture_can_be_decoded_in_batches() -> None:
    # 1024x1024 = 65,536 blocks. The old code created over one million
    # Python pixel loops; the new decoder should handle this as NumPy batches.
    red = bytes.fromhex("00f8000000000000")
    payload = red * (256 * 256)
    result = _decode_dxt(payload, 1024, 1024, "DXT1")
    assert result.shape == (1024, 1024, 4)
    assert tuple(result[100, 900]) == (255, 0, 0, 255)


class _CountingTextureResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def load_rgba(self, path: str, source: str) -> np.ndarray:
        self.calls.append((path, source))
        return np.full((8, 8, 4), (20, 40, 60, 255), dtype=np.uint8)


def test_mipmap_cache_reuses_levels_and_evicts_within_budget() -> None:
    resolver = _CountingTextureResolver()
    cache = MipmapCache(max_bytes=350)  # one 8x8 RGBA pyramid (340 bytes)
    one = cache.get("a.paa", "models.pbo", resolver)
    assert one is not None
    assert cache.get("a.paa", "models.pbo", resolver) is one
    assert len(resolver.calls) == 1
    cache.get("b.paa", "models.pbo", resolver)
    assert len(resolver.calls) == 2
    assert cache.bytes_used <= 350
    cache.get("a.paa", "models.pbo", resolver)
    assert len(resolver.calls) == 3
    assert cache.bytes_used <= 350
    cache.clear()
    assert cache.bytes_used == 0
    assert cache._items == {}


def test_mipmap_cache_does_not_confuse_two_sources() -> None:
    resolver = _CountingTextureResolver()
    cache = MipmapCache(max_bytes=2000)
    cache.get("ca\\data\\wall.paa", "one.pbo", resolver)
    cache.get("ca\\data\\wall.paa", "two.pbo", resolver)
    assert len(resolver.calls) == 2


def _asset(name: str, *, kind: str = "model", size: int = 12) -> catalog.BrowserAsset:
    return catalog.BrowserAsset(kind=kind, path=name, source="assets.pbo", size=size)


def test_model_preview_cache_reuses_parsed_mesh_and_respects_lru(monkeypatch) -> None:
    calls = []

    def fake_load(item, catalogue):
        calls.append(item.path)
        return SimpleNamespace(
            points=np.zeros((2, 3), dtype=np.float32),
            edges=np.zeros((1, 2), dtype=np.int32),
            faces=(), model_path=item.path,
        )

    monkeypatch.setattr(catalog, "load_model", fake_load)
    cache = catalog.ModelPreviewCache(max_models=2, max_bytes=1000)
    fixture = catalog.BrowserCatalogue((), (), {})
    a, b, c = (_asset(v) for v in ("a.p3d", "b.p3d", "c.p3d"))
    first = cache.get(a, fixture)
    assert cache.get(a, fixture) is first
    cache.get(b, fixture)
    assert calls == ["a.p3d", "b.p3d"]
    cache.get(a, fixture)  # mark A as recently used
    cache.get(c, fixture)  # B must be evicted
    cache.get(b, fixture)
    assert calls == ["a.p3d", "b.p3d", "c.p3d", "b.p3d"]
    assert len(cache._items) == 2
    assert cache.bytes_used <= cache.max_bytes
    cache.clear()
    assert cache.bytes_used == 0


def test_standalone_texture_cache_reuses_rgba_without_rereading_pbo(monkeypatch) -> None:
    calls = []

    def fake_load(item, catalogue):
        calls.append(item.path)
        return np.full((8, 8, 4), 173, dtype=np.uint8)

    monkeypatch.setattr(catalog, "load_texture", fake_load)
    cache = catalog.TexturePreviewCache(max_images=1, max_bytes=260)
    fixture = catalog.BrowserCatalogue((), (), {})
    a, b = _asset("a.paa", kind="texture"), _asset("b.paa", kind="texture")
    first = cache.get(a, fixture)
    assert cache.get(a, fixture) is first
    cache.get(b, fixture)
    cache.get(a, fixture)
    assert calls == ["a.paa", "b.paa", "a.paa"]
    assert len(cache._items) == 1
    assert cache.bytes_used == first.nbytes


def test_preview_render_size_uses_actual_tk_canvas_pixels() -> None:
    app = app_module.AssetBrowserApp.__new__(app_module.AssetBrowserApp)
    widget = SimpleNamespace(winfo_width=lambda: 640, winfo_height=lambda: 400)
    app.canvas = SimpleNamespace(get_tk_widget=lambda: widget)
    assert app._preview_render_size() == (800, 500)

    widget = SimpleNamespace(winfo_width=lambda: 4000, winfo_height=lambda: 3000)
    app.canvas = SimpleNamespace(get_tk_widget=lambda: widget)
    assert app._preview_render_size() == (1200, 840)

    widget = SimpleNamespace(winfo_width=lambda: 1, winfo_height=lambda: 1)
    app.canvas = SimpleNamespace(get_tk_widget=lambda: widget)
    assert app._preview_render_size() == (960, 720)


def test_model_render_uses_cache_and_canvas_dimensions(monkeypatch) -> None:
    app = app_module.AssetBrowserApp.__new__(app_module.AssetBrowserApp)
    app.ax_preview = Mock()
    app.canvas = Mock()
    app.canvas.get_tk_widget.return_value.winfo_width.return_value = 640
    app.canvas.get_tk_widget.return_value.winfo_height.return_value = 400
    app.figure = Mock()
    app.texture_resolver = object()
    app.mipmap_cache = MipmapCache(max_bytes=256)
    app.skip_textures_var = SimpleNamespace(get=lambda: False)
    app.azim, app.elev, app.zoom = 35.0, 25.0, 1.0
    arguments = {}

    def fake_render(*args, **kwargs):
        arguments.update(kwargs)
        return np.zeros((500, 800, 3), dtype=np.uint8), 0, 0

    monkeypatch.setattr(app_module, "render_textured_model", fake_render)
    model = SimpleNamespace(
        points=np.zeros((3, 3), dtype=np.float32),
        faces=(object(),),
        source="assets.pbo",
        model_path="house.p3d",
    )
    app._draw_model(model)
    assert arguments["width"] == 800
    assert arguments["height"] == 500
    assert arguments["mipmap_cache"] is app.mipmap_cache
    assert arguments["load_textures"] is True



def test_texture_resolver_bounds_raw_bytes_and_rgba_images() -> None:
    resolver = TextureResolver(())
    resolver._bytes_cache_limit = 10
    resolver._image_cache_limit = 160
    resolver._cache_max_entries = 2

    resolver._cache_raw("a", b"a" * 5)
    resolver._cache_raw("b", b"b" * 5)
    resolver.bytes_cache.move_to_end("a")
    resolver._cache_raw("c", b"c" * 5)
    assert list(resolver.bytes_cache) == ["a", "c"]
    assert resolver._bytes_cache_bytes == 10
    resolver._cache_raw("huge", b"x" * 11)
    assert "huge" not in resolver.bytes_cache

    first = np.full((5, 5, 4), 11, dtype=np.uint8)  # 100 bytes
    second = np.full((5, 5, 4), 22, dtype=np.uint8)
    resolver._cache_rgba("a", first)
    resolver._cache_rgba("b", second)
    assert list(resolver.image_cache) == ["b"]
    assert resolver._image_cache_bytes == 100
    resolver._cache_rgba("missing", None)
    assert list(resolver.image_cache) == ["b", "missing"]
    assert resolver._image_cache_bytes == 100
    resolver._cache_rgba("huge", np.zeros((100, 100, 4), dtype=np.uint8))
    assert "huge" not in resolver.image_cache
    assert resolver._image_cache_bytes <= resolver._image_cache_limit



def test_model_preview_cache_keeps_fast_and_full_lods_separate(monkeypatch) -> None:
    calls = []

    def fake_load(item, _catalogue, *, lod_quality="full"):
        calls.append(lod_quality)
        return SimpleNamespace(
            points=np.zeros((5 if lod_quality == "fast" else 20, 3), dtype=np.float32),
            edges=np.zeros((0, 2), dtype=np.int32),
            faces=(),
            model_path=item.path,
        )

    monkeypatch.setattr(catalog, "load_model", fake_load)
    cache = catalog.ModelPreviewCache(max_models=4, max_bytes=10_000)
    catalogue = catalog.BrowserCatalogue((), (), {})
    item = _asset("example.p3d")
    fast = cache.get(item, catalogue, lod_quality="fast")
    full = cache.get(item, catalogue, lod_quality="full")
    assert fast is not full
    assert len(fast.points) == 5
    assert len(full.points) == 20
    assert cache.get(item, catalogue, lod_quality="fast") is fast
    assert cache.get(item, catalogue, lod_quality="full") is full
    assert calls == ["fast", "full"]


def test_browser_changing_lod_reloads_selected_model_without_rescan() -> None:
    app = app_module.AssetBrowserApp.__new__(app_module.AssetBrowserApp)
    app.current = _asset("selected.p3d")
    app._show_model = Mock()
    app._scan_sources = Mock()
    app._change_preview_lod()
    app._show_model.assert_called_once_with(app.current)
    app._scan_sources.assert_not_called()



def test_browser_mip_selection_redraws_without_rescanning_or_reloading_mesh() -> None:
    app = app_module.AssetBrowserApp.__new__(app_module.AssetBrowserApp)
    app.current = _asset("house.p3d")
    app._mipmap_choices = {"Full (mip 0)": "auto", "Max 256 px": "max:256"}
    app.texture_mipmap_var = SimpleNamespace(get=lambda: "Max 256 px")
    app._redraw_current_model = Mock()
    app._scan_sources = Mock()
    app._show_model = Mock()

    assert app._selected_texture_mipmap() == "max:256"
    app._change_texture_mipmap()
    app._redraw_current_model.assert_called_once_with()
    app._scan_sources.assert_not_called()
    app._show_model.assert_not_called()


def test_texture_preview_cache_separates_embedded_mip_choices(monkeypatch) -> None:
    from p3d_asset_browser_catalog import TexturePreviewCache
    observed = []

    def fake_load(item, _catalogue, *, mipmap="auto"):
        observed.append(mipmap)
        width = 8 if mipmap == "auto" else 4
        return np.ones((width, width, 4), dtype=np.uint8)

    monkeypatch.setattr(catalog, "load_texture", fake_load)
    cache = TexturePreviewCache(max_images=4)
    item = _asset("data\\wall.paa", kind="texture")
    catalogue = catalog.BrowserCatalogue((), (), {})
    full = cache.get(item, catalogue)
    smaller = cache.get(item, catalogue, mipmap="max:256")
    assert full.shape == (8, 8, 4)
    assert smaller.shape == (4, 4, 4)
    assert cache.get(item, catalogue) is full
    assert cache.get(item, catalogue, mipmap="max:256") is smaller
    assert observed == ["auto", "max:256"]



def test_browser_reuses_raw_p3d_for_new_lod_without_reopening_archive(
    tmp_path: Path, monkeypatch,
) -> None:
    """Selecting a new LOD must read the original PBO once, not twice."""
    from cwr_worldgen.pbo import PboEntry, write_pbo
    from cwr_worldgen import assets
    from test_p3d_lod_selection import _mlod

    archive = tmp_path / "models.ifa"
    write_pbo(archive, (PboEntry("house.p3d", _mlod(((1.0, 6), (9.0, 3)))),))
    catalogue = catalog.scan_catalogue((archive,), use_cache=False)
    item, = catalogue.models
    reads = []
    actual_read = assets.read_asset_record_bytes

    def counted(record):
        reads.append(record.path)
        return actual_read(record)

    monkeypatch.setattr(assets, "read_asset_record_bytes", counted)
    cache = catalog.ModelPreviewCache(max_models=3, raw_limit=1_000_000)
    fast = cache.get(item, catalogue, lod_quality="fast")
    assert fast.original_vertex_count == 3
    assert not fast.archive_cached
    assert fast.source_bytes is None
    assert len(reads) == 1

    detailed = cache.get(item, catalogue, lod_quality="lod:0")
    assert detailed.original_vertex_count == 6
    assert detailed.archive_cached
    assert not detailed.preview_cache_hit
    assert detailed.read_seconds == 0.0
    assert detailed.parse_seconds >= 0
    assert detailed.source_bytes is None
    assert len(reads) == 1
    assert len(cache._raw_items) == 1

    repeat = cache.get(item, catalogue, lod_quality="fast")
    assert repeat is fast
    assert repeat.preview_cache_hit
    assert repeat.read_seconds == 0.0
    assert repeat.parse_seconds == 0.0
    assert len(reads) == 1


def test_browser_raw_p3d_cache_is_bounded_and_cleared_with_catalogue() -> None:
    cache = catalog.ModelPreviewCache(raw_limit=7, max_raw_models=2)
    a, b, c = (("models.pbo", f"{name}.p3d") for name in ("a", "b", "c"))
    cache._remember_raw(a, b"first")
    assert cache.raw_bytes_used == 5
    cache._remember_raw(b, b"second")
    assert a not in cache._raw_items
    assert cache.raw_bytes_used == 6
    cache._remember_raw(c, b"X" * 8)
    assert c not in cache._raw_items
    cache._remember_raw(b, b"one")
    assert cache.raw_bytes_used == 3
    cache.clear()
    assert cache._raw_items == {}
    assert cache.raw_bytes_used == 0


def test_browser_model_load_timing_breakdown_identifies_each_cache_kind() -> None:
    app = app_module.AssetBrowserApp.__new__(app_module.AssetBrowserApp)
    captured = []
    app.load_timing_var = SimpleNamespace(set=captured.append)
    model = SimpleNamespace(
        read_seconds=0.01, parse_seconds=0.02,
        archive_cached=False, preview_cache_hit=False,
    )
    app._update_model_timing(model, 0.065, 0.035)
    assert "P3D read: 10.0 ms" in captured[-1]
    assert "Mesh: 20.0 ms" in captured[-1]
    assert "Textures: 35.0 ms" in captured[-1]
    assert "Preview: 30.0 ms" in captured[-1]
    assert "Total: 95.0 ms" in captured[-1]

    model.archive_cached = True
    app._update_model_timing(model, 0.065, 0.035)
    assert "P3D RAM cache" in captured[-1]

    model.preview_cache_hit = True
    app._update_model_timing(model, 0.065, 0.035)
    assert "decoded LOD cache" in captured[-1]


def test_browser_render_forwards_texture_load_timing_and_reports_it(
    monkeypatch,
) -> None:
    app = app_module.AssetBrowserApp.__new__(app_module.AssetBrowserApp)
    app.ax_preview = Mock()
    app.figure = Mock()
    app.canvas = Mock()
    app.canvas.get_tk_widget.return_value.winfo_width.return_value = 640
    app.canvas.get_tk_widget.return_value.winfo_height.return_value = 400
    app.texture_resolver = object()
    app.mipmap_cache = MipmapCache()
    app.skip_textures_var = SimpleNamespace(get=lambda: False)
    app.azim, app.elev, app.zoom = 35.0, 25.0, 1.0
    app._mipmap_choices = {"Full (mip 0)": "auto"}
    app.texture_mipmap_var = SimpleNamespace(get=lambda: "Full (mip 0)")
    captured = []
    app.load_timing_var = SimpleNamespace(set=captured.append)
    passed = []

    def renderer(*_args, **kwargs):
        passed.append(kwargs)
        kwargs["texture_timing"]["seconds"] = 0.001
        return np.zeros((500, 800, 3), dtype=np.uint8), 1, 0

    monkeypatch.setattr(app_module, "render_textured_model", renderer)
    model = SimpleNamespace(
        points=np.zeros((3, 3), dtype=np.float32),
        faces=(object(),), source="models.pbo", model_path="house.p3d",
    )
    app._draw_model(model)
    assert "texture_timing" in passed[0]
    assert passed[0]["texture_mipmap"] == "auto"
    assert "Textures:" in captured[-1]
    assert "Total:" in captured[-1]


def test_browser_standalone_texture_preview_shows_load_time_and_cache_hit() -> None:
    app = app_module.AssetBrowserApp.__new__(app_module.AssetBrowserApp)
    app._busy = Mock()
    app._unbusy = Mock()
    app.ax_preview = Mock()
    app.figure = Mock()
    app.canvas = Mock()
    app.info_var = SimpleNamespace(set=Mock())
    app._set_related_models = Mock()
    app.catalogue = catalog.BrowserCatalogue((), (), {})
    cache = catalog.TexturePreviewCache()
    item = _asset("wall.paa", kind="texture")
    image = np.ones((4, 4, 4), dtype=np.uint8)
    cache._items[(item.source, item.path, "auto")] = image
    app.texture_preview_cache = cache
    app.texture_mipmap_var = SimpleNamespace(get=lambda: "Full (mip 0)")
    app._mipmap_choices = {"Full (mip 0)": "auto"}
    texts = []
    app.load_timing_var = SimpleNamespace(set=texts.append)
    app._show_texture(item)
    assert "Texture read/decode:" in texts[-1]
    assert "(RGBA cache)" in texts[-1]
    assert "Preview/details:" in texts[-1]
    assert "Total:" in texts[-1]
