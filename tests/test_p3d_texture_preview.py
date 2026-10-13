from __future__ import annotations

from pathlib import Path
import struct
import sys

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from p3d_texture_io import decode_paa, list_paa_mips, TextureResolver
from p3d_texture_render import (
    RenderFace,
    _project,
    _sample_bilinear,
    _sample_mipmapped,
    _mipmap_chain,
    _triangle_texture_lod,
    render_textured_model,
)


def test_decode_dxt1_paa() -> None:
    block = struct.pack("<HHI", 0xF800, 0x0000, 0)
    data = (
        struct.pack("<H", 0xFF01)
        + struct.pack("<H", 0)
        + struct.pack("<HH", 4, 4)
        + bytes((8, 0, 0))
        + block
    )
    image = decode_paa(data, is_paa=True)
    assert image.shape == (4, 4, 4)
    assert tuple(image[0, 0]) == (255, 0, 0, 255)


def test_decode_legacy_paletted_pac() -> None:
    data = bytearray()
    data += struct.pack("<H", 2)
    data += bytes((0, 0, 255, 0, 255, 0))  # BGR red, green
    data += struct.pack("<HH", 2, 2)
    data += bytes((5, 0, 0))
    data += bytes((3, 0, 1, 1, 0))  # 4 literal palette indices
    image = decode_paa(bytes(data), is_paa=False)
    assert tuple(image[0, 0]) == (255, 0, 0, 255)
    assert tuple(image[0, 1]) == (0, 255, 0, 255)


class _Resolver:
    def load_rgba(self, texture_path: str, source: str) -> np.ndarray:
        del texture_path, source
        image = np.empty((2, 2, 4), dtype=np.uint8)
        image[:] = (255, 0, 0, 255)
        return image


def test_textured_renderer_applies_face_texture() -> None:
    points = np.asarray(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        dtype=np.float32,
    )
    face = RenderFace(
        (0, 1, 2, 3),
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        r"data\red.paa",
    )
    image, hits, misses = render_textured_model(
        points,
        (face,),
        _Resolver(),
        "fixture",
        width=160,
        height=120,
    )
    assert hits > 0
    assert misses == 0
    red = (image[:, :, 0] > 240) & (image[:, :, 1] < 20) & (image[:, :, 2] < 20)
    assert int(red.sum()) > 100


def test_projection_zoom_doubles_screen_distance() -> None:
    points = np.asarray(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        dtype=np.float32,
    )
    normal = _project(points, 800, 600, 35.0, 25.0, 1.0)
    zoomed = _project(points, 800, 600, 35.0, 25.0, 2.0)
    centre = np.asarray((400.0, 300.0))
    normal_distance = np.linalg.norm(normal[:, :2] - centre, axis=1)
    zoomed_distance = np.linalg.norm(zoomed[:, :2] - centre, axis=1)
    assert np.allclose(zoomed_distance, normal_distance * 2.0)


def test_bilinear_sampling_blends_adjacent_texels() -> None:
    texture = np.asarray(
        [
            [(255, 0, 0, 255), (0, 255, 0, 255)],
            [(0, 0, 255, 255), (255, 255, 255, 255)],
        ],
        dtype=np.uint8,
    )
    sampled = _sample_bilinear(texture, np.asarray([[0.5]]), np.asarray([[0.5]]))[0, 0]
    assert 110 <= int(sampled[0]) <= 145
    assert 110 <= int(sampled[1]) <= 145
    assert 110 <= int(sampled[2]) <= 145
    assert int(sampled[3]) == 255



def _dense_checkerboard(size: int = 64) -> np.ndarray:
    image = np.zeros((size, size, 4), dtype=np.uint8)
    pixels = (np.indices((size, size)).sum(axis=0) & 1) * 255
    image[:, :, :3] = pixels[:, :, None]
    image[:, :, 3] = 255
    return image


def test_mipmapped_preview_averages_distant_checkerboard_instead_of_moire() -> None:
    """Large, repetitive Arma building textures need minification filtering."""
    texture = _dense_checkerboard()
    levels = _mipmap_chain(texture)
    assert levels[0].shape == (64, 64, 4)
    assert levels[-1].shape == (1, 1, 4)
    assert all(120 <= int(x) <= 135 for x in levels[-1][0, 0, :3])
    assert int(levels[-1][0, 0, 3]) == 255

    # A 64-pixel-wide image spanning 4 screen pixels needs around mip level 4.
    uv = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    screen_xy = np.array(((1.0, 1.0), (5.0, 1.0), (1.0, 5.0)))
    lod = _triangle_texture_lod(uv, screen_xy, 64, 64, len(levels) - 1)
    assert lod == 4.0
    samples = _sample_mipmapped(
        levels, np.array([[0.01, 0.25, 0.7]]),
        np.array([[0.01, 0.25, 0.7]]), lod,
    )
    assert np.ptp(samples[0, :, 0]) <= 1


def test_mipmapped_preview_keeps_sharp_texture_when_magnified() -> None:
    levels = _mipmap_chain(_dense_checkerboard())
    uv = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    screen_xy = np.array(((0.0, 0.0), (128.0, 0.0), (0.0, 128.0)))
    lod = _triangle_texture_lod(uv, screen_xy, 64, 64, len(levels) - 1)
    assert lod == 0.0
    sampled = _sample_mipmapped(levels, np.array([[1.5 / 64]]), np.array([[1.5 / 64]]), lod)
    assert np.array_equal(sampled, _sample_bilinear(levels[0], np.array([[1.5 / 64]]), np.array([[1.5 / 64]])))


def test_perspective_rasterizer_minifies_tiled_wall_texture() -> None:
    class CheckerResolver:
        def __init__(self):
            self.calls = 0
        def load_rgba(self, texture_path: str, source: str) -> np.ndarray:
            self.calls += 1
            return _dense_checkerboard()

    resolver = CheckerResolver()
    points = np.array([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
    ], dtype=np.float32)
    face = RenderFace(
        (0, 1, 2, 3),
        ((0.0, 0.0), (32.0, 0.0), (32.0, 32.0), (0.0, 32.0)),
        r"ca\structures\data\checker_co.paa",
    )
    rendered, hits, misses = render_textured_model(
        points, (face,), resolver, "example.pbo",
        width=320, height=240, azim_deg=0, elev_deg=0,
    )
    assert (hits, misses) == (2, 0)
    assert resolver.calls == 1
    assert np.max(np.abs(rendered[120, 160].astype(int) - 128)) <= 3


def test_browser_uses_antialiased_downsampling_for_3d_models(monkeypatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock
    import p3d_asset_browser_app as browser_app

    app = browser_app.AssetBrowserApp.__new__(browser_app.AssetBrowserApp)
    app.ax_preview = Mock()
    app.canvas = Mock()
    app.figure = Mock()
    app.texture_resolver = object()
    app.skip_textures_var = SimpleNamespace(get=lambda: False)
    app.azim = 35.0
    app.elev = 25.0
    app.zoom = 1.0

    def mock_render(*args, **kwargs):
        return np.zeros((40, 40, 3), dtype=np.uint8), 1, 0

    monkeypatch.setattr(browser_app, "render_textured_model", mock_render)
    model = SimpleNamespace(
        points=np.zeros((3, 3), dtype=np.float32),
        faces=(object(),),
        source="buildings.pbo", model_path="building.p3d",
    )
    app._draw_model(model)
    assert app.ax_preview.imshow.call_args.kwargs["interpolation"] == "hanning"
    assert app.ax_preview.imshow.call_args.kwargs["resample"] is True



def _embedded_dxt1_mips() -> bytes:
    """Three true PAA payloads with different colours at each mip level."""
    red = struct.pack("<HHI", 0xF800, 0x0000, 0)
    green = struct.pack("<HHI", 0x07E0, 0x0000, 0)
    blue = struct.pack("<HHI", 0x001F, 0x0000, 0)
    data = bytearray(struct.pack("<HH", 0xFF01, 0))
    for width, height, payload in (
        (8, 8, red * 4), (4, 4, green), (2, 2, blue),
    ):
        data.extend(struct.pack("<HH", width, height))
        data.extend(len(payload).to_bytes(3, "little"))
        data.extend(payload)
    data.extend(b"\0" * 7)  # end-of-chain sentinel
    return bytes(data)


def test_selects_original_embedded_paa_mip_without_decoding_base(monkeypatch) -> None:
    import p3d_texture_io as paa_io
    data = _embedded_dxt1_mips()
    assert list_paa_mips(data) == ((8, 8), (4, 4), (2, 2))
    original = paa_io._decode_dxt
    sizes = []

    def tracked(payload, width, height, kind):
        sizes.append((width, height))
        return original(payload, width, height, kind)

    monkeypatch.setattr(paa_io, "_decode_dxt", tracked)

    full = decode_paa(data)
    half = decode_paa(data, mipmap="mip:1")
    quarter = decode_paa(data, mipmap="mip:2")
    max_four = decode_paa(data, mipmap="max:4")
    max_one = decode_paa(data, mipmap="max:1")
    beyond = decode_paa(data, mipmap="mip:9")

    assert sizes == [(8, 8), (4, 4), (2, 2), (4, 4), (2, 2), (2, 2)]
    assert full.shape == (8, 8, 4) and tuple(full[0, 0]) == (255, 0, 0, 255)
    assert half.shape == (4, 4, 4) and tuple(half[0, 0]) == (0, 255, 0, 255)
    assert quarter.shape == (2, 2, 4) and tuple(quarter[0, 0]) == (0, 0, 255, 255)
    assert np.array_equal(max_four, half)
    assert np.array_equal(max_one, quarter)
    assert np.array_equal(beyond, quarter)


def test_mipmap_decoder_handles_single_mip_and_invalid_selection() -> None:
    block = struct.pack("<HHI", 0xF800, 0, 0)
    data = (
        struct.pack("<HHHH", 0xFF01, 0, 4, 4)
        + len(block).to_bytes(3, "little") + block
    )
    assert list_paa_mips(data) == ((4, 4),)
    assert np.array_equal(decode_paa(data), decode_paa(data, mipmap="max:64"))
    assert np.array_equal(decode_paa(data), decode_paa(data, mipmap="mip:8"))
    import pytest
    with pytest.raises(ValueError, match="invalid texture mip"):
        decode_paa(data, mipmap="mip:nope")


def test_texture_resolver_separates_full_and_small_mip_cache(monkeypatch) -> None:
    data = _embedded_dxt1_mips()
    resolver = TextureResolver(())
    from unittest.mock import Mock
    resolver.load_bytes = Mock(return_value=data)

    full = resolver.load_rgba("wall.paa", "models.pbo")
    smaller = resolver.load_rgba("wall.paa", "models.pbo", mipmap="max:4")
    assert full.shape == (8, 8, 4)
    assert smaller.shape == (4, 4, 4)
    assert resolver.load_rgba("wall.paa", "models.pbo") is full
    assert resolver.load_rgba("wall.paa", "models.pbo", mipmap="max:4") is smaller
    assert resolver.load_bytes.call_count == 2  # raw bytes reused by load_bytes in production
    assert "wall.paa" in resolver.image_cache
    assert "wall.paa|max:4" in resolver.image_cache


def test_model_renderer_forwards_selected_texture_mipmap(monkeypatch) -> None:
    import p3d_texture_render as rasterizer
    from types import SimpleNamespace
    import p3d_asset_browser_app as browser

    class CapturingResolver:
        def __init__(self):
            self.calls = []
        def load_rgba(self, path, source, *, mipmap="auto"):
            self.calls.append((path, source, mipmap))
            return np.full((4, 4, 4), (50, 100, 150, 255), dtype=np.uint8)

    monkeypatch.setattr(rasterizer, "_project", lambda *_a: np.array([
        [5.0, 5.0, 1.0], [26.0, 5.0, 1.0], [15.0, 26.0, 1.0],
    ]))
    points = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    face = RenderFace((0, 1, 2), ((0, 0), (1, 0), (0, 1)), "wall.paa")
    resolver = CapturingResolver()
    mipcache = rasterizer.MipmapCache()
    image, hits, misses = rasterizer.render_textured_model(
        points, (face,), resolver, "models.pbo",
        width=32, height=32, mipmap_cache=mipcache, texture_mipmap="max:256",
    )
    assert (hits, misses) == (1, 0)
    assert resolver.calls == [("wall.paa", "models.pbo", "max:256")]
    assert image.shape == (32, 32, 3)
    rasterizer.render_textured_model(
        points, (face,), resolver, "models.pbo",
        width=32, height=32, mipmap_cache=mipcache, texture_mipmap="max:256",
    )
    assert len(resolver.calls) == 1
