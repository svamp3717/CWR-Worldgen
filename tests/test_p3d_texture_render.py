from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import p3d_texture_render as render


class _Resolver:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def load_rgba(self, texture_path: str, source: str):
        self.calls.append((texture_path, source))
        if self.fail:
            raise AssertionError("texture resolver must not be called")
        texture = np.zeros((4, 4, 4), dtype=np.uint8)
        texture[:, :, 0] = 220
        texture[:, :, 3] = 255
        return texture


def _visible_projection(_points, _width, _height, _azim, _elev, _zoom):
    return np.asarray(
        (
            (5.0, 5.0, 1.0),
            (26.0, 5.0, 1.0),
            (15.0, 26.0, 1.0),
        ),
        dtype=float,
    )


def _triangle():
    points = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float32,
    )
    faces = (
        render.RenderFace(
            indices=(0, 1, 2),
            uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
            texture_path=r"inner\wall.paa",
        ),
    )
    return points, faces


def test_skip_textures_never_calls_texture_resolver(monkeypatch) -> None:
    monkeypatch.setattr(render, "_project", _visible_projection)
    points, faces = _triangle()
    resolver = _Resolver(fail=True)

    image, hits, misses = render.render_textured_model(
        points,
        faces,
        resolver,
        "mod.pbo.zst!addons\\inner.pbo!inner\\house.p3d",
        width=32,
        height=32,
        load_textures=False,
    )

    assert resolver.calls == []
    assert hits == 0
    assert misses == 0
    assert np.any(image != 236)


def test_textured_mode_still_loads_texture(monkeypatch) -> None:
    monkeypatch.setattr(render, "_project", _visible_projection)
    points, faces = _triangle()
    resolver = _Resolver()

    _image, hits, misses = render.render_textured_model(
        points,
        faces,
        resolver,
        "mod.pbo.zst!addons\\inner.pbo!inner\\house.p3d",
        width=32,
        height=32,
        load_textures=True,
    )

    assert resolver.calls == [
        (r"inner\wall.paa", "mod.pbo.zst!addons\\inner.pbo!inner\\house.p3d")
    ]
    assert hits == 1
    assert misses == 0


def _two_band_atlas() -> np.ndarray:
    """An asymmetric colour atlas: a vertical UV flip swaps the two bands."""
    image = np.empty((8, 8, 4), dtype=np.uint8)
    image[:4] = (225, 35, 20, 255)  # top: red
    image[4:] = (20, 40, 225, 255)  # bottom: blue
    return image


def test_p3d_top_origin_uvs_sample_correct_colour_atlas_rows() -> None:
    atlas = _two_band_atlas()
    u = np.asarray([0.25, 0.25, 0.25])
    v = np.asarray([0.25, 0.75, -0.25])
    samples = render._sample_bilinear(atlas, u, v)

    assert tuple(samples[0]) == (225, 35, 20, 255)
    assert tuple(samples[1]) == (20, 40, 225, 255)
    # UVs outside [0,1] still repeat; they must not flip when wrapped.
    assert tuple(samples[2]) == (20, 40, 225, 255)


def test_model_triangle_uses_correct_half_of_p3d_colour_atlas(monkeypatch) -> None:
    monkeypatch.setattr(render, "_project", _visible_projection)
    points, _faces = _triangle()

    class AtlasResolver:
        def load_rgba(self, texture_path: str, source: str):
            assert texture_path == r"inner\wall.paa"
            return _two_band_atlas()

    # Same polygon and image, but with UVs deliberately confined to the top
    # half of the atlas. Prior to the fix, these faces appeared blue.
    faces = (
        render.RenderFace(
            indices=(0, 1, 2),
            uvs=((0.125, 0.25), (0.375, 0.25), (0.25, 0.25)),
            texture_path=r"inner\wall.paa",
        ),
    )
    image, hits, misses = render.render_textured_model(
        points, faces, AtlasResolver(), "inner.pbo",
        width=32, height=32,
    )
    foreground = image[np.any(image != 236, axis=2)]
    assert hits == 1
    assert misses == 0
    assert len(foreground) > 0
    assert np.all(foreground == (225, 35, 20))



def test_texture_preparation_time_is_measured_separately_from_rasterization(
    monkeypatch,
) -> None:
    monkeypatch.setattr(render, "_project", _visible_projection)
    points, faces = _triangle()
    resolver = _Resolver()
    timings = {}
    image, hits, misses = render.render_textured_model(
        points, faces, resolver, "models.pbo",
        width=32, height=32, texture_timing=timings,
    )
    assert image.shape == (32, 32, 3)
    assert (hits, misses) == (1, 0)
    assert "seconds" in timings
    assert timings["seconds"] >= 0

    skipped = {}
    render.render_textured_model(
        points, faces, _Resolver(fail=True), "models.pbo",
        width=32, height=32, load_textures=False, texture_timing=skipped,
    )
    assert skipped == {}



def test_rasterizer_samples_only_covered_visible_fragment_pixels(monkeypatch) -> None:
    """A large triangle bounding box must not trigger full-box RGBA sampling."""
    monkeypatch.setattr(
        render, "_project",
        lambda *_args: np.array([
            (4.0, 4.0, 1.0),
            (123.0, 4.0, 1.0),
            (4.0, 123.0, 1.0),
        ]),
    )
    points, faces = _triangle()
    count = []
    original = render._sample_mipmapped

    def tracked(levels, u, v, lod):
        count.append(u.size)
        assert u.ndim == v.ndim == 1
        assert u.shape == v.shape
        return original(levels, u, v, lod)

    monkeypatch.setattr(render, "_sample_mipmapped", tracked)
    image, hits, misses = render.render_textured_model(
        points, faces, _Resolver(), "models.pbo", width=128, height=128,
    )
    assert (hits, misses) == (1, 0)
    bbox_pixels = 120 * 120
    covered_pixels = np.count_nonzero(image[:, :, 0] == 220)
    assert 0 < count[0] < bbox_pixels * 0.6
    assert count[0] == covered_pixels


def test_opaque_front_triangle_occludes_behind_without_sampling_it(
    monkeypatch,
) -> None:
    """Back-facing geometry must not decode or sample textures when hidden."""
    monkeypatch.setattr(
        render, "_project",
        lambda *_args: np.array([
            (5, 5, 1), (26, 5, 1), (15, 26, 1),
            (5, 5, 2), (26, 5, 2), (15, 26, 2),
        ], dtype=float),
    )
    points = np.zeros((6, 3), dtype=np.float32)
    a = render.RenderFace((0, 1, 2), ((0, 0), (0, 0), (0, 0)), "front.paa")
    b = render.RenderFace((3, 4, 5), ((0, 0), (0, 0), (0, 0)), "back.paa")

    class ColourResolver:
        def __init__(self):
            self.calls = []

        def load_rgba(self, path, source):
            self.calls.append(path)
            rgba = np.ones((4, 4, 4), dtype=np.uint8) * 255
            rgba[:, :, :3] = (220, 35, 45) if path == "front.paa" else (0, 0, 200)
            return rgba

    resolver = ColourResolver()
    image, hits, misses = render.render_textured_model(
        points, (a, b), resolver, "models.pbo", width=32, height=32,
    )
    assert (hits, misses) == (1, 0)
    assert resolver.calls == ["front.paa"]
    assert tuple(image[12, 15]) == (220, 35, 45)


def test_transparent_front_fragments_do_not_hide_opaque_background(
    monkeypatch,
) -> None:
    """Alpha-masked near geometry should leave already-rendered background visible."""
    monkeypatch.setattr(
        render, "_project",
        lambda *_args: np.array([
            (5, 5, 2), (26, 5, 2), (15, 26, 2),
            (5, 5, 1), (26, 5, 1), (15, 26, 1),
        ], dtype=float),
    )
    points = np.zeros((6, 3), dtype=np.float32)
    back = render.RenderFace((0, 1, 2), ((0, 0), (0, 0), (0, 0)), "back.paa")
    front = render.RenderFace((3, 4, 5), ((0, 0), (0, 0), (0, 0)), "front.paa")

    class TransparentResolver:
        def load_rgba(self, path, _source):
            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[:, :, :3] = (0, 100, 255) if path == "back.paa" else (255, 0, 0)
            rgba[:, :, 3] = 255 if path == "back.paa" else 0
            return rgba

    result, hits, misses = render.render_textured_model(
        points, (back, front), TransparentResolver(),
        "models.pbo", width=32, height=32,
    )
    assert (hits, misses) == (2, 0)
    assert tuple(result[12, 15]) == (0, 100, 255)


def test_small_visible_triangle_does_not_touch_bounding_box_corners(
    monkeypatch,
) -> None:
    monkeypatch.setattr(render, "_project", _visible_projection)
    points, faces = _triangle()
    image, _hits, _misses = render.render_textured_model(
        points, faces, _Resolver(), "models.pbo", width=32, height=32,
    )
    assert tuple(image[6, 6]) == (220, 0, 0)
    assert tuple(image[23, 24]) == (236, 236, 236)
