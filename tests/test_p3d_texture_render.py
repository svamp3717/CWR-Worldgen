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
