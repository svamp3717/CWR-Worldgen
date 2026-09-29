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
