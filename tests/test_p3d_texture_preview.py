from __future__ import annotations

from pathlib import Path
import struct
import sys

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from p3d_texture_io import decode_paa
from p3d_texture_render import RenderFace, render_textured_model


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
    face = RenderFace((0, 1, 2, 3), ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), r"data\red.paa")
    image, hits, misses = render_textured_model(points, (face,), _Resolver(), "fixture", width=160, height=120)
    assert hits > 0
    assert misses == 0
    red = (image[:, :, 0] > 240) & (image[:, :, 1] < 20) & (image[:, :, 2] < 20)
    assert int(red.sum()) > 100
