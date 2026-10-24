"""The configured stock gravel road is asf25, whose photo is AI88 asf_new."""
from __future__ import annotations

from hashlib import sha256
import io
from pathlib import Path
import struct
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from cwr_worldgen import parking_surface_policy as parking
from cwr_worldgen.stock_gravel_parking_texture import (
    STOCK_GRAVEL_MODEL, STOCK_GRAVEL_TEXTURE,
    STOCK_GRAVEL_MODEL_WIDTH_METRES,
    STOCK_GRAVEL_REPEAT_LENGTH_METRES,
    _load_from_roots, decode_stock_gravel_paa,
    load_stock_gravel_photo, stock_gravel_material,
)


def _fake_ai88_paa(*, width=16, height=16, checksum_error=False) -> bytes:
    # Native 0x8080 AI88. Each 16-bit pixel has 4-bit transparency and
    # 4-bit luminance. An all-literal LZSS stream is enough for a regression.
    words = [
        ((15 if x not in (0, width - 1) else 0) << 12)
        | (((2 + x + y) % 14) << 4)
        for y in range(height) for x in range(width)
    ]
    pixels = struct.pack("<" + "H" * len(words), *words)
    compressed = bytearray()
    for start in range(0, len(pixels), 8):
        chunk = pixels[start:start + 8]
        compressed.append((1 << len(chunk)) - 1)
        compressed.extend(chunk)
    signed_sum = sum(x if x < 128 else x - 256 for x in pixels)
    compressed += struct.pack("<i", signed_sum + int(checksum_error))
    header = (
        struct.pack("<H", 0x8080)
        + b"GGATSFFO" + struct.pack("<I", 4) + b"TEST"
        + struct.pack("<H", 0)
        + struct.pack("<HH", width, height)
        + len(compressed).to_bytes(3, "little")
    )
    return header + compressed + b"\x00" * 4


def _geometry():
    return parking._ParkingGeometry(
        osm_key="way/nonpaved",
        outer=((4.0, 4.0), (21.0, 4.0), (21.0, 21.0), (4.0, 21.0)),
        holes=(),
        centre_x=12.5, centre_z=12.5,
        half_length=8.5, half_width=8.5,
        ux=1.0, uz=0.0, px=0.0, pz=1.0,
        surface="gravel",
    )


def test_asf25_is_actual_configured_stock_gravel_donor() -> None:
    from cwr_worldgen.model import PlayabilitySpec
    assert PlayabilitySpec.__dataclass_fields__["gravel_road_model"].default == STOCK_GRAVEL_MODEL
    assert STOCK_GRAVEL_TEXTURE == r"o\road\asf_new.paa"
    assert STOCK_GRAVEL_MODEL_WIDTH_METRES == 7.0
    assert STOCK_GRAVEL_REPEAT_LENGTH_METRES == 6.25


def test_native_ai88_paa_decodes_intensity_alpha_and_checks_checksum() -> None:
    decoded = decode_stock_gravel_paa(_fake_ai88_paa())
    assert decoded.mode == "RGBA"
    assert decoded.size == (16, 16)
    pixels = np.asarray(decoded)
    assert np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
    assert np.array_equal(pixels[:, :, 0], pixels[:, :, 2])
    assert pixels[1, 0, 3] == 0
    assert pixels[1, 2, 3] == 255
    with pytest.raises(ValueError, match="checksum"):
        decode_stock_gravel_paa(_fake_ai88_paa(checksum_error=True))
    with pytest.raises(ValueError, match="AI88"):
        decode_stock_gravel_paa(b"not-a-paa")


def test_stock_gravel_repeat_uses_exact_native_7m_and_6_25m_uv_scale() -> None:
    decoded = decode_stock_gravel_paa(_fake_ai88_paa())
    geometry = _geometry()
    xs = np.array([[1.25, 1.25 + 2 * STOCK_GRAVEL_REPEAT_LENGTH_METRES]])
    zs = np.array([[2.0, 2.0]])
    result = stock_gravel_material(
        xs, zs, geometry=geometry, pixel_scale=0.1, image=decoded,
    )
    assert np.allclose(result[0, 0], result[0, 1])
    assert result.shape == (1, 2, 3)
    assert float(result.mean()) > 0


def test_stock_asf_texture_is_used_only_for_gravel_not_paved_lots() -> None:
    fake = Image.fromarray(
        np.full((16, 16, 4), (85, 85, 85, 255), dtype=np.uint8),
        mode="RGBA",
    )
    background = Image.new("RGB", (128, 128), (50, 110, 20))
    spec = SimpleNamespace(cells=1, cell_size=25.0)
    gravel = parking._render_parking_cell(
        cell_index=0, geometries=(_geometry(),), spec=spec,
        base=background, stock_gravel_photo=fake,
    )
    values = np.asarray(gravel)
    assert tuple(values[0, 0]) == (50, 110, 20)
    # Parking detail may darken grey uniformly for wheel tracks.
    assert values[65, 65, 0] == values[65, 65, 1] == values[65, 65, 2]
    assert values[65, 65, 0] > 40

    paved = parking._render_parking_cell(
        cell_index=0, geometries=(_geometry().__class__(
            **{field: getattr(_geometry(), field)
               for field in _geometry().__dataclass_fields__ if field != "surface"},
            surface="paved",
        ),), spec=spec, base=background, stock_gravel_photo=fake,
    )
    assert not np.array_equal(np.asarray(paved)[65, 65], values[65, 65])


def test_reference_zip_is_accepted_as_stock_gravel_asset_root(tmp_path: Path) -> None:
    archive = tmp_path / "Road(1).zip"
    data = _fake_ai88_paa()
    with ZipFile(archive, "w") as output:
        output.writestr("Road/asf_new.paa", data)
    _load_from_roots.cache_clear()
    loaded = load_stock_gravel_photo(SimpleNamespace(asset_roots=(archive,)))
    assert loaded is not None
    image, digest, source = loaded
    assert image.size == (16, 16)
    assert digest == sha256(data).hexdigest()
    assert "asf_new.paa" in source
