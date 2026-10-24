# SPDX-License-Identifier: GPL-3.0-or-later
"""Sample the *actual* stock CWA asf25 gravel road texture on parking terrain.

The model name "asf" is not a surface classification. CWR-Worldgen configures
o\\road\\asf25.p3d as its default gravel donor. Its visible diffuse texture is
o\\road\\asf_new.paa, stored in native legacy AI88 PAA format.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import io
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image

STOCK_GRAVEL_MODEL = r"o\road\asf25.p3d"
STOCK_GRAVEL_TEXTURE = r"o\road\asf_new.paa"
STOCK_GRAVEL_MODEL_WIDTH_METRES = 7.0
STOCK_GRAVEL_REPEAT_LENGTH_METRES = 25.0 / 4.0
# asf_new.paa is an edge-faded road ribbon. Repeating its transparent
# 10%-wide edges across a *parking polygon* would leave bands of grass.
STOCK_GRAVEL_SAFE_U_RANGE = (0.12, 0.88)
_STOCK_PAA_AI88 = 0x8080


def _read_lzss_image(stream: io.BytesIO, expected: int) -> bytes:
    """Decode old BIS signed-checksum LZSS PAA mips, with bounded output."""
    if not 0 < expected <= 64 * 1024 * 1024:
        raise ValueError("invalid legacy road PAA image size")
    window = bytearray(b" " * (4096 + 18 - 1))
    cursor, flags, checksum = 4096 - 18, 0, 0
    output = bytearray()
    while len(output) < expected:
        flags >>= 1
        if not flags & 256:
            raw = stream.read(1)
            if not raw:
                raise ValueError("truncated stock gravel PAA flags")
            flags = raw[0] | 0xFF00
        if flags & 1:
            raw = stream.read(1)
            if not raw:
                raise ValueError("truncated stock gravel PAA literal")
            values = raw
        else:
            pair = stream.read(2)
            if len(pair) != 2:
                raise ValueError("truncated stock gravel PAA back-reference")
            distance = pair[0] | ((pair[1] & 0xF0) << 4)
            count = (pair[1] & 15) + 3
            values = bytes(window[(cursor - distance + idx) & 4095]
                           for idx in range(count)) if distance >= count else None
            if values is None:
                # Overlapping copy: each byte can refer to one just written.
                for idx in range(count):
                    if len(output) >= expected:
                        break
                    value = window[(cursor - distance) & 4095]
                    output.append(value)
                    checksum += value if value < 128 else value - 256
                    window[cursor] = value
                    cursor = (cursor + 1) & 4095
                continue
        for value in values:
            if len(output) >= expected:
                break
            output.append(value)
            checksum += value if value < 128 else value - 256
            window[cursor] = value
            cursor = (cursor + 1) & 4095
    stored = stream.read(4)
    if len(stored) != 4 or struct.unpack("<i", stored)[0] != checksum:
        raise ValueError("stock gravel PAA checksum mismatch")
    return bytes(output)


def decode_stock_gravel_paa(data: bytes) -> Image.Image:
    """Decode a real `o\\road\\asf_new.paa` AI88 first mip as RGBA."""
    stream = io.BytesIO(data)
    if len(data) < 16 or struct.unpack("<H", stream.read(2))[0] != _STOCK_PAA_AI88:
        raise ValueError("stock asf gravel texture is not AI88 PAA")
    while True:
        before = stream.tell()
        if stream.read(4) != b"GGAT":
            stream.seek(before)
            break
        if len(stream.read(4)) != 4:
            raise ValueError("truncated stock gravel PAA tag")
        size_raw = stream.read(4)
        if len(size_raw) != 4:
            raise ValueError("truncated stock gravel PAA tag length")
        size = struct.unpack("<I", size_raw)[0]
        if size > len(data) - stream.tell():
            raise ValueError("oversized stock gravel PAA tag")
        stream.seek(size, io.SEEK_CUR)
    palette_raw = stream.read(2)
    if len(palette_raw) != 2:
        raise ValueError("missing stock gravel PAA palette header")
    palette_count = struct.unpack("<H", palette_raw)[0]
    if palette_count > 256:
        raise ValueError("implausible stock gravel PAA palette")
    stream.seek(palette_count * 3, io.SEEK_CUR)
    header = stream.read(7)
    if len(header) != 7:
        raise ValueError("missing stock gravel PAA first mip")
    width, height = struct.unpack_from("<HH", header)
    payload_length = int.from_bytes(header[4:7], "little")
    if not (16 <= width <= 2048 and 16 <= height <= 2048):
        raise ValueError("implausible stock gravel PAA dimensions")
    payload_start = stream.tell()
    if payload_length < 4 or payload_start + payload_length > len(data):
        raise ValueError("invalid stock gravel PAA payload")
    raw = _read_lzss_image(stream, width * height * 2)
    if stream.tell() > payload_start + payload_length:
        raise ValueError("stock gravel PAA mip exceeds stored payload")
    words = np.frombuffer(raw, dtype="<u2").reshape((height, width))
    # The project's PAA decoder encodes AI88 as an ARGB4444 image where
    # bits 4..7 are intensity and bits 12..15 are transparency.
    intensity = ((words >> 4) & 15).astype(np.uint8) * 17
    alpha = ((words >> 12) & 15).astype(np.uint8) * 17
    return Image.fromarray(
        np.stack((intensity, intensity, intensity, alpha), axis=2), mode="RGBA",
    )


@lru_cache(maxsize=8)
def _load_from_roots(roots: tuple[str, ...]):
    from . import runway_exact_background_policy as exact

    located = exact._read_external_asset_cached(roots, STOCK_GRAVEL_TEXTURE.casefold())
    if located is None:
        return None
    data, source = located
    image = decode_stock_gravel_paa(data)
    return image, sha256(data).hexdigest(), source


def load_stock_gravel_photo(spec):
    """Read installed stock O/road assets, never bundle proprietary texture bytes."""
    from . import runway_exact_background_policy as exact

    roots = tuple(str(path) for path in exact._candidate_asset_roots(spec))
    return _load_from_roots(roots) if roots else None


@lru_cache(maxsize=24)
def _prefiltered_strip(image_bytes: bytes, width: int, height: int, sample_x: int, sample_y: int):
    source = Image.frombytes("RGBA", (width, height), image_bytes)
    lo, hi = STOCK_GRAVEL_SAFE_U_RANGE
    crop = source.crop((
        int(round(lo * width)), 0, int(round(hi * width)), height,
    ))
    tile = crop.convert("RGB").resize(
        (sample_x, sample_y), Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(tile, dtype=np.float32)
    pixels.setflags(write=False)
    return pixels


def _mirror_index(value: np.ndarray, dimension: int):
    """Mirror alternate tiles to prevent visible borders in the gravel fill."""
    periodic = np.mod(value, 2.0)
    mirrored = np.minimum(periodic, 2.0 - periodic)
    return np.clip(mirrored * (dimension - 1), 0, dimension - 1)


def stock_gravel_material(
    world_x: np.ndarray,
    world_z: np.ndarray,
    *,
    geometry,
    pixel_scale: float,
    image: Image.Image,
) -> np.ndarray:
    """World-anchored, lot-oriented stock gravel at the real asf25 UV scale."""
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive")
    physical_width = STOCK_GRAVEL_MODEL_WIDTH_METRES * (
        STOCK_GRAVEL_SAFE_U_RANGE[1] - STOCK_GRAVEL_SAFE_U_RANGE[0]
    )
    physical_length = STOCK_GRAVEL_REPEAT_LENGTH_METRES
    sample_x = min(512, max(16, int(math.ceil(2 * physical_width / pixel_scale))))
    sample_y = min(512, max(16, int(math.ceil(2 * physical_length / pixel_scale))))
    rgba = image.convert("RGBA")
    pixels = _prefiltered_strip(
        rgba.tobytes(), rgba.width, rgba.height, sample_x, sample_y,
    )
    # Anchor UVs to the *world origin* through the lot axes, never the cell
    # origin. Adjacent WRP terrain cells thus sample the same pattern.
    u = (world_x * geometry.px + world_z * geometry.pz) / physical_width
    v = (world_x * geometry.ux + world_z * geometry.uz) / physical_length
    tx = _mirror_index(u, sample_x)
    ty = _mirror_index(v, sample_y)
    x0, y0 = np.floor(tx).astype(np.intp), np.floor(ty).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, sample_x - 1), np.minimum(y0 + 1, sample_y - 1)
    fx = (tx - x0).astype(np.float32)[..., None]
    fy = (ty - y0).astype(np.float32)[..., None]
    upper = pixels[y0, x0] * (1.0 - fx) + pixels[y0, x1] * fx
    lower = pixels[y1, x0] * (1.0 - fx) + pixels[y1, x1] * fx
    return upper * (1.0 - fy) + lower * fy
