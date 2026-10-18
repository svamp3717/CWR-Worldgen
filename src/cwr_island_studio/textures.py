# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode loose terrain images on demand for the visible island viewport.

PAA DXT1 and DXT5 are supported using Pillow's native block decoder.
Unavailable texture assets are left in the slot-colour preview, never invented.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import io
import struct

import numpy as np
from PIL import Image

from .pbo_textures import PboTextures

_PAA_FORMATS = {0xFF01: 1, 0xFF05: 3}  # DXT1, DXT5
_MAX_TEX_BYTES = 32 * 1024 * 1024


def decode_paa_preview(data: bytes) -> Image.Image:
    if len(data) < 4 or len(data) > _MAX_TEX_BYTES:
        raise ValueError("invalid texture size")
    magic = struct.unpack_from("<H", data)[0]
    if magic not in _PAA_FORMATS:
        raise ValueError("unsupported PAA compression")
    cursor = 2
    while data[cursor:cursor + 4] == b"GGAT":
        if cursor + 12 > len(data):
            raise ValueError("truncated PAA metadata")
        size = struct.unpack_from("<I", data, cursor + 8)[0]
        cursor += 12 + size
        if cursor > len(data):
            raise ValueError("truncated PAA metadata")
    if cursor + 2 > len(data):
        raise ValueError("missing PAA palette")
    palette_count = struct.unpack_from("<H", data, cursor)[0]
    cursor += 2 + palette_count * 3
    if palette_count or cursor + 7 > len(data):
        raise ValueError("unsupported PAA palette or mipmap")
    width, height = struct.unpack_from("<HH", data, cursor)
    length = int.from_bytes(data[cursor + 4:cursor + 7], "little")
    cursor += 7
    if not (4 <= width <= 8192 and 4 <= height <= 8192):
        raise ValueError("invalid PAA dimensions")
    if length > _MAX_TEX_BYTES or cursor + length > len(data):
        raise ValueError("truncated PAA image")
    return Image.frombytes(
        "RGBA", (width, height), data[cursor:cursor + length],
        "bcn", _PAA_FORMATS[magic],
    ).convert("RGB")


def _safe_relative(path: str) -> Path | None:
    normalized = path.replace("\\", "/").lstrip("/")
    pieces = normalized.split("/")
    if not pieces or any(part in {"", ".", ".."} or ":" in part for part in pieces):
        return None
    return Path(*pieces)


@lru_cache(maxsize=128)
def _image_tile(root: str, virtual_path: str) -> np.ndarray | None:
    """Return a bounded RGB tile or None for missing/unsupported loose assets."""
    relative = _safe_relative(virtual_path)
    if relative is None:
        return None
    base = Path(root)
    target = base / relative
    if not target.is_file():
        return None
    try:
        if target.suffix.casefold() == ".paa":
            image = decode_paa_preview(target.read_bytes())
        else:
            with Image.open(target) as img:
                image = img.convert("RGB")
        if max(image.size) > 128:
            image.thumbnail((128, 128), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
        array.flags.writeable = False
        return array
    except (ValueError, OSError, SyntaxError):
        return None


@lru_cache(maxsize=128)
def _pbo_image_tile(archives: PboTextures, virtual_path: str) -> np.ndarray | None:
    try:
        data = archives.read(virtual_path)
        if data is None:
            return None
        if virtual_path.casefold().endswith(".paa"):
            image = decode_paa_preview(data)
        else:
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert("RGB")
        if max(image.size) > 128:
            image.thumbnail((128, 128), Image.Resampling.BILINEAR)
        tile = np.asarray(image, dtype=np.uint8).copy()
        tile.flags.writeable = False
        return tile
    except (ValueError, OSError, SyntaxError):
        return None


def clear_archive_texture_cache() -> None:
    """Drop decoded tiles before replacing or closing selected archives."""
    _pbo_image_tile.cache_clear()


def render_real_textures(
    slots: np.ndarray, x: np.ndarray, z: np.ndarray,
    texture_paths: tuple[str, ...], asset_root: Path | str | None,
    fallback: np.ndarray, cell_size: float,
    *, archives: PboTextures | None = None,
) -> tuple[np.ndarray, int]:
    """Apply available actual texture pixels over a slot-colour viewport.

    Inputs are only viewport-sized. The renderer never allocates an image
    with the world's full dimensions. Missing assets keep the fallback colour.
    """
    if asset_root is None and archives is None:
        return fallback, 0
    if slots.shape != fallback.shape[:2]:
        raise ValueError("texture and fallback preview dimensions differ")
    output = fallback.copy()
    loaded = 0
    root = str(Path(asset_root).resolve()) if asset_root is not None else None
    valid = (slots >= 0) & (slots < len(texture_paths))
    for slot in np.unique(slots[valid]):
        path = texture_paths[int(slot)]
        if not path:
            continue
        tile = _image_tile(root, path) if root is not None else None
        if tile is None and archives is not None:
            tile = _pbo_image_tile(archives, path)
        if tile is None:
            continue
        rows, cols = np.nonzero(slots == slot)
        tx = np.mod(x[cols] / cell_size, 1.0)
        tz = np.mod(z[rows] / cell_size, 1.0)
        cx = np.minimum(tile.shape[1] - 1, (tx * tile.shape[1]).astype(np.intp))
        cz = np.minimum(tile.shape[0] - 1, (tz * tile.shape[0]).astype(np.intp))
        output[rows, cols] = tile[cz, cx]
        loaded += 1
    return output, loaded
