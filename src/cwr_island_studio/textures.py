# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode loose terrain images on demand for the visible island viewport.

PAA DXT1/DXT3/DXT5 and original OFP/CWA PAC textures are supported.
A missing texture retains its original diagnostic slot colour.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import io
import struct

import numpy as np
from PIL import Image

from .pbo_textures import PboTextures, canonical_path
from .asset_sources import roots
from cwr_worldgen.forest_ground_blends import STOCK_FOREST_BLEND_FAMILIES
from typing import Sequence

_PAA_FORMATS = {0xFF01: 1, 0xFF02: 2, 0xFF03: 2, 0xFF04: 3, 0xFF05: 3}
_RAW_FORMATS = {0x4444, 0x1555, 0x8888, 0x8080}
_MAX_DECODE_PIXELS = 8 * 1024 * 1024
_MAX_TEX_BYTES = 32 * 1024 * 1024

# Only the original stock directional forest transitions use this selector.
# Generated pitch/runway/surface PAA files are already world-aligned in
# south-to-north pixel row order and must never be rotated along with them.
FOREST_BORDER_ORIENTATIONS = (
    "Auto (stock artwork)",
    "North up",
    "90 clockwise",
    "180",
    "90 counterclockwise",
    "Mirror east/west",
    "Mirror north/south",
    "Transpose",
    "Anti-transpose",
)


def _stock_forest_components(path: str) -> tuple[str, tuple[str, ...]] | None:
    """Recognize stock quartets named NE, NW, SE, SW, including mixed ground.

    Original Nogova also has j1/b1 and j1/s2 boundaries, not just j1/t1.
    """
    normalized = path.replace("\\", "/").casefold()
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    namespace, sep, filename = normalized.rpartition("/")
    if not sep:
        return None
    stem, dot, extension = filename.rpartition(".")
    if not dot or extension not in {"paa", "pac"} or len(stem) != 8:
        return None
    for source_namespace, forest, _ground, _suffix, _masks in STOCK_FOREST_BLEND_FAMILIES.values():
        if namespace != source_namespace:
            continue
        parts = tuple(stem[i:i + 2] for i in range(0, 8, 2))
        if forest in parts and any(p != forest for p in parts) and all(
            len(p) == 2 and p.isascii() and p.isalnum() for p in parts
        ):
            return namespace, parts
    return None


def _is_stock_forest_border(path: str) -> bool:
    return _stock_forest_components(path) is not None


def _orient_stock_border(tile: np.ndarray, orientation: str) -> np.ndarray:
    """Rotate/mirror only the rendered tile; WRP indices are not edited."""
    if orientation == "North up":
        return tile
    if orientation == "90 clockwise":
        return np.rot90(tile, -1)
    if orientation == "180":
        return np.rot90(tile, 2)
    if orientation == "90 counterclockwise":
        return np.rot90(tile, 1)
    if orientation == "Mirror east/west":
        return np.fliplr(tile)
    if orientation == "Mirror north/south":
        return np.flipud(tile)
    if orientation == "Transpose":
        return np.transpose(tile, (1, 0, 2))
    if orientation == "Anti-transpose":
        return np.fliplr(np.flipud(np.transpose(tile, (1, 0, 2))))
    raise ValueError(f"unknown forest border preview orientation: {orientation}")


def decode_paa_preview(data: bytes) -> Image.Image:
    if len(data) < 4 or len(data) > _MAX_TEX_BYTES:
        raise ValueError("invalid texture size")
    magic = struct.unpack_from("<H", data)[0]
    if magic not in _PAA_FORMATS and magic not in _RAW_FORMATS:
        raise ValueError("unsupported PAA/PAC image format")
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
    if not (1 <= width <= 8192 and 1 <= height <= 8192):
        raise ValueError("invalid PAA dimensions")
    if width * height > _MAX_DECODE_PIXELS:
        raise ValueError("texture exceeds decoded-pixel limit")
    if length > _MAX_TEX_BYTES or cursor + length > len(data):
        raise ValueError("truncated PAA image")
    if magic in _PAA_FORMATS:
        return Image.frombytes(
            "RGBA", (width, height), data[cursor:cursor + length],
            "bcn", _PAA_FORMATS[magic],
        ).convert("RGB")
    if magic == 0x8888:
        needed = width * height * 4
        if length < needed:
            raise ValueError("truncated ARGB8888 texture")
        bgra = np.frombuffer(
            data, dtype=np.uint8, count=needed, offset=cursor,
        ).reshape((height, width, 4))
        return Image.fromarray(bgra[:, :, [2, 1, 0]].copy(), mode="RGB")

    # Reuse the same checked LZW/LZSS decoder used for original Worldgen PAC
    # ground textures. Older PAA/PAC ARGB4444/ARGB1555 images use it as well.
    from cwr_worldgen.runway_exact_background_policy import _decode_pac_lzw

    raw = _decode_pac_lzw(io.BytesIO(data[cursor:]), width * height * 2)
    words = np.frombuffer(raw, dtype="<u2").copy()
    if magic == 0x8080:  # AI88 uses a packed intensity/alpha encoding.
        intensity = words & 0x00F0
        words = (words & 0xF000) | (intensity << 4) | intensity | (intensity >> 4)
    if magic in {0x4444, 0x8080}:
        rgb = np.stack(
            [((words >> shift) & 15).astype(np.uint8) * 17
             for shift in (8, 4, 0)], axis=-1,
        ).reshape((height, width, 3))
    else:
        rgb = np.stack(
            [((words >> shift) & 31).astype(np.uint8)
             for shift in (10, 5, 0)], axis=-1,
        ).reshape((height, width, 3))
        rgb = (rgb.astype(np.uint16) * 255 // 31).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def decode_pac_preview(data: bytes) -> Image.Image:
    """Decode legacy OFP/CWA paletted PAC terrain textures.

    Reuse the PAC RLE/LZW decoder already used for original island textures
    by the Worldgen pipeline. Keeping it inside the installed cwr_worldgen
    package also works in standalone PyInstaller builds (no tools/ import).
    """
    if len(data) < 9 or len(data) > _MAX_TEX_BYTES:
        raise ValueError("invalid PAC texture size")
    # Some assets carry a typed DXT PAA header despite having a .pac name.
    # Keep those decodable instead of treating 0xFF01 as a palette count.
    if struct.unpack_from("<H", data)[0] in (*_PAA_FORMATS, *_RAW_FORMATS):
        return decode_paa_preview(data)
    from cwr_worldgen.runway_exact_background_policy import _decode_paletted_pac

    return _decode_paletted_pac(data)


def _safe_relative(path: str) -> Path | None:
    normalized = path.replace("\\", "/").lstrip("/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    pieces = normalized.split("/")
    if not pieces or any(part in {"", ".", ".."} or ":" in part for part in pieces):
        return None
    return Path(*pieces)


def _texture_candidates(virtual_path: str) -> tuple[str, ...]:
    """Try an alternative extension only when the WRP's literal name fails.

    OFP texture tables and asset PBOs sometimes disagree on .paa vs .pac.
    Preserve the exact asset first; never swap two existing valid textures.
    """
    if _safe_relative(virtual_path) is None:
        return ()
    suffix = Path(virtual_path).suffix.casefold()
    if suffix not in {".paa", ".pac"}:
        return (virtual_path,)
    alternate = ".pac" if suffix == ".paa" else ".paa"
    return (virtual_path, virtual_path[:-4] + alternate)


def _texture_asset_exists(
    candidates: tuple[str, ...], search_roots: tuple[Path, ...],
    archives: PboTextures | None,
) -> bool:
    for path in candidates:
        relative = _safe_relative(path)
        if relative is not None and any((root / relative).is_file() for root in search_roots):
            return True
        if archives is not None:
            try:
                if canonical_path(path) in archives.members:
                    return True
            except ValueError:
                pass
    return False


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
        suffix = target.suffix.casefold()
        if suffix == ".paa":
            image = decode_paa_preview(target.read_bytes())
        elif suffix == ".pac":
            image = decode_pac_preview(target.read_bytes())
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
        suffix = Path(virtual_path).suffix.casefold()
        if suffix == ".paa":
            image = decode_paa_preview(data)
        elif suffix == ".pac":
            image = decode_pac_preview(data)
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


def _resolve_tile(
    path: str, search_roots: tuple[Path, ...], archives: PboTextures | None,
) -> np.ndarray | None:
    """Resolve the exact WRP path before trying its alternate extension."""
    candidates = _texture_candidates(path)
    for root in reversed(search_roots):
        for candidate in candidates:
            tile = _image_tile(str(root), candidate)
            if tile is not None:
                return tile
    if archives is not None:
        for candidate in candidates:
            tile = _pbo_image_tile(archives, candidate)
            if tile is not None:
                return tile
    return None


def _corner_means(tile: np.ndarray) -> np.ndarray:
    height, width = tile.shape[:2]
    dy, dx = max(1, height // 5), max(1, width // 5)
    return np.asarray((
        tile[:dy, :dx].mean(axis=(0, 1)),       # NW
        tile[:dy, -dx:].mean(axis=(0, 1)),      # NE
        tile[-dy:, :dx].mean(axis=(0, 1)),      # SW
        tile[-dy:, -dx:].mean(axis=(0, 1)),     # SE
    ), dtype=np.float64)


@lru_cache(maxsize=512)
def _auto_stock_border_orientation(
    path: str, search_roots: tuple[Path, ...], archives: PboTextures | None,
) -> str:
    """Match stock quartet corners with their real source material colours.

    The names give NE/NW/SE/SW materials, but the PAC/PAA storage orientation
    must be checked against the decoded artwork, not assumed from WRP rows.
    """
    decoded = _stock_forest_components(path)
    if decoded is None:
        return "North up"
    namespace, parts = decoded
    tile = _resolve_tile(path, search_roots, archives)
    if tile is None:
        return "North up"

    colours: dict[str, np.ndarray] = {}
    for part in set(parts):
        base = _resolve_tile(rf"{namespace}\\{part}.paa", search_roots, archives)
        if base is None:
            return "North up"
        colours[part] = np.asarray(base, dtype=np.float64).mean(axis=(0, 1))
    if max(float(np.linalg.norm(a - b)) for a in colours.values()
           for b in colours.values()) < 10.0:
        return "North up"

    ne, nw, se, sw = parts
    expected = np.asarray([colours[p] for p in (nw, ne, sw, se)])
    best_name, best_score = "North up", float("inf")
    for orientation in FOREST_BORDER_ORIENTATIONS[1:]:
        candidate = _orient_stock_border(tile, orientation)
        difference = _corner_means(candidate) - expected
        score = float(np.sum(difference * difference))
        if score < best_score:
            best_name, best_score = orientation, score
    return best_name


def clear_archive_texture_cache() -> None:
    """Drop decoded tiles before replacing or closing selected archives."""
    _pbo_image_tile.cache_clear()
    _auto_stock_border_orientation.cache_clear()


def clear_loose_texture_cache() -> None:
    """Discard cached tiles when folders are added or removed."""
    _image_tile.cache_clear()
    _auto_stock_border_orientation.cache_clear()


def render_real_textures(
    slots: np.ndarray, x: np.ndarray, z: np.ndarray,
    texture_paths: tuple[str, ...],
    asset_root: Path | str | Sequence[Path | str] | None,
    fallback: np.ndarray, cell_size: float,
    *, archives: PboTextures | None = None,
    diagnostics: dict[str, str] | None = None,
    forest_border_orientation: str = "North up",
) -> tuple[np.ndarray, int]:
    """Apply available actual texture pixels over a slot-colour viewport.

    Inputs are only viewport-sized. The renderer never allocates an image
    with the world's full dimensions. Missing assets keep the fallback colour.
    """
    search_roots = roots(asset_root)
    if forest_border_orientation not in FOREST_BORDER_ORIENTATIONS:
        raise ValueError("invalid forest border preview orientation")
    if slots.shape != fallback.shape[:2]:
        raise ValueError("texture and fallback preview dimensions differ")
    output = fallback.copy()
    loaded = 0
    valid = (slots >= 0) & (slots < len(texture_paths))
    for slot in np.unique(slots[valid]):
        path = texture_paths[int(slot)]
        if not path:
            continue
        candidates = _texture_candidates(path)
        tile = _resolve_tile(path, search_roots, archives)
        if tile is None:
            if diagnostics is not None:
                diagnostics[path] = (
                    "unreadable" if _texture_asset_exists(candidates, search_roots, archives)
                    else "missing"
                )
            continue
        stock_border = _is_stock_forest_border(path)
        if stock_border:
            orientation = (
                _auto_stock_border_orientation(path, search_roots, archives)
                if forest_border_orientation == "Auto (stock artwork)"
                else forest_border_orientation
            )
            tile = _orient_stock_border(tile, orientation)
        rows, cols = np.nonzero(slots == slot)
        tx = np.mod(x[cols] / cell_size, 1.0)
        tz = np.mod(z[rows] / cell_size, 1.0)
        cx = np.minimum(tile.shape[1] - 1, (tx * tile.shape[1]).astype(np.intp))
        if stock_border:
            # Original stock quadrant blends are inspected with north at the
            # top of the decoded tile, independent of the selected rotation.
            cz = np.minimum(tile.shape[0] - 1, ((1.0 - tz) * tile.shape[0]).astype(np.intp))
        else:
            # Generated sports/road/runway images are authored with pixel row 0
            # at the SOUTH edge (see sports_pitch_surface_policy._render_pitch_cell).
            # Flipping those in the viewer breaks cross-cell field markings.
            cz = np.minimum(tile.shape[0] - 1, (tz * tile.shape[0]).astype(np.intp))
        output[rows, cols] = tile[cz, cx]
        loaded += 1
    return output, loaded
