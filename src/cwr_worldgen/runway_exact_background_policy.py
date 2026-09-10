# SPDX-License-Identifier: GPL-3.0-or-later
"""Composite generated runway slices over the exact terrain texture they replace.

The runway renderer has to replace a complete RVW4 terrain cell. Colour-matched
fallbacks are useful when no source artwork is available, but they inevitably
show seams beside stock terrain. This policy resolves the actual WRP texture from
local generated files or configured/game-adjacent asset roots, decodes DXT1 PAA,
and uses those pixels as the runway-cell background for every ground preset.

When the source PAA and generated slice have the same native dimensions, unchanged
DXT1 blocks are copied verbatim into the generated PAA at every mip level. Only
blocks whose footprint intersects the runway are recompressed. Thus the outer
part of a runway cell remains the same compressed terrain artwork as its neighbour.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import io
import json
import math
import os
import struct
from typing import Sequence

import numpy as np
from PIL import Image, ImageStat

from . import runway_surface_policy as _runway
from .paa import _compress_dxt1_rgb_bytes


_DXT1_MAGIC = 0xFF01
_TAG_SIGNATURE = b"GGAT"
_TAG_AVERAGE = b"CGVA"
_TAG_OFFSETS = b"SFFO"
_PBO_FIELDS = struct.Struct("<IIIII")
_PBO_PROPERTIES = 0x56657273
_PBO_COMPRESSED = 0x43707273
_SURFACE_CACHE_V20 = "surface-pipeline-v20-exact-runway-background-textures"
_INSTALLED = False
_ORIGINAL_BACKGROUND_TEXTURE = None
_ORIGINAL_RENDER_RUNWAY_CELL = None
_ORIGINAL_APPLY_RUNWAY_TEXTURES = None
_ORIGINAL_WRITE_RUNWAY_PAA = None


@dataclass(frozen=True, slots=True)
class _PaaMip:
    width: int
    height: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class _ExactTexture:
    wire_path: str
    source: str
    data: bytes
    mips: tuple[_PaaMip, ...]
    top_image: Image.Image


@dataclass(slots=True)
class _ExactBuildState:
    source_dir: Path
    spec: object
    textures: dict[str, _ExactTexture]
    exact_cell_assignments: int = 0
    fallback_cell_assignments: int = 0


_EXACT_STATE: ContextVar[_ExactBuildState | None] = ContextVar(
    "cwr_runway_exact_background_state", default=None
)
_LAST_EXACT_TEXTURE: ContextVar[_ExactTexture | None] = ContextVar(
    "cwr_runway_last_exact_background", default=None
)


def _canonical(value: object) -> str:
    path = str(value or "").replace("/", "\\").strip().lstrip("\\").casefold()
    while "\\\\" in path:
        path = path.replace("\\\\", "\\")
    return path


def _decode_rgb565(value: int) -> tuple[int, int, int]:
    red = (value >> 11) & 0x1F
    green = (value >> 5) & 0x3F
    blue = value & 0x1F
    return red * 255 // 31, green * 255 // 63, blue * 255 // 31


def _parse_dxt1_paa(data: bytes) -> tuple[_PaaMip, ...]:
    """Read legacy DXT1 PAA mip payloads without requiring external tools."""
    stream = io.BytesIO(data)
    raw_magic = stream.read(2)
    if len(raw_magic) != 2 or struct.unpack("<H", raw_magic)[0] != _DXT1_MAGIC:
        raise ValueError("terrain source is not a DXT1 PAA")

    while True:
        marker = stream.read(4)
        if marker != _TAG_SIGNATURE:
            stream.seek(-len(marker), io.SEEK_CUR)
            break
        name = stream.read(4)
        size_raw = stream.read(4)
        if len(name) != 4 or len(size_raw) != 4:
            raise ValueError("truncated PAA tag")
        tag_size = struct.unpack("<I", size_raw)[0]
        if len(stream.read(tag_size)) != tag_size:
            raise ValueError("truncated PAA tag payload")

    palette_raw = stream.read(2)
    if len(palette_raw) != 2:
        raise ValueError("truncated PAA palette header")
    palette_count = struct.unpack("<H", palette_raw)[0]
    if palette_count:
        raise ValueError("palette DXT1 PAA is not supported for runway compositing")

    mips: list[_PaaMip] = []
    while True:
        dimensions = stream.read(4)
        if len(dimensions) != 4:
            raise ValueError("truncated PAA mip dimensions")
        width, height = struct.unpack("<HH", dimensions)
        if width == 0 and height == 0:
            break
        size_raw = stream.read(3)
        if len(size_raw) != 3:
            raise ValueError("truncated PAA mip size")
        payload_size = int.from_bytes(size_raw, "little")
        payload = stream.read(payload_size)
        if len(payload) != payload_size:
            raise ValueError("truncated PAA mip payload")
        expected = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * 8
        if payload_size != expected:
            raise ValueError("unsupported compressed/non-DXT1 PAA mip payload")
        mips.append(_PaaMip(width, height, payload))
    if not mips:
        raise ValueError("PAA contains no mipmaps")
    return tuple(mips)


def _decode_dxt1_payload(width: int, height: int, payload: bytes) -> Image.Image:
    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    if len(payload) != blocks_x * blocks_y * 8:
        raise ValueError("invalid DXT1 payload length")
    output = bytearray(width * height * 3)
    cursor = 0
    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            colour0, colour1, indices = struct.unpack_from("<HHI", payload, cursor)
            cursor += 8
            p0 = _decode_rgb565(colour0)
            p1 = _decode_rgb565(colour1)
            if colour0 > colour1:
                p2 = tuple((2 * a + b) // 3 for a, b in zip(p0, p1))
                p3 = tuple((a + 2 * b) // 3 for a, b in zip(p0, p1))
            else:
                p2 = tuple((a + b) // 2 for a, b in zip(p0, p1))
                p3 = (0, 0, 0)
            palette = (p0, p1, p2, p3)
            for local_y in range(4):
                y = block_y * 4 + local_y
                if y >= height:
                    break
                for local_x in range(4):
                    x = block_x * 4 + local_x
                    if x >= width:
                        break
                    pixel = local_y * 4 + local_x
                    colour = palette[(indices >> (pixel * 2)) & 0x3]
                    offset = (y * width + x) * 3
                    output[offset : offset + 3] = bytes(colour)
    return Image.frombytes("RGB", (width, height), bytes(output))


def decode_dxt1_paa(data: bytes) -> Image.Image:
    """Decode the largest mip of an OFP/CWA DXT1 PAA into RGB."""
    mip = _parse_dxt1_paa(data)[0]
    return _decode_dxt1_payload(mip.width, mip.height, mip.payload)


def _read_cstring(stream: io.BytesIO) -> str:
    value = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("truncated PBO string")
        if char == b"\0":
            return value.decode("latin-1")
        value.extend(char)


def _decompress_pbo_payload(data: bytes, output_size: int) -> bytes:
    """Decode OFP's Cprs/LZSS entry packing."""
    if output_size < 0:
        raise ValueError("negative PBO output size")
    payload_end = max(0, len(data) - 4)
    output = bytearray()
    cursor = 0
    while len(output) < output_size:
        if cursor >= payload_end:
            raise ValueError("truncated compressed PBO payload")
        flags = data[cursor]
        cursor += 1
        for bit in range(8):
            if len(output) >= output_size:
                break
            if flags & (1 << bit):
                if cursor >= payload_end:
                    raise ValueError("truncated compressed PBO literal")
                output.append(data[cursor])
                cursor += 1
                continue
            if cursor + 2 > payload_end:
                raise ValueError("truncated compressed PBO pointer")
            word = data[cursor] | (data[cursor + 1] << 8)
            cursor += 2
            distance = (word & 0x00FF) + ((word & 0xF000) >> 4)
            run_length = ((word & 0x0F00) >> 8) + 3
            source = len(output) - distance
            for _ in range(run_length):
                if len(output) >= output_size:
                    break
                if source < 0:
                    output.append(0x20)
                elif source < len(output):
                    output.append(output[source])
                else:
                    raise ValueError("invalid compressed PBO back-reference")
                source += 1
    if len(data) >= 4:
        expected = struct.unpack_from("<I", data, len(data) - 4)[0]
        actual = sum(output) & 0xFFFFFFFF
        if expected != actual:
            raise ValueError("compressed PBO checksum mismatch")
    return bytes(output)


def _extract_pbo_asset(path: Path, target: str) -> bytes | None:
    raw = path.read_bytes()
    stream = io.BytesIO(raw)
    entries: list[tuple[str, int, int, int]] = []
    properties: dict[str, str] = {}
    while True:
        name = _read_cstring(stream)
        fields = stream.read(_PBO_FIELDS.size)
        if len(fields) != _PBO_FIELDS.size:
            raise ValueError("truncated PBO header")
        packing, original_size, _reserved, _timestamp, data_size = _PBO_FIELDS.unpack(fields)
        if not name:
            if packing == _PBO_PROPERTIES:
                while True:
                    key = _read_cstring(stream)
                    if not key:
                        break
                    properties[key.casefold()] = _read_cstring(stream)
                continue
            # Resistance PBOs may use a non-zero end-of-header marker. Filename
            # emptiness, rather than the remaining fields, is the reliable fence.
            break
        entries.append((name, packing, original_size, data_size))

    prefix = properties.get("prefix", "").replace("/", "\\").strip("\\")
    if not prefix:
        prefix = path.stem
    data_cursor = stream.tell()
    for name, packing, original_size, data_size in entries:
        stored = raw[data_cursor : data_cursor + data_size]
        if len(stored) != data_size:
            raise ValueError(f"truncated PBO entry {name}")
        data_cursor += data_size
        combined = name.replace("/", "\\").lstrip("\\")
        if prefix and not _canonical(combined).startswith(_canonical(prefix) + "\\"):
            combined = prefix + "\\" + combined
        if _canonical(combined) != target:
            continue
        if packing == 0:
            return stored
        if packing == _PBO_COMPRESSED and original_size > 0:
            return _decompress_pbo_payload(stored, original_size)
        return None
    return None


def _casefold_child(parent: Path, name: str) -> Path | None:
    direct = parent / name
    if direct.exists():
        return direct
    if not parent.is_dir():
        return None
    folded = name.casefold()
    try:
        for child in parent.iterdir():
            if child.name.casefold() == folded:
                return child
    except OSError:
        return None
    return None


def _casefold_relative(root: Path, canonical_path: str) -> Path | None:
    current = root
    for part in canonical_path.split("\\"):
        current = _casefold_child(current, part)
        if current is None:
            return None
    return current if current.is_file() else None


def _candidate_asset_roots(spec) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for value in getattr(spec, "asset_roots", ()) or ():
        try:
            candidates.append(Path(value).expanduser().resolve())
        except OSError:
            continue
    deploy = getattr(spec, "deploy_mod_dir", None)
    if deploy:
        try:
            # A normal @mod directory lives directly under the CWA installation.
            candidates.append(Path(deploy).expanduser().resolve().parent)
        except OSError:
            pass
    env_root = os.environ.get("CWR_GAME_ROOT", "").strip()
    if env_root:
        try:
            candidates.append(Path(env_root).expanduser().resolve())
        except OSError:
            pass
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key not in seen and candidate.exists():
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _likely_pbos(root: Path, prefix: str) -> tuple[Path, ...]:
    if root.is_file():
        return (root,) if root.suffix.casefold() == ".pbo" else ()
    wanted = f"{prefix}.pbo".casefold()
    candidates: list[Path] = []
    for relative in (
        Path(f"{prefix}.pbo"),
        Path("Res") / "AddOns" / f"{prefix}.pbo",
        Path("AddOns") / f"{prefix}.pbo",
        Path("res") / "addons" / f"{prefix}.pbo",
        Path("addons") / f"{prefix}.pbo",
    ):
        candidate = root / relative
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    if candidates:
        return tuple(candidates)
    try:
        for candidate in root.rglob("*.pbo"):
            if candidate.name.casefold() == wanted:
                candidates.append(candidate)
    except OSError:
        pass
    return tuple(candidates)


@lru_cache(maxsize=512)
def _read_external_asset_cached(root_names: tuple[str, ...], canonical_path: str) -> tuple[bytes, str] | None:
    prefix = canonical_path.split("\\", 1)[0]
    for root_name in root_names:
        root = Path(root_name)
        if root.is_dir():
            loose = _casefold_relative(root, canonical_path)
            if loose is None and root.name.casefold() == prefix:
                remainder = canonical_path.split("\\", 1)[1] if "\\" in canonical_path else ""
                loose = _casefold_relative(root, remainder) if remainder else None
            if loose is not None:
                try:
                    return loose.read_bytes(), str(loose)
                except OSError:
                    pass
        for pbo in _likely_pbos(root, prefix):
            try:
                value = _extract_pbo_asset(pbo, canonical_path)
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            if value is not None:
                return value, str(pbo)
    return None


def _local_asset(source_dir: Path, world_name: str, canonical_path: str) -> tuple[bytes, str] | None:
    parts = canonical_path.split("\\")
    relatives: list[str] = []
    if parts and parts[0] == str(world_name).casefold() and len(parts) > 1:
        relatives.append("\\".join(parts[1:]))
    # Also accept source-local wire paths such as data\\g.paa.
    relatives.append(canonical_path)
    for relative in relatives:
        candidate = _casefold_relative(source_dir, relative)
        if candidate is None:
            continue
        try:
            return candidate.read_bytes(), str(candidate)
        except OSError:
            continue
    return None


def _load_exact_texture(source_dir: Path, spec, wire_path: str) -> _ExactTexture | None:
    canonical_path = _canonical(wire_path)
    if not canonical_path or not canonical_path.endswith(".paa"):
        return None
    located = _local_asset(source_dir, str(getattr(spec, "name", "")), canonical_path)
    if located is None:
        roots = _candidate_asset_roots(spec)
        located = _read_external_asset_cached(tuple(str(root) for root in roots), canonical_path)
    if located is None:
        return None
    data, source = located
    try:
        mips = _parse_dxt1_paa(data)
        top = _decode_dxt1_payload(mips[0].width, mips[0].height, mips[0].payload)
    except (ValueError, struct.error):
        return None
    if top.width != top.height or top.width < 16 or top.width & (top.width - 1):
        return None
    return _ExactTexture(str(wire_path), source, data, mips, top)


def _tag(name: bytes, payload: bytes) -> bytes:
    return _TAG_SIGNATURE + name + struct.pack("<I", len(payload)) + payload


def _u24(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFF:
        raise ValueError("PAA mip payload exceeds 24-bit size field")
    return value.to_bytes(3, "little")


def _rgb_block_bytes(image: Image.Image, block_x: int, block_y: int) -> bytes:
    image = image.convert("RGB")
    raw = image.tobytes()
    width, _height = image.size
    stride = width * 3
    x = block_x * 4
    y = block_y * 4
    offset = y * stride + x * 3
    return (
        raw[offset : offset + 12]
        + raw[offset + stride : offset + stride + 12]
        + raw[offset + stride * 2 : offset + stride * 2 + 12]
        + raw[offset + stride * 3 : offset + stride * 3 + 12]
    )


def _changed_in_source_region(
    changed: np.ndarray,
    *,
    block_x: int,
    block_y: int,
    mip_width: int,
    mip_height: int,
) -> bool:
    top_height, top_width = changed.shape
    x0 = int(math.floor(block_x * 4 * top_width / mip_width))
    x1 = int(math.ceil(min(mip_width, block_x * 4 + 4) * top_width / mip_width))
    y0 = int(math.floor(block_y * 4 * top_height / mip_height))
    y1 = int(math.ceil(min(mip_height, block_y * 4 + 4) * top_height / mip_height))
    return bool(np.any(changed[y0:y1, x0:x1]))


def _write_with_preserved_background(path: Path, image: Image.Image, source: _ExactTexture) -> None:
    """Write runway PAA while copying untouched source DXT1 blocks verbatim."""
    image = image.convert("RGB")
    if image.size != source.top_image.size:
        _ORIGINAL_WRITE_RUNWAY_PAA(path, image)
        return

    source_top = np.asarray(source.top_image, dtype=np.uint8)
    final_top = np.asarray(image, dtype=np.uint8)
    changed = np.any(source_top != final_top, axis=2)
    source_by_size = {(mip.width, mip.height): mip for mip in source.mips}
    levels: list[tuple[int, int, bytes]] = []
    current = image
    while True:
        width, height = current.size
        if width < 4 or height < 4 or width % 4 or height % 4:
            break
        blocks_x, blocks_y = width // 4, height // 4
        source_mip = source_by_size.get((width, height))
        payload = bytearray(blocks_x * blocks_y * 8)
        cursor = 0
        for block_y in range(blocks_y):
            for block_x in range(blocks_x):
                preserve = (
                    source_mip is not None
                    and not _changed_in_source_region(
                        changed,
                        block_x=block_x,
                        block_y=block_y,
                        mip_width=width,
                        mip_height=height,
                    )
                )
                if preserve:
                    source_offset = (block_y * blocks_x + block_x) * 8
                    block = source_mip.payload[source_offset : source_offset + 8]
                else:
                    block = _compress_dxt1_rgb_bytes(
                        _rgb_block_bytes(current, block_x, block_y)
                    )
                payload[cursor : cursor + 8] = block
                cursor += 8
        levels.append((width, height, bytes(payload)))
        if width == 4 and height == 4:
            break
        current = current.resize((max(4, width // 2), max(4, height // 2)), Image.Resampling.BOX)
        if len(levels) >= 16:
            break

    if not levels or levels[-1][0] != 4 or levels[-1][1] != 4:
        _ORIGINAL_WRITE_RUNWAY_PAA(path, image)
        return

    fixed_prefix_size = 2 + (12 + 4) + (12 + 64) + 2
    offsets: list[int] = []
    cursor = fixed_prefix_size
    for width, height, payload in levels:
        offsets.append(cursor)
        cursor += 4 + 3 + len(payload)
    offsets.extend([0] * (16 - len(offsets)))
    mean = ImageStat.Stat(image).mean
    red, green, blue = int(mean[0]), int(mean[1]), int(mean[2])
    output = bytearray(struct.pack("<H", _DXT1_MAGIC))
    output += _tag(_TAG_AVERAGE, bytes((blue, green, red, 255)))
    output += _tag(_TAG_OFFSETS, struct.pack("<16I", *offsets))
    output += struct.pack("<H", 0)
    for width, height, payload in levels:
        output += struct.pack("<HH", width, height)
        output += _u24(len(payload))
        output += payload
    output += struct.pack("<HH", 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(output))


def _background_with_exact_source(material, profile, seed, size, *, ground_path="") -> Image.Image:
    state = _EXACT_STATE.get()
    exact = state.textures.get(_canonical(ground_path)) if state is not None else None
    if exact is None:
        return _ORIGINAL_BACKGROUND_TEXTURE(
            material, profile, seed, size, ground_path=ground_path
        )
    image = exact.top_image
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BOX)
    return image.copy()


def _render_with_exact_source(*args, **kwargs):
    state = _EXACT_STATE.get()
    exact = None
    if state is not None:
        original_index = int(kwargs.get("original_wrp_texture_index", 0))
        material_index = original_index - 1
        ground_path = _runway._ground_path_for_material(state.spec, material_index)
        exact = state.textures.get(_canonical(ground_path))
        if exact is not None:
            state.exact_cell_assignments += 1
        else:
            state.fallback_cell_assignments += 1
    _LAST_EXACT_TEXTURE.set(exact)
    return _ORIGINAL_RENDER_RUNWAY_CELL(*args, **kwargs)


def _write_runway_paa(path: Path, image: Image.Image, *args, **kwargs) -> None:
    exact = _LAST_EXACT_TEXTURE.get()
    if exact is None or args or kwargs:
        _ORIGINAL_WRITE_RUNWAY_PAA(path, image, *args, **kwargs)
        return
    _write_with_preserved_background(path, image, exact)


def _apply_with_exact_backgrounds(
    source_dir: Path,
    dataset,
    projection,
    spec,
    texture_indices: Sequence[int],
    texture_paths: Sequence[str],
):
    source_dir = Path(source_dir)
    touched = _runway.runway_texture_cell_indices(dataset, projection, spec)
    exact_by_path: dict[str, _ExactTexture] = {}
    for cell_index in touched:
        original_index = int(texture_indices[cell_index])
        material_index = original_index - 1
        ground_path = _runway._ground_path_for_material(spec, material_index)
        canonical = _canonical(ground_path)
        if not canonical or canonical in exact_by_path:
            continue
        exact = _load_exact_texture(source_dir, spec, ground_path)
        if exact is not None:
            exact_by_path[canonical] = exact

    state = _ExactBuildState(source_dir, spec, exact_by_path)
    token = _EXACT_STATE.set(state)
    old_size = _runway.RUNWAY_TEXTURE_SIZE
    # If every exact source agrees on one native size, render at that size so
    # unchanged DXT1 blocks can remain byte-for-byte identical to the stock PAA.
    native_sizes = {texture.top_image.width for texture in exact_by_path.values()}
    if len(native_sizes) == 1:
        candidate = next(iter(native_sizes))
        if 16 <= candidate <= 1024 and not (candidate & (candidate - 1)):
            _runway.RUNWAY_TEXTURE_SIZE = candidate
    try:
        result = _ORIGINAL_APPLY_RUNWAY_TEXTURES(
            source_dir,
            dataset,
            projection,
            spec,
            texture_indices,
            texture_paths,
        )
    finally:
        _runway.RUNWAY_TEXTURE_SIZE = old_size
        _EXACT_STATE.reset(token)
        _LAST_EXACT_TEXTURE.set(None)

    report_path = source_dir / "runway-textures.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update({
                "background_source_mode": (
                    "exact-terrain-texture" if state.exact_cell_assignments and not state.fallback_cell_assignments
                    else "mixed-exact-and-fallback" if state.exact_cell_assignments
                    else "profile-fallback"
                ),
                "exact_background_cell_assignments": state.exact_cell_assignments,
                "fallback_background_cell_assignments": state.fallback_cell_assignments,
                "exact_background_paths": sorted(
                    texture.wire_path for texture in state.textures.values()
                ),
                "exact_background_sources": sorted(
                    {texture.source for texture in state.textures.values()}
                ),
            })
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return result


def install_runway_exact_background_policy() -> None:
    """Use exact source terrain beneath runways for every ground preset."""
    global _INSTALLED
    global _ORIGINAL_BACKGROUND_TEXTURE, _ORIGINAL_RENDER_RUNWAY_CELL
    global _ORIGINAL_APPLY_RUNWAY_TEXTURES, _ORIGINAL_WRITE_RUNWAY_PAA
    if _INSTALLED:
        return

    _ORIGINAL_BACKGROUND_TEXTURE = _runway._background_texture
    _ORIGINAL_RENDER_RUNWAY_CELL = _runway._render_runway_cell
    _ORIGINAL_APPLY_RUNWAY_TEXTURES = _runway.apply_generated_runway_texture_table
    _ORIGINAL_WRITE_RUNWAY_PAA = _runway.write_rgb_dxt1_paa

    _runway._background_texture = _background_with_exact_source
    _runway._render_runway_cell = _render_with_exact_source
    _runway.apply_generated_runway_texture_table = _apply_with_exact_backgrounds
    _runway.write_rgb_dxt1_paa = _write_runway_paa

    # The runway cache wrapper resolves this module global at call time. Replace
    # whichever v18/v19 value is currently installed with the exact-source gate.
    _runway._SURFACE_CACHE_V18 = _SURFACE_CACHE_V20
    _INSTALLED = True
