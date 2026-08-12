# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import struct
from typing import Sequence

from PIL import Image, ImageStat

_DXT1_MAGIC = 0xFF01
_TAG_SIGNATURE = b"GGAT"
_TAG_AVERAGE = b"CGVA"  # "AVGC" on disk in little-endian tag notation
_TAG_OFFSETS = b"SFFO"  # "OFFS"
_MINIMUM_LEGACY_MIP = 4


@dataclass(frozen=True, slots=True)
class PaaSummary:
    magic: int
    width: int
    height: int
    mipmap_count: int
    minimum_mip_width: int
    minimum_mip_height: int
    tags: tuple[str, ...]


def _rgb565(red: int, green: int, blue: int) -> int:
    for label, value in (("red", red), ("green", green), ("blue", blue)):
        if not 0 <= value <= 255:
            raise ValueError(f"{label} channel must be within 0..255")
    return ((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3)


def _u24(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFF:
        raise ValueError("PAA mipmap payload exceeds the 24-bit size field")
    return value.to_bytes(3, "little")


def _solid_dxt1_payload(width: int, height: int, colour565: int) -> bytes:
    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    # Keep color0 > color1 so the block is decoded in opaque four-colour mode.
    color0 = colour565
    color1 = colour565 - 1 if colour565 > 0 else 0
    if color0 <= color1:
        color0, color1 = 1, 0
    block = struct.pack("<HHI", color0, color1, 0)
    return block * (blocks_x * blocks_y)


def _tag(name: bytes, payload: bytes) -> bytes:
    if len(name) != 4:
        raise ValueError("PAA tag names must be four bytes")
    return _TAG_SIGNATURE + name + struct.pack("<I", len(payload)) + payload


def write_solid_dxt1_paa(
    path: Path,
    *,
    width: int = 128,
    height: int = 128,
    colour: tuple[int, int, int] = (86, 125, 70),
    max_levels: int = 16,
) -> None:
    if width < _MINIMUM_LEGACY_MIP or height < _MINIMUM_LEGACY_MIP:
        raise ValueError("legacy-compatible PAA dimensions must be at least 4x4")
    if width & (width - 1) or height & (height - 1):
        raise ValueError("PAA dimensions must be powers of two")
    if max_levels <= 0:
        raise ValueError("PAA must contain at least one mipmap")

    colour565 = _rgb565(*colour)
    levels: list[tuple[int, int, bytes]] = []
    mip_width = width
    mip_height = height
    for _ in range(max_levels):
        levels.append((mip_width, mip_height, _solid_dxt1_payload(mip_width, mip_height, colour565)))
        if mip_width == _MINIMUM_LEGACY_MIP and mip_height == _MINIMUM_LEGACY_MIP:
            break
        mip_width = max(_MINIMUM_LEGACY_MIP, mip_width // 2)
        mip_height = max(_MINIMUM_LEGACY_MIP, mip_height // 2)
    else:
        raise ValueError("max_levels is too small to reach the 4x4 legacy mip")

    if len(levels) > 16:
        raise ValueError("PAA OFFS metadata supports at most 16 mipmaps")

    # Match legacy OFP/CWA PAA files: magic, AVGC tag, OFFS tag, palette,
    # then mipmaps. The OFFS entries point to each mipmap's width field.
    fixed_prefix_size = 2 + (12 + 4) + (12 + 64) + 2
    offsets: list[int] = []
    cursor = fixed_prefix_size
    for mip_width, mip_height, payload in levels:
        offsets.append(cursor)
        cursor += 4 + 3 + len(payload)
    offsets.extend([0] * (16 - len(offsets)))

    red, green, blue = colour
    average_payload = bytes((blue, green, red, 255))
    offsets_payload = struct.pack("<16I", *offsets)

    output = bytearray()
    output += struct.pack("<H", _DXT1_MAGIC)
    output += _tag(_TAG_AVERAGE, average_payload)
    output += _tag(_TAG_OFFSETS, offsets_payload)
    output += struct.pack("<H", 0)  # no palette for DXT1
    for mip_width, mip_height, payload in levels:
        output += struct.pack("<HH", mip_width, mip_height)
        output += _u24(len(payload))
        output += payload
    output += struct.pack("<HH", 0, 0)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(output))



def _decode_rgb565(value: int) -> tuple[int, int, int]:
    red = (value >> 11) & 0x1F
    green = (value >> 5) & 0x3F
    blue = value & 0x1F
    return (red * 255 // 31, green * 255 // 63, blue * 255 // 31)


def _compress_dxt1_rgb_bytes(raw: bytes) -> bytes:
    """Compress one 4x4 RGB block with a fast deterministic endpoint fit.

    The previous encoder exhaustively compared all 120 pixel pairs in every
    block. A 1024px overview with mipmaps contains roughly 87,000 blocks, so
    that quality heuristic dominated the entire preview stage. The bounding-box
    axis below selects endpoints in linear time and preserves opaque DXT1 mode.
    """

    if len(raw) != 48:
        raise ValueError("DXT1 RGB blocks require exactly 48 bytes")
    r_min = g_min = b_min = 255
    r_max = g_max = b_max = 0
    for offset in range(0, 48, 3):
        red, green, blue = raw[offset], raw[offset + 1], raw[offset + 2]
        if red < r_min: r_min = red
        if red > r_max: r_max = red
        if green < g_min: g_min = green
        if green > g_max: g_max = green
        if blue < b_min: b_min = blue
        if blue > b_max: b_max = blue

    dr, dg, db = r_max - r_min, g_max - g_min, b_max - b_min
    if dr == 0 and dg == 0 and db == 0:
        color = _rgb565(r_min, g_min, b_min)
        color0 = min(0xFFFF, color + 1) if color < 0xFFFF else color
        color1 = max(0, color - 1) if color > 0 else 0
        if color0 <= color1:
            color0, color1 = 1, 0
        return struct.pack("<HHI", color0, color1, 0)

    low_projection: int | None = None
    high_projection: int | None = None
    low = (r_min, g_min, b_min)
    high = (r_max, g_max, b_max)
    for offset in range(0, 48, 3):
        red, green, blue = raw[offset], raw[offset + 1], raw[offset + 2]
        projection = red * dr + green * dg + blue * db
        if low_projection is None or projection < low_projection:
            low_projection = projection
            low = (red, green, blue)
        if high_projection is None or projection > high_projection:
            high_projection = projection
            high = (red, green, blue)

    color0 = _rgb565(*high)
    color1 = _rgb565(*low)
    if color0 == color1:
        color0 = min(0xFFFF, color0 + 1) if color0 < 0xFFFF else color0
        color1 = max(0, color1 - 1) if color1 > 0 else color1
    if color0 <= color1:
        color0, color1 = color1, color0

    e0r, e0g, e0b = _decode_rgb565(color0)
    e1r, e1g, e1b = _decode_rgb565(color1)
    p2r, p2g, p2b = (2 * e0r + e1r) // 3, (2 * e0g + e1g) // 3, (2 * e0b + e1b) // 3
    p3r, p3g, p3b = (e0r + 2 * e1r) // 3, (e0g + 2 * e1g) // 3, (e0b + 2 * e1b) // 3
    palette = ((e0r, e0g, e0b), (e1r, e1g, e1b), (p2r, p2g, p2b), (p3r, p3g, p3b))

    indices = 0
    pixel_index = 0
    for offset in range(0, 48, 3):
        red, green, blue = raw[offset], raw[offset + 1], raw[offset + 2]
        best_index = 0
        best_distance = 1 << 30
        for candidate, (pr, pg, pb) in enumerate(palette):
            rd, gd, bd = red - pr, green - pg, blue - pb
            distance = rd * rd + gd * gd + bd * bd
            if distance < best_distance:
                best_distance = distance
                best_index = candidate
        indices |= best_index << (pixel_index * 2)
        pixel_index += 1
    return struct.pack("<HHI", color0, color1, indices)


def _compress_dxt1_block(pixels: Sequence[tuple[int, int, int]]) -> bytes:
    if len(pixels) != 16:
        raise ValueError("DXT1 blocks require exactly 16 pixels")
    raw = bytearray(48)
    cursor = 0
    for pixel in pixels:
        if len(pixel) != 3 or any(not 0 <= value <= 255 for value in pixel):
            raise ValueError("RGB pixels must contain three channels within 0..255")
        raw[cursor] = int(pixel[0])
        raw[cursor + 1] = int(pixel[1])
        raw[cursor + 2] = int(pixel[2])
        cursor += 3
    return _compress_dxt1_rgb_bytes(bytes(raw))


def _image_dxt1_payload(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    width, height = image.size
    if width % 4 or height % 4:
        raise ValueError("DXT1 image dimensions must be multiples of four")
    data = image.tobytes()
    stride = width * 3
    output = bytearray((width // 4) * (height // 4) * 8)
    cursor = 0
    cache: dict[bytes, bytes] = {}
    cache_limit = 65_536
    for block_y in range(0, height, 4):
        row0 = block_y * stride
        for block_x in range(0, width, 4):
            offset = row0 + block_x * 3
            block = (
                data[offset : offset + 12]
                + data[offset + stride : offset + stride + 12]
                + data[offset + stride * 2 : offset + stride * 2 + 12]
                + data[offset + stride * 3 : offset + stride * 3 + 12]
            )
            compressed = cache.get(block)
            if compressed is None:
                compressed = _compress_dxt1_rgb_bytes(block)
                if len(cache) < cache_limit:
                    cache[block] = compressed
            output[cursor : cursor + 8] = compressed
            cursor += 8
    return bytes(output)




def _compress_dxt1_rgba_bytes(raw: bytes, *, alpha_threshold: int = 128) -> bytes:
    """Compress one 4x4 RGBA block using DXT1's one-bit transparency mode."""

    if len(raw) != 64:
        raise ValueError("DXT1 RGBA blocks require exactly 64 bytes")
    opaque = []
    for offset in range(0, 64, 4):
        if raw[offset + 3] >= alpha_threshold:
            opaque.append((raw[offset], raw[offset + 1], raw[offset + 2]))
    if len(opaque) == 16:
        rgb = bytearray(48)
        cursor = 0
        for offset in range(0, 64, 4):
            rgb[cursor:cursor+3] = raw[offset:offset+3]
            cursor += 3
        return _compress_dxt1_rgb_bytes(bytes(rgb))
    if not opaque:
        return struct.pack("<HHI", 0, 0, 0xFFFFFFFF)

    r_min = min(pixel[0] for pixel in opaque)
    g_min = min(pixel[1] for pixel in opaque)
    b_min = min(pixel[2] for pixel in opaque)
    r_max = max(pixel[0] for pixel in opaque)
    g_max = max(pixel[1] for pixel in opaque)
    b_max = max(pixel[2] for pixel in opaque)
    dr, dg, db = r_max-r_min, g_max-g_min, b_max-b_min
    if dr == 0 and dg == 0 and db == 0:
        low = high = opaque[0]
    else:
        low = min(opaque, key=lambda pixel: pixel[0]*dr + pixel[1]*dg + pixel[2]*db)
        high = max(opaque, key=lambda pixel: pixel[0]*dr + pixel[1]*dg + pixel[2]*db)
    color0 = _rgb565(*low)
    color1 = _rgb565(*high)
    # DXT1 transparency mode requires color0 <= color1. Equal endpoints are OK.
    if color0 > color1:
        color0, color1 = color1, color0
    c0 = _decode_rgb565(color0)
    c1 = _decode_rgb565(color1)
    c2 = tuple((a+b)//2 for a,b in zip(c0,c1))
    palette = (c0,c1,c2)
    indices = 0
    pixel_index = 0
    for offset in range(0, 64, 4):
        if raw[offset + 3] < alpha_threshold:
            best = 3
        else:
            red, green, blue = raw[offset], raw[offset+1], raw[offset+2]
            best = min(range(3), key=lambda candidate: (red-palette[candidate][0])**2 + (green-palette[candidate][1])**2 + (blue-palette[candidate][2])**2)
        indices |= best << (pixel_index * 2)
        pixel_index += 1
    return struct.pack("<HHI", color0, color1, indices)


def _image_dxt1_rgba_payload(image: Image.Image) -> bytes:
    image = image.convert("RGBA")
    width, height = image.size
    if width % 4 or height % 4:
        raise ValueError("DXT1 image dimensions must be multiples of four")
    data = image.tobytes()
    stride = width * 4
    output = bytearray((width // 4) * (height // 4) * 8)
    cursor = 0
    cache: dict[bytes, bytes] = {}
    for block_y in range(0, height, 4):
        row0 = block_y * stride
        for block_x in range(0, width, 4):
            offset = row0 + block_x * 4
            block = (
                data[offset:offset+16]
                + data[offset+stride:offset+stride+16]
                + data[offset+stride*2:offset+stride*2+16]
                + data[offset+stride*3:offset+stride*3+16]
            )
            compressed = cache.get(block)
            if compressed is None:
                compressed = _compress_dxt1_rgba_bytes(block)
                if len(cache) < 65_536:
                    cache[block] = compressed
            output[cursor:cursor+8] = compressed
            cursor += 8
    return bytes(output)

def write_rgb_dxt1_paa(
    path: Path,
    image: Image.Image,
    *,
    max_levels: int = 16,
) -> None:
    image = image.convert("RGB")
    width, height = image.size
    if width < _MINIMUM_LEGACY_MIP or height < _MINIMUM_LEGACY_MIP:
        raise ValueError("legacy-compatible PAA dimensions must be at least 4x4")
    if width & (width - 1) or height & (height - 1):
        raise ValueError("PAA dimensions must be powers of two")
    if max_levels <= 0:
        raise ValueError("PAA must contain at least one mipmap")

    levels: list[tuple[int, int, bytes]] = []
    current = image
    for _ in range(max_levels):
        mip_width, mip_height = current.size
        levels.append((mip_width, mip_height, _image_dxt1_payload(current)))
        if mip_width == _MINIMUM_LEGACY_MIP and mip_height == _MINIMUM_LEGACY_MIP:
            break
        next_width = max(_MINIMUM_LEGACY_MIP, mip_width // 2)
        next_height = max(_MINIMUM_LEGACY_MIP, mip_height // 2)
        current = current.resize((next_width, next_height), Image.Resampling.BOX)
    else:
        raise ValueError("max_levels is too small to reach the 4x4 legacy mip")

    if len(levels) > 16:
        raise ValueError("PAA OFFS metadata supports at most 16 mipmaps")

    fixed_prefix_size = 2 + (12 + 4) + (12 + 64) + 2
    offsets: list[int] = []
    cursor = fixed_prefix_size
    for mip_width, mip_height, payload in levels:
        offsets.append(cursor)
        cursor += 4 + 3 + len(payload)
    offsets.extend([0] * (16 - len(offsets)))

    mean = ImageStat.Stat(image).mean
    red, green, blue = (int(mean[0]), int(mean[1]), int(mean[2]))

    output = bytearray()
    output += struct.pack("<H", _DXT1_MAGIC)
    output += _tag(_TAG_AVERAGE, bytes((blue, green, red, 255)))
    output += _tag(_TAG_OFFSETS, struct.pack("<16I", *offsets))
    output += struct.pack("<H", 0)
    for mip_width, mip_height, payload in levels:
        output += struct.pack("<HH", mip_width, mip_height)
        output += _u24(len(payload))
        output += payload
    output += struct.pack("<HH", 0, 0)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(output))



def write_rgba_dxt1_paa(
    path: Path,
    image: Image.Image,
    *,
    max_levels: int = 16,
) -> None:
    """Write an RGBA image as legacy-compatible DXT1 with one-bit alpha."""

    image = image.convert("RGBA")
    width, height = image.size
    if width < _MINIMUM_LEGACY_MIP or height < _MINIMUM_LEGACY_MIP:
        raise ValueError("legacy-compatible PAA dimensions must be at least 4x4")
    if width & (width - 1) or height & (height - 1):
        raise ValueError("PAA dimensions must be powers of two")
    if max_levels <= 0:
        raise ValueError("PAA must contain at least one mipmap")

    levels: list[tuple[int, int, bytes]] = []
    current = image
    for _ in range(max_levels):
        mip_width, mip_height = current.size
        levels.append((mip_width, mip_height, _image_dxt1_rgba_payload(current)))
        if mip_width == _MINIMUM_LEGACY_MIP and mip_height == _MINIMUM_LEGACY_MIP:
            break
        next_width = max(_MINIMUM_LEGACY_MIP, mip_width // 2)
        next_height = max(_MINIMUM_LEGACY_MIP, mip_height // 2)
        current = current.resize((next_width, next_height), Image.Resampling.BOX)
    else:
        raise ValueError("max_levels is too small to reach the 4x4 legacy mip")
    if len(levels) > 16:
        raise ValueError("PAA OFFS metadata supports at most 16 mipmaps")

    fixed_prefix_size = 2 + (12 + 4) + (12 + 64) + 2
    offsets: list[int] = []
    cursor = fixed_prefix_size
    for mip_width, mip_height, payload in levels:
        offsets.append(cursor)
        cursor += 4 + 3 + len(payload)
    offsets.extend([0] * (16 - len(offsets)))

    mean = ImageStat.Stat(image).mean
    red, green, blue, alpha = (int(mean[0]), int(mean[1]), int(mean[2]), int(mean[3]))
    output = bytearray()
    output += struct.pack("<H", _DXT1_MAGIC)
    output += _tag(_TAG_AVERAGE, bytes((blue, green, red, alpha)))
    output += _tag(_TAG_OFFSETS, struct.pack("<16I", *offsets))
    output += struct.pack("<H", 0)
    for mip_width, mip_height, payload in levels:
        output += struct.pack("<HH", mip_width, mip_height)
        output += _u24(len(payload))
        output += payload
    output += struct.pack("<HH", 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(output))

def inspect_paa(path: Path) -> PaaSummary:
    stream = io.BytesIO(path.read_bytes())
    magic_raw = stream.read(2)
    if len(magic_raw) != 2:
        raise ValueError("truncated PAA magic")
    magic = struct.unpack("<H", magic_raw)[0]
    if magic != _DXT1_MAGIC:
        raise ValueError(f"expected DXT1 PAA magic 0x{_DXT1_MAGIC:04X}, got 0x{magic:04X}")

    tags: list[str] = []
    offsets_from_tag: tuple[int, ...] | None = None
    while True:
        marker = stream.read(4)
        if marker != _TAG_SIGNATURE:
            stream.seek(-len(marker), io.SEEK_CUR)
            break
        name = stream.read(4)
        size_raw = stream.read(4)
        if len(name) != 4 or len(size_raw) != 4:
            raise ValueError("truncated PAA tag")
        size = struct.unpack("<I", size_raw)[0]
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError("truncated PAA tag payload")
        tags.append(name[::-1].decode("ascii", "replace"))
        if name == _TAG_OFFSETS:
            if size != 64:
                raise ValueError("PAA OFFS tag must contain 16 offsets")
            offsets_from_tag = struct.unpack("<16I", payload)

    palette_raw = stream.read(2)
    if len(palette_raw) != 2:
        raise ValueError("truncated PAA palette header")
    palette_count = struct.unpack("<H", palette_raw)[0]
    if palette_count != 0:
        raise ValueError("unexpected palette in DXT1 PAA")

    first_width = 0
    first_height = 0
    minimum_width = 0
    minimum_height = 0
    mipmap_count = 0
    actual_offsets: list[int] = []
    while True:
        actual_offsets.append(stream.tell())
        dimensions = stream.read(4)
        if len(dimensions) != 4:
            raise ValueError("truncated PAA mipmap dimensions")
        width, height = struct.unpack("<HH", dimensions)
        if width == 0 and height == 0:
            actual_offsets.pop()
            break
        size_bytes = stream.read(3)
        if len(size_bytes) != 3:
            raise ValueError("truncated PAA mipmap size")
        size = int.from_bytes(size_bytes, "little")
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError("truncated PAA mipmap payload")
        expected = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * 8
        if size != expected:
            raise ValueError(f"invalid DXT1 payload size {size}, expected {expected}")
        if mipmap_count == 0:
            first_width = width
            first_height = height
        minimum_width = width
        minimum_height = height
        mipmap_count += 1

    if mipmap_count == 0:
        raise ValueError("PAA contains no mipmaps")
    if minimum_width < _MINIMUM_LEGACY_MIP or minimum_height < _MINIMUM_LEGACY_MIP:
        raise ValueError("PAA contains sub-4x4 mipmaps unsupported by the legacy compatibility profile")
    if offsets_from_tag is not None:
        declared = tuple(value for value in offsets_from_tag if value)
        if declared != tuple(actual_offsets):
            raise ValueError("PAA OFFS metadata does not match mipmap locations")
    trailer = stream.read()
    # Some original OFP/CWA PAA files carry one extra zero word after the
    # 0x0000/0x0000 mip terminator.  The game accepts this legacy padding, so
    # the validator should reject actual trailing data rather than authentic
    # zero fill from shipped/community assets.
    if trailer and any(trailer):
        raise ValueError("unexpected non-zero data after PAA terminator")

    return PaaSummary(
        magic=magic,
        width=first_width,
        height=first_height,
        mipmap_count=mipmap_count,
        minimum_mip_width=minimum_width,
        minimum_mip_height=minimum_height,
        tags=tuple(tags),
    )
