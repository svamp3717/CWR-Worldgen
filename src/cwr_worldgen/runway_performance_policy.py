# SPDX-License-Identifier: GPL-3.0-or-later
"""Make runway preparation explicit and keep first-run texture generation cheap."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json
import struct

import numpy as np
from PIL import Image


_INSTALLED = False
_DXT1_RECORD = np.dtype([
    ("color0", "<u2"),
    ("color1", "<u2"),
    ("indices", "<u4"),
])
assert _DXT1_RECORD.itemsize == 8


def _extract_pbo_asset_streaming(exact, path: Path, target: str) -> bytes | None:
    """Read only the selected PBO entry instead of loading the whole package."""
    target = exact._canonical(target)
    with Path(path).open("rb") as stream:
        entries: list[tuple[str, int, int, int]] = []
        properties: dict[str, str] = {}
        while True:
            name = exact._read_cstring(stream)
            fields = stream.read(exact._PBO_FIELDS.size)
            if len(fields) != exact._PBO_FIELDS.size:
                raise ValueError("truncated PBO header")
            packing, original_size, _reserved, _timestamp, data_size = (
                exact._PBO_FIELDS.unpack(fields)
            )
            if not name:
                if packing == exact._PBO_PROPERTIES:
                    while True:
                        key = exact._read_cstring(stream)
                        if not key:
                            break
                        properties[key.casefold()] = exact._read_cstring(stream)
                    continue
                break
            entries.append((name, packing, original_size, data_size))

        prefix = properties.get("prefix", "").replace("/", "\\").strip("\\")
        if not prefix:
            prefix = Path(path).stem
        data_cursor = stream.tell()

        for name, packing, original_size, data_size in entries:
            combined = name.replace("/", "\\").lstrip("\\")
            if prefix and not exact._canonical(combined).startswith(
                exact._canonical(prefix) + "\\"
            ):
                combined = prefix + "\\" + combined

            if exact._canonical(combined) != target:
                data_cursor += data_size
                continue

            stream.seek(data_cursor)
            stored = stream.read(data_size)
            if len(stored) != data_size:
                raise ValueError(f"truncated PBO entry {name}")
            if packing == 0:
                return stored
            if packing == exact._PBO_COMPRESSED and original_size > 0:
                return exact._decompress_pbo_payload(stored, original_size)
            return None

    return None


def _rgb565_array(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.uint16)
    return (
        ((values[:, 0] >> 3) << 11)
        | ((values[:, 1] >> 2) << 5)
        | (values[:, 2] >> 3)
    ).astype(np.uint16, copy=False)


def _decode_rgb565_array(values: np.ndarray) -> np.ndarray:
    packed = np.asarray(values, dtype=np.uint16).astype(np.uint32)
    result = np.empty((packed.size, 3), dtype=np.int32)
    result[:, 0] = ((packed >> 11) & 0x1F).astype(np.int32) * 255 // 31
    result[:, 1] = ((packed >> 5) & 0x3F).astype(np.int32) * 255 // 63
    result[:, 2] = (packed & 0x1F).astype(np.int32) * 255 // 31
    return result


def _compress_dxt1_blocks_numpy(blocks: np.ndarray) -> np.ndarray:
    """Compress N RGB 4x4 blocks with the same endpoint fit as paa.py."""
    values = np.asarray(blocks, dtype=np.uint8)
    if values.ndim != 3 or values.shape[1:] != (16, 3):
        raise ValueError("DXT1 block batch must have shape (N, 16, 3)")
    count = values.shape[0]
    if count == 0:
        return np.empty((0, 8), dtype=np.uint8)

    work = values.astype(np.int32)
    minimum = work.min(axis=1)
    maximum = work.max(axis=1)
    span = maximum - minimum
    solid = np.all(span == 0, axis=1)

    projection = np.sum(work * span[:, None, :], axis=2, dtype=np.int64)
    row = np.arange(count)
    low = work[row, np.argmin(projection, axis=1)]
    high = work[row, np.argmax(projection, axis=1)]

    color0 = _rgb565_array(high).astype(np.int64)
    color1 = _rgb565_array(low).astype(np.int64)

    equal = color0 == color1
    bump_up = equal & (color0 < 0xFFFF)
    bump_down = equal & (color1 > 0)
    color0[bump_up] += 1
    color1[bump_down] -= 1

    swap = color0 <= color1
    old0 = color0.copy()
    color0[swap] = color1[swap]
    color1[swap] = old0[swap]

    if np.any(solid):
        solid_color = _rgb565_array(minimum[solid]).astype(np.int64)
        solid0 = np.where(solid_color < 0xFFFF, solid_color + 1, solid_color)
        solid1 = np.where(solid_color > 0, solid_color - 1, 0)
        bad = solid0 <= solid1
        solid0[bad] = 1
        solid1[bad] = 0
        color0[solid] = solid0
        color1[solid] = solid1

    endpoint0 = _decode_rgb565_array(color0.astype(np.uint16))
    endpoint1 = _decode_rgb565_array(color1.astype(np.uint16))
    palette = np.empty((count, 4, 3), dtype=np.int32)
    palette[:, 0, :] = endpoint0
    palette[:, 1, :] = endpoint1
    palette[:, 2, :] = (2 * endpoint0 + endpoint1) // 3
    palette[:, 3, :] = (endpoint0 + 2 * endpoint1) // 3

    difference = work[:, :, None, :] - palette[:, None, :, :]
    distance = np.sum(difference * difference, axis=3, dtype=np.int64)
    best = np.argmin(distance, axis=2).astype(np.uint32)
    shifts = (np.arange(16, dtype=np.uint32) * 2)[None, :]
    indices = np.bitwise_or.reduce(best << shifts, axis=1)
    indices[solid] = 0

    records = np.empty(count, dtype=_DXT1_RECORD)
    records["color0"] = color0.astype(np.uint16)
    records["color1"] = color1.astype(np.uint16)
    records["indices"] = indices
    return records.view(np.uint8).reshape(count, 8).copy()


def _image_blocks(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, channels = pixels.shape
    if channels != 3 or width % 4 or height % 4:
        raise ValueError("DXT1 image dimensions must be multiples of four")
    blocks_y, blocks_x = height // 4, width // 4
    return (
        pixels.reshape(blocks_y, 4, blocks_x, 4, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(blocks_y * blocks_x, 16, 3)
    )


def _changed_block_mask(changed: np.ndarray, width: int, height: int) -> np.ndarray:
    top_height, top_width = changed.shape
    if top_width % width or top_height % height:
        return np.ones((height // 4) * (width // 4), dtype=bool)
    scale_x = top_width // width
    scale_y = top_height // height
    region_w = 4 * scale_x
    region_h = 4 * scale_y
    blocks_x, blocks_y = width // 4, height // 4
    try:
        return (
            changed.reshape(blocks_y, region_h, blocks_x, region_w)
            .any(axis=(1, 3))
            .reshape(-1)
        )
    except ValueError:
        return np.ones(blocks_x * blocks_y, dtype=bool)


def _write_with_preserved_background_fast(exact, path: Path, image: Image.Image, source) -> None:
    """Vectorized version of exact-background runway PAA generation."""
    image = image.convert("RGB")
    if image.size != source.top_image.size:
        exact._ORIGINAL_WRITE_RUNWAY_PAA(path, image)
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
        blocks = _image_blocks(current)
        source_mip = source_by_size.get((width, height))
        if source_mip is not None:
            payload = np.frombuffer(source_mip.payload, dtype=np.uint8).reshape(-1, 8).copy()
            changed_blocks = _changed_block_mask(changed, width, height)
        else:
            payload = np.empty((blocks.shape[0], 8), dtype=np.uint8)
            changed_blocks = np.ones(blocks.shape[0], dtype=bool)

        selected = np.flatnonzero(changed_blocks)
        if selected.size:
            payload[selected] = _compress_dxt1_blocks_numpy(blocks[selected])
        levels.append((width, height, payload.tobytes(order="C")))

        if width == 4 and height == 4:
            break
        current = current.resize(
            (max(4, width // 2), max(4, height // 2)),
            Image.Resampling.BOX,
        )
        if len(levels) >= 16:
            break

    if not levels or levels[-1][0] != 4 or levels[-1][1] != 4:
        exact._ORIGINAL_WRITE_RUNWAY_PAA(path, image)
        return

    fixed_prefix_size = 2 + (12 + 4) + (12 + 64) + 2
    offsets: list[int] = []
    cursor = fixed_prefix_size
    for width, height, payload in levels:
        offsets.append(cursor)
        cursor += 4 + 3 + len(payload)
    offsets.extend([0] * (16 - len(offsets)))

    mean = exact.ImageStat.Stat(image).mean
    red, green, blue = int(mean[0]), int(mean[1]), int(mean[2])
    output = bytearray(struct.pack("<H", exact._DXT1_MAGIC))
    output += exact._tag(exact._TAG_AVERAGE, bytes((blue, green, red, 255)))
    output += exact._tag(exact._TAG_OFFSETS, struct.pack("<16I", *offsets))
    output += struct.pack("<H", 0)
    for width, height, payload in levels:
        output += struct.pack("<HH", width, height)
        output += exact._u24(len(payload))
        output += payload
    output += struct.pack("<HH", 0, 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(bytes(output))


def _runway_report_detail(source_dir: Path) -> str:
    path = Path(source_dir) / "runway-textures.json"
    if not path.is_file():
        return ""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ""

    cells = int(report.get("runway_cells", 0) or 0)
    hits = int(report.get("cache_hits", 0) or 0)
    misses = int(report.get("cache_misses", 0) or 0)
    generated = int(report.get("generated_runway_textures", 0) or 0)
    return (
        f"{cells} cells; cache {hits} hits/{misses} rendered; "
        f"{generated} unique PAAs"
    )


def install_runway_performance_policy() -> None:
    """Keep runway preparation fast and visible before RVW4 serialization."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import runway_exact_background_policy as exact
    from . import runway_surface_policy as runway
    from .progress import report_progress

    # Read only the selected stock terrain PAA payload from its PBO.
    exact._extract_pbo_asset = lambda path, target: _extract_pbo_asset_streaming(
        exact, path, target
    )
    try:
        exact._read_external_asset_cached.cache_clear()
    except AttributeError:
        pass

    # The old exact-background writer recompressed changed 4x4 DXT1 blocks one at
    # a time in Python. Batch each mip through NumPy while preserving untouched
    # source blocks byte-for-byte.
    exact._write_with_preserved_background = (
        lambda path, image, source: _write_with_preserved_background_fast(
            exact, path, image, source
        )
    )

    original_apply = runway.apply_generated_runway_texture_table

    def apply_runway_textures_with_progress(
        source_dir, dataset, projection, spec, texture_indices, texture_paths
    ):
        touched = runway.runway_texture_cell_indices(dataset, projection, spec)
        if not touched:
            return original_apply(
                source_dir, dataset, projection, spec, texture_indices, texture_paths
            )

        report_progress(
            86,
            f"Preparing runway terrain textures ({len(touched):,} touched cells)",
        )
        started = perf_counter()
        result = original_apply(
            source_dir, dataset, projection, spec, texture_indices, texture_paths
        )
        elapsed = perf_counter() - started
        detail = _runway_report_detail(Path(source_dir))
        report_progress(
            86,
            "Runway terrain textures ready"
            + (f" ({detail}; {elapsed:.2f}s)" if detail else f" ({elapsed:.2f}s)"),
        )
        return result

    runway.apply_generated_runway_texture_table = apply_runway_textures_with_progress
    _INSTALLED = True
