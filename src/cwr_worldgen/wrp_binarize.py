# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental CWA OPRW v3 serializer for generated 4WVR worlds.

Based on Poseidon Landscape::SerializeBin / SSCompress and WrpReader.
The original 4WVR is never modified. Terrain geography/random/sound layers
are initially zero-filled (unbinarized RVW4 does not contain those layers);
engine-runtime testing remains required before production distribution.
"""
from __future__ import annotations

import mmap
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .model import CWA_RVW4_HEIGHT_SCALE

_HEADER = struct.Struct("<4sii")
_OBJECT = struct.Struct("<12fi76s")
_WORD = struct.Struct("<I")
_OBJECT_BIN = struct.Struct("<II12f")
_MAX_CELLS = 2048 * 2048
_TEXTURE_TABLE_BYTES = 512 * 32
_MIN_COMPRESS_SIZE = 1024



def _packed(raw: bytes) -> bytes:
    """Emit Poseidon SSCompress-compatible distance tokens and checksum.

    Runs and repeated 4-byte values are encoded as references. Other data
    remains literal; this bounded-time encoder does not build search trees.
    A deliberately conservative stream is valid even if not optimally small.
    """
    n = len(raw)
    if n < _MIN_COMPRESS_SIZE:
        return raw
    result = bytearray()
    at = 0
    while at < n:
        flags = 0
        payload = bytearray()
        for bit in range(8):
            if at >= n:
                break
            distance = 0
            length = 0
            if at >= 1 and raw[at] == raw[at - 1]:
                end = min(n, at + 18)
                while at + length < end and raw[at + length] == raw[at]:
                    length += 1
                if length >= 3:
                    distance = 1
            if length < 3 and at >= 4 and raw[at:at + 4] == raw[at - 4:at]:
                end = min(n, at + 18)
                while at + length < end and raw[at + length] == raw[at - 4 + length]:
                    length += 1
                if length >= 4:
                    distance = 4
            if length >= 3:
                payload.append(distance & 255)
                payload.append(((distance >> 4) & 0xF0) | (length - 3))
                at += length
            else:
                flags |= 1 << bit
                payload.append(raw[at])
                at += 1
        result.append(flags)
        result.extend(payload)
    result.extend(_WORD.pack(sum(raw) & 0xFFFFFFFF))
    return bytes(result)


def binarize_rvw4(
    source: Path, target: Path, *, height_scale: float = CWA_RVW4_HEIGHT_SCALE,
) -> int:
    """Write OPRW v3 from 4WVR, retaining the original.

    CWR-CE decodes 4WVR int16 terrain at a fixed 0.045 metres per sample
    (LANDDATA_SCALE). Worldgen uses the same scale for quantization and
    object grounding. OPRW stores physical metre heights as float32.
    Explicit overrides remain available for compatibility tests.
    """
    if not np.isfinite(height_scale) or height_scale <= 0:
        raise ValueError("height_scale must be positive and finite")
    source = Path(source)
    target = Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("binarized output must not overwrite the editable 4WVR source")
    temporary_name: str | None = None
    with source.open("rb") as src, mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if len(data) < _HEADER.size:
            raise ValueError("truncated 4WVR header")
        magic, width, height = _HEADER.unpack_from(data)
        cells = width * height
        if magic != b"4WVR" or min(width, height) <= 0 or cells > _MAX_CELLS:
            raise ValueError("expected a supported unbinarized 4WVR world")
        heights_at = _HEADER.size
        textures_at = heights_at + cells * 2
        names_at = textures_at + cells * 2
        objects_at = names_at + _TEXTURE_TABLE_BYTES
        if len(data) < objects_at + _OBJECT.size or (len(data) - objects_at) % _OBJECT.size:
            raise ValueError("truncated or misaligned 4WVR object table")

        raw_indices = np.frombuffer(data, dtype="<i2", count=cells, offset=textures_at)
        if raw_indices.size and (np.min(raw_indices) < 0 or np.max(raw_indices) > 511):
            raise ValueError("invalid 4WVR texture index")
        slots = [
            data[names_at + i * 32:names_at + (i + 1) * 32].split(b"\0", 1)[0]
            for i in range(512)
        ]
        highest = max([i for i, name in enumerate(slots) if name] + [0])
        if raw_indices.size:
            highest = max(highest, int(np.max(raw_indices)))
        texture_names = slots[:highest + 1]
        # Do not hold exported NumPy views when closing the source mmap.
        del raw_indices

        # First pass: shared model table instead of 76-byte names per object.
        model_ids: dict[bytes, int] = {}
        object_count = 0
        for pos in range(objects_at, len(data), _OBJECT.size):
            model = data[pos + 52:pos + 128].split(b"\0", 1)[0]
            if not model:
                if pos + _OBJECT.size != len(data):
                    raise ValueError("4WVR has data after object terminator")
                break
            if model not in model_ids:
                if len(model_ids) >= 65536:
                    raise ValueError("too many unique WRP object models")
                model_ids[model] = len(model_ids)
            object_count += 1
        else:
            raise ValueError("4WVR is missing its object terminator")

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.",
                suffix=".tmp", delete=False,
            ) as out:
                temporary_name = out.name
                out.write(b"OPRW")
                out.write(_WORD.pack(3))
                out.write(struct.pack("<IIII", width, height, width, height))
                # 4WVR does not include these serialized geography layers.
                for size in (cells * 4, cells):
                    out.write(_packed(bytes(size)))
                out.write(_WORD.pack(0))  # mountain peaks
                out.write(_packed(data[textures_at:names_at]))
                out.write(_packed(bytes(cells * 4)))  # random/environment flags
                elevations = (
                    np.frombuffer(data, dtype="<i2", count=cells, offset=heights_at)
                    .astype("<f4") * np.float32(height_scale)
                )
                out.write(_packed(elevations.tobytes()))
                del elevations
                out.write(_WORD.pack(len(texture_names)))
                for name in texture_names:
                    out.write(name + b"\0\0")  # null-terminated name + UV bool
                out.write(_WORD.pack(len(model_ids)))
                for name in model_ids:
                    out.write(name + b"\0")
                for pos in range(objects_at, objects_at + object_count * _OBJECT.size, _OBJECT.size):
                    values = _OBJECT.unpack_from(data, pos)
                    if values[12] < 0 or values[12] > (16 << 20):
                        raise ValueError("WRP object ID exceeds the CWR-CE engine limit")
                    name = values[13].split(b"\0", 1)[0]
                    out.write(_OBJECT_BIN.pack(values[12], model_ids[name], *values[:12]))
                out.write(_WORD.pack(0xFFFFFFFF))
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
    return object_count


def repack_game_pbo_with_oprw(
    source_dir: Path,
    source_wrp: Path,
    binarized_wrp: Path,
    pbo_path: Path,
    *, backend: str = "auto",
    poseidon_tools_path: Path | None = None,
) -> None:
    """Repack runtime PBO atomically without changing the editable 4WVR.

    Stage both the original WRP and replacement PBO outside the packed source
    directory, restore the source file even when packing fails, and replace
    the final game PBO only after the pack succeeds.
    """
    import shutil
    from .pbo import pack_directory

    source_dir = Path(source_dir)
    source_wrp = Path(source_wrp)
    binarized_wrp = Path(binarized_wrp)
    pbo_path = Path(pbo_path)
    if source_wrp.parent.resolve() != source_dir.resolve():
        raise ValueError("WRP must live in the packed source directory")
    with binarized_wrp.open("rb") as stream:
        if stream.read(8) != b"OPRW\x03\x00\x00\x00":
            raise ValueError("binarized world is not OFP/CWA OPRW v3")
    # The work directory must not appear in source_dir, or it would be
    # accidentally included in the output archive.
    with tempfile.TemporaryDirectory(
        dir=source_dir.parent, prefix=".wrp-binarize-",
    ) as work_dir:
        work = Path(work_dir)
        original_backup = work / "original-4wvr.wrp"
        pending_pbo = work / pbo_path.name
        os.replace(source_wrp, original_backup)
        try:
            shutil.copyfile(binarized_wrp, source_wrp)
            pack_directory(
                source_dir, pending_pbo, backend=backend,
                poseidon_tools_path=poseidon_tools_path,
            )
        finally:
            os.replace(original_backup, source_wrp)
        os.replace(pending_pbo, pbo_path)
