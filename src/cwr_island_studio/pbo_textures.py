# SPDX-License-Identifier: GPL-3.0-or-later
"""Indexed, read-only PBO texture lookup for the standalone map editor.

Archive headers are scanned once; individual texture members are fetched only
when visible in the viewport. A whole-file Zstandard wrapper is expanded once
to temporary disk storage, avoiding any eager decoding of texture members.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import shutil
import struct
import tempfile
from typing import Sequence

import zstandard as zstd

_ENTRY = struct.Struct("<IIIII")
_VERSION = 0x56657273
_LZSS = 0x43707273
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_MAX_MEMBER = 32 * 1024 * 1024
_MAX_HEADER_NAME = 4096


def canonical_path(value: str) -> str:
    parts = value.replace("/", "\\").strip("\\").split("\\")
    if not parts or any(not p or p in (".", "..") or ":" in p for p in parts):
        raise ValueError("unsafe PBO entry path")
    return "\\".join(parts).casefold()


def _cstring(source) -> str:
    result = bytearray()
    for _ in range(_MAX_HEADER_NAME):
        ch = source.read(1)
        if not ch:
            raise ValueError("truncated PBO header string")
        if ch == b"\0":
            return result.decode("latin-1")
        result.extend(ch)
    raise ValueError("PBO header string is too long")


@dataclass(frozen=True, slots=True)
class TextureMember:
    path: Path
    offset: int
    stored_size: int
    original_size: int
    packing: int


def _scan(path: Path, archive_stem: str) -> dict[str, TextureMember]:
    """Read the PBO header and record offsets without extracting member data."""
    records: list[tuple[str, int, int, int]] = []
    prefix = ""
    with path.open("rb") as source:
        while True:
            name = _cstring(source)
            fields = source.read(_ENTRY.size)
            if len(fields) != _ENTRY.size:
                raise ValueError("truncated PBO header")
            packing, original_size, reserved, timestamp, size = _ENTRY.unpack(fields)
            if not name:
                if packing == _VERSION:
                    while True:
                        key = _cstring(source)
                        if not key:
                            break
                        value = _cstring(source)
                        if key.casefold() == "prefix":
                            prefix = value
                    continue
                if (
                    packing not in (0, _LZSS) or original_size != 0
                    or reserved != 0 or timestamp not in (0, 0xFFFFFFFF)
                    or size != 0
                ):
                    raise ValueError("unsupported PBO header extension")
                break
            records.append((name, packing, original_size, size))
        start = source.tell()
        source.seek(0, 2)
        end = source.tell()

    if not prefix:
        prefix = archive_stem
    prefix = prefix.strip("\\/")
    result: dict[str, TextureMember] = {}
    offset = start
    for name, packing, original_size, size in records:
        if size < 0 or offset + size > end:
            raise ValueError("truncated PBO member data")
        try:
            clean = canonical_path(name)
            if prefix:
                clean_prefix = canonical_path(prefix)
                if not clean.startswith(clean_prefix + "\\"):
                    clean = clean_prefix + "\\" + clean
        except ValueError:
            offset += size
            continue
        if clean.endswith((".paa", ".pac", ".png", ".jpg", ".jpeg", ".webp", ".p3d")):
            if packing in (0, _LZSS):
                result[clean] = TextureMember(path, offset, size, original_size, packing)
        offset += size
    return result


def decode_lzss_stream(stream: io.BytesIO, expected: int) -> bytes:
    """Decode exactly one compressed array, leaving following bytes untouched."""
    if not 0 < expected <= _MAX_MEMBER:
        raise ValueError("compressed entry exceeds preview size limit")
    window = bytearray(b" ") * 4096
    result = bytearray(expected)
    at = output_at = 0
    while output_at < expected:
        flags_raw = stream.read(1)
        if not flags_raw:
            raise ValueError("truncated compressed texture")
        flags = flags_raw[0]
        for bit in range(8):
            if output_at >= expected:
                break
            if flags & (1 << bit):
                value = stream.read(1)
                if not value:
                    raise ValueError("truncated compressed literal")
                result[output_at] = window[at] = value[0]
                output_at += 1
                at = (at + 1) & 4095
            else:
                pair = stream.read(2)
                if len(pair) != 2:
                    raise ValueError("truncated compressed backreference")
                position = pair[0] | ((pair[1] & 0xF0) << 4)
                length = min(3 + (pair[1] & 15), expected - output_at)
                source_at = at - position
                for i in range(length):
                    value = window[(source_at + i) & 4095]
                    result[output_at + i] = value
                    window[(at + i) & 4095] = value
                output_at += length
                at = (at + length) & 4095
    checksum = stream.read(4)
    if len(checksum) != 4 or struct.unpack("<I", checksum)[0] != sum(result) & 0xFFFFFFFF:
        raise ValueError("compressed texture checksum mismatch")
    return bytes(result)


def _lzss_decode(data: bytes, expected: int) -> bytes:
    stream = io.BytesIO(data)
    decoded = decode_lzss_stream(stream, expected)
    if stream.read(1):
        raise ValueError("unexpected compressed texture trailer")
    return decoded


def _read_member(member: TextureMember) -> bytes:
    if member.stored_size > _MAX_MEMBER:
        raise ValueError("texture is too large for preview")
    with member.path.open("rb") as source:
        source.seek(member.offset)
        data = source.read(member.stored_size)
    if len(data) != member.stored_size:
        raise ValueError("truncated PBO texture")
    if member.packing == _LZSS:
        return _lzss_decode(data, member.original_size)
    return data


class PboTextures:
    """Open selected PBOs for virtual-path lookups; no eager image decoding."""

    def __init__(self, paths: Sequence[Path | str]) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="cwr-island-pbos-")
        self.members: dict[str, TextureMember] = {}
        self.models: dict[str, TextureMember] = {}
        self.archives: tuple[Path, ...] = tuple(Path(p).resolve() for p in paths)
        try:
            for index, original in enumerate(self.archives):
                path = original
                with path.open("rb") as source:
                    wrapper = source.read(4) == _ZSTD_MAGIC
                if wrapper:
                    path = Path(self._temp.name) / f"archive-{index}.pbo"
                    with original.open("rb") as source, path.open("wb") as target:
                        with zstd.ZstdDecompressor().stream_reader(source) as decoder:
                            shutil.copyfileobj(decoder, target, length=1024 * 1024)
                name = original.name
                for suffix in (".zst", ".pbo", ".ifa"):
                    if name.lower().endswith(suffix):
                        name = name[:-len(suffix)]
                for virtual, member in _scan(path, name).items():
                    if virtual.endswith(".p3d"):
                        self.models[virtual] = member
                    else:
                        self.members[virtual] = member
        except BaseException:
            self.close()
            raise

    def read(self, virtual_path: str) -> bytes | None:
        try:
            member = self.members.get(canonical_path(virtual_path))
        except ValueError:
            return None
        return None if member is None else _read_member(member)

    def read_model(self, virtual_path: str) -> bytes | None:
        """Read a P3D member lazily, with the same compressed-size limits."""
        try:
            member = self.models.get(canonical_path(virtual_path))
        except ValueError:
            return None
        return None if member is None else _read_member(member)

    def close(self) -> None:
        self.members.clear()
        self.models.clear()
        self._temp.cleanup()
