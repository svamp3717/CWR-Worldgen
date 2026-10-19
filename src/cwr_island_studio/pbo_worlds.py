# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, bounded-memory RVW4 loading from selected PBO/IFA archives.

The archive and its entries are never modified. Members are extracted to a
temporary directory and mapped using the ordinary IslandDocument reader;
no complete island needs to fit in Python memory.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from concurrent.futures import CancelledError
import threading
import struct
import tempfile
import time
from typing import BinaryIO

import zstandard as zstd

from .pbo_textures import (
    TextureMember, _LZSS, _ZSTD_MAGIC, _scan,
)

_COPY_CHUNK = 1024 * 1024
_MAX_WORLD_BYTES = 32 * 1024**3


def _copy_exact(
    source: BinaryIO, target: BinaryIO, count: int,
    cancel_event: threading.Event | None = None,
) -> None:
    remaining = count
    while remaining:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        chunk = source.read(min(_COPY_CHUNK, remaining))
        if not chunk:
            raise ValueError("truncated WRP member in PBO")
        target.write(chunk)
        remaining -= len(chunk)


def _decode_lzss_world(
    source: BinaryIO, target: BinaryIO, expected: int,
    cancel_event: threading.Event | None = None,
) -> None:
    """Decode legacy PBO LZSS into fixed-size output buffers, not RAM."""
    if not 0 < expected <= _MAX_WORLD_BYTES:
        raise ValueError("WRP uncompressed size is invalid or too large")
    window = bytearray(b" " * 4096)
    output = bytearray()
    at = written = checksum = 0

    def store(value: int) -> None:
        nonlocal at, written, checksum
        window[at] = value
        at = (at + 1) & 4095
        output.append(value)
        written += 1
        if written % (64 * 1024) == 0 and cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        checksum = (checksum + value) & 0xFFFFFFFF
        if len(output) >= _COPY_CHUNK:
            target.write(output)
            output.clear()

    while written < expected:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        flags_raw = source.read(1)
        if not flags_raw:
            raise ValueError("truncated compressed WRP flags")
        flags = flags_raw[0]
        for bit in range(8):
            if written >= expected:
                break
            if flags & (1 << bit):
                value = source.read(1)
                if not value:
                    raise ValueError("truncated compressed WRP literal")
                store(value[0])
            else:
                pair = source.read(2)
                if len(pair) != 2:
                    raise ValueError("truncated compressed WRP backreference")
                offset = pair[0] | ((pair[1] & 0xF0) << 4)
                length = min(3 + (pair[1] & 15), expected - written)
                # Use the same moving dictionary semantics as the PBO reader.
                pointer = at - offset
                for i in range(length):
                    store(window[(pointer + i) & 4095])

    if output:
        target.write(output)
    digest = source.read(4)
    if len(digest) != 4 or struct.unpack("<I", digest)[0] != checksum:
        raise ValueError("compressed WRP checksum mismatch")
    if source.read(1):
        raise ValueError("unexpected compressed WRP trailer")


class PboWorldArchive:
    """A selectable list of WRP entries in one archive, with owned temp files."""

    def __init__(
        self, source: Path | str, *,
        progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.path = Path(source).resolve()
        self._cancel_event = cancel_event
        report = progress or (lambda _message: None)
        self._temporary = tempfile.TemporaryDirectory(prefix="cwr-island-wrp-")
        self._closed = False
        try:
            self._check_cancel()
            archive = self.path
            with archive.open("rb") as stream:
                wrapped = stream.read(4) == _ZSTD_MAGIC
            if wrapped:
                report(f"Unpacking compressed archive {self.path.name}…")
                archive = Path(self._temporary.name) / "container.pbo"
                with self.path.open("rb") as source_file, archive.open("wb") as target:
                    with zstd.ZstdDecompressor().stream_reader(
                        source_file, read_across_frames=True,
                    ) as decoder:
                        reported = 0
                        while True:
                            self._check_cancel()
                            chunk = decoder.read(_COPY_CHUNK)
                            if not chunk:
                                break
                            target.write(chunk)
                            if target.tell() - reported >= 64 * _COPY_CHUNK:
                                reported = target.tell()
                                report(
                                    f"Unpacking {self.path.name}: "
                                    f"{reported // _COPY_CHUNK:,} MiB…"
                                )
            self._check_cancel()
            report(f"Reading PBO directory in {self.path.name}…")
            stem = self.path.name
            for suffix in (".zst", ".pbo", ".ifa"):
                if stem.casefold().endswith(suffix):
                    stem = stem[:-len(suffix)]
            self.members: dict[str, TextureMember] = _scan(
                archive, stem, extensions=(".wrp",),
                cancel_event=self._cancel_event,
                on_progress=lambda count: report(
                    f"Indexing WRP entries in {self.path.name}: "
                    f"{count:,} directory entries scanned…"
                ),
            )
            self._check_cancel()
            if not self.members:
                raise ValueError("No .wrp files were found in this archive")
        except BaseException:
            self.close()
            raise

    def _check_cancel(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise CancelledError()

    def extract(self, member_name: str) -> Path:
        self._check_cancel()
        if self._closed:
            raise RuntimeError("PBO archive has already been closed")
        member = self.members[member_name]
        expected = (
            member.original_size if member.packing == _LZSS
            else member.stored_size
        )
        if not 0 < expected <= _MAX_WORLD_BYTES:
            raise ValueError("WRP member exceeds the supported size limit")
        destination = Path(self._temporary.name) / "selected-world.wrp"
        # Each archive instance is used for one selected world.
        with member.path.open("rb") as source, destination.open("wb") as target:
            source.seek(member.offset)
            if member.packing == _LZSS:
                # Prevent decompression from reading the next PBO member.
                with _LimitedReader(source, member.stored_size) as limited:
                    _decode_lzss_world(
                        limited, target, expected, self._cancel_event,
                    )
            else:
                _copy_exact(source, target, member.stored_size, self._cancel_event)
        self._check_cancel()
        with destination.open("rb") as stream:
            if stream.read(4) != b"4WVR":
                destination.unlink(missing_ok=True)
                raise ValueError("Selected WRP is not an unbinarized 4WVR world")
        return destination

    def open_world(
        self, member_name: str, *, cell_size: float = 50.0,
        height_scale: float = 1.0,
    ):
        """Return an IslandDocument which owns this temporary archive source."""
        from .core import IslandDocument

        try:
            filename = self.extract(member_name)
            document = IslandDocument(
                filename, cell_size=cell_size, height_scale=height_scale,
                read_only=True,
            )
            document.source_label = f"{self.path.name} / {member_name}"
            document._archive_owner = self
            return document
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        temporary = self._temporary
        try:
            temporary.cleanup()
        except PermissionError:
            # Windows can temporarily lock a newly extracted WRP (for example
            # while a scanner or an mmap reader still has it open). Do not
            # replace the original WRP parsing error with WinError 32, or
            # block the Tk event loop when the user closes the application.
            def retry_cleanup() -> None:
                for delay in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
                    time.sleep(delay)
                    try:
                        temporary.cleanup()
                        return
                    except PermissionError:
                        continue

            threading.Thread(
                target=retry_cleanup,
                daemon=True,
                name="island-studio-temp-cleanup",
            ).start()


class _LimitedReader:
    """Restrict compressed decoding to the selected member's stored bytes."""

    def __init__(self, stream: BinaryIO, count: int) -> None:
        self.stream = stream
        self.remaining = count

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.remaining
        wanted = min(self.remaining, size)
        chunk = self.stream.read(wanted)
        self.remaining -= len(chunk)
        return chunk
