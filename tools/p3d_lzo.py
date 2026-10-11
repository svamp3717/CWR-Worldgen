"""Read Arma-era LZO1X blocks used by ODOL and PAA textures.

BIS stores unframed LZO streams without a length prefix; the caller supplies
the expected decompressed byte count. The decoder consumes the end marker and
rejects inconsistent sizes, truncation and invalid backreferences.
"""
from __future__ import annotations

import io
from typing import Protocol


class LzoReadError(ValueError):
    """Invalid or truncated LZO-compressed stream."""


class LzoReader(Protocol):
    limit: int

    def tell(self) -> int: ...
    def seek(self, offset: int) -> None: ...
    def take(self, size: int, what: str = "data") -> bytes: ...
    def u8(self) -> int: ...


class _PayloadReader:
    def __init__(self, payload: bytes):
        self.stream = io.BytesIO(payload)
        self.limit = len(payload)

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int) -> None:
        if not 0 <= offset <= self.limit:
            raise LzoReadError("LZO input offset outside payload")
        self.stream.seek(offset)

    def take(self, size: int, what: str = "LZO input") -> bytes:
        if size < 0 or size > self.limit - self.tell():
            raise LzoReadError(f"truncated {what}")
        return self.stream.read(size)

    def u8(self) -> int:
        return self.take(1)[0]


def decode_lzo_bytes(payload: bytes, expected: int) -> bytes:
    """Decode a complete LZO payload and reject trailing bytes."""
    if not 0 <= expected <= 256 * 1024 * 1024:
        raise LzoReadError(f"invalid LZO uncompressed size {expected}")
    reader = _PayloadReader(payload)
    result = decode_lzo_stream(reader, expected)
    if reader.tell() != reader.limit:
        raise LzoReadError("trailing data after LZO end marker")
    return result


def decode_lzo_stream(reader: LzoReader, expected: int) -> bytes:
    """Expand an unframed BI LZO1X array, consuming its end marker.

    The compressed size is not stored in the ODOL array header, so decoding
    must consume the stream through the LZO end-of-stream instruction.
    """
    out = bytearray()

    def byte() -> int:
        return reader.u8()

    def literals(n: int) -> None:
        if n < 0 or n > expected - len(out):
            raise LzoReadError("LZO literal run exceeds expected output")
        out.extend(reader.take(n, "LZO literals"))

    def copy(position: int, n: int) -> None:
        if position < 0 or position >= len(out) or n < 0 or n > expected - len(out):
            raise LzoReadError("invalid LZO match offset or output size")
        for offset in range(n):
            out.append(out[position + offset])

    def extended(base: int) -> int:
        amount = base
        while True:
            value = byte()
            if value:
                return amount + value
            amount += 255
            if amount > expected:
                raise LzoReadError("excessively long LZO literal/match run")

    def after_match() -> int:
        # The two last bytes of a match encode its following literal count.
        pos = reader.tell()
        reader.seek(pos - 2)
        value = reader.u8() & 3
        reader.seek(pos)
        return value

    state = "start"
    token = 0
    if reader.tell() < reader.limit:
        first = reader.take(1)[0]
        reader.seek(reader.tell() - 1)
        if first > 17:
            token = byte() - 17
            if token < 4:
                state = "match_next"
            else:
                literals(token)
                state = "first_literal"
        else:
            state = "literal"
    while True:
        if state == "literal":
            token = byte()
            if token >= 16:
                state = "match"
                continue
            if token == 0:
                token = extended(15)
            literals(token + 3)
            state = "first_literal"
            continue
        if state == "first_literal":
            token = byte()
            if token >= 16:
                state = "match"
                continue
            offset = 1 + 0x800 + (token >> 2) + (byte() << 2)
            copy(len(out) - offset, 3)
            state = "after_match"
            continue
        if state == "match_next":
            literals(token)
            token = byte()
            state = "match"
            continue
        if state == "match":
            if token >= 64:
                offset = 1 + ((token >> 2) & 7) + (byte() << 3)
                copy(len(out) - offset, (token >> 5) + 1)
            elif token >= 32:
                length = token & 31
                if not length:
                    length = extended(31)
                first, second = byte(), byte()
                offset = 1 + (first >> 2) + (second << 6)
                copy(len(out) - offset, length + 2)
            elif token >= 16:
                high = (token & 8) << 11
                length = token & 7
                if not length:
                    length = extended(7)
                first, second = byte(), byte()
                distance = high + (first >> 2) + (second << 6)
                if distance == 0:
                    if len(out) != expected:
                        raise LzoReadError("LZO end marker before expected output size")
                    return bytes(out)
                copy(len(out) - distance - 0x4000, length + 2)
            else:
                offset = 1 + (token >> 2) + (byte() << 2)
                copy(len(out) - offset, 2)
            state = "after_match"
            continue
        if state == "after_match":
            token = after_match()
            state = "match_next" if token else "literal"
            continue
        raise LzoReadError(f"unknown LZO decoder state {state}")

