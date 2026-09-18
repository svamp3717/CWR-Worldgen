"""Resolve OFP/CWA model textures and decode PAA first mip levels."""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import struct
import sys
from typing import Sequence

import numpy as np

import measure_p3d_models as measure

_PAA_FORMATS = {
    0x8080: "AI88", 0x4444: "ARGB4444", 0x1555: "ARGB1555", 0x8888: "ARGB8888",
    0xFF01: "DXT1", 0xFF02: "DXT2", 0xFF03: "DXT3", 0xFF04: "DXT4", 0xFF05: "DXT5",
}


@dataclass(frozen=True, slots=True)
class PBOAssetRef:
    pbo_path: Path
    offset: int
    packing: int
    original_size: int
    data_size: int


@dataclass(frozen=True, slots=True)
class LooseAssetRef:
    path: Path


AssetRef = PBOAssetRef | LooseAssetRef


def _canonical(value: str) -> str:
    return measure._canonical_model_path(value)


def _read_cstring_file(handle, label: str) -> str:
    out = bytearray()
    while True:
        byte = handle.read(1)
        if not byte:
            raise measure.ModelReadError(f"truncated {label}")
        if byte == b"\0":
            return out.decode("latin-1")
        out.extend(byte)
        if len(out) > 1024 * 1024:
            raise measure.ModelReadError(f"implausibly long {label}")


class TextureResolver:
    """Resolve canonical PAA/PAC paths from loose files or PBOs, lazily."""

    def __init__(self, inputs: Sequence[Path]) -> None:
        self.inputs = tuple(path.expanduser() for path in inputs)
        self.assets: dict[str, AssetRef] = {}
        self.basename_assets: dict[str, AssetRef | None] = {}
        self.indexed_pbos: set[Path] = set()
        self.indexed_all = False
        self.bytes_cache: dict[str, bytes | None] = {}
        self.image_cache: dict[str, np.ndarray | None] = {}

    def _remember(self, canonical: str, ref: AssetRef) -> None:
        canonical = _canonical(canonical)
        if not canonical.endswith((".paa", ".pac")):
            return
        self.assets.setdefault(canonical, ref)
        basename = canonical.rsplit("\\", 1)[-1]
        if basename not in self.basename_assets:
            self.basename_assets[basename] = ref
        elif self.basename_assets[basename] != ref:
            self.basename_assets[basename] = None

    def _index_pbo(self, pbo_path: Path) -> None:
        resolved = pbo_path.resolve()
        if resolved in self.indexed_pbos or not pbo_path.is_file():
            return
        self.indexed_pbos.add(resolved)
        try:
            with pbo_path.open("rb") as handle:
                metadata: list[tuple[str, int, int, int]] = []
                properties: dict[str, str] = {}
                while True:
                    name = _read_cstring_file(handle, "PBO entry name")
                    fields = handle.read(measure._PBO_ENTRY.size)
                    if len(fields) != measure._PBO_ENTRY.size:
                        raise measure.ModelReadError("truncated PBO entry header")
                    packing, original_size, _reserved, _timestamp, data_size = measure._PBO_ENTRY.unpack(fields)
                    if data_size > measure._MAX_PBO_ENTRY_SIZE or original_size > measure._MAX_PBO_ENTRY_SIZE:
                        raise measure.ModelReadError(f"implausible PBO entry size in {pbo_path}")
                    if not name:
                        if packing == measure._PBO_PROPERTIES:
                            while True:
                                key = _read_cstring_file(handle, "PBO property key")
                                if not key:
                                    break
                                properties[key.casefold()] = _read_cstring_file(handle, "PBO property value")
                            continue
                        if any((packing, original_size, data_size)):
                            raise measure.ModelReadError("unsupported PBO extension record")
                        break
                    metadata.append((name, packing, original_size, data_size))

                prefix = properties.get("prefix", "").replace("/", "\\").strip("\\") or pbo_path.stem
                canonical_prefix = _canonical(prefix)
                cursor = handle.tell()
                for name, packing, original_size, data_size in metadata:
                    combined = name.replace("/", "\\").lstrip("\\")
                    if canonical_prefix and not _canonical(combined).startswith(canonical_prefix + "\\"):
                        combined = prefix + "\\" + combined
                    canonical = _canonical(combined)
                    if canonical.endswith((".paa", ".pac")):
                        self._remember(canonical, PBOAssetRef(pbo_path, cursor, packing, original_size, data_size))
                    cursor += data_size
        except (OSError, ValueError, struct.error) as exc:
            print(f"[texture index warning] {pbo_path}: {exc}", file=sys.stderr, flush=True)

    def _index_all(self) -> None:
        if self.indexed_all:
            return
        self.indexed_all = True
        print("[textures] indexing PAA/PAC assets...", file=sys.stderr, flush=True)
        for raw_input in self.inputs:
            path = raw_input.expanduser()
            if not path.exists():
                continue
            if path.is_file():
                if path.suffix.casefold() == ".pbo":
                    self._index_pbo(path)
                elif path.suffix.casefold() in {".paa", ".pac"}:
                    self._remember(path.name, LooseAssetRef(path))
                continue
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                suffix = child.suffix.casefold()
                if suffix == ".pbo":
                    self._index_pbo(child)
                elif suffix in {".paa", ".pac"}:
                    try:
                        relative = child.relative_to(path).as_posix()
                    except ValueError:
                        relative = child.name
                    self._remember(relative, LooseAssetRef(child))
        print(f"[textures] indexed {len(self.assets):,} texture asset(s)", file=sys.stderr, flush=True)

    def _find_ref(self, texture_path: str, source: str) -> AssetRef | None:
        canonical = _canonical(texture_path)
        if "!" in source:
            source_pbo = Path(source.split("!", 1)[0])
            if source_pbo.suffix.casefold() == ".pbo":
                self._index_pbo(source_pbo)
        ref = self.assets.get(canonical)
        if ref is not None:
            return ref
        self._index_all()
        ref = self.assets.get(canonical)
        if ref is not None:
            return ref
        return self.basename_assets.get(canonical.rsplit("\\", 1)[-1])

    def load_bytes(self, texture_path: str, source: str) -> bytes | None:
        canonical = _canonical(texture_path)
        if canonical in self.bytes_cache:
            return self.bytes_cache[canonical]
        ref = self._find_ref(canonical, source)
        if ref is None:
            self.bytes_cache[canonical] = None
            return None
        try:
            if isinstance(ref, LooseAssetRef):
                data = ref.path.read_bytes()
            else:
                with ref.pbo_path.open("rb") as handle:
                    handle.seek(ref.offset)
                    stored = handle.read(ref.data_size)
                if len(stored) != ref.data_size:
                    raise measure.ModelReadError("truncated texture PBO entry")
                if ref.packing == 0:
                    data = stored
                elif ref.packing == measure._PBO_COMPRESSED:
                    data = measure._decompress_lzss_pbo(stored, ref.original_size)
                else:
                    raise measure.ModelReadError(f"unsupported texture PBO packing method {ref.packing:#x}")
        except (OSError, ValueError, struct.error) as exc:
            print(f"[texture error] {canonical}: {exc}", file=sys.stderr, flush=True)
            data = None
        self.bytes_cache[canonical] = data
        return data

    def load_rgba(self, texture_path: str, source: str) -> np.ndarray | None:
        canonical = _canonical(texture_path)
        if canonical in self.image_cache:
            return self.image_cache[canonical]
        data = self.load_bytes(canonical, source)
        if data is None:
            print(f"[texture missing] {canonical}", file=sys.stderr, flush=True)
            self.image_cache[canonical] = None
            return None
        try:
            image = decode_paa(data, is_paa=canonical.endswith(".paa"))
        except (ValueError, struct.error, IndexError) as exc:
            print(f"[texture decode error] {canonical}: {exc}", file=sys.stderr, flush=True)
            image = None
        self.image_cache[canonical] = image
        return image


def _u16(stream: io.BytesIO) -> int:
    raw = stream.read(2)
    if len(raw) != 2:
        raise ValueError("truncated PAA u16")
    return struct.unpack("<H", raw)[0]


def _u32(stream: io.BytesIO) -> int:
    raw = stream.read(4)
    if len(raw) != 4:
        raise ValueError("truncated PAA u32")
    return struct.unpack("<I", raw)[0]


def _u24(stream: io.BytesIO) -> int:
    raw = stream.read(3)
    if len(raw) != 3:
        raise ValueError("truncated PAA u24")
    return raw[0] | (raw[1] << 8) | (raw[2] << 16)


def _decode_paa_lzw(stream: io.BytesIO, expected_size: int) -> bytes:
    """BIS 4K-window LZW/LZSS with the signed-byte checksum used by PAA."""
    if expected_size < 0 or expected_size > 512 * 1024 * 1024:
        raise ValueError("implausible PAA decompressed size")
    n, f, threshold = 4096, 18, 2
    text = bytearray(b" " * (n + f - 1))
    r, flags, checksum = n - f, 0, 0
    out = bytearray()
    while len(out) < expected_size:
        flags >>= 1
        if (flags & 256) == 0:
            raw = stream.read(1)
            if not raw:
                raise ValueError("truncated PAA LZW flags")
            flags = raw[0] | 0xFF00
        if flags & 1:
            raw = stream.read(1)
            if not raw:
                raise ValueError("truncated PAA LZW literal")
            c = raw[0]
            out.append(c)
            checksum += c if c < 128 else c - 256
            text[r] = c
            r = (r + 1) & (n - 1)
        else:
            pair = stream.read(2)
            if len(pair) != 2:
                raise ValueError("truncated PAA LZW reference")
            offset = pair[0] | ((pair[1] & 0xF0) << 4)
            run = (pair[1] & 0x0F) + threshold + 1
            pos = r - offset
            for _ in range(run):
                if len(out) >= expected_size:
                    break
                c = text[pos & (n - 1)]
                pos += 1
                out.append(c)
                checksum += c if c < 128 else c - 256
                text[r] = c
                r = (r + 1) & (n - 1)
    raw_checksum = stream.read(4)
    if len(raw_checksum) != 4:
        raise ValueError("truncated PAA LZW checksum")
    stored = struct.unpack("<i", raw_checksum)[0]
    if stored != checksum:
        raise ValueError(f"PAA LZW checksum mismatch: stored {stored}, calculated {checksum}")
    return bytes(out)


def _expand565(value: int) -> np.ndarray:
    r, g, b = (value >> 11) & 0x1F, (value >> 5) & 0x3F, value & 0x1F
    return np.array([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2), 255], dtype=np.uint8)


def _dxt1_block(block: bytes, punch_through: bool) -> np.ndarray:
    c0, c1 = struct.unpack_from("<HH", block, 0)
    colors = np.empty((4, 4), dtype=np.uint8)
    colors[0], colors[1] = _expand565(c0), _expand565(c1)
    if c0 > c1 or not punch_through:
        colors[2, :3] = ((2 * colors[0, :3].astype(np.uint16) + colors[1, :3] + 1) // 3).astype(np.uint8)
        colors[3, :3] = ((colors[0, :3].astype(np.uint16) + 2 * colors[1, :3] + 1) // 3).astype(np.uint8)
        colors[2:, 3] = 255
    else:
        colors[2, :3] = ((colors[0, :3].astype(np.uint16) + colors[1, :3]) // 2).astype(np.uint8)
        colors[2, 3], colors[3] = 255, 0
    bits = struct.unpack_from("<I", block, 4)[0]
    pixels = np.empty((4, 4, 4), dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            pixels[y, x] = colors[(bits >> ((y * 4 + x) * 2)) & 3]
    return pixels


def _decode_dxt(data: bytes, width: int, height: int, kind: str) -> np.ndarray:
    block_bytes = 8 if kind == "DXT1" else 16
    bw, bh = (width + 3) // 4, (height + 3) // 4
    needed = bw * bh * block_bytes
    if len(data) < needed:
        raise ValueError(f"truncated {kind} PAA mip: need {needed}, got {len(data)}")
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    offset = 0
    for by in range(bh):
        for bx in range(bw):
            block = data[offset : offset + block_bytes]
            if kind == "DXT1":
                pixels = _dxt1_block(block, True)
            elif kind in {"DXT2", "DXT3"}:
                pixels = _dxt1_block(block[8:], False)
                for i in range(16):
                    nibble = (block[i // 2] >> (4 if i & 1 else 0)) & 0xF
                    pixels[i // 4, i % 4, 3] = (nibble << 4) | nibble
            else:
                a0, a1 = block[0], block[1]
                alphas = [a0, a1]
                if a0 > a1:
                    alphas.extend([((7 - i) * a0 + i * a1 + 3) // 7 for i in range(1, 7)])
                else:
                    alphas.extend([((5 - i) * a0 + i * a1 + 2) // 5 for i in range(1, 5)])
                    alphas.extend([0, 255])
                alpha_bits = int.from_bytes(block[2:8], "little")
                pixels = _dxt1_block(block[8:], False)
                for i in range(16):
                    pixels[i // 4, i % 4, 3] = alphas[(alpha_bits >> (i * 3)) & 7]
            offset += block_bytes
            y0, x0 = by * 4, bx * 4
            h, w = min(4, height - y0), min(4, width - x0)
            rgba[y0 : y0 + h, x0 : x0 + w] = pixels[:h, :w]
    return rgba


def _decode_4444(raw: bytes, width: int, height: int, ai88: bool = False) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2", count=width * height).copy()
    if ai88:
        a, i = words & 0xF000, words & 0x00F0
        words = a | (i << 4) | i | (i >> 4)
    channels = [((words >> shift) & 0xF).astype(np.uint8) * 17 for shift in (8, 4, 0, 12)]
    return np.stack(channels, axis=1).reshape((height, width, 4))


def _decode_1555(raw: bytes, width: int, height: int) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2", count=width * height)
    r, g, b = (words >> 10) & 0x1F, (words >> 5) & 0x1F, words & 0x1F
    a = np.where(words & 0x8000, 255, 0)
    channels = ((r << 3) | (r >> 2), (g << 3) | (g >> 2), (b << 3) | (b >> 2), a)
    return np.stack(channels, axis=1).astype(np.uint8).reshape((height, width, 4))


def _decode_p8_payload(stream: io.BytesIO, width: int, height: int, palette: np.ndarray, *, lzw: bool) -> np.ndarray:
    expected = width * height
    if lzw:
        indices = np.frombuffer(_decode_paa_lzw(stream, expected), dtype=np.uint8)
    else:
        out = bytearray()
        while len(out) < expected:
            raw = stream.read(1)
            if not raw:
                raise ValueError("truncated PAC RLE control byte")
            control = raw[0]
            if control & 0x80:
                value = stream.read(1)
                if not value:
                    raise ValueError("truncated PAC RLE value")
                count = (control & 0x7F) + 1
                out.extend(value * min(count, expected - len(out)))
            else:
                count = control + 1
                chunk = stream.read(count)
                if len(chunk) != count:
                    raise ValueError("truncated PAC RLE literal run")
                out.extend(chunk[: expected - len(out)])
        indices = np.frombuffer(bytes(out), dtype=np.uint8)
    if len(palette) == 0:
        raise ValueError("paletted PAC has no palette")
    if int(indices.max(initial=0)) >= len(palette):
        raise ValueError("PAC palette index out of range")
    return palette[indices].reshape((height, width, 4)).copy()


def decode_paa(data: bytes, *, is_paa: bool = True) -> np.ndarray:
    """Decode the first PAA/PAC mip to HxWx4 RGBA, following CWR-CE's decoder."""
    stream = io.BytesIO(data)
    magic = _u16(stream)
    kind = _PAA_FORMATS.get(magic)
    if kind is None:
        stream.seek(0)
        kind = "ARGB4444" if is_paa else "P8"
    while True:
        pos = stream.tell()
        if stream.read(4) != b"GGAT":
            stream.seek(pos)
            break
        if len(stream.read(4)) != 4:
            raise ValueError("truncated PAA TAGG name")
        size = _u32(stream)
        if size > len(data) - stream.tell():
            raise ValueError("truncated PAA TAGG payload")
        stream.seek(size, io.SEEK_CUR)
    palette_count = _u16(stream)
    if palette_count > 256:
        raise ValueError(f"invalid PAA/PAC palette size {palette_count}")
    palette_raw = stream.read(palette_count * 3)
    if len(palette_raw) != palette_count * 3:
        raise ValueError("truncated PAA/PAC palette")
    palette = np.empty((palette_count, 4), dtype=np.uint8)
    for index in range(palette_count):
        b, g, r = palette_raw[index * 3 : index * 3 + 3]
        packed = b | (g << 8) | (r << 16)
        palette[index] = (r, g, b, 0 if packed in {0xFF00FF, 0x00FFFF} else 255)
    width, height = _u16(stream), _u16(stream)
    lzw_p8 = width == 1234 and height == 8765
    if lzw_p8:
        width, height = _u16(stream), _u16(stream)
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise ValueError(f"invalid PAA dimensions {width}x{height}")
    data_size = _u24(stream)
    payload_start = stream.tell()
    if payload_start + data_size > len(data):
        raise ValueError(f"truncated PAA/PAC mip payload: need {data_size} bytes")
    if kind == "P8":
        return _decode_p8_payload(stream, width, height, palette, lzw=lzw_p8)
    if kind.startswith("DXT"):
        return _decode_dxt(stream.read(data_size), width, height, kind)
    if kind == "ARGB8888":
        payload = stream.read(data_size)
        needed = width * height * 4
        if len(payload) < needed:
            raise ValueError("truncated ARGB8888 PAA mip")
        bgra = np.frombuffer(payload[:needed], dtype=np.uint8).reshape((height, width, 4))
        return bgra[:, :, [2, 1, 0, 3]].copy()
    stream.seek(payload_start)
    raw = _decode_paa_lzw(stream, width * height * 2)
    if kind == "ARGB4444":
        return _decode_4444(raw, width, height)
    if kind == "AI88":
        return _decode_4444(raw, width, height, True)
    if kind == "ARGB1555":
        return _decode_1555(raw, width, height)
    raise ValueError(f"unsupported PAA format {kind}")
