#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure OFP/CWA P3D model bounds from loose files or PBO archives.

The scanner is intentionally standalone. It does not participate in world
building; it is an offline catalogue tool for discovering the dimensions of
stock game models before adding them to placement policies.

Examples::

    python tools/measure_p3d_models.py "C:\\Games\\ColdWarAssault" \
        --include "o\\hous\\*.p3d" --output stock-houses.json

    python tools/measure_p3d_models.py O.pbo Data3D.pbo --output all-models.json

The native reader supports OFP/CWA-era ODOL v6/v7 first-LOD vertex tables and
OFP MLOD/SP3X files. PBO entries using the legacy Cprs LZSS packing method are
decompressed in memory, so an extracted game data tree is not required.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fnmatch
import io
import json
import math
from pathlib import Path
import struct
import sys
from typing import Iterator, Sequence


_U32 = struct.Struct("<I")
_VEC3 = struct.Struct("<fff")
_PBO_ENTRY = struct.Struct("<IIIII")
_PBO_PROPERTIES = 0x56657273  # 'Vers'
_PBO_COMPRESSED = 0x43707273  # 'Cprs'

_MAX_VERTEX_COUNT = 10_000_000
_MAX_PBO_ENTRY_SIZE = 2_000_000_000
_LZSS_WINDOW_SIZE = 0x1000
_LZSS_WINDOW_MASK = _LZSS_WINDOW_SIZE - 1
_LZSS_FILL_BYTE = 0x20


class ModelReadError(ValueError):
    """Raised when a P3D is unsupported or malformed."""


@dataclass(frozen=True, slots=True)
class ModelMeasurement:
    model_path: str
    source: str
    format: str
    version: int
    lod: str
    vertex_count: int
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    width_m: float
    height_m: float
    length_m: float
    footprint_area_m2: float
    aspect_ratio: float
    origin_to_bottom_m: float


@dataclass(frozen=True, slots=True)
class ModelFailure:
    model_path: str
    source: str
    error: str


def _canonical_model_path(value: str) -> str:
    path = value.replace("/", "\\").strip().lstrip("\\")
    while "\\\\" in path:
        path = path.replace("\\\\", "\\")
    return path.casefold()


def _read_exact(stream: io.BytesIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ModelReadError(f"truncated {label}: wanted {size} bytes, got {len(data)}")
    return data


def _read_u32(stream: io.BytesIO, label: str) -> int:
    return _U32.unpack(_read_exact(stream, _U32.size, label))[0]


def _read_cstring(stream: io.BytesIO, label: str) -> str:
    out = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            raise ModelReadError(f"truncated {label}")
        if byte == b"\0":
            return out.decode("latin-1")
        out.extend(byte)


def _decompress_lzss_stream(stream: io.BytesIO, expected_size: int) -> bytes:
    """Expand legacy BIS LZSS when only the output length is known.

    OFP ODOL compressed arrays and Cprs PBO entries use BIS' 4 KiB-window LZSS.
    The dictionary starts filled with ASCII spaces. This matters for references
    encountered before 4 KiB of real output exists: those references are valid
    and read from the pre-filled dictionary rather than from already-emitted
    output bytes.

    ODOL blocks store only the uncompressed size, so decoding stops when exactly
    ``expected_size`` bytes have been produced. The following four bytes are the
    additive checksum for that decompressed block.
    """
    if expected_size < 0:
        raise ModelReadError("negative LZSS output size")
    if expected_size == 0:
        return b""

    window = bytearray([_LZSS_FILL_BYTE]) * _LZSS_WINDOW_SIZE
    window_pos = 0
    out = bytearray()
    checksum = 0

    def emit(value: int) -> None:
        nonlocal window_pos, checksum
        out.append(value)
        checksum = (checksum + value) & 0xFFFFFFFF
        window[window_pos] = value
        window_pos = (window_pos + 1) & _LZSS_WINDOW_MASK

    while len(out) < expected_size:
        flags_raw = stream.read(1)
        if not flags_raw:
            raise ModelReadError("truncated LZSS flag byte")
        flags = flags_raw[0]

        for bit in range(8):
            if len(out) >= expected_size:
                break

            if flags & (1 << bit):
                literal = stream.read(1)
                if not literal:
                    raise ModelReadError("truncated LZSS literal")
                emit(literal[0])
                continue

            pair = stream.read(2)
            if len(pair) != 2:
                raise ModelReadError("truncated LZSS reference")
            b1, b2 = pair
            offset = b1 | ((b2 & 0xF0) << 4)
            run_length = (b2 & 0x0F) + 3

            # BIS' decoder copies from the circular dictionary relative to the
            # current write cursor. Keep the source cursor advancing separately:
            # overlapping matches intentionally consume bytes written earlier in
            # this same run, which is how repeated patterns are represented.
            source_pos = window_pos
            for index in range(run_length):
                if len(out) >= expected_size:
                    break
                value = window[(source_pos - offset + index) & _LZSS_WINDOW_MASK]
                emit(value)

    checksum_raw = _read_exact(stream, 4, "LZSS checksum")
    stored_checksum = _U32.unpack(checksum_raw)[0]
    if stored_checksum != checksum:
        raise ModelReadError(
            f"LZSS checksum mismatch: stored {stored_checksum:#x}, calculated {checksum:#x}"
        )
    return bytes(out)


def _decompress_lzss_pbo(data: bytes, expected_size: int) -> bytes:
    stream = io.BytesIO(data)
    result = _decompress_lzss_stream(stream, expected_size)
    trailing = stream.read()
    if trailing:
        raise ModelReadError(f"compressed PBO entry has {len(trailing)} trailing bytes")
    return result


def _read_odol_array(stream: io.BytesIO, count: int, item_size: int, label: str) -> bytes:
    if count < 0 or count > _MAX_VERTEX_COUNT:
        raise ModelReadError(f"implausible {label} count {count}")
    expected = count * item_size
    if expected < 1024:
        return _read_exact(stream, expected, label)
    return _decompress_lzss_stream(stream, expected)


def _measurement(
    *,
    model_path: str,
    source: str,
    format_name: str,
    version: int,
    lod: str,
    points: Sequence[tuple[float, float, float]],
) -> ModelMeasurement:
    if not points:
        raise ModelReadError("selected LOD has no vertices")
    if any(not all(math.isfinite(value) for value in point) for point in points):
        raise ModelReadError("selected LOD contains non-finite vertex coordinates")

    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    max_z = max(point[2] for point in points)
    width = max_x - min_x
    height = max_y - min_y
    length = max_z - min_z
    if width < 0.0 or height < 0.0 or length < 0.0:
        raise ModelReadError("invalid negative model extent")
    smaller = min(width, length)
    larger = max(width, length)
    aspect = larger / smaller if smaller > 1.0e-6 else 0.0

    def clean(value: float) -> float:
        return round(float(value), 4)

    return ModelMeasurement(
        model_path=_canonical_model_path(model_path),
        source=source,
        format=format_name,
        version=int(version),
        lod=lod,
        vertex_count=len(points),
        min_x=clean(min_x),
        min_y=clean(min_y),
        min_z=clean(min_z),
        max_x=clean(max_x),
        max_y=clean(max_y),
        max_z=clean(max_z),
        width_m=clean(width),
        height_m=clean(height),
        length_m=clean(length),
        footprint_area_m2=clean(width * length),
        aspect_ratio=clean(aspect),
        origin_to_bottom_m=clean(-min_y),
    )


def _measure_odol(data: bytes, *, model_path: str, source: str) -> ModelMeasurement:
    stream = io.BytesIO(data)
    if _read_exact(stream, 4, "ODOL signature") != b"ODOL":
        raise ModelReadError("not an ODOL P3D")
    version = _read_u32(stream, "ODOL version")
    lod_count = _read_u32(stream, "ODOL LOD count")
    if version not in {6, 7}:
        raise ModelReadError(f"unsupported ODOL version {version}; expected OFP/CWA v6/v7")
    if lod_count <= 0 or lod_count > 128:
        raise ModelReadError(f"implausible ODOL LOD count {lod_count}")

    # OFP/CWA ODOL v6/v7 starts with the first LOD vertex table. For catalogue
    # sizing we only need its vertices, not faces, selections, animations, etc.
    flag_count = _read_u32(stream, "ODOL point-flag count")
    _read_odol_array(stream, flag_count, 4, "ODOL point flags")

    uv_count = _read_u32(stream, "ODOL UV count")
    _read_odol_array(stream, uv_count, 8, "ODOL UV coordinates")

    point_count = _read_u32(stream, "ODOL point count")
    if point_count <= 0 or point_count > _MAX_VERTEX_COUNT:
        raise ModelReadError(f"implausible ODOL point count {point_count}")
    if flag_count != point_count or uv_count != point_count:
        raise ModelReadError(
            "ODOL first-LOD vertex table count mismatch "
            f"(flags={flag_count}, uv={uv_count}, points={point_count})"
        )
    raw_points = _read_exact(stream, point_count * _VEC3.size, "ODOL points")
    points = [
        _VEC3.unpack_from(raw_points, offset)
        for offset in range(0, len(raw_points), _VEC3.size)
    ]

    normal_count = _read_u32(stream, "ODOL normal count")
    if normal_count != point_count:
        raise ModelReadError(
            f"ODOL first-LOD normal count {normal_count} does not match {point_count} vertices"
        )
    _read_exact(stream, normal_count * _VEC3.size, "ODOL normals")

    return _measurement(
        model_path=model_path,
        source=source,
        format_name="ODOL",
        version=version,
        lod="first",
        points=points,
    )


def _measure_mlod(data: bytes, *, model_path: str, source: str) -> ModelMeasurement:
    stream = io.BytesIO(data)
    if _read_exact(stream, 4, "MLOD signature") != b"MLOD":
        raise ModelReadError("not an MLOD P3D")
    version = _read_u32(stream, "MLOD version")
    lod_count = _read_u32(stream, "MLOD LOD count")
    if lod_count <= 0 or lod_count > 128:
        raise ModelReadError(f"implausible MLOD LOD count {lod_count}")

    signature = _read_exact(stream, 4, "MLOD first-LOD signature")
    if signature != b"SP3X":
        try:
            readable = signature.decode("ascii")
        except UnicodeDecodeError:
            readable = repr(signature)
        raise ModelReadError(f"unsupported MLOD LOD format {readable}; expected SP3X")

    major = _read_u32(stream, "SP3X major version")
    _minor = _read_u32(stream, "SP3X minor version")
    point_count = _read_u32(stream, "SP3X point count")
    normal_count = _read_u32(stream, "SP3X normal count")
    _face_count = _read_u32(stream, "SP3X face count")
    _flags = _read_u32(stream, "SP3X flags")
    if point_count <= 0 or point_count > _MAX_VERTEX_COUNT:
        raise ModelReadError(f"implausible SP3X point count {point_count}")
    if major not in {27, 28}:
        raise ModelReadError(f"unsupported SP3X major version {major}")

    points: list[tuple[float, float, float]] = []
    for index in range(point_count):
        raw = _read_exact(stream, 16, f"SP3X point {index}")
        x, y, z, _point_flags = struct.unpack("<fffi", raw)
        points.append((x, y, z))

    _read_exact(stream, normal_count * _VEC3.size, "SP3X normals")

    return _measurement(
        model_path=model_path,
        source=source,
        format_name="MLOD/SP3X",
        version=version,
        lod="first",
        points=points,
    )


def measure_p3d(data: bytes, *, model_path: str, source: str) -> ModelMeasurement:
    if len(data) < 12:
        raise ModelReadError("P3D is too small to contain a valid header")
    signature = data[:4]
    if signature == b"ODOL":
        return _measure_odol(data, model_path=model_path, source=source)
    if signature == b"MLOD":
        return _measure_mlod(data, model_path=model_path, source=source)
    raise ModelReadError(f"unsupported P3D signature {signature!r}")


def _pbo_entries(path: Path) -> Iterator[tuple[str, bytes]]:
    raw = path.read_bytes()
    stream = io.BytesIO(raw)
    metadata: list[tuple[str, int, int, int]] = []
    properties: dict[str, str] = {}

    while True:
        name = _read_cstring(stream, "PBO entry name")
        fields = _read_exact(stream, _PBO_ENTRY.size, "PBO entry header")
        packing, original_size, reserved, timestamp, data_size = _PBO_ENTRY.unpack(fields)
        del reserved, timestamp
        if data_size > _MAX_PBO_ENTRY_SIZE or original_size > _MAX_PBO_ENTRY_SIZE:
            raise ModelReadError(f"implausible PBO entry size for {name!r}")
        if not name:
            if packing == _PBO_PROPERTIES:
                while True:
                    key = _read_cstring(stream, "PBO property key")
                    if not key:
                        break
                    properties[key.casefold()] = _read_cstring(stream, "PBO property value")
                continue
            if any((packing, original_size, data_size)):
                raise ModelReadError("unsupported PBO extension record")
            break
        metadata.append((name, packing, original_size, data_size))

    prefix = properties.get("prefix", "").replace("/", "\\").strip("\\") or path.stem
    cursor = stream.tell()
    for name, packing, original_size, data_size in metadata:
        end = cursor + data_size
        if end > len(raw):
            raise ModelReadError(f"truncated PBO entry {name}")
        stored = raw[cursor:end]
        cursor = end

        combined = name.replace("/", "\\").lstrip("\\")
        canonical_prefix = _canonical_model_path(prefix)
        if canonical_prefix and not _canonical_model_path(combined).startswith(canonical_prefix + "\\"):
            combined = prefix + "\\" + combined
        model_path = _canonical_model_path(combined)
        if not model_path.endswith(".p3d"):
            continue

        if packing == 0:
            data = stored
        elif packing == _PBO_COMPRESSED:
            if original_size <= 0:
                raise ModelReadError(f"compressed PBO entry {model_path} has no original size")
            data = _decompress_lzss_pbo(stored, original_size)
        else:
            raise ModelReadError(
                f"unsupported PBO packing method {packing:#x} for {model_path}"
            )
        yield model_path, data


def _matches(model_path: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    folded = _canonical_model_path(model_path)
    return any(fnmatch.fnmatchcase(folded, _canonical_model_path(pattern)) for pattern in patterns)


def _iter_models(inputs: Sequence[Path], patterns: Sequence[str]) -> Iterator[tuple[str, str, bytes]]:
    seen_loose: set[Path] = set()
    seen_pbos: set[Path] = set()

    def loose(path: Path, root: Path | None = None) -> Iterator[tuple[str, str, bytes]]:
        resolved = path.resolve()
        if resolved in seen_loose:
            return
        seen_loose.add(resolved)
        if root is not None:
            try:
                model_path = path.relative_to(root).as_posix()
            except ValueError:
                model_path = path.name
        else:
            model_path = path.name
        model_path = _canonical_model_path(model_path)
        if _matches(model_path, patterns):
            yield model_path, str(path), path.read_bytes()

    def pbo(path: Path) -> Iterator[tuple[str, str, bytes]]:
        resolved = path.resolve()
        if resolved in seen_pbos:
            return
        seen_pbos.add(resolved)
        for model_path, data in _pbo_entries(path):
            if _matches(model_path, patterns):
                yield model_path, f"{path}!{model_path}", data

    for raw_input in inputs:
        path = raw_input.expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            suffix = path.suffix.casefold()
            if suffix == ".p3d":
                yield from loose(path)
            elif suffix == ".pbo":
                yield from pbo(path)
            else:
                raise ValueError(f"unsupported input file type: {path}")
            continue

        for child in sorted(path.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if not child.is_file():
                continue
            suffix = child.suffix.casefold()
            if suffix == ".p3d":
                yield from loose(child, path)
            elif suffix == ".pbo":
                yield from pbo(child)


def scan_models(
    inputs: Sequence[Path], patterns: Sequence[str] = ()
) -> tuple[list[ModelMeasurement], list[ModelFailure]]:
    measurements: list[ModelMeasurement] = []
    failures: list[ModelFailure] = []
    for model_path, source, data in _iter_models(inputs, patterns):
        try:
            measurements.append(measure_p3d(data, model_path=model_path, source=source))
        except (ModelReadError, OSError, struct.error, OverflowError, IndexError) as exc:
            # A malformed/unsupported model should be recorded and scanning should
            # continue. IndexError is defensive here: the native parser must not
            # abort an entire catalogue because one legacy model has odd data.
            failures.append(ModelFailure(model_path=model_path, source=source, error=str(exc)))
    measurements.sort(key=lambda item: item.model_path)
    failures.sort(key=lambda item: item.model_path)
    return measurements, failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure OFP/CWA P3D model bounds from loose models, directories, or PBO archives."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="P3D, PBO, or directory to scan")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Only include canonical model paths matching GLOB. May be repeated; "
            "example: --include 'o\\hous\\*.p3d'."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, help="Write JSON report to this file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit status if any model could not be measured",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        measurements, failures = scan_models(args.inputs, args.include)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = {
        "schema": 1,
        "model_count": len(measurements),
        "failure_count": len(failures),
        "models": [asdict(item) for item in measurements],
        "failures": [asdict(item) for item in failures],
    }
    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(
            f"Measured {len(measurements):,} model(s); "
            f"{len(failures):,} failure(s); wrote {args.output}",
            file=sys.stderr,
        )

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
