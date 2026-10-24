#!/usr/bin/env python3
"""Create two LOCAL ODOL paved-road probe PBOs from a user's stock Road.zip.

Both copies preserve original geometry, textures, Roadway and other LODs.
The experimental copy changes ONLY drawable visual LOD flags from original
stock values to the flag values measured from working SFP paved roads.

No proprietary P3D or PAA is bundled with this script or the repository.

Usage:
  python tools/build_paved_flag_probe.py --source-zip "Road(1).zip" \
      --output-dir ./probe-output

Place both generated PBOs in a test CWA addon directory. In separate world
builds, configure the paved donor and curve donor to the control or experiment
paths. Keep all other settings, terrain and gravel models identical.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

_STOCK_POINT_FLAG = 0x13F
_STOCK_FACE_FLAGS = frozenset((0x24102, 0x2C102))
_PAVED_NAMES = (
    "sil25.p3d", "sil12.p3d", "sil6.p3d",
    "sil10 25.p3d", "sil10 50.p3d", "sil10 75.p3d",
    "sil10 100.p3d",
)


def _u32(reader: io.BytesIO) -> int:
    raw = reader.read(4)
    if len(raw) != 4:
        raise ValueError("truncated ODOL integer")
    return struct.unpack("<I", raw)[0]


def _cstring(reader: io.BytesIO) -> str:
    raw = bytearray()
    while True:
        character = reader.read(1)
        if not character:
            raise ValueError("truncated ODOL string")
        if character == b"\0":
            return raw.decode("latin-1")
        raw += character


def _skip_small_array(reader: io.BytesIO, size: int, *, item_size: int) -> None:
    if size < 0 or size * item_size >= 1024:
        raise ValueError("probe accepts only uncompressed native small-road arrays")
    reader.seek(size * item_size, io.SEEK_CUR)


def visual_flag_spans(data: bytes):
    """Locate every native ODOL LOD's point and face flag offsets.

    Parse rather than replace every matching byte pattern: unrelated Roadway
    or Geometry LOD flags must not be overwritten by the experiment.
    """
    reader = io.BytesIO(data)
    if reader.read(4) != b"ODOL":
        raise ValueError("source road must be original ODOL, not MLOD")
    version, count = _u32(reader), _u32(reader)
    if version != 7 or not 1 <= count <= 12:
        raise ValueError("expected stock ODOL v7 road model")

    spans = []
    for lod in range(count):
        count_flags = _u32(reader)
        if count_flags > 255:
            raise ValueError("compressed point flags unsupported")
        points_start = reader.tell()
        flags = [_u32(reader) for _ in range(count_flags)]

        _skip_small_array(reader, _u32(reader), item_size=8)  # UV
        reader.seek(_u32(reader) * 12, io.SEEK_CUR)  # points
        reader.seek(_u32(reader) * 12 + 48, io.SEEK_CUR)  # normals, bounds
        for _ in range(_u32(reader)):
            _cstring(reader)  # texture path
        for _ in range(2):
            _skip_small_array(reader, _u32(reader), item_size=2)

        face_count = _u32(reader)
        _u32(reader)  # face allocation size
        faces = []
        for _ in range(face_count):
            offset = reader.tell()
            flag = _u32(reader)
            reader.seek(2, io.SEEK_CUR)  # texture index
            num_vertices = reader.read(1)
            if len(num_vertices) != 1:
                raise ValueError("truncated face")
            reader.seek(num_vertices[0] * 2, io.SEEK_CUR)
            faces.append((offset, flag))

        for _ in range(_u32(reader)):
            reader.seek(18, io.SEEK_CUR)  # section
        for _ in range(_u32(reader)):
            _cstring(reader)  # selection name
            for item_size in (2, 1, 4):
                _skip_small_array(reader, _u32(reader), item_size=item_size)
            reader.seek(1, io.SEEK_CUR)
            for item_size in (4, 2, 1):
                _skip_small_array(reader, _u32(reader), item_size=item_size)
        for _ in range(_u32(reader)):
            _cstring(reader)
            _cstring(reader)
        for _ in range(_u32(reader)):
            reader.seek(4, io.SEEK_CUR)  # animation time
            reader.seek(_u32(reader) * 12, io.SEEK_CUR)  # animation points
        reader.seek(12, io.SEEK_CUR)  # colors and flags
        if _u32(reader):
            raise ValueError("road proxy model unexpectedly contains proxies")
        if reader.tell() > len(data):
            raise ValueError("ODOL LOD exceeds model size")
        spans.append((points_start, flags, faces))

    resolutions = []
    for _ in range(count):
        raw = reader.read(4)
        if len(raw) != 4:
            raise ValueError("missing ODOL LOD resolutions")
        resolutions.append(struct.unpack("<f", raw)[0])
    # Some stock ODOL models append native metadata after this table. Keep it.
    return tuple(zip(spans, resolutions))


def patch_stock_visual_flags(data: bytes) -> tuple[bytes, int]:
    """Change only visual LOD flags, retaining native mesh/contact bytes."""
    spans = visual_flag_spans(data)
    result = bytearray(data)
    changed_lods = 0
    for ((flag_start, point_flags, faces), resolution) in spans:
        if not 0 <= resolution < 1.0e9 or not faces:
            continue
        if not point_flags or any(value != _STOCK_POINT_FLAG for value in point_flags):
            raise ValueError("visual point flags are not the expected original stock flags")
        if any(value not in _STOCK_FACE_FLAGS for _, value in faces):
            raise ValueError("visual face flags are not expected stock values")
        for index in range(len(point_flags)):
            struct.pack_into("<I", result, flag_start + index * 4, 1)
        for offset, _ in faces:
            struct.pack_into("<I", result, offset, 0)
        changed_lods += 1
    if changed_lods == 0:
        raise ValueError("no original-stock visual LODs to patch")

    reparsed = visual_flag_spans(result)
    for (original, resolution), (modified, new_resolution) in zip(spans, reparsed):
        if resolution != new_resolution:
            raise ValueError("LOD resolution changed unexpectedly")
        if 0 <= resolution < 1.0e9 and original[2]:
            if set(modified[1]) != {1} or any(flag for _, flag in modified[2]):
                raise ValueError("flag patch did not apply cleanly")
        elif original != modified:
            raise ValueError("nonvisual LOD was modified")
    return bytes(result), changed_lods


def _write_uncompressed_pbo(output: Path, entries: list[tuple[str, bytes]]) -> None:
    """Write the legacy simple PBO layout, without extra embedded files."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        for name, payload in entries:
            stream.write(name.replace("/", "\\").encode("ascii") + b"\0")
            stream.write(struct.pack("<IIIII", 0, len(payload), 0, 0, len(payload)))
        stream.write(b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0))
        for _name, payload in entries:
            stream.write(payload)


def build_probe_pair(source_zip: Path, output_dir: Path) -> dict[str, object]:
    control: list[tuple[str, bytes]] = []
    experimental: list[tuple[str, bytes]] = []
    model_results = {}
    with ZipFile(source_zip) as archive:
        for name in _PAVED_NAMES:
            original = archive.read("Road/" + name)
            patched, lod_count = patch_stock_visual_flags(original)
            control.append(("roads/" + name, original))
            experimental.append(("roads/" + name, patched))
            model_results[name] = {
                "visual_lods_patched": lod_count,
                "original_sha256": sha256(original).hexdigest(),
                "patched_sha256": sha256(patched).hexdigest(),
                "changed_bytes": sum(a != b for a, b in zip(original, patched)),
                "length_unchanged": len(original) == len(patched),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    first = output_dir / "cwr_paved_stock_control.pbo"
    second = output_dir / "cwr_paved_flag_probe.pbo"
    _write_uncompressed_pbo(first, control)
    _write_uncompressed_pbo(second, experimental)
    report = {
        "hypothesis": "stock-vs-SFP visual flags alone affect gravel/asphalt ordering",
        "stock_reference_sha256": sha256(source_zip.read_bytes()).hexdigest(),
        "control_pbo": first.name,
        "experimental_pbo": second.name,
        "changed_properties": "visual ODOL point/face flags only",
        "models": model_results,
        "configured_paved_models": {
            "control": r"cwr_paved_stock_control\roads\sil25.p3d",
            "experiment": r"cwr_paved_flag_probe\roads\sil25.p3d",
        },
        "configured_paved_curves": {
            "control": r"cwr_paved_stock_control\roads\sil10 25.p3d",
            "experiment": r"cwr_paved_flag_probe\roads\sil10 25.p3d",
        },
        "game_rendering_verified": False,
    }
    (output_dir / "paved-flag-probe-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = build_probe_pair(args.source_zip, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
