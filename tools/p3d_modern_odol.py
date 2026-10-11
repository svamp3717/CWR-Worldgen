"""Read safe model bounds for newer (Arma-era) ODOL files.

This is a *bounds-only* preview, not an ODOL mesh decoder. The Arma-era
format places ModelInfo ahead of the LODs, unlike OFP's ODOL v6/v7. Reading
its early bounding-box fields lets the browser identify and visualize assets
without interpreting compressed vertices or inventing faces.

Reference: BIS P3D ODOL ModelInfo and ODOL versioned headers.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
import struct


class ModernOdolError(ValueError):
    """Invalid or unsupported modern ODOL header."""


@dataclass(frozen=True)
class OdolBounds:
    version: int
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    lod_count: int


# Known Arma-era version families, including Arma 1 and intervening releases.
# ODOL v6/v7 are handled by the existing full-mesh reader instead.
MIN_MODERN_VERSION = 40
MAX_MODERN_VERSION = 75
MAX_LODS = 128
MAX_MUZZLE_FLASH_BYTES = 4096

_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_VEC3 = struct.Struct("<3f")


def _take(stream: io.BytesIO, length: int, label: str) -> bytes:
    if length < 0:
        raise ModernOdolError(f"invalid {label} length")
    data = stream.read(length)
    if len(data) != length:
        raise ModernOdolError(f"truncated {label}")
    return data


def _u32(stream: io.BytesIO, label: str) -> int:
    return _U32.unpack(_take(stream, 4, label))[0]


def _cstring(stream: io.BytesIO, label: str) -> None:
    for _ in range(MAX_MUZZLE_FLASH_BYTES):
        byte = _take(stream, 1, label)
        if byte == b"\\0":
            return
    raise ModernOdolError(f"{label} exceeds {MAX_MUZZLE_FLASH_BYTES} bytes")


def read_model_bounds(data: bytes) -> OdolBounds:
    """Extract ModelInfo bounds from an Arma-era ODOL without parsing its LODs.

    v58 adds a muzzle-flash string, v59 an app ID, and v75 an encryption
    preamble. Reject encrypted and incomplete headers rather than guessing.
    """
    stream = io.BytesIO(data)
    if _take(stream, 4, "ODOL signature") != b"ODOL":
        raise ModernOdolError("not an ODOL model")
    version = _u32(stream, "ODOL version")
    if not MIN_MODERN_VERSION <= version <= MAX_MODERN_VERSION:
        raise ModernOdolError(f"unsupported Arma-era ODOL version {version}")
    if version >= 75:
        if _u32(stream, "encryption marker") or _u32(stream, "encryption marker"):
            raise ModernOdolError(f"encrypted ODOL v{version} cannot be previewed")
    if version >= 59:
        _u32(stream, "application ID")
    if version >= 58:
        _cstring(stream, "muzzle-flash string")

    lod_count = _u32(stream, "LOD resolution count")
    if not 1 <= lod_count <= MAX_LODS:
        raise ModernOdolError(f"invalid ODOL LOD count {lod_count}")
    _take(stream, 4 * lod_count, "LOD resolutions")

    # ModelInfo starts with six 4-byte values, aiming centre (12 bytes),
    # two packed colours (8), and view density (4). Next are BboxMin/BboxMax.
    _take(stream, 48, "ModelInfo fields preceding model bounds")
    minimum = _VEC3.unpack(_take(stream, 12, "ModelInfo BboxMin"))
    maximum = _VEC3.unpack(_take(stream, 12, "ModelInfo BboxMax"))
    if not all(math.isfinite(x) for x in (*minimum, *maximum)):
        raise ModernOdolError("non-finite ODOL model bounds")
    if any(low > high for low, high in zip(minimum, maximum)):
        raise ModernOdolError("inverted ODOL model bounds")
    return OdolBounds(version, minimum, maximum, lod_count)


def box_vertices(bounds: OdolBounds) -> tuple[tuple[float, float, float], ...]:
    """Eight corners in a stable order for a bounds-only wireframe."""
    ax, ay, az = bounds.minimum
    bx, by, bz = bounds.maximum
    return (
        (ax, ay, az), (bx, ay, az), (bx, ay, bz), (ax, ay, bz),
        (ax, by, az), (bx, by, az), (bx, by, bz), (ax, by, bz),
    )


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
