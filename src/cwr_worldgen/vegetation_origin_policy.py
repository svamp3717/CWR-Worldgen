# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Ground stock vegetation at its visible base, not its centered P3D origin.

CWA ODOL tree/bush vertices are authored about the model's bounding centre.
Worldgen placement routines emit terrain-height Y values, which are root heights,
whereas WRP ObjectCreate expects the model centre in its transform.  This affects
house-yard fruit trees, street trees, standalone forests and CWA-flattened field
bushes equally. Use measured P3D geometry when available, but fall back to
CWR-CE's own model-aware ground snapping without requiring game files locally.
"""

from dataclasses import dataclass, replace
import io
import math
import struct
from typing import Mapping, Sequence

from .assets import AssetRecord, canonical_asset_path, read_asset_record_bytes
from .legacy_proxy_models import (
    ProxyCloneError, _read_exact, _read_odol_array, _read_u32,
)
from .model import WorldObject
from .vegetation_clearance_policy import (
    _configured_vegetation_models, _is_tree_or_bush,
)

_VEC3 = struct.Struct("<fff")

# CWR-CE Landscape::ObjectCreate() moves a model down when its authored
# origin (object PositionModelToWorld(-shape->BoundingCenter())) starts
# above SurfaceY(). Giving stock vegetation a substantial positive WRP Y
# ensures the game grounds it using its OWN loaded P3D, not a guessed pivot.
# Raw WRP editors which do not emulate ObjectCreate() will see this bias.
RUNTIME_SNAP_BIAS_METRES = 128.0


@dataclass(frozen=True, slots=True)
class VegetationOriginReport:
    lifted_objects: int
    lifted_models: int
    measured_models: int
    runtime_snap_models: tuple[str, ...]
    unavailable_models: tuple[str, ...]
    unreadable_models: tuple[str, ...]


def _odol_visual_base_offset(data: bytes) -> float:
    """Read first-LOD vertex Y without parsing P3D faces or proxy children.

    ODOL v6/v7 have already-centred vertices.  A negative first-LOD bottom
    therefore means a positive WRP-model-centre offset is required to place
    the visible bottom on the supplied terrain-height origin.
    """
    stream = io.BytesIO(data)
    if _read_exact(stream, 4, "P3D magic") != b"ODOL":
        raise ValueError("only centred ODOL v6/v7 vegetation is supported")
    version = _read_u32(stream, "ODOL version")
    lod_count = _read_u32(stream, "ODOL LOD count")
    if version not in {6, 7} or not 1 <= lod_count <= 128:
        raise ValueError("unsupported ODOL version or LOD count")
    flags = _read_u32(stream, "point flags count")
    _read_odol_array(stream, flags, 4, "ODOL point flags")
    uv = _read_u32(stream, "UV count")
    _read_odol_array(stream, uv, 8, "ODOL UVs")
    count = _read_u32(stream, "point count")
    if not 0 < count <= 2_000_000 or count != flags or count != uv:
        raise ValueError("invalid ODOL first-LOD vertex table")
    raw = _read_exact(stream, count * _VEC3.size, "ODOL vertices")
    lowest = min(
        _VEC3.unpack_from(raw, i * _VEC3.size)[1] for i in range(count)
    )
    if not math.isfinite(lowest) or lowest < -100.0:
        raise ValueError("implausible ODOL vegetation bottom")
    return max(0.0, -float(lowest))


def _direct_vegetation(model_path: str, spec: object, configured: set[str]) -> bool:
    """Skip forest blocks and procedural proxy carriers with different pivots."""
    normalized = canonical_asset_path(model_path)
    basename = normalized.rsplit("\\", 1)[-1]
    if "les_" in basename or basename.startswith(("les ", "les-")):
        return False
    from .procedural_forests import is_generated_cluster_model
    if is_generated_cluster_model(str(getattr(spec, "name", "")), model_path):
        return False
    return _is_tree_or_bush(model_path, spec, configured)


def vegetation_origin_lifts(
    objects: Sequence[WorldObject],
    records: Sequence[AssetRecord],
    spec: object,
) -> tuple[dict[str, float], VegetationOriginReport]:
    """Prefer measured model pivots, otherwise use CWR-CE runtime grounding."""
    configured = _configured_vegetation_models(spec)
    models: dict[str, str] = {}
    for obj in objects:
        key = canonical_asset_path(obj.model_path)
        if key not in models and _direct_vegetation(obj.model_path, spec, configured):
            models[key] = obj.model_path
    available = {canonical_asset_path(record.path): record for record in records}
    lifts: dict[str, float] = {}
    unavailable: list[str] = []
    unreadable: list[str] = []
    runtime_snap: list[str] = []
    measured = 0
    for key, original in sorted(models.items()):
        record = available.get(key)
        if record is None:
            unavailable.append(original)
            runtime_snap.append(original)
            lifts[key] = RUNTIME_SNAP_BIAS_METRES
            continue
        try:
            lift = _odol_visual_base_offset(read_asset_record_bytes(record))
        except (ValueError, OSError, ProxyCloneError) as exc:
            unreadable.append(f"{original}: {exc}")
            runtime_snap.append(original)
            lifts[key] = RUNTIME_SNAP_BIAS_METRES
            continue
        measured += 1
        if lift > 0.01:
            lifts[key] = lift
    lifted = sum(1 for obj in objects if canonical_asset_path(obj.model_path) in lifts)
    return lifts, VegetationOriginReport(
        lifted, len(lifts), measured, tuple(runtime_snap),
        tuple(unavailable), tuple(unreadable)
    )


def apply_vegetation_origin_lifts(
    objects: Sequence[WorldObject], lifts: Mapping[str, float],
) -> tuple[WorldObject, ...]:
    """Shift model centres or defer final grounding to CWR-CE ObjectCreate()."""
    if not lifts:
        return tuple(objects)
    return tuple(
        replace(
            obj, y=obj.y + lifts[canonical_asset_path(obj.model_path)]
        )
        if canonical_asset_path(obj.model_path) in lifts else obj
        for obj in objects
    )
