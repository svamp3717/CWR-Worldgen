# SPDX-License-Identifier: GPL-3.0-or-later
"""Resolve a straight, drivable OFP/CWA bridge module from a selected P3D."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from .assets import canonical_asset_path, read_asset_record_bytes
from .fast_asset_scan_policy import locate_assets_fast
from .fence_donor_geometry import inspect_fence_donor_geometry
from .legacy_proxy_models import ProxyCloneError, inspect_roadway_deck_geometry


@dataclass(frozen=True, slots=True)
class BridgeDonor:
    model: str
    length_metres: float
    width_metres: float
    heading_offset_degrees: float
    roadway_local_y_metres: float
    source_format: str
    content_sha256: str
    roadway_centre_x: float = 0.0
    roadway_centre_z: float = 0.0
    leveling_pitch_degrees: float = 0.0
    roadway_origin_z_after_pitch: float = 0.0
    roadway_end_difference_metres: float = 0.0
    leveled_relief_metres: float = 0.0


def configured_bridge_donor(spec: object) -> BridgeDonor | None:
    """Auto-measure visual footprint and actual Roadway LOD height.

    Unlike an ordinary fence, a bridge must have a usable Roadway LOD. The
    donor's visible railing height is not a safe substitute for its drivable
    surface. Only straight, approximately level bridge modules are supported.
    """
    model = str(getattr(spec, "bridge_donor_model", "") or "").strip().replace("/", "\\")
    if not model:
        return None
    overrides = {}
    for name, positive in (
        ("length", True),
        ("heading_offset", False),
        ("deck_height", False),
    ):
        raw = getattr(spec, f"bridge_donor_{name}", None)
        if raw in ("", None):
            overrides[name] = None
            continue
        value = float(raw)
        if not math.isfinite(value) or (positive and value <= 0):
            raise ValueError(
                f"bridge_donor_{name} must be finite"
                + (" and positive" if positive else "")
            )
        overrides[name] = value
    roots = tuple(Path(p) for p in getattr(spec, "asset_roots", ()) or ())
    if not roots:
        raise ValueError(
            f"Bridge donor {model!r} needs Asset roots for Roadway LOD validation. "
            "Add its PBO or folder to Asset roots."
        )
    scan = locate_assets_fast(
        roots, (model,),
        cache_dir=getattr(spec, "cache_dir", None),
        use_cache=bool(getattr(spec, "cache_enabled", True)),
        refresh=bool(getattr(spec, "cache_refresh", False)),
    )
    record = next(
        (item for item in scan.records if item.path == canonical_asset_path(model)),
        None,
    )
    if record is None:
        raise ValueError(
            f"Bridge donor {model!r} not found in Asset roots. "
            "Use its virtual P3D path and select the containing addon PBO/folder."
        )
    try:
        data = read_asset_record_bytes(record)
        dimensions = inspect_fence_donor_geometry(data, allow_ambiguous=True)
        roadway = inspect_roadway_deck_geometry(data)
    except (OSError, ValueError, ProxyCloneError) as exc:
        raise ValueError(f"Cannot inspect bridge donor {model!r}: {exc}") from exc
    roadway_length = max(roadway.span_x, roadway.span_z)
    roadway_width = min(roadway.span_x, roadway.span_z)
    if roadway_width >= roadway_length * 0.85:
        raise ValueError(
            f"Bridge donor {model!r} has no clear straight Roadway axis "
            f"({roadway.span_x:.2f} x {roadway.span_z:.2f} m)"
        )
    length = overrides["length"] if overrides["length"] is not None else roadway_length
    heading = (
        overrides["heading_offset"]
        if overrides["heading_offset"] is not None
        else (90.0 if roadway.span_x > roadway.span_z else 0.0)
    )
    deck_y = (
        overrides["deck_height"]
        if overrides["deck_height"] is not None
        else roadway.leveled_local_y
    )
    if not 2.0 <= length <= 200.0:
        raise ValueError(f"Bridge module length {length:.2f} m is outside 2..200 m")
    if roadway_width < 2.0:
        raise ValueError(
            f"Bridge donor {model!r} Roadway is only {roadway_width:.2f} m wide; "
            "select a drivable bridge module"
        )
    return BridgeDonor(
        model=model,
        length_metres=length,
        width_metres=roadway_width,
        heading_offset_degrees=heading,
        roadway_local_y_metres=deck_y,
        source_format=dimensions.source_format,
        content_sha256=hashlib.sha256(data).hexdigest(),
        roadway_centre_x=roadway.centre_x,
        roadway_centre_z=roadway.centre_z,
        leveling_pitch_degrees=roadway.leveling_pitch_degrees,
        roadway_origin_z_after_pitch=(
            roadway.centre_z * math.cos(math.radians(roadway.leveling_pitch_degrees))
            - roadway.local_y * math.sin(math.radians(roadway.leveling_pitch_degrees))
        ),
        roadway_end_difference_metres=roadway.roadway_end_difference_metres,
        leveled_relief_metres=roadway.leveled_relief_metres,
    )
