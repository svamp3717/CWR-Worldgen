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
from .legacy_proxy_models import ProxyCloneError, inspect_roadway_deck_height


@dataclass(frozen=True, slots=True)
class BridgeDonor:
    model: str
    length_metres: float
    width_metres: float
    heading_offset_degrees: float
    roadway_local_y_metres: float
    source_format: str
    content_sha256: str


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
        dimensions = inspect_fence_donor_geometry(data)
        deck_height = inspect_roadway_deck_height(data)
    except (OSError, ValueError, ProxyCloneError) as exc:
        raise ValueError(f"Cannot inspect bridge donor {model!r}: {exc}") from exc
    length = overrides["length"] if overrides["length"] is not None else dimensions.length_metres
    heading = (
        overrides["heading_offset"]
        if overrides["heading_offset"] is not None
        else dimensions.heading_offset_degrees
    )
    deck_y = (
        overrides["deck_height"]
        if overrides["deck_height"] is not None
        else deck_height
    )
    if not 2.0 <= length <= 200.0:
        raise ValueError(f"Bridge module length {length:.2f} m is outside 2..200 m")
    if dimensions.width_metres < 2.0:
        raise ValueError(
            f"Bridge donor {model!r} is only {dimensions.width_metres:.2f} m wide; "
            "select a drivable bridge module"
        )
    return BridgeDonor(
        model=model,
        length_metres=length,
        width_metres=dimensions.width_metres,
        heading_offset_degrees=heading,
        roadway_local_y_metres=deck_y,
        source_format=dimensions.source_format,
        content_sha256=hashlib.sha256(data).hexdigest(),
    )
