# SPDX-License-Identifier: GPL-3.0-or-later
"""Finish paved outside-miter coverage without touching dirt/gravel.

The first wedge pass fixed the large asphalt holes, but Lundby40 exposed two
remaining mistakes:

* the final seam planner still measured unpitched/approximate stock endpoints;
* the generated triangular helper was inset into the road without compensating
  for the triangle taper, so its visible cross-section could be narrower than
  the grass opening it was meant to cover.

Keep this policy paved-only (sil/asf/kos).  It gives the final seam pass physical
WRP connector positions, installs the corrected wedge geometry, uses strict
surface containment when deciding that a wedge is already hidden, and rewrites
wedge assets in the parent process after procedural asset workers finish.  The
last step matters on Windows because asset workers use multiprocessing spawn and
would otherwise import the unpatched module in a fresh interpreter.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path

from . import procedural_infrastructure as _infra
from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_visual_finish_policy as _finish
from .paved_wedge_geometry import paved_wedge_local_points as _wide_wedge_points


_STRICT_SURFACE_MARGIN_METRES = 0.003
_ORIGINAL_WRITE_ASSETS = None
_INSTALLED = False


def paved_wedge_local_points(
    turn_degrees: float,
) -> tuple[tuple[float, float, float], ...]:
    return _wide_wedge_points(
        turn_degrees,
        radius_metres=_infra.GENERATED_PAVED_FILL_RADIUS_METRES,
        maximum_turn_degrees=_infra.GENERATED_PAVED_MITER_MAXIMUM_DEGREES,
    )


def _project_local_point(obj, point: tuple[float, float]) -> tuple[float, float]:
    """Project a P3D local X/Z point through the actual WRP yaw/pitch pose."""

    local_x, local_z = float(point[0]), float(point[1])
    heading = math.radians(float(obj.heading_degrees))
    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(obj.pitch_degrees)))
    return (
        float(obj.x)
        + local_x * cosine_heading
        + local_z * sine_heading * cosine_pitch,
        float(obj.z)
        - local_x * sine_heading
        + local_z * cosine_heading * cosine_pitch,
    )


def _physical_piece_axis(obj):
    straight = _emitted._geometry.stock_straight_match(str(obj.model_path))
    if straight is not None:
        family = straight.group("family").casefold()
        length = float(
            _emitted._geometry.STOCK_STRAIGHT_LENGTHS_METRES[
                int(straight.group("length"))
            ]
        )
        return (
            family,
            False,
            _project_local_point(obj, (0.0, -length * 0.5)),
            _project_local_point(obj, (0.0, length * 0.5)),
        )

    curve = _emitted._geometry.stock_curve_connectors(str(obj.model_path))
    if curve is not None:
        return (
            curve.family,
            True,
            _project_local_point(obj, curve.begin),
            _project_local_point(obj, curve.end),
        )
    return None


def _physical_seam_endpoints(report):
    """Return final paved/dirt stock endpoints exactly as RVW4 will render them."""

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    endpoints = []
    for obj in report.objects[cap_count:]:
        geometry = _physical_piece_axis(obj)
        if geometry is None:
            continue
        family, is_curve, first_point, second_point = geometry
        axis = (first_point, second_point)
        for endpoint_index, point in enumerate(axis):
            tangent = (
                (float(obj.heading_degrees) + endpoint_index * _emitted._geometry.STOCK_CURVE_ANGLE_DEGREES)
                % 180.0
                if is_curve
                else float(obj.heading_degrees) % 180.0
            )
            other = axis[1 - endpoint_index]
            outward_vector = (
                float(point[0]) - float(other[0]),
                float(point[1]) - float(other[1]),
            )
            tangent_unit = (
                math.sin(math.radians(tangent)),
                math.cos(math.radians(tangent)),
            )
            outward_heading = (
                tangent
                if (
                    outward_vector[0] * tangent_unit[0]
                    + outward_vector[1] * tangent_unit[1]
                )
                >= 0.0
                else tangent + 180.0
            )
            endpoints.append(
                _finish._SeamEndpoint(
                    point=(float(point[0]), float(point[1])),
                    object_id=int(obj.object_id),
                    endpoint_index=endpoint_index,
                    family=family,
                    tangent_axis_degrees=tangent,
                    is_curve=is_curve,
                    outward_heading_degrees=outward_heading % 360.0,
                )
            )
    return tuple(endpoints)


def _strict_straight_contains(obj, point: tuple[float, float]) -> bool:
    match = _emitted._geometry.stock_straight_match(str(obj.model_path))
    if match is None:
        return False
    family = match.group("family").casefold()
    if family not in _emitted._PAVED_FAMILIES:
        return False
    geometry = _physical_piece_axis(obj)
    if geometry is None:
        return False
    _family, _is_curve, start, end = geometry
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    length = math.hypot(dx, dz)
    if length <= 1.0e-9:
        return False
    ux, uz = dx / length, dz / length
    px = float(point[0]) - float(start[0])
    pz = float(point[1]) - float(start[1])
    along = px * ux + pz * uz
    lateral = abs(px * -uz + pz * ux)
    return (
        -_STRICT_SURFACE_MARGIN_METRES
        <= along
        <= length + _STRICT_SURFACE_MARGIN_METRES
        and lateral
        <= float(_emitted._geometry.STOCK_HALF_WIDTHS_METRES[family])
        + _STRICT_SURFACE_MARGIN_METRES
    )


def _strict_generated_miter_contains(obj, point: tuple[float, float]) -> bool:
    turn = _emitted.paved_miter_angle_degrees(str(obj.model_path))
    if turn is None:
        return False
    dx = float(point[0]) - float(obj.x)
    dz = float(point[1]) - float(obj.z)
    heading = math.radians(float(obj.heading_degrees))
    local_x = dx * math.cos(heading) - dz * math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(obj.pitch_degrees)))
    if abs(cosine_pitch) <= 1.0e-9:
        return False
    local_z = (
        dx * math.sin(heading) + dz * math.cos(heading)
    ) / cosine_pitch
    radius = (
        _infra.GENERATED_PAVED_FILL_RADIUS_METRES
        + _infra.GENERATED_PAVED_MITER_SAFETY_METRES
    )
    margin = _STRICT_SURFACE_MARGIN_METRES
    if math.hypot(local_x, local_z) <= radius + margin:
        return True
    half_angle = math.radians(float(turn) * 0.5)
    cosine = math.cos(half_angle)
    if cosine <= 1.0e-9:
        return False
    base_x = radius * cosine
    apex_x = radius / cosine
    absolute_x = abs(local_x)
    if absolute_x < base_x - margin or absolute_x > apex_x + margin:
        return False
    depth = apex_x - base_x
    if depth <= 1.0e-9:
        return False
    fraction = max(0.0, min(1.0, (apex_x - absolute_x) / depth))
    return (
        abs(local_z)
        <= radius * math.sin(half_angle) * fraction + margin
    )


def _generated_wedge_contains(obj, point: tuple[float, float]) -> bool:
    turn = _emitted.paved_wedge_angle_degrees(str(obj.model_path))
    if turn is None:
        return False
    dx = float(point[0]) - float(obj.x)
    dz = float(point[1]) - float(obj.z)
    heading = math.radians(float(obj.heading_degrees))
    local_x = dx * math.cos(heading) - dz * math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(obj.pitch_degrees)))
    if abs(cosine_pitch) <= 1.0e-9:
        return False
    local_z = (
        dx * math.sin(heading) + dz * math.cos(heading)
    ) / cosine_pitch
    points = paved_wedge_local_points(turn)
    depth = float(points[0][2])
    base_half_width = abs(float(points[1][0]))
    margin = _STRICT_SURFACE_MARGIN_METRES
    if local_z < -margin or local_z > depth + margin:
        return False
    fraction = max(0.0, min(1.0, 1.0 - local_z / max(1.0e-9, depth)))
    return abs(local_x) <= base_half_width * fraction + margin


def _surface_contains(obj, point: tuple[float, float]) -> bool:
    if _strict_straight_contains(obj, point):
        return True
    filename = str(obj.model_path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    if filename == "paved_fill.p3d":
        return math.dist((float(obj.x), float(obj.z)), point) <= (
            _infra.GENERATED_PAVED_FILL_RADIUS_METRES
            + _STRICT_SURFACE_MARGIN_METRES
        )
    if _emitted.paved_miter_angle_degrees(filename) is not None:
        return _strict_generated_miter_contains(obj, point)
    return _generated_wedge_contains(obj, point)


def _surface_is_paved(obj) -> bool:
    match = _emitted._geometry.stock_straight_match(str(obj.model_path))
    if match is not None:
        return match.group("family").casefold() in _emitted._PAVED_FAMILIES
    filename = str(obj.model_path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    return (
        filename == "paved_fill.p3d"
        or _emitted.paved_miter_angle_degrees(filename) is not None
        or _emitted.paved_wedge_angle_degrees(filename) is not None
    )


def _terrain_wedge_cover_plans(report, elevations=None, spec=None):
    """Plan terrain-clear outside triangles for sil/asf/kos seams only."""

    endpoints = tuple(
        endpoint
        for endpoint in _finish._seam_endpoints(report)
        if endpoint.family in _emitted._PAVED_FAMILIES
    )
    plans = []
    for distance, first, second, _pair_key in _emitted._nearest_endpoint_pairs(endpoints):
        if first.family != second.family:
            continue
        if not _emitted._pair_is_unambiguous(endpoints, first, second, distance):
            continue
        turn = _finish._axis_heading_difference(
            first.tangent_axis_degrees,
            second.tangent_axis_degrees,
        )
        if turn < _emitted.MINIMUM_EMITTED_TANGENT_ERROR_DEGREES:
            continue
        if turn > _emitted.MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES:
            continue
        seam_centre = (
            (float(first.point[0]) + float(second.point[0])) * 0.5,
            (float(first.point[1]) + float(second.point[1])) * 0.5,
        )
        involved_ids = {int(first.object_id), int(second.object_id)}
        if any(
            int(candidate.object_id) not in involved_ids
            and math.dist(candidate.point, seam_centre)
            <= _emitted.TERRAIN_WEDGE_JUNCTION_EXCLUSION_METRES
            for candidate in endpoints
        ):
            continue
        geometry = _emitted._outer_miter_geometry(first, second)
        if geometry is None:
            continue
        _area, apex, centroid = geometry
        coverage_samples = _emitted._gap_samples(first, second) + (apex, centroid)
        if _emitted._terrain_wedge_already_visible(
            report,
            first,
            second,
            coverage_samples,
            elevations,
            spec,
        ):
            continue
        plans.append(
            _finish._SeamCoverPlan(
                model_path=rf"o\road\{first.family}6.p3d",
                centre=seam_centre,
                tangent_axis_degrees=_emitted._plan_heading(first, second),
                turn_degrees=turn,
                outer_miter_apex=apex,
            )
        )
    return tuple(plans)


def _refresh_wedge_catalogue(
    source_dir: Path,
    catalogue_path: Path,
    result,
):
    document = json.loads(catalogue_path.read_text(encoding="utf-8"))
    changed = False
    for model in document.get("models", []):
        if not isinstance(model, dict):
            continue
        key = model.get("key")
        subtype = str(key.get("subtype", "")) if isinstance(key, dict) else ""
        if _infra.paved_wedge_angle_degrees(subtype + ".p3d") is None:
            continue
        relative = str(model.get("relative_path", ""))
        path = source_dir / relative
        if not path.is_file():
            continue
        model["sha256"] = sha256(path.read_bytes()).hexdigest()
        model["lod_resolutions"] = _infra.inspect_mlod(path).resolutions
        changed = True
    if not changed:
        return result

    document.pop("catalogue_sha256", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    document["catalogue_sha256"] = digest
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    catalogue_path.write_text(payload, encoding="utf-8")
    embedded = source_dir / "i" / "infrastructure.json"
    embedded.parent.mkdir(parents=True, exist_ok=True)
    embedded.write_text(payload, encoding="utf-8")
    return replace(result, catalogue_sha256=digest)


def _write_assets_with_final_wedges(self, source_dir: Path, catalogue_path: Path):
    if _ORIGINAL_WRITE_ASSETS is None:
        raise RuntimeError("paved wedge asset policy is not installed")
    result = _ORIGINAL_WRITE_ASSETS(self, source_dir, catalogue_path)

    rewritten = False
    for key in tuple(self._usage):
        if key.kind != "road":
            continue
        if _infra.paved_wedge_angle_degrees(key.subtype + ".p3d") is None:
            continue
        wire = self.model_path(key)
        relative = wire.split("\\", 1)[1].replace("\\", "/")
        destination = source_dir / relative
        _infra.write_infrastructure_mlod(destination, key, self._texture_path(key))
        rewritten = True

    if not rewritten:
        return result
    return _refresh_wedge_catalogue(source_dir, catalogue_path, result)


def install_stock_road_paved_wedge_policy() -> None:
    """Make final paved wedge generation and visibility tests physically exact."""

    global _ORIGINAL_WRITE_ASSETS, _INSTALLED
    if _INSTALLED:
        return

    # These globals are looked up at call time in the parent process.  P3D files
    # created in spawned workers are rewritten serially by the write_assets
    # wrapper below so Windows receives the same geometry too.
    _infra.paved_wedge_local_points = paved_wedge_local_points
    _emitted.paved_wedge_local_points = paved_wedge_local_points
    _finish._seam_endpoints = _physical_seam_endpoints
    _emitted._surface_contains = _surface_contains
    # Keep the historical private name because the emitted policy calls it.
    _emitted._surface_is_sil = _surface_is_paved
    _emitted._terrain_wedge_cover_plans = _terrain_wedge_cover_plans

    _ORIGINAL_WRITE_ASSETS = _infra.ProceduralInfrastructureLibrary.write_assets
    _infra.ProceduralInfrastructureLibrary.write_assets = _write_assets_with_final_wedges
    _INSTALLED = True
