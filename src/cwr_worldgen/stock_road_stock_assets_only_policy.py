# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep paved and dirt roads stock-only while allowing procedural gravel.

Paved production roads use stock ``sil/asf/kos`` pieces and measured Resistance
junctions. Dirt roads use stock ``ces``. Generated gravel remains a supported,
separate family when requested.

This final policy also completes the reference-WRP construction strategy:

* native paved curves are allowed to smooth hard OSM vertices inside a bounded,
  obstacle-checked road corridor;
* tangent-compatible ordinary paved straight neighbours use a small Kodiak-style
  longitudinal overlap instead of exact butt joints;
* a T junction is chosen from the WrpTool stock family catalogue before approach
  headings are regularised, and only the resulting obstacle-safe geometry has to
  pass the strict final connector matcher; and
* a generic ``*6`` cap is therefore a fallback only when the stock T cannot be
  made representable inside those safety bounds.

Generated paved and generated dirt road P3Ds remain forbidden at serialization.
"""
from __future__ import annotations

import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_inspector_candidate_enforcement_policy as _enforcement
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local
from . import stock_road_model_geometry as _geometry
from . import stock_road_sharp_turn_policy as _sharp
from .procedural_infrastructure import (
    paved_miter_angle_degrees,
    paved_wedge_angle_degrees,
)


STOCK_CURVE_SOURCE_CORRIDOR_METRES = 1.25
ORDINARY_PAVED_OVERLAP_METRES = 0.45
ORDINARY_PAVED_OVERLAP_TANGENT_TOLERANCE_DEGREES = 0.75
ORDINARY_PAVED_OVERLAP_ENDPOINT_TOLERANCE_METRES = 0.15
ORDINARY_PAVED_OVERLAP_MAXIMUM_END_SHORTFALL_METRES = 0.55

_ORIGINAL_NATIVE_T = None
_ORIGINAL_CHAIN = None
_ORIGINAL_FIT = None
_INSTALLED = False


def _normalised_path(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


def _generated_dirt_model(model_path: str) -> bool:
    """Reject any future world-local dirt P3D without confusing gravel with dirt."""

    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    return "\\i\\" in path and filename.startswith("dirt") and filename.endswith(".p3d")


def _family_first_native_t(incidents):
    """Choose the stock T from road families before judging source headings.

    Heading error is intentionally not an acceptance gate here. This function is
    used only while the all-or-nothing relaxation transaction is planning. The
    connector policy then inserts measured connector-aligned approach points,
    applies its two-metre lateral bound and obstacle checks, and the transaction
    finally reruns the strict 0.90-degree matcher on the edited geometry.
    """

    if len(incidents) != 3:
        return None
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    main_family = incidents[first].family
    if main_family is None or incidents[second].family != main_family:
        return None
    branch_family = incidents[branch].family
    if branch_family is None:
        return None
    model = _junction._T_JUNCTION_MODELS.get((main_family, branch_family))
    if model is None:
        return None

    main_a = _junction._heading(incidents[first].direction)
    main_b = _junction._heading(incidents[second].direction)
    branch_heading = _junction._heading(incidents[branch].direction)
    fits = []
    for actual_zero, actual_180 in ((main_a, main_b), (main_b, main_a)):
        rotation, maximum_error = _junction._best_rotation(
            (
                (0.0, actual_zero),
                (180.0, actual_180),
                (_candidate.MEASURED_T_BRANCH_LOCAL_HEADING_DEGREES, branch_heading),
            )
        )
        fits.append((maximum_error, rotation))
    maximum_error, rotation = min(fits)
    return _junction._NativeJunction(
        model,
        rotation,
        maximum_error,
        main_family,
    )


def _stock_native_t_dispatch(incidents):
    """Select stock T first while planning, then enforce the strict final fit."""

    if _ORIGINAL_NATIVE_T is None:
        raise RuntimeError("stock paved/dirt policy is not installed")

    if _enforcement._contains_generated_gravel(incidents):
        return _ORIGINAL_NATIVE_T(incidents)

    if _local._PLANNING_RELAXED_JUNCTION.get():
        return _family_first_native_t(incidents)

    return _candidate._measured_native_t_junction(incidents)


def _stock_paved_straight(piece) -> bool:
    match = _geometry.stock_straight_match(str(piece.model_path))
    return match is not None and match.group("family").casefold() in {"sil", "asf", "kos"}


def _heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _tangent_compatible_straight_chain(fitted) -> bool:
    """Only overlap a chain whose existing ordinary seams are already healthy."""

    if len(fitted) < 2 or any(not _stock_paved_straight(item[0]) for item in fitted):
        return False
    for previous, current in zip(fitted, fitted[1:]):
        previous_heading = _heading(previous[1], previous[2])
        current_heading = _heading(current[1], current[2])
        if _p._heading_difference(previous_heading, current_heading) > (
            ORDINARY_PAVED_OVERLAP_TANGENT_TOLERANCE_DEGREES + 1.0e-9
        ):
            return False
        if math.dist(previous[2], current[1]) > (
            ORDINARY_PAVED_OVERLAP_ENDPOINT_TOLERANCE_METRES + 1.0e-9
        ):
            return False
    return True


def _valid_overlapped_straight_chain(fitted) -> bool:
    if len(fitted) < 2 or any(not _stock_paved_straight(item[0]) for item in fitted):
        return False
    seam_limit = (
        ORDINARY_PAVED_OVERLAP_METRES
        + ORDINARY_PAVED_OVERLAP_ENDPOINT_TOLERANCE_METRES
    )
    for previous, current in zip(fitted, fitted[1:]):
        if _p._heading_difference(
            _heading(previous[1], previous[2]),
            _heading(current[1], current[2]),
        ) > ORDINARY_PAVED_OVERLAP_TANGENT_TOLERANCE_DEGREES + 1.0e-9:
            return False
        if math.dist(previous[2], current[1]) > seam_limit + 1.0e-9:
            return False
    return True


def _overlapped_stock_chain(
    measure,
    pieces,
    fitted,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    """Re-space healthy stock straights with real 0.45 m axial overlap."""

    if not _tangent_compatible_straight_chain(fitted):
        return fitted
    sequence = tuple(item[0] for item in fitted)
    if any(not _stock_paved_straight(piece) for piece in sequence):
        return fitted

    current = float(start_distance)
    rebuilt = []
    final_progress = current
    for index, piece in enumerate(sequence):
        endpoint = measure.chord_endpoint(
            current,
            float(piece.length_metres),
            float(maximum_end_distance),
        )
        if endpoint is None:
            return fitted
        end_distance, end_x, end_z, _chord_heading = endpoint
        start_x, start_z, _start_heading = measure.point(current)
        rebuilt.append((piece, (start_x, start_z), (end_x, end_z)))
        final_progress = float(end_distance)
        if index + 1 < len(sequence):
            next_progress = final_progress - ORDINARY_PAVED_OVERLAP_METRES
            if next_progress <= current + 0.05:
                return fitted
            current = next_progress

    if final_progress < (
        float(minimum_end_distance)
        - ORDINARY_PAVED_OVERLAP_MAXIMUM_END_SHORTFALL_METRES
    ):
        return fitted
    if final_progress > float(maximum_end_distance) + 1.0e-6:
        return fitted
    if not _valid_overlapped_straight_chain(tuple(rebuilt)):
        return fitted
    return tuple(rebuilt)


def _stock_overlap_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock paved overlap policy is not installed")
    fitted = _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    return _overlapped_stock_chain(
        measure,
        pieces,
        fitted,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )


def _generated_road_model(model_path: str) -> bool:
    """Return True only for generated paved/dirt road models forbidden in output."""

    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    return (
        filename == "paved_fill.p3d"
        or paved_miter_angle_degrees(filename) is not None
        or paved_wedge_angle_degrees(filename) is not None
        or _generated_dirt_model(path)
    )


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    """Run the complete road fitter and reject generated paved/dirt survivors."""

    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock paved/dirt final guard is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    forbidden = tuple(
        obj for obj in report.objects if _generated_road_model(str(obj.model_path))
    )
    if forbidden:
        sample = ", ".join(sorted({str(obj.model_path) for obj in forbidden})[:6])
        raise ValueError(
            "stock paved/dirt policy violation: generated paved or dirt road P3Ds "
            f"survived final fitting ({len(forbidden)} objects; {sample})"
        )
    return report


def install_stock_road_stock_assets_only_policy() -> None:
    """Install final stock paved/dirt rules while preserving generated gravel."""

    global _ORIGINAL_NATIVE_T, _ORIGINAL_CHAIN, _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    if not _enforcement._FINAL_INSTALLED or not _curve_usage._INSTALLED:
        raise RuntimeError("candidate enforcement and curve usage must install first")

    _ORIGINAL_NATIVE_T = _enforcement._junction._native_t_junction

    _ORIGINAL_CHAIN = _curve_usage._ORIGINAL_CHAIN
    _curve_usage._ORIGINAL_CHAIN = _stock_overlap_chain

    _ORIGINAL_FIT = _p.fit_road_objects
    _generator.fit_road_objects = _fit

    _enforcement._junction._native_t_junction = _stock_native_t_dispatch

    _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES = STOCK_CURVE_SOURCE_CORRIDOR_METRES

    _INSTALLED = True
