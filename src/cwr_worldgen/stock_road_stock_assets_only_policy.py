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
from . import stock_road_inspector_candidate_enforcement_policy as _enforcement
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_junction_endpoint_policy as _endpoint
from . import stock_road_junction_policy as _junction
from . import stock_road_kodiak_reference_policy as _kodiak
from . import stock_road_model_geometry as _geometry
from . import stock_road_relaxation_transaction_policy as _transaction
from . import stock_road_sharp_turn_policy as _sharp
from .procedural_infrastructure import (
    paved_miter_angle_degrees,
    paved_wedge_angle_degrees,
)


STOCK_CURVE_SOURCE_CORRIDOR_METRES = 1.25
ORDINARY_PAVED_OVERLAP_METRES = 0.45
ORDINARY_PAVED_OVERLAP_TANGENT_TOLERANCE_DEGREES = 0.75
ORDINARY_PAVED_OVERLAP_ENDPOINT_TOLERANCE_METRES = 0.15

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

    # Generated gravel keeps its existing junction path. The stock-first planner
    # below is only for stock paved and stock-ces combinations.
    if _enforcement._contains_generated_gravel(incidents):
        return _ORIGINAL_NATIVE_T(incidents)

    if _transaction._PLANNING_RELAXED_JUNCTION.get():
        return _family_first_native_t(incidents)

    # Final placement judges the geometry that actually survived the obstacle
    # transaction. Raw OSM headings are no longer the reason a valid T disappears.
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
    """Re-space a healthy paved straight chain with bounded axial overlap.

    The guide chord for each stock slab is 0.45 m shorter than the actual P3D.
    The model is not scaled, so its physical ends extend about 0.225 m beyond
    each guide end and neighbouring slabs overlap longitudinally. The fitter may
    add another normal stock piece when needed, so the chain still reaches its
    required endpoint instead of accumulating an uncovered length deficit.
    """

    if not _tangent_compatible_straight_chain(fitted):
        return fitted

    ordered = tuple(sorted(
        pieces,
        key=lambda piece: (-float(piece.length_metres), str(piece.model_path).casefold()),
    ))
    if not ordered or any(not _stock_paved_straight(piece) for piece in ordered):
        return fitted

    original_sequence = [item[0] for item in fitted]
    shortest_effective = max(
        0.50,
        min(float(piece.length_metres) for piece in ordered) - ORDINARY_PAVED_OVERLAP_METRES,
    )
    maximum_objects = max(
        len(original_sequence) + 2,
        int(math.ceil((float(maximum_end_distance) - float(start_distance)) / shortest_effective)) + 2,
    )

    current = float(start_distance)
    rebuilt = []
    for index in range(maximum_objects):
        # The physical last model extends half the overlap beyond its guide end.
        if (
            current + ORDINARY_PAVED_OVERLAP_METRES * 0.5
            >= float(preferred_end_distance) - 0.05
            and current + ORDINARY_PAVED_OVERLAP_METRES * 0.5
            >= float(minimum_end_distance) - 0.05
        ):
            break

        remaining = max(0.0, float(preferred_end_distance) - current)
        if index < len(original_sequence):
            piece = original_sequence[index]
        else:
            preferred = _p._road_piece_sequence(remaining, ordered)
            piece = preferred[0] if preferred else ordered[-1]

        effective = max(
            0.50,
            float(piece.length_metres) - ORDINARY_PAVED_OVERLAP_METRES,
        )
        endpoint = measure.chord_endpoint(
            current,
            effective,
            float(maximum_end_distance),
        )
        if endpoint is None:
            return fitted
        end_distance, end_x, end_z, _chord_heading = endpoint
        start_x, start_z, _start_heading = measure.point(current)
        rebuilt.append((piece, (start_x, start_z), (end_x, end_z)))
        if end_distance <= current + 1.0e-7:
            return fitted
        current = float(end_distance)

    if (
        current + ORDINARY_PAVED_OVERLAP_METRES * 0.5
        < float(minimum_end_distance) - 0.05
    ):
        return fitted
    if not _tangent_compatible_straight_chain(tuple(rebuilt)):
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
    """Run the inner fitter and reject generated paved/dirt survivors."""

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
    if not _enforcement._FINAL_INSTALLED:
        raise RuntimeError("Inspector candidate final policy must install first")
    if not _endpoint._INSTALLED or not _kodiak._INSTALLED:
        raise RuntimeError("endpoint and Kodiak road policies must install first")

    _ORIGINAL_NATIVE_T = _enforcement._junction._native_t_junction

    # Keep junction-endpoint enforcement as the public outer chain wrapper. Put
    # the ordinary overlap stage immediately inside it so effective T/X endpoint
    # windows still get the final word. This preserves the established wrapper
    # contract while changing the actual straight-chain construction.
    _ORIGINAL_CHAIN = _endpoint._ORIGINAL_CHAIN
    _endpoint._ORIGINAL_CHAIN = _stock_overlap_chain

    # Likewise keep Kodiak's final ownership cleanup outermost. The guard runs
    # immediately inside it, after all older generation layers but before Kodiak
    # removes redundant stock node stubs. Kodiak never adds road models, so no
    # generated paved/dirt P3D can appear after this check.
    _ORIGINAL_FIT = _kodiak._ORIGINAL_FIT
    _kodiak._ORIGINAL_FIT = _fit

    # During planning, choose the family-compatible WrpTool T first. The normal
    # connector relaxation transaction decides whether its measured arms can be
    # reached safely; final selection remains strict.
    _enforcement._junction._native_t_junction = _stock_native_t_dispatch

    # Exact paved curves may smooth the source inside a road-width-bounded
    # corridor. The curve-first policy additionally checks the obstacle index.
    _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES = STOCK_CURVE_SOURCE_CORRIDOR_METRES

    # Leave the established public outer wrappers in place.
    _p._stock_piece_chain = _endpoint._junction_endpoint_chain
    _p.fit_road_objects = _kodiak._fit
    _generator.fit_road_objects = _kodiak._fit
    _INSTALLED = True
