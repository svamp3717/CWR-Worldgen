# SPDX-License-Identifier: GPL-3.0-or-later
"""Make Road Inspector candidate decisions authoritative at selection and output.

Two older layers can otherwise undo the candidate policy. The mixed gravel/paved
selector legitimately needs its historical paved-overlay fallback for generated
gravel, while the many composed fit wrappers can promote a rigid junction again
after an earlier candidate audit. Keep generated gravel on its existing path,
apply the Inspector tolerance to stock paved/native choices, and run one final
candidate ownership pass over the exact report that will be serialized to WRP.

A strict final selector and a useful connector-relaxation planner are not the
same thing. During the transaction's planning phase a near-straight paved T may
provisionally use the wider historical heading budget so the approach points can
be moved onto the measured Memory-LOD connectors. The transaction then re-runs
the strict selector against that edited geometry before committing anything.
A T whose through road is itself turning never receives that planning exception;
Road Inspector's turning-cap candidate keeps those approaches authoritative.
"""
from __future__ import annotations

from . import generator as _generator
from . import gravel_asphalt_transition_policy as _mixed
from . import playability as _p
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local
from . import stock_road_measured_junction_policy as _measured
from . import stock_road_native_junction_ownership_policy as _ownership
from . import stock_road_relaxation_transaction_policy as _transaction


# A rigid T has a straight 0/180 main axis. If the source through road already
# turns more than this at the node, bending both approaches onto that axis merely
# moves the visible defect away from the centre. The current Inspector report
# starts flagging this failure just above this bound.
MAXIMUM_NATIVE_THROUGH_TURN_DEGREES = 1.25

_ORIGINAL_NATIVE_X = None
_ORIGINAL_FINAL_FIT = None
_SELECTOR_INSTALLED = False
_FINAL_INSTALLED = False


def _contains_generated_gravel(incidents) -> bool:
    return any(
        _p.is_generated_gravel_road_model(str(incident.model_path))
        for incident in incidents
    )


def _through_turn_degrees(incidents) -> float:
    if len(incidents) != 3:
        return 180.0
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return 180.0
    first, second = pair
    first_heading = _junction._heading(incidents[first].direction)
    second_heading = _junction._heading(incidents[second].direction)
    separation = _junction._angular_distance(first_heading, second_heading)
    return abs(180.0 - separation)


def _stock_ces_mixed_t(incidents) -> bool:
    layered = _mixed._layered_mixed_t_components(incidents)
    if layered is None:
        return False
    _family, _first, _second, _unpaved, generated_gravel = layered
    return not generated_gravel


def _planning_tolerance_degrees(incidents) -> float | None:
    """Return the provisional T tolerance allowed only inside a transaction."""

    if _local._same_family_paved_t(incidents):
        return float(_local.MAXIMUM_PAVED_T_HEADING_ERROR_DEGREES)
    if _stock_ces_mixed_t(incidents):
        return float(_mixed.MAXIMUM_STOCK_CES_NATIVE_HEADING_ERROR_DEGREES)
    return None


def _measured_native_t_with_limit(incidents, limit_degrees: float):
    """Evaluate measured -X T geometry with a temporary planning-only limit."""

    previous = _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES
    try:
        _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = float(limit_degrees)
        return _measured._native_t_junction(incidents)
    finally:
        _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = previous


def _candidate_native_t_dispatch(incidents):
    """Preserve gravel fallback and separate provisional from final T matching."""

    if _contains_generated_gravel(incidents):
        return _mixed._native_t_junction(incidents)

    # A genuine bend through the intersection follows the Inspector's low-fill
    # candidate rather than forcing a straight rigid main axis.
    if _through_turn_degrees(incidents) > MAXIMUM_NATIVE_THROUGH_TURN_DEGREES:
        return None

    if _transaction._PLANNING_RELAXED_JUNCTION.get():
        limit = _planning_tolerance_degrees(incidents)
        if limit is not None:
            return _measured_native_t_with_limit(incidents, limit)

    # Outside the planning ContextVar, only geometry already within the visible
    # 0.90-degree connector tolerance can select a native stock T.
    return _candidate._measured_native_t_junction(incidents)


def _candidate_native_x_dispatch(incidents):
    """Accept a stock paved X only inside the Inspector connector tolerance."""

    if _ORIGINAL_NATIVE_X is None:
        raise RuntimeError("Inspector candidate X selector is not installed")
    native = _ORIGINAL_NATIVE_X(incidents)
    if native is None:
        return None
    if not all(
        getattr(incident, "family", None) in _candidate._PAVED_FAMILIES
        for incident in incidents
    ):
        return native
    if float(native.maximum_heading_error_degrees) > (
        _candidate.INSPECTOR_NATIVE_CONNECTOR_TOLERANCE_DEGREES + 1.0e-9
    ):
        return None
    return native


def install_stock_road_inspector_candidate_selector_policy() -> None:
    """Install final T/X selection semantics from Road Inspector candidates."""

    global _ORIGINAL_NATIVE_X, _SELECTOR_INSTALLED
    if _SELECTOR_INSTALLED:
        return
    if not _candidate._INSTALLED:
        raise RuntimeError("Inspector candidate policy must install first")

    _ORIGINAL_NATIVE_X = _junction._native_x_junction
    _junction._native_t_junction = _candidate_native_t_dispatch
    _junction._native_x_junction = _candidate_native_x_dispatch
    _SELECTOR_INSTALLED = True


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _ORIGINAL_FINAL_FIT is None:
        raise RuntimeError("Inspector candidate final enforcement is not installed")
    report = _ORIGINAL_FINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return report

    # This is intentionally the last road-fit mutation. If an older wrapper has
    # reintroduced a skewed native cap or a centre-crossing approach, apply the
    # same native-or-low-fill and connector-ownership decision the Inspector
    # reports against the final WRP geometry.
    return _candidate._native_owner_realign(
        report,
        dataset,
        projection,
        elevations,
        spec,
    )


def install_stock_road_inspector_candidate_final_policy() -> None:
    """Run Inspector candidate enforcement after every older road fit wrapper."""

    global _ORIGINAL_FINAL_FIT, _FINAL_INSTALLED
    if _FINAL_INSTALLED:
        return
    if not _SELECTOR_INSTALLED:
        raise RuntimeError("Inspector candidate selector policy must install first")
    if not _ownership._INSTALLED:
        raise RuntimeError("native junction ownership policy must install first")

    _ORIGINAL_FINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _FINAL_INSTALLED = True
