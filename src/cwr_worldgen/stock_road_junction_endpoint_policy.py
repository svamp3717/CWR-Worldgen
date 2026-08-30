# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep late exact paved fits on the effective junction endpoint window.

The base stock-road fitter reserves a short approach trim around each junction.
The local-fit policy may expand paved approaches back underneath a generic cap,
while purpose-built native T/X junctions now keep that measured connector trim.
Several late exact-curve policies can return their recovered stock actions
directly, however, and historically made that decision from the caller's raw
window before the final junction policy had a chance to adjust it.

Lundby32 exposed the expansion side of that failure: a successful exact fit could
stop at an obsolete trimmed endpoint even though a generic cap required node
coverage. Native-junction ownership exposes the opposite case: an exact straight
fit can continue all the way to the logical node even though a purpose-built T/X
owns the centre and its approaches must stop at the measured connector.

Treat the current quality window as authoritative in both directions. Retry the
same complete exact policy stack when it expands a covered endpoint, and also
when it trims a native-junction endpoint inward. No helper or overlapping road
object is added.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import road_quality_policy as _quality


MINIMUM_ENDPOINT_RECOVERY_IMPROVEMENT_METRES = 0.05
_WINDOW_EPSILON_METRES = 1.0e-6

_ORIGINAL_CHAIN = None
_INSTALLED = False


def _effective_window(
    measure,
    pieces,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
):
    context = _quality._CONTEXT.get()
    if context is None:
        return (
            float(start_distance),
            float(preferred_end_distance),
            float(minimum_end_distance),
            float(maximum_end_distance),
        )
    return tuple(
        float(value)
        for value in _quality._quality_window(
            measure,
            pieces,
            start_distance,
            preferred_end_distance,
            minimum_end_distance,
            maximum_end_distance,
            context,
        )
    )


def _covered_endpoint_errors(
    measure,
    fitted,
    *,
    recover_start: bool,
    recover_end: bool,
    effective_start: float,
    effective_end: float,
) -> tuple[float, ...]:
    if not fitted:
        return (math.inf,) if recover_start or recover_end else ()

    errors = []
    if recover_start:
        desired = measure.point(effective_start)[:2]
        errors.append(math.dist(tuple(fitted[0][1]), desired))
    if recover_end:
        desired = measure.point(effective_end)[:2]
        errors.append(math.dist(tuple(fitted[-1][2]), desired))
    return tuple(errors)


def _junction_endpoint_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road junction-endpoint policy is not installed")

    raw_start = float(start_distance)
    raw_preferred = float(preferred_end_distance)
    baseline = _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )

    effective = _effective_window(
        measure,
        pieces,
        start_distance,
        preferred_end_distance,
        minimum_end_distance,
        maximum_end_distance,
    )
    effective_start, effective_preferred, effective_minimum, effective_maximum = effective

    trim_start = effective_start > raw_start + _WINDOW_EPSILON_METRES
    trim_end = effective_preferred < raw_preferred - _WINDOW_EPSILON_METRES
    if trim_start or trim_end:
        # A native T/X owns the centre up to its measured connectors. Exact
        # wrappers underneath this policy are allowed to optimize the selected
        # stock sequence, not to ignore that final physical ownership window.
        return _ORIGINAL_CHAIN(
            measure,
            pieces,
            start_distance=effective_start,
            preferred_end_distance=effective_preferred,
            minimum_end_distance=effective_minimum,
            maximum_end_distance=effective_maximum,
        )

    recover_start = effective_start < raw_start - _WINDOW_EPSILON_METRES
    recover_end = effective_preferred > raw_preferred + _WINDOW_EPSILON_METRES
    if not recover_start and not recover_end:
        return baseline

    baseline_errors = _covered_endpoint_errors(
        measure,
        baseline,
        recover_start=recover_start,
        recover_end=recover_end,
        effective_start=effective_start,
        effective_end=effective_preferred,
    )
    if baseline_errors and max(baseline_errors) <= MINIMUM_ENDPOINT_RECOVERY_IMPROVEMENT_METRES:
        return baseline

    recovered = _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=effective_start,
        preferred_end_distance=effective_preferred,
        minimum_end_distance=effective_minimum,
        maximum_end_distance=effective_maximum,
    )
    if not recovered:
        return baseline

    recovered_errors = _covered_endpoint_errors(
        measure,
        recovered,
        recover_start=recover_start,
        recover_end=recover_end,
        effective_start=effective_start,
        effective_end=effective_preferred,
    )
    if not recovered_errors:
        return baseline

    baseline_max = max(baseline_errors) if baseline_errors else math.inf
    recovered_max = max(recovered_errors)
    if recovered_max + MINIMUM_ENDPOINT_RECOVERY_IMPROVEMENT_METRES < baseline_max:
        return recovered
    return baseline


def install_stock_road_junction_endpoint_policy() -> None:
    """Install endpoint-window enforcement outside every late exact wrapper."""

    global _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _junction_endpoint_chain
    _INSTALLED = True
