# SPDX-License-Identifier: GPL-3.0-or-later
"""Allow native stock junctions to absorb bounded mixed-surface skew.

The ordinary native-junction matcher intentionally accepts only near-orthogonal
T/X nodes. A mixed paved/gravel T can still use the measured stock T connector
geometry when the mismatch stays inside the existing road surfaces. This policy
enables that one narrow exception without making all stock intersections
globally more permissive.

Generated gravel is treated as the dirt (``ces``) family only as a connector-
geometry surrogate during matching. A later transition policy keeps the visible
central apron paved, so no brown dirt road is inserted between generated gravel
and the paved main road. A relaxed match is attempted only when exactly one
generated-gravel arm joins two same-family paved/asphalt/cobble arms, and the
maximum connector heading error remains within 18 degrees. The measured native
connector radius is 6.25 m, so the maximum lateral correction is about 1.93 m,
inside the 2.30 m half-width of the generated gravel road.
"""
from __future__ import annotations

import re

from . import stock_road_junction_policy as _junction

MAXIMUM_RELAXED_JUNCTION_HEADING_ERROR_DEGREES = 18.0
_INSTALLED = False
_ORIGINAL_FAMILY = None
_ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS = None

_GENERATED_GRAVEL_FILENAME = re.compile(
    r"^gravel(?:25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d$",
    re.IGNORECASE,
)


def _is_generated_gravel_model(model_path: str) -> bool:
    filename = str(model_path).replace("/", "\\").rsplit("\\", 1)[-1]
    return _GENERATED_GRAVEL_FILENAME.fullmatch(filename) is not None


def _family_with_generated_gravel(model_path: str) -> str | None:
    if _ORIGINAL_FAMILY is None:
        raise RuntimeError("stock road skew policy is not installed")
    family = _ORIGINAL_FAMILY(model_path)
    if family is not None:
        return family
    if _is_generated_gravel_model(model_path):
        # Resistance has no generated-gravel T family. Reuse ces only as the
        # connector-geometry surrogate; the visible apron is replaced with the
        # paved main-family T model by gravel_asphalt_transition_policy.
        return "ces"
    return None


def _eligible_relaxed_mixed_t(incidents) -> bool:
    if len(incidents) != 3:
        return False
    gravel = [incident for incident in incidents if _is_generated_gravel_model(incident.model_path)]
    if len(gravel) != 1:
        return False
    stock_families = [
        incident.family
        for incident in incidents
        if not _is_generated_gravel_model(incident.model_path)
    ]
    return (
        len(stock_families) == 2
        and stock_families[0] == stock_families[1]
        and stock_families[0] in {"sil", "asf", "kos"}
    )


def _native_junction_with_bounded_mixed_skew(incidents):
    if _ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS is None:
        raise RuntimeError("stock road skew policy is not installed")

    native = _ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS(incidents)
    if native is not None or not _eligible_relaxed_mixed_t(incidents):
        return native

    original_limit = _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES
    try:
        _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = (
            MAXIMUM_RELAXED_JUNCTION_HEADING_ERROR_DEGREES
        )
        return _ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS(incidents)
    finally:
        _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = original_limit


def install_stock_road_skew_policy() -> None:
    global _ORIGINAL_FAMILY, _ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FAMILY = _junction._family
    _ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS = _junction._native_junction_for_incidents
    _junction._family = _family_with_generated_gravel
    _junction._native_junction_for_incidents = _native_junction_with_bounded_mixed_skew
    _INSTALLED = True
