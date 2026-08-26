# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep generated gravel visually gravel-to-paved at mixed T junctions.

Generated gravel is classified as the stock ``ces`` family only as a connector-
geometry surrogate so the Resistance T-junction matcher can orient the native
mesh. Rendering that literal paved/dirt model inserts a brown dirt strip between
the generated gravel branch and the paved main road.

The same-surface T meshes share the measured 6.25 m connector layout with their
mixed-surface siblings. Replace only the visible central model with the paved
main-family T; the generated gravel chain still terminates at the same measured
branch connector.
"""
from __future__ import annotations

from dataclasses import replace

from . import playability as _p
from . import stock_road_junction_policy as _junction
from . import stock_road_measured_junction_policy as _measured

_ORIGINAL_NATIVE_T = None
_INSTALLED = False


def _native_t_junction(incidents):
    if _ORIGINAL_NATIVE_T is None:
        raise RuntimeError("gravel/paved transition policy is not installed")
    native = _ORIGINAL_NATIVE_T(incidents)
    if native is None or len(incidents) != 3:
        return native

    gravel = [
        incident
        for incident in incidents
        if _p.is_generated_gravel_road_model(incident.model_path)
    ]
    if len(gravel) != 1:
        return native

    main_family = native.cap_family
    paved_apron = _junction._T_JUNCTION_MODELS.get((main_family, main_family))
    if paved_apron is None:
        return native
    return replace(native, model_path=paved_apron)


def install_gravel_asphalt_transition_policy() -> None:
    global _ORIGINAL_NATIVE_T, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_NATIVE_T = _measured._native_t_junction
    _measured._native_t_junction = _native_t_junction
    _junction._native_t_junction = _native_t_junction
    _INSTALLED = True
