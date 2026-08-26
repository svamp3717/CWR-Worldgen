# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep generated gravel visually gravel-to-paved at mixed T junctions.

Generated gravel was temporarily classified as the stock ``ces`` family so the
native T-junction matcher could reuse a Resistance junction mesh. That made the
central branch visibly turn into brown dirt before reaching the paved road.

The stock same-surface T meshes share the same measured connector geometry, so a
mixed generated-gravel branch can use the paved main-road T model as its apron.
The generated gravel chain still terminates at the measured branch connector;
only the small central junction surface stays paved instead of becoming dirt.
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
    _INSTALLED = True
