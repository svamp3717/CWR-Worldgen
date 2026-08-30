# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical Resistance road catalogue mirrored from WrpTool configuration.

WrpTool's ``RoadDef.xml`` groups the Resistance ``o\\road`` straight, curve and
terminator P3Ds into the four stock road families. Its ``objects.ini`` also
classifies the purpose-built Resistance T/crossroad models as roads. Keep that
inventory in one place so the generator and Road Inspector cannot quietly drift
onto different model lists.

Only models whose connector geometry is measured elsewhere are exposed as
placeable junctions. ``kr_new_kos.p3d`` is retained in the WrpTool road inventory
as a special stock road object, but is deliberately not treated as a junction
until its Memory-LOD connectors are measured.
"""
from __future__ import annotations

from dataclasses import dataclass


STOCK_ROAD_FAMILIES = ("sil", "asf", "kos", "ces")

# Resistance road-family members listed by WrpTool RoadDef.xml.
WRPTOOL_ROAD_FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    "sil": (
        r"o\road\sil25.p3d",
        r"o\road\sil12.p3d",
        r"o\road\sil6.p3d",
        r"o\road\sil10 25.p3d",
        r"o\road\sil10 50.p3d",
        r"o\road\sil10 75.p3d",
        r"o\road\sil10 100.p3d",
        r"o\road\sil6konec.p3d",
    ),
    "asf": (
        r"o\road\asf25.p3d",
        r"o\road\asf12.p3d",
        r"o\road\asf6.p3d",
        r"o\road\asf10 25.p3d",
        r"o\road\asf10 50.p3d",
        r"o\road\asf10 75.p3d",
        r"o\road\asf10 100.p3d",
        r"o\road\asf6konec.p3d",
    ),
    "kos": (
        r"o\road\kos25.p3d",
        r"o\road\kos12.p3d",
        r"o\road\kos6.p3d",
        r"o\road\kos10 25.p3d",
        r"o\road\kos10 50.p3d",
        r"o\road\kos10 75.p3d",
        r"o\road\kos10 100.p3d",
        r"o\road\kos6konec.p3d",
    ),
    "ces": (
        r"o\road\ces25.p3d",
        r"o\road\ces12.p3d",
        r"o\road\ces6.p3d",
        r"o\road\ces10 25.p3d",
        r"o\road\ces10 50.p3d",
        r"o\road\ces10 75.p3d",
        r"o\road\ces10 100.p3d",
        r"o\road\ces6konec.p3d",
    ),
}

# Purpose-built Resistance junction P3Ds listed in WrpTool objects.ini.
WRPTOOL_T_JUNCTION_MODELS: dict[tuple[str, str], str] = {
    ("sil", "sil"): r"o\road\kr_new_sil_sil_t.p3d",
    ("sil", "asf"): r"o\road\kr_new_sil_asf_t.p3d",
    ("sil", "ces"): r"o\road\kr_new_sil_ces_t.p3d",
    ("sil", "kos"): r"o\road\kr_new_sil_kos_t.p3d",
    ("asf", "asf"): r"o\road\kr_new_asf_asf_t.p3d",
    ("asf", "ces"): r"o\road\kr_new_asf_ces_t.p3d",
    ("asf", "sil"): r"o\road\kr_new_asf_sil_t.p3d",
    ("kos", "kos"): r"o\road\kr_new_kos_kos_t.p3d",
    ("kos", "sil"): r"o\road\kr_new_kos_sil_t.p3d",
}
WRPTOOL_X_JUNCTION_MODELS: dict[str, str] = {
    "sil": r"o\road\kr_new_silxsil.p3d",
}
WRPTOOL_SPECIAL_ROAD_MODELS: tuple[str, ...] = (
    r"o\road\kr_new_kos.p3d",
)

WRPTOOL_NATIVE_JUNCTION_MODELS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (*WRPTOOL_T_JUNCTION_MODELS.values(), *WRPTOOL_X_JUNCTION_MODELS.values())
    )
)
WRPTOOL_RESISTANCE_ROAD_MODELS: tuple[str, ...] = tuple(
    dict.fromkeys(
        model
        for values in WRPTOOL_ROAD_FAMILY_MODELS.values()
        for model in values
    )
) + WRPTOOL_NATIVE_JUNCTION_MODELS + WRPTOOL_SPECIAL_ROAD_MODELS


def normalise_stock_road_path(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


@dataclass(frozen=True, slots=True)
class NativeJunctionAsset:
    model_path: str
    kind: str
    main_family: str
    branch_family: str | None


_NATIVE_JUNCTIONS_BY_PATH: dict[str, NativeJunctionAsset] = {}
for (main, branch), path in WRPTOOL_T_JUNCTION_MODELS.items():
    _NATIVE_JUNCTIONS_BY_PATH[normalise_stock_road_path(path)] = NativeJunctionAsset(
        path, "t", main, branch
    )
for family, path in WRPTOOL_X_JUNCTION_MODELS.items():
    _NATIVE_JUNCTIONS_BY_PATH[normalise_stock_road_path(path)] = NativeJunctionAsset(
        path, "x", family, None
    )

_WRPTOOL_ROADS_BY_PATH = {
    normalise_stock_road_path(path): path for path in WRPTOOL_RESISTANCE_ROAD_MODELS
}


def native_junction_asset(model_path: str) -> NativeJunctionAsset | None:
    return _NATIVE_JUNCTIONS_BY_PATH.get(normalise_stock_road_path(model_path))


def is_wrptool_resistance_road(model_path: str) -> bool:
    return normalise_stock_road_path(model_path) in _WRPTOOL_ROADS_BY_PATH


def wrptool_family_models(family: str) -> tuple[str, ...]:
    return WRPTOOL_ROAD_FAMILY_MODELS.get(str(family).casefold(), ())
