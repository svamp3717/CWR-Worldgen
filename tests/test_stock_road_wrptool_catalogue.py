# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from cwr_worldgen.model import WorldObject
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_wrp_catalogue as _catalogue
from cwr_worldgen import stock_road_wrptool_catalogue_policy as _policy


_EXPECTED_T = {
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


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str):
    return _junction._Incident(
        _direction(heading_degrees),
        family,
        rf"o\road\{family}25.p3d",
    )


def test_wrptool_catalogue_contains_every_resistance_t_and_crossroad() -> None:
    assert _catalogue.WRPTOOL_T_JUNCTION_MODELS == _EXPECTED_T
    assert _catalogue.WRPTOOL_X_JUNCTION_MODELS == {
        "sil": r"o\road\kr_new_silxsil.p3d"
    }
    assert r"o\road\kr_new_kos.p3d" in _catalogue.WRPTOOL_SPECIAL_ROAD_MODELS


def test_generator_can_select_every_wrptool_t_combination() -> None:
    for (main, branch), expected_model in _EXPECTED_T.items():
        incidents = (
            _incident(0.0, main),
            _incident(180.0, main),
            _incident(90.0, branch),
        )
        native = _junction._native_t_junction(incidents)
        assert native is not None, (main, branch)
        assert native.model_path.casefold() == expected_model.casefold()


def test_generator_can_select_wrptool_sil_crossroad() -> None:
    incidents = tuple(
        _incident(heading, "sil") for heading in (0.0, 90.0, 180.0, 270.0)
    )
    native = _junction._native_x_junction(incidents)
    assert native is not None
    assert native.model_path.casefold() == r"o\road\kr_new_silxsil.p3d"


@dataclass(frozen=True)
class _Report:
    objects: tuple[WorldObject, ...]
    junction_cap_objects: int


def test_wrptool_native_model_wins_even_if_legacy_cap_family_differs(monkeypatch) -> None:
    node = (20.0, 30.0)
    old = WorldObject(1, r"o\road\sil6.p3d", node[0], 0.035, node[1], 0.0, 0.0)
    native = _junction._NativeJunction(
        r"o\road\kr_new_asf_sil_t.p3d",
        0.0,
        0.0,
        "asf",
    )
    report = _Report((old,), 1)

    monkeypatch.setattr(
        _junction,
        "_junction_incidents",
        lambda dataset, projection, spec: {
            _junction._p._road_node_key(node): (node, native)
        },
    )
    monkeypatch.setattr(
        _junction,
        "_native_junction_object",
        lambda old, native, elevations, spec: replace(
            old, model_path=native.model_path, heading_degrees=native.heading_degrees
        ),
    )

    result = _policy._replace_stock_junction_caps(
        report,
        dataset=object(),
        projection=object(),
        elevations=(),
        spec=object(),
    )

    assert result.objects[0].model_path.casefold() == r"o\road\kr_new_asf_sil_t.p3d"
