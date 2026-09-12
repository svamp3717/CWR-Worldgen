from __future__ import annotations

from cwr_worldgen import cli
from cwr_worldgen import parking_surface_policy as parking
from cwr_worldgen import runway_surface_policy as runway
from cwr_worldgen import sports_pitch_surface_policy as sports
from cwr_worldgen.dynamic_surface_controls import (
    DYNAMIC_PARKING_KEY,
    DYNAMIC_RUNWAYS_KEY,
    DYNAMIC_SPORTS_KEY,
    DYNAMIC_SURFACE_OPTIONS,
    _append_disable_flags,
    _defaults_with_dynamic_surfaces,
    _dynamic_enabled,
)


def test_dynamic_surface_controls_default_on_and_emit_only_disabled_flags() -> None:
    values = _defaults_with_dynamic_surfaces({})
    assert values[DYNAMIC_PARKING_KEY] is True
    assert values[DYNAMIC_SPORTS_KEY] is True
    assert values[DYNAMIC_RUNWAYS_KEY] is True
    assert _append_disable_flags(["python", "milestone9"], values) == ["python", "milestone9"]

    values[DYNAMIC_PARKING_KEY] = False
    values[DYNAMIC_RUNWAYS_KEY] = False
    command = _append_disable_flags(["python", "milestone9"], values)
    assert "--no-dynamic-parking-lots" in command
    assert "--no-dynamic-runways" in command
    assert "--no-dynamic-sports-fields" not in command


def test_milestone9_parser_accepts_all_dynamic_surface_disable_flags() -> None:
    args = cli._parser().parse_args([
        "milestone9",
        "--output", "build/test",
        "--source-dir", "source-data/test",
        "--no-dynamic-parking-lots",
        "--no-dynamic-sports-fields",
        "--no-dynamic-runways",
    ])
    assert args.dynamic_parking_lots is False
    assert args.dynamic_sports_fields is False
    assert args.dynamic_runways is False


def test_disabled_surface_gates_return_no_dynamic_geometry(monkeypatch) -> None:
    environments = {key: environment for key, _label, _flag, environment in DYNAMIC_SURFACE_OPTIONS}
    monkeypatch.setenv(environments[DYNAMIC_PARKING_KEY], "0")
    monkeypatch.setenv(environments[DYNAMIC_SPORTS_KEY], "false")
    monkeypatch.setenv(environments[DYNAMIC_RUNWAYS_KEY], "off")

    assert parking._parking_geometries(None, None) == ()
    assert sports._pitch_geometries(None, None) == ()
    assert runway.runway_texture_cell_indices(None, None, None) == ()


def test_dynamic_enabled_defaults_to_true_and_understands_false_values(monkeypatch) -> None:
    environment = "CWR_TEST_DYNAMIC_SURFACE"
    monkeypatch.delenv(environment, raising=False)
    assert _dynamic_enabled(environment)
    monkeypatch.setenv(environment, "no")
    assert not _dynamic_enabled(environment)
    monkeypatch.setenv(environment, "1")
    assert _dynamic_enabled(environment)
