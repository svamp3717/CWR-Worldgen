"""Manual generation seed: reproducible world placements across PBO formats."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cwr_worldgen.cli import _parser
from cwr_worldgen.gui import (
    WorldgenGui, build_milestone9_command, default_gui_values,
    defaults_with_recent_source, load_gui_state,
)
from cwr_worldgen.milestone9 import Milestone9Spec, build_milestone9, resolve_world_seed
from cwr_worldgen.osm import (
    BboxProjection, _forest_single_tree_candidates,
    _selected_farmland_fence_field_keys,
)
from cwr_island_studio.oprw import open_island_document


def test_blank_seed_keeps_automatic_fingerprint_and_manual_seed_ignores_packaging(
    tmp_path: Path,
) -> None:
    source = Milestone9Spec(source_dir=tmp_path)
    assert resolve_world_seed(source, "abc123") == "milestone9:abc123"
    assert resolve_world_seed(source, "def456") == "milestone9:def456"
    manual = replace(source, world_seed="  314159  ")
    assert resolve_world_seed(manual, "abc123") == "314159"
    assert resolve_world_seed(manual, "def456") == "314159"
    for wrp in (False, True):
        for config in (False, True):
            for odol in (False, True):
                mode = replace(
                    manual, binarize_world=wrp, binarize_config=config,
                    odol_buildings=odol,
                )
                assert resolve_world_seed(mode, "abc123") == "314159"
    for invalid in ("x" * 129, "bad\nseed", "hello\tworld"):
        with pytest.raises(ValueError, match="seed"):
            replace(source, world_seed=invalid).validate()
    with pytest.raises(ValueError, match="seed"):
        replace(source, world_seed=123).validate()


def test_seed_controls_tree_candidates_and_fence_fields_consistently() -> None:
    projection = BboxProjection.create((0.0, 0.0, 0.03, 0.03), 3200.)
    latitude, longitude = 0.01, 0.01
    p1 = _forest_single_tree_candidates(
        "314159", projection, 12, 14, latitude, longitude, 24.0,
    )
    p2 = _forest_single_tree_candidates(
        "314159", projection, 12, 14, latitude, longitude, 24.0,
    )
    p3 = _forest_single_tree_candidates(
        "271828", projection, 12, 14, latitude, longitude, 24.0,
    )
    assert p1 == p2 and p1 != p3

    farms = [
        SimpleNamespace(
            osm_key=f"way/{i}", tags={"landuse": "farmland"},
            polygons=(object(),),
        )
        for i in range(40)
    ]
    keys1 = _selected_farmland_fence_field_keys(farms, "314159", 45.0)
    keys2 = _selected_farmland_fence_field_keys(farms, "314159", 45.0)
    keys3 = _selected_farmland_fence_field_keys(farms, "271828", 45.0)
    assert len(keys1) == 18
    assert keys1 == keys2 and keys1 != keys3


def test_gui_seed_is_optional_persisted_and_forwarded_to_cli(tmp_path: Path) -> None:
    values = default_gui_values()
    assert values["world_seed"] == ""
    values.update(
        source_dir=str(tmp_path), output=str(tmp_path / "out"),
        name="same_world", display_name="Same World",
    )
    automatic = build_milestone9_command(values, python="python")
    assert "--world-seed" not in automatic
    values["world_seed"] = "314159"
    manual = build_milestone9_command(values, python="python")
    assert manual[manual.index("--world-seed") + 1] == "314159"
    args = _parser().parse_args(manual[manual.index("milestone9"):])
    assert args.world_seed == "314159"

    class Value:
        def __init__(self) -> None:
            self.value = ""
            self.callbacks = []
        def get(self):
            return self.value
        def set(self, value):
            self.value = value
            for callback in self.callbacks:
                callback()
        def trace_add(self, mode, callback):
            assert mode == "write"
            self.callbacks.append(callback)

    gui = object.__new__(WorldgenGui)
    gui.state_path = tmp_path / "state.json"
    variable = Value()
    gui.vars = {"world_seed": variable}
    WorldgenGui._install_world_seed_persistence(gui)
    variable.set("314159")
    state = load_gui_state(gui.state_path)
    assert state["world_seed"] == "314159"
    assert defaults_with_recent_source(default_gui_values(), state)["world_seed"] == "314159"
    variable.set("")
    assert load_gui_state(gui.state_path)["world_seed"] == ""


def test_generation_seed_and_editable_wrp_do_not_change_with_binarized_packaging(
    tmp_path: Path,
) -> None:
    # A small real frozen source includes roads, water, forests, buildings.
    # Generates twice with the same seed but different game PBO formats.
    import test_milestone8 as milestone8_tests

    source = milestone8_tests.Milestone8BuildTests()._source(tmp_path / "source")
    spec = Milestone9Spec(
        source_dir=source,
        name="cwr_seed_test", display_name="Seed Test",
        world_seed="314159",
        solver_iterations=3,
        world_edge_blend_cells=2,
        forest_maximum_block_relief=3.0,
        ground_texture_profile="generated",
        surface_overview_size=128,
        surface_texture_size=32,
        verify_regeneration=False,
    )
    editable = build_milestone9(tmp_path / "editable", spec)
    packed = build_milestone9(
        tmp_path / "binarized",
        replace(spec, binarize_world=True, binarize_config=True),
    )
    assert editable.wrp_path.read_bytes() == packed.wrp_path.read_bytes()
    assert editable.pbo_path.read_bytes() != packed.pbo_path.read_bytes()
    with open_island_document(editable.wrp_path) as document:
        assert document.object_count > 0
    for result in (editable, packed):
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["world_seed"]["mode"] == "manual"
        assert manifest["world_seed"]["requested"] == "314159"
        assert manifest["world_seed"]["effective"] == "314159"
        assert manifest["surface_visual_pass"]["deterministic_seed"] == "314159"
    with open_island_document(packed.wrp_path) as document:
        assert document.object_count > 0
