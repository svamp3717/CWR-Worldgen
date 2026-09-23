from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen.asset_mapping import default_osm_asset_mapping
from cwr_worldgen.fast_wrp_write_policy import _fast_write_rvw4
from cwr_worldgen.parking_surface_policy import apply_parking_textures
from cwr_worldgen.runway_surface_policy import runway_texture_cell_indices
from cwr_worldgen.sports_pitch_surface_policy import apply_sports_pitch_textures
from cwr_worldgen.surface_pass import (
    MILESTONE9_MATERIALS,
    surface_texture_wire_paths,
    write_surface_textures,
)
from cwr_worldgen.terrain import GROUND_TEXTURE_PROFILES, ground_texture_path
from cwr_worldgen import wrp


def test_none_ground_profile_uses_blank_material_paths_and_writes_no_paas(
    tmp_path: Path,
) -> None:
    assert "none" in GROUND_TEXTURE_PROFILES
    assert ground_texture_path("blank_world", "g", "none") == ""
    assert surface_texture_wire_paths("blank_world", "none") == (
        "",
    ) * len(MILESTONE9_MATERIALS)
    assert write_surface_textures(tmp_path, "blank_world", "none", "seed", 128) == ()
    assert not tuple(tmp_path.rglob("*.paa"))


def test_rvw4_writers_accept_intentional_blank_texture_slots(tmp_path: Path) -> None:
    scalar = tmp_path / "scalar.wrp"
    fast = tmp_path / "fast.wrp"
    texture_paths = (r"blank_world\data\d.paa", "")
    texture_indices = (1, 1, 1, 1)
    elevations = (0.0, 0.0, 0.0, 0.0)

    wrp.write_rvw4(
        scalar,
        2,
        2,
        elevations,
        texture_indices,
        texture_paths,
        (),
        height_scale=0.01,
        renumber_object_ids=True,
    )
    _fast_write_rvw4(
        wrp.write_rvw4,
        wrp,
        fast,
        2,
        2,
        elevations,
        texture_indices,
        texture_paths,
        (),
        height_scale=0.01,
        renumber_object_ids=True,
    )

    assert scalar.read_bytes() == fast.read_bytes()
    summary = wrp.inspect_rvw4(scalar)
    assert summary.texture_slots[0] == r"blank_world\data\d.paa"
    assert summary.texture_slots[1] == ""
    assert summary.texture_index_counts[1] == 4
    assert "" not in summary.texture_paths


def test_none_ground_profile_disables_generated_surface_painting(tmp_path: Path) -> None:
    spec = SimpleNamespace(ground_texture_profile="none")
    indices = (1, 1, 1, 1)
    paths = (r"blank_world\data\d.paa", "")

    assert runway_texture_cell_indices(None, None, spec) == ()
    assert apply_parking_textures(tmp_path, None, None, spec, indices, paths) == (
        indices,
        paths,
        (),
    )
    assert apply_sports_pitch_textures(tmp_path, None, None, spec, indices, paths) == (
        indices,
        paths,
        (),
    )


def test_zero_vegetation_limits_remove_forest_and_mapped_tree_asset_rules() -> None:
    spec = SimpleNamespace(
        name="blank_world",
        max_forest_objects=0,
        maximum_mapped_tree_objects=0,
        procedural_gravel_roads=False,
        barriers_enabled=False,
        bus_stops_enabled=False,
        cemeteries_enabled=False,
        wetland_reeds_enabled=False,
    )
    mapping = default_osm_asset_mapping(spec, 9)
    rule_ids = {rule.rule_id for rule in mapping.rules}

    assert "primary-forest" not in rule_ids
    assert "mapped-individual-trees" not in rule_ids
