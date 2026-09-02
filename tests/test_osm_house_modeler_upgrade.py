# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace

from cwr_worldgen import procedural_buildings as pb
from cwr_worldgen import osm_house_modeler_upgrade as upgrade
from cwr_worldgen.osm_house_modeler_upgrade import detail_plan_for_key


def _house(*, interiors: bool = False, regional_style: str = "sweden_red") -> pb.BuildingVariantKey:
    return pb.BuildingVariantKey(
        "residential",
        "gabled",
        10.0,
        14.0,
        6.0,
        foundation_depth_m=0.5,
        regional_style=regional_style,
        interiors=interiors,
    )


def test_default_variants_remain_byte_compatible() -> None:
    key = _house(regional_style="default")
    plan = detail_plan_for_key(key, foundation_depth=0.5)
    assert not plan.enabled

    original = upgrade._ORIGINAL_VISUAL_LOD
    assert original is not None
    baseline = original(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        None,
        "foundation.paa",
        0.5,
    )
    upgraded = pb._visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        None,
        "foundation.paa",
        0.5,
    )
    assert upgraded == baseline


def test_exterior_upgrade_reads_legacy_positional_foundation_arguments() -> None:
    key = _house(interiors=False)
    plan = detail_plan_for_key(key, foundation_depth=0.5)
    # Exterior entrances get the modeler-style terrain staircase whenever a
    # visible foundation is present, so this assertion is probability-free.
    assert plan.stairs

    original = upgrade._ORIGINAL_VISUAL_LOD
    assert original is not None
    baseline = original(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        None,
        "foundation.paa",
        0.5,
    )
    upgraded = pb._visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        None,
        "foundation.paa",
        0.5,
    )
    assert len(upgraded.points) > len(baseline.points)
    assert len(upgraded.faces) > len(baseline.faces)
    assert upgraded.resolution == baseline.resolution == 1.0
    assert {face.texture for face in upgraded.faces} <= {
        "wall.paa",
        "roof.paa",
        "foundation.paa",
        "",
    }


def test_enterable_upgrade_keeps_collision_sensitive_details_safe() -> None:
    key = _house(interiors=True)
    plan = detail_plan_for_key(key, foundation_depth=0.5)
    assert not plan.stairs
    assert plan.balcony_count == 0
    # This deterministic Swedish key receives safe secondary architecture, so
    # the enterable visual path is genuinely exercised instead of only testing
    # the two safety exclusions above.
    assert plan.porch or plan.chimney_count or plan.gutters

    original = upgrade._ORIGINAL_VISUAL_LOD
    assert original is not None
    baseline = original(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        None,
        "foundation.paa",
        0.5,
        0.0,
        "inside.paa",
    )
    upgraded = pb._visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        35.0,
        None,
        "foundation.paa",
        0.5,
        0.0,
        "inside.paa",
    )
    assert len(upgraded.points) > len(baseline.points)
    assert len(upgraded.faces) > len(baseline.faces)
    # Detail geometry must not replace CWR's selections/properties, which are
    # later used by the animated door and engine-specific LOD writer.
    assert upgraded.selections == baseline.selections
    assert upgraded.properties == baseline.properties


def test_polygon_native_exterior_receives_modeler_details() -> None:
    key = replace(
        _house(interiors=False),
        width_m=10.0,
        length_m=14.0,
        footprint_vertices=(
            (-5.0, -7.0),
            (5.0, -7.0),
            (5.0, 7.0),
            (-5.0, 7.0),
        ),
        entrance_edge=0,
        entrance_fraction=0.5,
    )
    plan = detail_plan_for_key(key, foundation_depth=0.5)
    assert plan.stairs

    original = upgrade._ORIGINAL_POLYGON_VISUAL_LOD
    assert original is not None
    baseline = original(
        key,
        "wall.paa",
        "roof.paa",
        roof_pitch_degrees=35.0,
        foundation_texture="foundation.paa",
        foundation_depth=0.5,
        plain_wall_texture="plain.paa",
    )
    upgraded = pb._polygon_native_visual_lod(
        key,
        "wall.paa",
        "roof.paa",
        roof_pitch_degrees=35.0,
        foundation_texture="foundation.paa",
        foundation_depth=0.5,
        plain_wall_texture="plain.paa",
    )
    assert len(upgraded.points) > len(baseline.points)
    assert len(upgraded.faces) > len(baseline.faces)
    assert upgraded.resolution == baseline.resolution == 1.0
