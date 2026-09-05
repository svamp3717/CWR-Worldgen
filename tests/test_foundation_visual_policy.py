from __future__ import annotations

from cwr_worldgen import procedural_buildings as buildings
from cwr_worldgen.foundation_visual_policy import (
    FOUNDATION_SKIN_PROJECTION_M,
    _offset_polygon_foundation_skin,
)


def test_rectangular_visible_foundation_projects_beyond_wall_footprint() -> None:
    points, faces = buildings._add_foundation_skirt(
        (),
        (),
        half_width=5.0,
        half_length=7.0,
        texture="world\\d\\f.paa",
        depth=0.5,
        top_height=0.1,
    )
    assert len(points) == 8
    assert len(faces) == 4
    assert max(abs(point[0]) for point in points) >= 5.0 + FOUNDATION_SKIN_PROJECTION_M - 1.0e-6
    assert max(abs(point[2]) for point in points) >= 7.0 + FOUNDATION_SKIN_PROJECTION_M - 1.0e-6


def test_polygon_foundation_skin_moves_only_vertical_foundation_face() -> None:
    foundation = "world\\d\\f.paa"
    wall = "world\\d\\w00.paa"
    points = (
        (-2.0, -0.5, -3.0), (-2.0, 0.1, -3.0), (2.0, 0.1, -3.0), (2.0, -0.5, -3.0),
        (-2.0, 0.0, -3.0), (-2.0, 2.5, -3.0), (2.0, 2.5, -3.0), (2.0, 0.0, -3.0),
    )
    normals = ((0.0, 0.0, -1.0),)
    foundation_face = buildings._Face(
        foundation,
        tuple((index, 0, 0.0, 0.0) for index in (0, 1, 2, 3)),
    )
    wall_face = buildings._Face(
        wall,
        tuple((index, 0, 0.0, 0.0) for index in (4, 5, 6, 7)),
    )
    lod = buildings._Lod(points, normals, (foundation_face, wall_face), 1.0)

    shifted = _offset_polygon_foundation_skin(
        lod,
        foundation_texture=foundation,
        foundation_depth=0.5,
        foundation_top=0.1,
    )

    for index in range(4):
        assert shifted.points[index][2] == points[index][2] - FOUNDATION_SKIN_PROJECTION_M
    assert shifted.points[4:] == points[4:]


def test_foundation_policy_revises_only_building_model_cache_namespace() -> None:
    payload = {"world_name": "test", "variant": {"family": "residential"}}
    old = buildings.cache_key(
        "procedural-building-model-v49-robust-polygon-roof-triangulation",
        payload,
    )
    revised_direct = buildings.cache_key(
        "procedural-building-model-v50-foundation-skin-offset",
        payload,
    )
    assert old == revised_direct

    texture_payload = {"family": "residential", "texture_size": 128}
    assert buildings.cache_key(
        "procedural-building-wall-modeler-v1-cwa78", texture_payload
    ) == buildings.cache_key(
        "procedural-building-wall-modeler-v1-cwa78", texture_payload
    )
