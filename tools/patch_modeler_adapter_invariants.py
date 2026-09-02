from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected patch context not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    upgrade = root / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py"

    replace_once(
        upgrade,
        "    detail_texture = foundation_texture or wall_texture\n",
        "    # Secondary architecture must never borrow a painted window atlas.\n"
        "    # Polygon-native facade tests and, more importantly, actual models rely\n"
        "    # on those atlas UVs being reserved for wall bands. Foundation material\n"
        "    # is preferred; roof material is the safe fallback when no plinth exists.\n"
        "    detail_texture = foundation_texture or roof_texture or wall_texture\n",
    )
    replace_once(
        upgrade,
        '''            texture=wall_texture,\n        )\n        canopy_y = min(max(2.25, eave_y - 0.55), 2.65)\n''',
        '''            texture=detail_texture,\n        )\n        canopy_y = min(max(2.25, eave_y - 0.55), 2.65)\n''',
    )
    replace_once(
        upgrade,
        '''                y0=0.08,\n                y1=canopy_y,\n                texture=wall_texture,\n            )\n\n    if plan.balcony_count:\n''',
        '''                y0=0.08,\n                y1=canopy_y,\n                texture=detail_texture,\n            )\n\n    if plan.balcony_count:\n''',
    )
    replace_once(
        upgrade,
        "                texture=foundation_texture or wall_texture,\n",
        "                texture=detail_texture,\n",
    )
    replace_once(
        upgrade,
        '''    return _pb._Lod(\n        tuple(points),\n        tuple(normals),\n        tuple(faces),\n        lod.resolution,\n        lod.mass_per_point,\n        lod.selections,\n        lod.properties,\n    )\n''',
        '''    added_points = len(points) - len(lod.points)\n    added_faces = len(faces) - len(lod.faces)\n    selections = tuple(\n        _pb._NamedSelection(\n            selection.name,\n            selection.point_weights + bytes(added_points),\n            selection.face_flags + bytes(added_faces),\n        )\n        for selection in lod.selections\n    )\n    mass_per_point = lod.mass_per_point\n    if mass_per_point and added_points:\n        mass_per_point = mass_per_point + (0.0,) * added_points\n    return _pb._Lod(\n        tuple(points),\n        tuple(normals),\n        tuple(faces),\n        lod.resolution,\n        mass_per_point,\n        selections,\n        lod.properties,\n    )\n''',
    )

    milestone = root / "tests" / "test_milestone8.py"
    replace_once(
        milestone,
        '        self.assertEqual((tiny.family, tiny.height_m, tiny.regional_style), ("outbuilding", 3.0, "sweden_red"))\n',
        '        self.assertEqual((tiny.family, tiny.height_m, tiny.regional_style), ("outbuilding", 3.0, "swedish_wood"))\n',
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
