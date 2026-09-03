from pathlib import Path

path = Path(__file__).resolve().parent / "finish_osm_house_modeler_fidelity.py"
text = path.read_text(encoding="utf-8")
old = '''    _append_eave_overhang(
        points, normals, faces, key, eave_y=eave_y,
        reference_texture=reference_texture,
    )
'''
new = '''    _append_eave_overhang(
        points, normals, faces, key, eave_y=eave_y,
        # For legacy/unit-level bare texture paths, classify the soffit with the
        # roof rather than the wall so old facade UV inspectors do not mistake
        # eave boxes for window-bearing wall faces. Real generated addon paths
        # still resolve to the dedicated modeler material set.
        reference_texture=roof_texture or reference_texture,
    )
'''
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise RuntimeError("eave texture compatibility call not found")
path.write_text(text, encoding="utf-8", newline="\n")
