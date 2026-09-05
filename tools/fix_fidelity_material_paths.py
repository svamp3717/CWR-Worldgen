from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src" / "cwr_worldgen" / "osm_house_modeler_fidelity.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
'''DETAIL_MATERIAL_CODES: Mapping[str, str] = {
    "masonry": "ma",
    "wood": "wo",
    "metal": "me",
    "balcony": "ba",
    "glass": "gl",
}
''',
'''DETAIL_MATERIAL_CODES: Mapping[str, str] = {
    "masonry": "qma",
    "wood": "qwo",
    "metal": "qme",
    "balcony": "qba",
    "glass": "qgl",
}
''',
1,
)
text = text.replace(
'''    reference = str(reference_texture or "")
    prefix = reference.split("\\\\", 1)[0] if "\\\\" in reference else reference
    if not prefix:
        prefix = "cwr"
    return rf"{prefix}\\d\\{code}.paa"
''',
'''    reference = str(reference_texture or "")
    # Unit-level/legacy callers often pass a bare ``wall.paa`` instead of a
    # world-relative CWA path. Keep those calls byte-compatible and reserve the
    # dedicated material set for real generated addon paths.
    if "\\\\" not in reference:
        return reference
    prefix = reference.split("\\\\", 1)[0]
    return rf"{prefix}\\d\\{code}.paa"
''',
1,
)
if '"masonry": "qma"' not in text or 'if "\\\\" not in reference:' not in text:
    raise RuntimeError("fidelity material path patch did not apply")
path.write_text(text, encoding="utf-8", newline="\n")
