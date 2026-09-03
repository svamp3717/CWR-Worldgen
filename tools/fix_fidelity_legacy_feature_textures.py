from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py"
text = path.read_text(encoding="utf-8")

anchor = '''    detail_texture = foundation_texture or roof_texture or wall_texture
    eave_y = _roof_base_y(key, roof_pitch_degrees)
    reference_texture = wall_texture or roof_texture or foundation_texture
'''
replacement = '''    detail_texture = foundation_texture or roof_texture or wall_texture
    eave_y = _roof_base_y(key, roof_pitch_degrees)
    reference_texture = wall_texture or roof_texture or foundation_texture
    generated_material_paths = "\\\\" in str(reference_texture or "")

    def feature_material(material, kind: str, legacy_texture: str) -> str:
        if generated_material_paths:
            return material_texture_path(reference_texture, material, kind)
        return legacy_texture

    def feature_detail(kind: str, legacy_texture: str) -> str:
        if generated_material_paths:
            return detail_texture_path(reference_texture, kind)
        return legacy_texture
'''
if anchor in text:
    text = text.replace(anchor, replacement, 1)
elif replacement not in text:
    raise RuntimeError("feature material fallback anchor not found")

replacements = {
    'stair_texture = material_texture_path(reference_texture, stair_spec.get("material"), "masonry")':
        'stair_texture = feature_material(stair_spec.get("material"), "masonry", foundation_texture or roof_texture or detail_texture)',
    'porch_texture = material_texture_path(reference_texture, porch_spec.get("material"), "wood")':
        'porch_texture = feature_material(porch_spec.get("material"), "wood", foundation_texture or roof_texture or detail_texture)',
    'balcony_texture = detail_texture_path(reference_texture, "balcony")':
        'balcony_texture = feature_detail("balcony", roof_texture or foundation_texture or detail_texture)',
    'chimney_texture = material_texture_path(reference_texture, chimney_spec.get("material"), "masonry")':
        'chimney_texture = feature_material(chimney_spec.get("material"), "masonry", foundation_texture or roof_texture or detail_texture)',
    'rainwater_texture = material_texture_path(reference_texture, rainwater_spec.get("material"), "metal")':
        'rainwater_texture = feature_material(rainwater_spec.get("material"), "metal", roof_texture or foundation_texture or detail_texture)',
}
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError(f"legacy material replacement missing: {old}")

path.write_text(text, encoding="utf-8", newline="\n")
