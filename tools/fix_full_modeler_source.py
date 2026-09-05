from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "cwr_worldgen" / "procedural_buildings.py"
text = path.read_text(encoding="utf-8")

old = '''        material_slot = (
            int.from_bytes(sha256(token.encode("utf-8")).digest()[:2], "big") % 16
            if "|" in token else 0
        )
        value = (roof_index * 16 + material_slot) * self.texture_variants
'''
new = '''        material_text = token.casefold()
        if "|" not in token:
            material_slot = 0
        elif "tile" in material_text or "clay" in material_text or "terracotta" in material_text:
            material_slot = 1
        elif "metal" in material_text or "steel" in material_text or "zinc" in material_text:
            material_slot = 2
        elif "slate" in material_text:
            material_slot = 3
        elif "thatch" in material_text or "reed" in material_text:
            material_slot = 4
        elif "concrete" in material_text or "cement" in material_text:
            material_slot = 5
        elif "bitumen" in material_text or "asphalt" in material_text:
            material_slot = 6
        elif "wood" in material_text or "timber" in material_text or "shingle" in material_text:
            material_slot = 7
        elif "stone" in material_text:
            material_slot = 8
        else:
            material_slot = 16 + int.from_bytes(sha256(token.encode("utf-8")).digest()[:2], "big") % 48
        # 64 slots per roof form still fit the historic three-character base36
        # filename budget even with a maximum 20-character world name.
        value = (roof_index * 64 + material_slot) * self.texture_variants
'''
if old not in text and new not in text:
    raise RuntimeError("roof material slot anchor not found")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8", newline="\n")
