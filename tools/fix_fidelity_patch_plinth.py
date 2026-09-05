from pathlib import Path

path = Path(__file__).resolve().parent / "finish_osm_house_modeler_fidelity.py"
text = path.read_text(encoding="utf-8")
old = '''    old_foundation = \'\'\'    foundation_top = plinth_height + (\n        FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0\n    )\n\'\'\'\n    new_foundation = \'\'\'    style_plinth = max(0.0, float(getattr(key, "visible_plinth_m", 0.0) or 0.0))\n    foundation_top = plinth_height + max(\n        style_plinth,\n        FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0,\n    )\n\'\'\'\n    text = replace_all_required(text, old_foundation, new_foundation, "rectangular visible plinth", minimum=2)\n'''
new = '''    text = replace_once(\n        text,\n        \'\'\'        foundation_top = plinth_height + (\n            FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0\n        )\n\'\'\',\n        \'\'\'        style_plinth = max(0.0, float(getattr(key, "visible_plinth_m", 0.0) or 0.0))\n        foundation_top = plinth_height + max(\n            style_plinth,\n            FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0,\n        )\n\'\'\',\n        "nested rectangular visible plinth",\n    )\n    text = replace_once(\n        text,\n        \'\'\'    foundation_top = plinth_height + (\n        FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0\n    )\n\'\'\',\n        \'\'\'    style_plinth = max(0.0, float(getattr(key, "visible_plinth_m", 0.0) or 0.0))\n    foundation_top = plinth_height + max(\n        style_plinth,\n        FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0,\n    )\n\'\'\',\n        "gabled rectangular visible plinth",\n    )\n'''
if old not in text:
    if '"nested rectangular visible plinth"' not in text:
        raise RuntimeError("fidelity plinth patch block not found")
else:
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8", newline="\n")
