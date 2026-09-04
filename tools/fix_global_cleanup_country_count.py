from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / "tools" / "apply_global_country_visual_cleanup.py"
text = helper.read_text(encoding="utf-8")

old = '''    paths = sorted(country_dir.glob("*.json"))\n    if len(paths) != 249:\n        raise RuntimeError(f"expected 249 country profiles, found {len(paths)}")\n\n    context_count = 0\n    for path in paths:\n        document = json.loads(path.read_text(encoding="utf-8"))\n'''
new = '''    documents: list[tuple[Path, dict]] = []\n    for path in sorted(country_dir.glob("*.json")):\n        document = json.loads(path.read_text(encoding="utf-8"))\n        if document.get("iso_alpha2"):\n            documents.append((path, document))\n    if len(documents) != 249:\n        raise RuntimeError(f"expected 249 country profiles, found {len(documents)}")\n\n    context_count = 0\n    for path, document in documents:\n'''
if text.count(old) != 1:
    raise RuntimeError("permanent populator country-count block changed")
text = text.replace(old, new, 1)
text = text.replace('    return len(paths), context_count\\n', '    return len(documents), context_count\\n', 1)

old_test = '''    paths = sorted(COUNTRY_DIR.glob("*.json"))\n    assert len(paths) == 249\n    context_count = 0\n    for path in paths:\n        document = json.loads(path.read_text(encoding="utf-8"))\n'''
new_test = '''    profiles = []\n    for path in sorted(COUNTRY_DIR.glob("*.json")):\n        document = json.loads(path.read_text(encoding="utf-8"))\n        if document.get("iso_alpha2"):\n            profiles.append((path, document))\n    assert len(profiles) == 249\n    context_count = 0\n    for path, document in profiles:\n'''
if text.count(old_test) != 1:
    raise RuntimeError("global test country-count block changed")
text = text.replace(old_test, new_test, 1)

helper.write_text(text, encoding="utf-8", newline="\n")
print("Fixed global cleanup to count actual country profiles rather than metadata JSON files.")
