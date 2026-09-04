from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / "tools" / "apply_global_country_visual_cleanup.py"
text = helper.read_text(encoding="utf-8")

old = '''    paths = sorted(country_dir.glob("*.json"))
    if len(paths) != 249:
        raise RuntimeError(f"expected 249 country profiles, found {len(paths)}")

    context_count = 0
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
'''
new = '''    documents: list[tuple[Path, dict]] = []
    for path in sorted(country_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("iso_alpha2"):
            documents.append((path, document))
    if len(documents) != 249:
        raise RuntimeError(f"expected 249 country profiles, found {len(documents)}")

    context_count = 0
    for path, document in documents:
'''
if text.count(old) != 1:
    raise RuntimeError("permanent populator country-count block changed")
text = text.replace(old, new, 1)
if text.count("    return len(paths), context_count\n") != 1:
    raise RuntimeError("permanent populator return count changed")
text = text.replace("    return len(paths), context_count\n", "    return len(documents), context_count\n", 1)

old_test = '''    paths = sorted(COUNTRY_DIR.glob("*.json"))
    assert len(paths) == 249
    context_count = 0
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
'''
new_test = '''    profiles = []
    for path in sorted(COUNTRY_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("iso_alpha2"):
            profiles.append((path, document))
    assert len(profiles) == 249
    context_count = 0
    for path, document in profiles:
'''
if text.count(old_test) != 1:
    raise RuntimeError("global test country-count block changed")
text = text.replace(old_test, new_test, 1)

helper.write_text(text, encoding="utf-8", newline="\n")
print("Fixed global cleanup to count actual country profiles rather than metadata JSON files.")
