from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "cwr_worldgen" / "house_style_catalogue.py"
text = path.read_text(encoding="utf-8")

constants = '''_LEGACY_IDENTIFIER_BY_STYLE_IDENTIFIER = {
    "mediterranean_europe": "western_europe",
    "eastern_europe_balkans": "eastern_europe",
    "north_africa": "africa",
    "west_africa": "africa",
    "east_africa": "africa",
    "central_southern_africa": "africa",
}
_LEGACY_DEFAULT_STYLE_IDENTIFIERS = frozenset({
    "western_europe", "eastern_europe_balkans", "west_africa",
})


'''
if "_LEGACY_DEFAULT_STYLE_IDENTIFIERS =" not in text:
    marker = "def _load_profiles() -> tuple[RegionProfile, ...]:\n"
    if marker not in text:
        raise RuntimeError("house style loader anchor not found")
    text = text.replace(marker, constants + marker, 1)

old = '        legacy_identifier = str(document.get("legacy_identifier", "")).strip()\n'
new = '''        legacy_identifier = str(
            document.get("legacy_identifier")
            or _LEGACY_IDENTIFIER_BY_STYLE_IDENTIFIER.get(style_identifier.casefold(), "")
        ).strip()
'''
if old in text:
    text = text.replace(old, new, 1)

old_default = '            legacy_default=bool(document.get("legacy_default", False)),\n'
new_default = '''            legacy_default=bool(
                document.get("legacy_default", False)
                or style_identifier.casefold() in _LEGACY_DEFAULT_STYLE_IDENTIFIERS
            ),
'''
if old_default in text:
    text = text.replace(old_default, new_default, 1)

# The first patch may already have expanded the line. Keep it valid, but the
# constants above now exist before REGION_PROFILES is initialized.
path.write_text(text, encoding="utf-8", newline="\n")
