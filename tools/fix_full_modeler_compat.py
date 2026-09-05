from __future__ import annotations

from pathlib import Path
import re

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

# The original compatibility helper could place this block below REGION_PROFILES,
# which is too late because _load_profiles() consults it during import. Remove any
# generated copy and put exactly one copy immediately before the loader.
block_pattern = re.compile(
    r'_LEGACY_IDENTIFIER_BY_STYLE_IDENTIFIER = \{.*?\}\n'
    r'_LEGACY_DEFAULT_STYLE_IDENTIFIERS = frozenset\(\{.*?\}\)\n\n\n',
    re.S,
)
text = block_pattern.sub("", text)
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

# Normalize either the untouched line or the long line produced by the first helper.
text = re.sub(
    r'            legacy_default=bool\([^\n]*\),\n',
    '''            legacy_default=bool(
                document.get("legacy_default", False)
                or style_identifier.casefold() in _LEGACY_DEFAULT_STYLE_IDENTIFIERS
            ),
''',
    text,
    count=1,
)

path.write_text(text, encoding="utf-8", newline="\n")
