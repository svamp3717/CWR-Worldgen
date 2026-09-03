# Branch-local texture migration helper.
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDINGS = ROOT / "src/cwr_worldgen/procedural_buildings.py"
FIDELITY = ROOT / "src/cwr_worldgen/osm_house_modeler_fidelity.py"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_buildings() -> None:
    text = BUILDINGS.read_text(encoding="utf-8")
    start = text.index("    def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:\n")
    prefix, body = text[:start], text[start:]
    import_anchor = "        selected = sorted(self._usage)\n        model_assets: list[GeneratedBuildingAsset] = []\n"
    import_block = "        selected = sorted(self._usage)\n        from .osm_house_modeler_texture_bridge import (\n            modeler_door_texture_image,\n            modeler_foundation_texture_image,\n            modeler_front_texture_image,\n            modeler_interior_wall_texture_image,\n            modeler_open_wall_texture_image,\n            modeler_roof_texture_image,\n            modeler_wall_texture_image,\n        )\n        model_assets: list[GeneratedBuildingAsset] = []\n"
    body = replace_once(body, import_anchor, import_block, "texture bridge imports")
    # Replace the two names that contain ``_wall_texture_image`` first so the
    # final base-wall replacement cannot accidentally match inside them.
    replacements = (
        (
            "_open_wall_texture_image(\n                            family,",
            "modeler_open_wall_texture_image(\n                            family,",
        ),
        (
            "_interior_wall_texture_image(\n                            family,",
            "modeler_interior_wall_texture_image(\n                            family,",
        ),
        (
            "_wall_texture_image(\n                            family,",
            "modeler_wall_texture_image(\n                            family,",
        ),
        (
            "_foundation_texture_image(self.texture_size)",
            "modeler_foundation_texture_image(self.texture_size)",
        ),
        (
            "_door_texture_image(\n                        self.texture_size,",
            "modeler_door_texture_image(\n                        self.texture_size,",
        ),
        (
            "_front_texture_image(\n                            family,",
            "modeler_front_texture_image(\n                            family,",
        ),
        (
            "_roof_texture_image(\n                            roof, size=self.texture_size,",
            "modeler_roof_texture_image(\n                            roof, size=self.texture_size,",
        ),
    )
    for old, new in replacements:
        if new in body:
            continue
        count = body.count(old)
        if count != 1:
            raise RuntimeError(f"production texture replacement {old!r}: found {count}")
        body = body.replace(old, new, 1)
    body = replace_once(
        body,
        '                "roof_pitch_degrees": (key.roof_pitch_degrees or self.roof_pitch_degrees),\n',
        '                "roof_pitch_degrees": self.roof_pitch_degrees,\n',
        "empty-building roof pitch",
    )
    BUILDINGS.write_text(prefix + body, encoding="utf-8")


def patch_fidelity() -> None:
    text = FIDELITY.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from .paa import write_rgb_dxt1_paa\n",
        "from .paa import write_rgb_dxt1_paa\nfrom .osm_house_modeler_texture_bridge import modeler_detail_texture_image\n",
        "detail bridge import",
    )
    text = replace_once(
        text,
        "        write_rgb_dxt1_paa(path, _DETAIL_IMAGE_FACTORIES[kind](size))\n",
        "        write_rgb_dxt1_paa(path, modeler_detail_texture_image(kind, size))\n",
        "detail material generator",
    )
    FIDELITY.write_text(text, encoding="utf-8")


def patch_notices() -> None:
    text = NOTICES.read_text(encoding="utf-8")
    marker = "`src/cwr_worldgen/osm_house_modeler_styles.py`"
    if "osm_house_modeler_textures.py" in text:
        return
    if marker not in text:
        raise RuntimeError("third-party notice style marker not found")
    text = text.replace(
        marker,
        marker + " and `src/cwr_worldgen/osm_house_modeler_textures.py`",
        1,
    )
    NOTICES.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_buildings()
    patch_fidelity()
    patch_notices()
