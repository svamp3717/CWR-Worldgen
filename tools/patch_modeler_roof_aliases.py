from __future__ import annotations

from pathlib import Path


OLD = '''def _normalise_roof_defaults(value: Any, *, filename: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{filename}: roof_defaults must be an object")
    clean_roofs = {str(key): str(item) for key, item in value.items()}
    unsupported = sorted(set(clean_roofs.values()) - _ALLOWED_ROOF_STYLES)
    if unsupported:
        raise ValueError(
            f"{filename}: roof_defaults contains unsupported roof style(s): {', '.join(unsupported)}"
        )
    return clean_roofs
'''

NEW = '''def _normalise_roof_defaults(value: Any, *, filename: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{filename}: roof_defaults must be an object")
    # osm-house-modeler has a slightly richer roof vocabulary than CWR's legacy
    # P3D renderer. Preserve the upstream data verbatim and translate only at the
    # compatibility boundary, using the same aliases as the modeler's own style
    # engine: shed/skillion roofs use CWR's gabled implementation and mansards use
    # its hipped implementation until native shapes are added to the renderer.
    aliases = {
        "gable": "gabled",
        "gabled": "gabled",
        "saltbox": "gabled",
        "skillion": "gabled",
        "shed": "gabled",
        "hip": "hipped",
        "hipped": "hipped",
        "half_hipped": "hipped",
        "mansard": "hipped",
        "pyramid": "pyramidal",
        "pyramidal": "pyramidal",
        "flat": "flat",
        "dome": "dome",
        "onion": "onion",
    }
    clean_roofs = {
        str(key): aliases.get(str(item).casefold().replace("-", "_"), str(item))
        for key, item in value.items()
    }
    unsupported = sorted(set(clean_roofs.values()) - _ALLOWED_ROOF_STYLES)
    if unsupported:
        raise ValueError(
            f"{filename}: roof_defaults contains unsupported roof style(s): {', '.join(unsupported)}"
        )
    return clean_roofs
'''


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "src" / "cwr_worldgen" / "house_style_catalogue.py"
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return 0
    if OLD not in text:
        raise RuntimeError("house_style_catalogue roof-default normalizer has changed unexpectedly")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
