from __future__ import annotations

import argparse
import json
from pathlib import Path

GLOBAL_VISUAL_BALANCE_REVISION = "2026-09-global-country-visual-balance-v1"


def _dedupe(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _distribution(values: object, field: str) -> list[dict[str, object]]:
    # Do not invent national frequency statistics. The profile's curated list is
    # authoritative; the generic baseline simply gives each listed option an
    # explicit equal chance. Countries can override this with hand-tuned weights
    # (Sweden already does).
    return [{field: value, "weight": 100} for value in _dedupe(values)]


def _weighted(field: str, values: list[tuple[str, int]]) -> list[dict[str, object]]:
    return [{field: value, "weight": weight} for value, weight in values]


def _tune_sweden(document: dict) -> None:
    document["parent_region_identifier"] = "northern_europe"
    document["parent_region_name"] = "Northern Europe"
    document["detail_revision"] = "2026-09-sweden-colour-balance-v4"
    provenance = document.setdefault("data_provenance", {})
    provenance["architectural_basis"] = (
        "curated national tuning over the Northern Europe regional baseline; "
        "Sweden itself is defined only in country_styles"
    )

    for context_name, context in (document.get("contexts") or {}).items():
        selection = context.get("selection") or {}
        families = selection.get("family_distributions") or {}
        rural = str(context_name).casefold() == "rural"
        families["residential"] = (
            [{"lt": 78, "style": "swedish_wood"}, {"lt": 96, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]
            if rural else
            [{"lt": 58, "style": "swedish_wood"}, {"lt": 92, "style": "western_stucco"}, {"lt": 100, "style": "western_brick"}]
        )
        families["agricultural"] = (
            [{"lt": 84, "style": "swedish_wood"}, {"lt": 100, "style": "western_brick"}]
            if rural else
            [{"lt": 72, "style": "swedish_wood"}, {"lt": 100, "style": "western_brick"}]
        )
        selection["family_distributions"] = families
        context["selection"] = selection

        details = context.get("architectural_details") or {}
        materials = details.get("materials") or {}
        materials["common_wall_material_distribution"] = _weighted(
            "material",
            [
                ("painted vertical timber cladding", 74 if rural else 52),
                ("stucco/render", 22 if rural else 40),
                ("brick", 4 if rural else 8),
            ],
        )
        materials["wall_material_colour_distributions"] = {
            "painted vertical timber cladding": _weighted(
                "colour",
                [
                    ("falun red", 48 if rural else 30),
                    ("ochre yellow", 30 if rural else 28),
                    ("white", 8 if rural else 16),
                    ("cream", 5 if rural else 10),
                    ("grey", 2 if rural else 5),
                    ("dark green", 3 if rural else 4),
                    ("natural timber", 4 if rural else 7),
                ],
            ),
            "stucco/render": _weighted(
                "colour",
                [
                    ("cream", 42 if rural else 35),
                    ("white", 35 if rural else 36),
                    ("grey", 8 if rural else 10),
                    ("ochre yellow", 13 if rural else 17),
                    ("falun red", 2),
                ],
            ),
        }

        materials["facade_colour_distribution"] = _weighted(
            "colour",
            [
                ("falun red", 38 if rural else 24),
                ("ochre yellow", 28 if rural else 24),
                ("white", 12 if rural else 20),
                ("cream", 8 if rural else 14),
                ("natural timber", 5),
                ("dark green", 4),
                ("grey", 3 if rural else 6),
                ("black", 2 if rural else 3),
            ],
        )

        overrides = materials.get("building_class_overrides") or {}
        barn = overrides.get("barn") or {}
        barn["facade_colour_distribution"] = _weighted(
            "colour",
            [
                ("falun red", 78 if rural else 68),
                ("ochre yellow", 8 if rural else 10),
                ("natural timber", 7 if rural else 8),
                ("dark green", 3 if rural else 4),
                ("grey", 1 if rural else 2),
                ("white", 2 if rural else 5),
                ("cream", 1 if rural else 3),
            ],
        )
        overrides["barn"] = barn
        shed = overrides.get("shed") or {}
        shed["facade_colour_distribution"] = _weighted(
            "colour",
            [
                ("falun red", 58 if rural else 44),
                ("ochre yellow", 18),
                ("natural timber", 12),
                ("grey", 3 if rural else 6),
                ("dark green", 4 if rural else 5),
                ("white", 3 if rural else 9),
                ("cream", 2 if rural else 6),
            ],
        )
        overrides["shed"] = shed
        materials["building_class_overrides"] = overrides
        details["materials"] = materials
        context["architectural_details"] = details


def populate(repo_root: Path) -> tuple[int, int]:
    package = repo_root / "src" / "cwr_worldgen"
    country_dir = package / "country_styles"
    documents: list[tuple[Path, dict]] = []
    for path in sorted(country_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("iso_alpha2"):
            documents.append((path, document))
    if len(documents) != 249:
        raise RuntimeError(f"expected 249 country profiles, found {len(documents)}")

    context_count = 0
    for path, document in documents:
        if document.get("identifier") == "se_sweden":
            _tune_sweden(document)

        contexts = document.get("contexts") or {}
        if not isinstance(contexts, dict):
            raise RuntimeError(f"{path.name}: contexts must be an object")
        for context_name, context in contexts.items():
            if not isinstance(context, dict):
                raise RuntimeError(f"{path.name}: context {context_name!r} must be an object")
            details = context.get("architectural_details") or {}
            materials = details.get("materials") or {}
            if not isinstance(materials, dict):
                raise RuntimeError(f"{path.name}: {context_name} materials must be an object")

            wall_values = _dedupe(materials.get("common_wall_materials"))
            if wall_values and not materials.get("common_wall_material_distribution"):
                materials["common_wall_material_distribution"] = _distribution(
                    wall_values, "material"
                )

            colour_values = _dedupe(materials.get("typical_colour_palette"))
            if colour_values and not materials.get("facade_colour_distribution"):
                materials["facade_colour_distribution"] = _distribution(
                    colour_values, "colour"
                )

            materials["global_visual_balance_revision"] = GLOBAL_VISUAL_BALANCE_REVISION
            materials["global_visual_balance_provenance"] = (
                "Explicit ordinary-wall and facade-colour distributions derived from "
                "this country/context's curated lists. Hand-tuned distributions take "
                "precedence; otherwise listed options use an equal-weight baseline. "
                "Explicit OSM building:material/building:colour tags remain authoritative."
            )
            context_count += 1

        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    # Sweden is a country profile, not a special 24th regional style. Keeping both
    # caused two authorities for the same country and made sync behavior brittle.
    sweden_region = package / "house_styles" / "24_sweden.json"
    sweden_region.unlink(missing_ok=True)
    return len(documents), context_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    countries, contexts = populate(args.repo_root.resolve())
    print(f"Applied global visual balance to {countries} countries / {contexts} contexts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
