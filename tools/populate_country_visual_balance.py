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
            document["parent_region_identifier"] = "northern_europe"
            document["parent_region_name"] = "Northern Europe"
            provenance = document.setdefault("data_provenance", {})
            provenance["architectural_basis"] = (
                "curated national tuning over the Northern Europe regional baseline; "
                "Sweden itself is defined only in country_styles"
            )

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
