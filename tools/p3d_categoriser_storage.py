"""Persistent storage helpers for the P3D model categoriser."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import measure_p3d_models as measure
from p3d_categoriser_app import Classification, PLACEMENTS


def incomplete_state_path(path: Path) -> Path:
    """Return the companion file used for reviewed but incomplete models."""
    suffix = path.suffix or ".json"
    stem = path.name[:-len(suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.unclassified{suffix}")


def _read_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read existing classification file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"classification file {path} must contain a JSON object")
    return raw


def _classification_from_item(item: object) -> tuple[str, Classification] | None:
    if not isinstance(item, dict) or "model_path" not in item:
        return None
    key = measure._canonical_model_path(str(item["model_path"]))
    placement = str(item.get("placement", "")).strip()
    if placement not in PLACEMENTS:
        placement = ""
    categories = [str(value) for value in item.get("categories", []) if str(value).strip()]
    return key, Classification(
        categories=categories,
        placement=placement,
        reviewed=bool(item.get("reviewed", True)),
    )


def load_state(path: Path) -> tuple[dict[str, Classification], list[str]]:
    """Load complete state plus the reviewed-incomplete companion file.

    Legacy files that still contain incomplete entries in their main ``models`` array
    are accepted. Legacy diagnostic fields such as ``failures`` are simply ignored.
    The next save migrates incomplete entries to the companion file and writes no
    scan failures to either catalogue file.
    """
    result: dict[str, Classification] = {}
    categories: list[str] = []

    companion = incomplete_state_path(path)
    if companion.exists():
        raw = _read_json(companion)
        for item in raw.get("models", []):
            parsed = _classification_from_item(item)
            if parsed is not None:
                key, classification = parsed
                result[key] = classification

    if path.exists():
        raw = _read_json(path)
        categories = [str(value) for value in raw.get("categories", []) if str(value).strip()]
        # Main file wins if a stale companion contains the same model.
        for item in raw.get("models", []):
            parsed = _classification_from_item(item)
            if parsed is not None:
                key, classification = parsed
                result[key] = classification

    return result, categories


def _complete(classification: Classification) -> bool:
    return bool(classification.categories) and classification.placement in PLACEMENTS


def _model_record(model_path: str, classification: Classification) -> dict:
    return {
        "model_path": model_path,
        "categories": list(classification.categories),
        "placement": classification.placement,
        "reviewed": classification.reviewed,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def save_split_state(
    path: Path,
    *,
    categories: Sequence[str],
    state: dict[str, Classification],
    resume_model_path: str = "",
) -> tuple[int, int]:
    """Save completed models separately from reviewed-but-incomplete models.

    Runtime scan/parser failures are intentionally not persisted. They remain console
    diagnostics only. Returns ``(complete_count, incomplete_count)``.
    """
    complete_keys = [key for key in sorted(state) if _complete(state[key])]
    incomplete_keys = [
        key
        for key in sorted(state)
        if state[key].reviewed and not _complete(state[key])
    ]

    companion = incomplete_state_path(path)
    main_report = {
        "schema": 4,
        "kind": "completed_model_classifications",
        "categories": list(categories),
        "placements": list(PLACEMENTS),
        "resume_model_path": resume_model_path or "",
        "complete_count": len(complete_keys),
        "reviewed_count": sum(1 for key in complete_keys if state[key].reviewed),
        "incomplete_reviewed_count": len(incomplete_keys),
        "incomplete_reviewed_file": companion.name,
        "models": [_model_record(key, state[key]) for key in complete_keys],
    }
    _atomic_write_json(path, main_report)

    if incomplete_keys:
        incomplete_report = {
            "schema": 1,
            "kind": "reviewed_incomplete_model_classifications",
            "source_catalogue": path.name,
            "model_count": len(incomplete_keys),
            "models": [_model_record(key, state[key]) for key in incomplete_keys],
        }
        _atomic_write_json(companion, incomplete_report)
    elif companion.exists():
        companion.unlink()

    return len(complete_keys), len(incomplete_keys)
