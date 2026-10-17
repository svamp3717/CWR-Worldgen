"""Identify DayZ-namespaced P3D models without guessing from ODOL versions alone.

DayZ shares P3D/ODOL version families with Arma. Only the virtual asset
namespace reliably signals that a particular model belongs to DayZ.
"""
from __future__ import annotations


def is_dayz_model_path(path: str) -> bool:
    normalized = str(path).replace("/", "\\").strip("\\").casefold()
    return normalized.startswith("dz\\")


def model_format_label(model_path: str, format_name: str, version: int) -> str:
    """Readable label for both the asset browser and categoriser."""
    formatted = f"{format_name} v{version}"
    return f"DayZ asset ({formatted})" if is_dayz_model_path(model_path) else formatted
