"""Game-specific P3D asset labels based on virtual namespaces, not ODOL versions.

Several Bohemia games share ODOL version families. The path carried in the
PBO prefix is more reliable than the ODOL revision for identifying a game.
"""
from __future__ import annotations


def _namespace(path: str) -> str:
    normalized = str(path).replace("/", "\\").strip("\\").casefold()
    return normalized.partition("\\")[0] if "\\" in normalized else ""


def is_dayz_model_path(path: str) -> bool:
    return _namespace(path) == "dz"


def is_take_on_helicopters_model_path(path: str) -> bool:
    return _namespace(path) == "hsim"


def model_format_label(model_path: str, format_name: str, version: int) -> str:
    """Annotate known game namespaces, never guessing from format version."""
    formatted = f"{format_name} v{version}"
    if is_dayz_model_path(model_path):
        return f"DayZ asset ({formatted})"
    if is_take_on_helicopters_model_path(model_path):
        return f"Take On Helicopters asset ({formatted})"
    return formatted
