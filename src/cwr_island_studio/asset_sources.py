# SPDX-License-Identifier: GPL-3.0-or-later
"""Ordered loose-asset and archive selections for Island Studio.

Sources are intentionally not scanned at selection time. Last-added loose
folders win on collisions, ahead of the indexed PBOs (also last-added wins).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def roots(value: Path | str | Sequence[Path | str] | None) -> tuple[Path, ...]:
    """Convert either the former single-root API or a sequence to paths."""
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        value = (value,)
    return tuple(Path(item).resolve() for item in value)


def append_unique(
    existing: Sequence[Path | str], incoming: Iterable[Path | str],
) -> tuple[Path, ...]:
    """Append distinct paths without replacing earlier selections."""
    combined = list(roots(existing))
    used = {str(path).casefold() for path in combined}
    for item in incoming:
        path = Path(item).resolve()
        key = str(path).casefold()
        if key not in used:
            combined.append(path)
            used.add(key)
    return tuple(combined)


def remove_selected(
    existing: Sequence[Path | str], selected: Iterable[int],
) -> tuple[Path, ...]:
    """Remove only selected zero-based list entries, preserving source order."""
    excluded = set(selected)
    return tuple(path for i, path in enumerate(roots(existing)) if i not in excluded)
