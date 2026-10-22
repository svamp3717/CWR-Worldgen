"""Consistent, filesystem-safe island-type names for export ZIPs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .core import IslandDocument

_STOCK_NAMES = {
    "nogova": "nogova", "noe": "nogova", "o": "nogova",
    "everon": "everon", "eden": "everon",
    "malden": "malden", "abel": "malden",
    "kolgujev": "kolgujev", "cain": "kolgujev",
}


def island_export_stem(doc: IslandDocument) -> str:
    """Prefer known original island names, then source texture namespace.

    Archived WRPs can have a temporary extracted filename such as
    selected-world.wrp. Their source_label retains the actual member path.
    For generated worlds with generic names, the stock texture namespace
    identifies which island's assets were used.
    """
    source = str(getattr(doc, "source_label", "") or "")
    member = source.rsplit(" / ", 1)[-1].replace("\\", "/").rsplit("/", 1)[-1]
    name = Path(member).stem if member.casefold().endswith(".wrp") else doc.path.stem
    explicit = _STOCK_NAMES.get(name.casefold())
    if explicit is not None:
        return explicit

    families: Counter[str] = Counter()
    for path in getattr(doc, "texture_paths", ()):
        token = str(path).replace("\\", "/").lstrip("/").split("/", 1)
        if len(token) == 2 and token[1]:
            kind = _STOCK_NAMES.get(token[0].casefold())
            if kind is not None:
                families[kind] += 1
    if families:
        return sorted(families, key=lambda x: (-families[x], x))[0]

    safe = "".join(
        "_" if ord(char) < 32 or char in '<>"\\/|?*:' else char
        for char in name.casefold()
    ).strip(" .")
    return safe or "island"
