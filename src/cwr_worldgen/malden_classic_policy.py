# SPDX-License-Identifier: GPL-3.0-or-later
"""Use the installed original Malden/Abel terrain artwork for Malden classic.

The historical malden profile generated colour-matched local PAAs. That was
useful as a dependency-free approximation, but it made "Malden classic" behave
unlike "Everon classic", which references stock island terrain directly.

Resolve the original Malden WRP from the configured CWA installation, use its own
terrain texture table as the source of truth, and choose a compact authentic
Malden palette by average colour. If the source island or readable texture
metadata is unavailable, callers retain the old generated Malden fallback.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import io
import math
import struct
from typing import Mapping, Sequence


_WRP_HEADER = struct.Struct("<4sii")
_TEXTURE_PATH_BYTES = 32
_AVERAGE_TAG_MARKER = b"GGAT"
_AVERAGE_TAG_NAME = b"CGVA"  # AVGC on disk, little-endian tag notation.
_MALDEN_TEXTURE_CANDIDATE_LIMIT = 24

# Keep the semantic palette compact, like Everon classic. Original island WRP
# tables contain many transition tiles intended to be arranged in a specific
# neighbourhood. Reusing arbitrary transition tiles per OSM class produces
# seams, so select four frequently-used source archetypes and reuse them.
_MALDEN_GROUP_TARGETS: Mapping[str, tuple[int, int, int]] = {
    "grass": (109, 118, 70),
    "sand": (184, 162, 109),
    "rock": (110, 105, 96),
    "earth": (120, 91, 59),
}
_MALDEN_MATERIAL_GROUPS: Mapping[str, str] = {
    "w": "earth",
    "q": "earth",
    "s": "sand",
    "g": "grass",
    "h": "grass",
    "r": "rock",
    "k": "rock",
    "f": "grass",
    "e": "grass",
    "a": "grass",
    "b": "grass",
    "c": "grass",
    "u": "earth",
    "i": "earth",
    "p": "earth",
    "o": "earth",
    "d": "earth",
    "t": "grass",
    "v": "earth",
    "j": "grass",
    "y": "grass",
    "x": "sand",
}

_INSTALLED = False
_ORIGINAL_GROUND_TEXTURE_PROFILE = None
_ORIGINAL_GROUND_TEXTURE_PATHS = None
_ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS = None
_ORIGINAL_WRITE_SURFACE_TEXTURES = None
_ORIGINAL_RUNWAY_BACKGROUND_TEXTURE_PATH = None
_ORIGINAL_EXTERNAL_RUNWAY_TEXTURE_PATHS = None


class _StockMaldenProfile(str):
    """String marker making old generated-vs-stock predicates treat Malden as stock."""

    def __new__(cls):
        return super().__new__(cls, "malden")

    def __hash__(self) -> int:
        return hash("everon")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str(other).casefold() in {"malden", "everon"}
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


_STOCK_MALDEN_PROFILE = _StockMaldenProfile()


def _is_malden(value: object) -> bool:
    return str(value or "").strip().casefold() == "malden"


def _casefold_child(parent: Path, name: str) -> Path | None:
    direct = parent / name
    if direct.exists():
        return direct
    if not parent.is_dir():
        return None
    folded = name.casefold()
    try:
        for child in parent.iterdir():
            if child.name.casefold() == folded:
                return child
    except OSError:
        return None
    return None


def _casefold_relative(root: Path, parts: Sequence[str]) -> Path | None:
    current = root
    for part in parts:
        current = _casefold_child(current, part)
        if current is None:
            return None
    return current if current.is_file() else None


def _candidate_roots(spec) -> tuple[Path, ...]:
    # Reuse the exact-background resolver so game-folder, --asset-root,
    # deployment-adjacent and CWR_GAME_ROOT discovery remain one system.
    from . import runway_exact_background_policy as exact

    return exact._candidate_asset_roots(spec)


def _original_malden_wrp(spec) -> Path | None:
    for root in _candidate_roots(spec):
        if root.is_file():
            if root.suffix.casefold() == ".wrp" and root.stem.casefold() == "abel":
                return root
            if root.suffix.casefold() == ".pbo" and root.stem.casefold() == "abel":
                for base in (root.parent.parent, root.parent):
                    path = _casefold_relative(base, ("Worlds", "abel.wrp"))
                    if path is not None:
                        return path
            continue
        candidates = [
            (root, ("Worlds", "abel.wrp")),
            (root, ("worlds", "abel.wrp")),
        ]
        if root.name.casefold() == "worlds":
            candidates.insert(0, (root, ("abel.wrp",)))
        if root.name.casefold() in {"dta", "addons"}:
            candidates.extend(
                (
                    (root.parent, ("Worlds", "abel.wrp")),
                    (root.parent.parent, ("Worlds", "abel.wrp")),
                )
            )
        for base, parts in candidates:
            path = _casefold_relative(base, parts)
            if path is not None:
                return path
    return None


def _wrp_texture_usage(path: Path) -> tuple[tuple[str, int], ...]:
    """Read only the terrain grids/table from a stock 1WVR or 4WVR island."""
    data = path.read_bytes()
    stream = io.BytesIO(data)
    raw = stream.read(_WRP_HEADER.size)
    if len(raw) != _WRP_HEADER.size:
        raise ValueError("truncated stock Malden WRP header")
    magic, width, height = _WRP_HEADER.unpack(raw)
    if magic not in {b"1WVR", b"4WVR"} or width <= 0 or height <= 0:
        raise ValueError("unsupported stock Malden WRP")
    cells = width * height
    heights = stream.read(cells * 2)
    if len(heights) != cells * 2:
        raise ValueError("truncated stock Malden WRP height grid")
    raw_indices = stream.read(cells * 2)
    if len(raw_indices) != cells * 2:
        raise ValueError("truncated stock Malden WRP texture grid")
    signed = magic == b"4WVR"
    format_code = "h" if signed else "H"
    indices = struct.unpack(f"<{cells}{format_code}", raw_indices)
    records = 512 if magic == b"4WVR" else 256
    slots: list[str] = []
    for _ in range(records):
        record = stream.read(_TEXTURE_PATH_BYTES)
        if len(record) != _TEXTURE_PATH_BYTES:
            raise ValueError("truncated stock Malden WRP texture table")
        value = record.split(b"\0", 1)[0]
        slots.append(value.decode("ascii", "replace") if value else "")
    counts = Counter(int(index) for index in indices if 0 <= int(index) < records)
    ranked = [
        (slots[index], int(count))
        for index, count in counts.items()
        if slots[index]
        and Path(slots[index].replace("\\", "/")).suffix.casefold() in {".paa", ".pac"}
    ]
    ranked.sort(key=lambda item: (-item[1], item[0].casefold()))
    return tuple(ranked)


def _average_colour_from_texture(data: bytes) -> tuple[int, int, int] | None:
    """Read AVGC without caring whether the legacy texture payload is PAA or PAC."""
    if len(data) < 2:
        return None
    stream = io.BytesIO(data)
    stream.read(2)  # legacy texture type/magic
    while True:
        marker = stream.read(4)
        if marker != _AVERAGE_TAG_MARKER:
            return None
        name = stream.read(4)
        raw_size = stream.read(4)
        if len(name) != 4 or len(raw_size) != 4:
            return None
        size = struct.unpack("<I", raw_size)[0]
        payload = stream.read(size)
        if len(payload) != size:
            return None
        if name == _AVERAGE_TAG_NAME and len(payload) >= 3:
            blue, green, red = payload[:3]
            return int(red), int(green), int(blue)


def _colour_distance(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> int:
    return sum((int(a) - int(b)) ** 2 for a, b in zip(left, right))


@lru_cache(maxsize=16)
def _resolved_palette_cached(
    wrp_name: str,
    wrp_size: int,
    wrp_mtime_ns: int,
    root_names: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    del wrp_size, wrp_mtime_ns
    from . import runway_exact_background_policy as exact

    usage = _wrp_texture_usage(Path(wrp_name))
    if not usage:
        return ()
    # Frequent entries are overwhelmingly more likely to be full base tiles than
    # one-off transition pieces. Restrict colour matching to those archetypes.
    source = usage[:_MALDEN_TEXTURE_CANDIDATE_LIMIT]
    coloured: list[tuple[str, int, tuple[int, int, int]]] = []
    for wire_path, count in source:
        canonical = exact._canonical(wire_path)
        located = exact._read_external_asset_cached(root_names, canonical)
        if located is None:
            continue
        colour = _average_colour_from_texture(located[0])
        if colour is not None:
            coloured.append((wire_path, count, colour))

    if coloured:
        maximum_count = max(count for _path, count, _colour in coloured)

        def choose(target: tuple[int, int, int]) -> str:
            # Colour drives the classification, while a small frequency penalty
            # biases ties/near-ties toward ordinary base tiles over transitions.
            def score(item):
                path, count, colour = item
                frequency_penalty = int(
                    900.0 * math.log2(maximum_count / max(1, count))
                )
                return (
                    _colour_distance(colour, target) + frequency_penalty,
                    -count,
                    path.casefold(),
                )

            return min(coloured, key=score)[0]

        groups = {
            name: choose(target)
            for name, target in _MALDEN_GROUP_TARGETS.items()
        }
    else:
        # The WRP remains authoritative even if old PAC artwork lacks AVGC.
        # Reuse its most common tile rather than fabricating an unverified path.
        base = source[0][0]
        groups = {name: base for name in _MALDEN_GROUP_TARGETS}

    return tuple(
        (code, groups[group])
        for code, group in _MALDEN_MATERIAL_GROUPS.items()
    )


def resolved_malden_surface_textures(spec) -> dict[str, str] | None:
    wrp = _original_malden_wrp(spec)
    if wrp is None:
        return None
    try:
        stat = wrp.stat()
        roots = _candidate_roots(spec)
        resolved = dict(
            _resolved_palette_cached(
                str(wrp.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                tuple(str(root) for root in roots),
            )
        )
    except (OSError, ValueError, struct.error):
        return None
    if not resolved or any(code not in resolved for code in _MALDEN_MATERIAL_GROUPS):
        return None

    # The single-runway asset filter intentionally permits only one terrain
    # texture. Register the dynamically discovered Malden grass tile as that one.
    try:
        from . import single_runway_background_policy as single

        single._STOCK_RUNWAY_BACKGROUND_SET.add(
            str(resolved["g"]).replace("/", "\\").lstrip("\\").casefold()
        )
    except (AttributeError, KeyError):
        pass
    return resolved


def _stock_malden_ground_texture_profile(spec):
    profile = _ORIGINAL_GROUND_TEXTURE_PROFILE(spec)
    if _is_malden(profile) and resolved_malden_surface_textures(spec) is not None:
        return _STOCK_MALDEN_PROFILE
    return profile


def _stock_malden_ground_texture_paths(spec) -> tuple[str, ...]:
    mapping = (
        resolved_malden_surface_textures(spec)
        if _is_malden(getattr(spec, "ground_texture_profile", ""))
        else None
    )
    if mapping is None:
        return _ORIGINAL_GROUND_TEXTURE_PATHS(spec)
    from . import generator

    return tuple(
        mapping[str(getattr(material, "code", ""))]
        for material in generator._material_definitions(spec)
    )


def _stock_malden_external_ground_texture_paths(spec) -> tuple[str, ...]:
    mapping = (
        resolved_malden_surface_textures(spec)
        if _is_malden(getattr(spec, "ground_texture_profile", ""))
        else None
    )
    if mapping is None:
        return _ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS(spec)
    return (mapping["g"],)


def _malden_runway_background_texture_path(spec) -> str:
    mapping = (
        resolved_malden_surface_textures(spec)
        if _is_malden(getattr(spec, "ground_texture_profile", ""))
        else None
    )
    if mapping is not None:
        return mapping["g"]
    return _ORIGINAL_RUNWAY_BACKGROUND_TEXTURE_PATH(spec)


def _malden_external_runway_texture_paths(spec) -> tuple[str, ...]:
    mapping = (
        resolved_malden_surface_textures(spec)
        if _is_malden(getattr(spec, "ground_texture_profile", ""))
        else None
    )
    if mapping is not None:
        return (mapping["g"],)
    return _ORIGINAL_EXTERNAL_RUNWAY_TEXTURE_PATHS(spec)


def _skip_stock_malden_surface_generation(
    source_dir: Path,
    world_name: str,
    profile: str,
    seed: str,
    size: int,
):
    if isinstance(profile, _StockMaldenProfile):
        return ()
    return _ORIGINAL_WRITE_SURFACE_TEXTURES(
        source_dir, world_name, profile, seed, size
    )


def install_malden_classic_policy() -> None:
    """Make Malden classic use stock Malden textures when the game is available."""
    global _INSTALLED
    global _ORIGINAL_GROUND_TEXTURE_PROFILE, _ORIGINAL_GROUND_TEXTURE_PATHS
    global _ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS, _ORIGINAL_WRITE_SURFACE_TEXTURES
    global _ORIGINAL_RUNWAY_BACKGROUND_TEXTURE_PATH
    global _ORIGINAL_EXTERNAL_RUNWAY_TEXTURE_PATHS
    if _INSTALLED:
        return

    from . import generator
    from . import single_runway_background_policy as single
    from . import surface_pass as surface

    _ORIGINAL_GROUND_TEXTURE_PROFILE = generator._ground_texture_profile
    _ORIGINAL_GROUND_TEXTURE_PATHS = generator._ground_texture_paths
    _ORIGINAL_EXTERNAL_GROUND_TEXTURE_PATHS = generator._external_ground_texture_paths
    _ORIGINAL_WRITE_SURFACE_TEXTURES = generator.write_surface_textures
    _ORIGINAL_RUNWAY_BACKGROUND_TEXTURE_PATH = single.runway_background_texture_path
    _ORIGINAL_EXTERNAL_RUNWAY_TEXTURE_PATHS = single.external_runway_texture_paths

    generator._ground_texture_profile = _stock_malden_ground_texture_profile
    generator._ground_texture_paths = _stock_malden_ground_texture_paths
    generator._external_ground_texture_paths = _stock_malden_external_ground_texture_paths
    generator.write_surface_textures = _skip_stock_malden_surface_generation
    surface.write_surface_textures = _skip_stock_malden_surface_generation

    single.runway_background_texture_path = _malden_runway_background_texture_path
    single.external_runway_texture_paths = _malden_external_runway_texture_paths

    _INSTALLED = True
