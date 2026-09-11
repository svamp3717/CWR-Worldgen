# SPDX-License-Identifier: GPL-3.0-or-later
"""Use one predictable terrain texture as the runway background for each preset.

Runway compositing needs source pixels, but ordinary WRP terrain references do not.
Keep that distinction deliberately simple: each stock ground preset has one known
runway-background PAA, while generated/Malden profiles use the world-local grass
PAA. Asset validation checks stock models plus that one texture and trusts texture
dependencies shipped inside those model packages instead of recursively proving
every PAA/PAC reference.

This policy also disables recursive PBO discovery for the normal game-folder path.
Stock packages are looked up only in the standard CWA/OFP locations. A nonstandard
layout therefore reports an unresolved asset quickly instead of crawling the whole
game installation for several minutes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence


STOCK_RUNWAY_BACKGROUND_TEXTURES: dict[str, str] = {
    "nogova": r"o\t1.paa",
    "everon": r"Eden\zbh.paa",
    "desert": r"o\ps.paa",
}
_STOCK_RUNWAY_BACKGROUND_SET = {
    value.casefold() for value in STOCK_RUNWAY_BACKGROUND_TEXTURES.values()
}
_INSTALLED = False


def _profile_name(spec) -> str:
    return str(getattr(spec, "ground_texture_profile", "generated") or "generated").strip().casefold()


def runway_background_texture_path(spec) -> str:
    """Return the only texture whose pixels runway generation is allowed to open."""
    stock = STOCK_RUNWAY_BACKGROUND_TEXTURES.get(_profile_name(spec))
    if stock is not None:
        return stock
    return rf"{getattr(spec, 'name', '')}\data\g.paa"


def external_runway_texture_paths(spec) -> tuple[str, ...]:
    """Return the one stock texture that asset validation should locate."""
    stock = STOCK_RUNWAY_BACKGROUND_TEXTURES.get(_profile_name(spec))
    return (stock,) if stock is not None else ()


def _metadata_record_from_loose(fast, path: Path, canonical_path: str):
    stat = path.stat()
    return fast._assets.AssetRecord(
        path=canonical_path,
        source=str(path),
        size=int(stat.st_size),
        sha256=None,
        dependencies=(),
        readable=True,
    )


def _metadata_record_from_pbo(fast, path: Path, entry):
    # Header metadata is enough for validation here. Do not seek/read the P3D or
    # PAA payload merely to hash it or discover stock texture dependencies.
    return fast._assets.AssetRecord(
        path=entry.canonical_path,
        source=str(path),
        size=int(entry.data_size),
        sha256=None,
        dependencies=(),
        readable=True,
    )


def _standard_pbos(module, root: Path, prefix: str) -> tuple[Path, ...]:
    """Resolve one package prefix in known CWA layouts without recursive walking."""
    if root.is_file():
        return (
            (root,)
            if root.suffix.casefold() == ".pbo" and root.stem.casefold() == prefix.casefold()
            else ()
        )
    filename = f"{prefix}.pbo"
    layouts = (
        (filename,),
        ("Dta", filename),
        ("Res", "Dta", filename),
        ("AddOns", filename),
        ("Res", "AddOns", filename),
    )
    result: list[Path] = []
    seen: set[str] = set()
    for parts in layouts:
        current = root
        for part in parts:
            current = module._casefold_child(current, part)
            if current is None:
                break
        if current is None or not current.is_file():
            continue
        try:
            key = os.path.normcase(str(current.resolve()))
        except OSError:
            key = os.path.normcase(str(current))
        if key not in seen:
            seen.add(key)
            result.append(current)
    return tuple(result)


def _standard_model_candidate(fast, roots: Sequence[Path], canonical_path: str) -> bool:
    """Return whether a selected P3D has a cheap standard-layout lookup target."""
    prefix = canonical_path.split("\\", 1)[0]
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        try:
            root = root.resolve()
        except OSError:
            pass
        if root.is_file():
            if root.suffix.casefold() == ".pbo" and root.stem.casefold() == prefix.casefold():
                return True
            continue
        if root.is_dir():
            loose = fast._casefold_relative(root, canonical_path)
            if loose is None and root.name.casefold() == prefix.casefold() and "\\" in canonical_path:
                loose = fast._casefold_relative(root, canonical_path.split("\\", 1)[1])
            if loose is not None:
                return True
        if fast._likely_pbos(root, prefix):
            return True
    return False


def _filter_scan_assets(fast, roots: Sequence[Path], selected: Iterable[str]) -> tuple[str, ...]:
    """Keep stock P3Ds and at most the fixed runway-background stock texture."""
    result: list[str] = []
    seen: set[str] = set()
    for value in selected:
        canonical = fast._assets.canonical_asset_path(str(value))
        suffix = Path(canonical).suffix.casefold()
        keep = False
        if suffix == ".p3d":
            # Generated world-local P3Ds have no stock package and do not belong
            # in a game-installation scan. Custom nonstandard package layouts are
            # intentionally not crawled in this performance-first mode.
            keep = _standard_model_candidate(fast, roots, canonical)
        elif suffix in {".paa", ".pac"}:
            keep = canonical in _STOCK_RUNWAY_BACKGROUND_SET
        if keep and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return tuple(result)


def install_single_runway_background_policy() -> None:
    """Install one-texture runway compositing and existence-only asset validation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import fast_asset_scan_policy as fast
    from . import generator
    from . import runway_exact_background_policy as exact
    from . import runway_surface_policy as runway

    # Every runway cell now composites against the same texture for its preset.
    # This one replacement feeds the renderer, exact-background resolver and the
    # persistent per-cell cache fingerprint because all three call this helper.
    runway._ground_path_for_material = lambda spec, _material_index: runway_background_texture_path(spec)

    # Only the texture whose pixels Worldgen actually opens is an external ground
    # dependency. Other stock WRP texture references are left for CWA to resolve.
    generator._external_ground_texture_paths = external_runway_texture_paths

    # Targeted validation becomes header/existence-only. In particular, opening a
    # stock P3D solely to regex every embedded texture name is no longer necessary.
    fast._record_from_loose = lambda path, canonical_path: _metadata_record_from_loose(
        fast, path, canonical_path
    )
    fast._record_from_pbo = lambda path, entry: _metadata_record_from_pbo(fast, path, entry)
    fast._fallback_named_pbos = lambda _root, _prefix: ()

    # Exact runway backgrounds use the same standard relative PBO locations and
    # never rglob the installation when a stock package is absent/misplaced.
    exact._likely_pbos = lambda root, prefix: _standard_pbos(exact, root, prefix)
    try:
        exact._read_external_asset_cached.cache_clear()
    except AttributeError:
        pass

    # The progress wrapper remains useful, but feed it only stock models that can
    # be checked cheaply plus the single allowed terrain texture.
    original_scan = generator.scan_assets

    def scan_assets_simplified(
        roots: Sequence[Path],
        selected_models: Iterable[str],
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        refresh: bool = False,
    ):
        filtered = _filter_scan_assets(fast, roots, selected_models)
        return original_scan(
            roots,
            filtered,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh=refresh,
        )

    generator.scan_assets = scan_assets_simplified
    _INSTALLED = True
