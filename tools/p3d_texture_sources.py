"""Texture-source discovery for P3D previews.

Adds OFP/CWA and Arma-friendly sibling PBO discovery on top of the base texture resolver.
A model in Data3D.pbo can, for example, reference ``data\\foo.pac``; in that case
we automatically try Data.pbo next to Data3D.pbo before falling back to the
configured texture search inputs.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

from p3d_texture_io import AssetRef, TextureResolver as BaseTextureResolver, _canonical
from cwr_worldgen.pbo import is_pbo_path, pbo_stem


def sibling_namespace_pbos(
    source_pbo: Path,
    texture_path: str,
    *,
    available_pbos: Sequence[Path] | None = None,
) -> tuple[Path, ...]:
    r"""Find sibling PBOs for CWA and Arma texture namespaces.

    CWA data\foo.pac commonly lives in Data.pbo, whereas Arma textures
    such as ca\structures\data\foo.paa live in Structures.pbo. Try the addon
    component before broad ca / a3 so that an unrelated CA.pbo doesn't
    prevent finding the correct archive.
    """
    segments = _canonical(texture_path).split("\\")
    if len(segments) < 2:
        return ()
    names = [segments[0]]
    if len(segments) >= 3 and segments[0] in {"ca", "a3", "ca_e"}:
        names.insert(0, segments[1])
    names = list(dict.fromkeys(name for name in names if name not in {"", ".", ".."}))

    if available_pbos is None:
        try:
            available_pbos = tuple(source_pbo.parent.iterdir())
        except OSError:
            return ()

    by_stem: dict[str, list[Path]] = {}
    for path in available_pbos:
        if path.is_file() and is_pbo_path(path):
            by_stem.setdefault(pbo_stem(path).casefold(), []).append(path)
    return tuple(
        path
        for name in names
        for path in sorted(by_stem.get(name, ()), key=lambda p: str(p).casefold())
    )


def sibling_namespace_pbo(source_pbo: Path, texture_path: str) -> Path | None:
    """Return the first matching sibling PBO, preserving the existing API."""
    matches = sibling_namespace_pbos(source_pbo, texture_path)
    return matches[0] if matches else None


class TextureResolver(BaseTextureResolver):
    """Base resolver plus automatic sibling namespace PBO discovery."""

    def __init__(self, inputs: Sequence[Path]) -> None:
        super().__init__(inputs)
        self.auto_sibling_pbos: set[Path] = set()
        self._sibling_directory_cache: dict[Path, tuple[Path, ...]] = {}

    def _find_ref(self, texture_path: str, source: str) -> AssetRef | None:
        canonical = _canonical(texture_path)

        if "!" in source:
            source_pbo = Path(source.split("!", 1)[0]).expanduser()
            if is_pbo_path(source_pbo):
                self._index_model_source(source)
                ref = self.assets.get(canonical)
                if ref is not None:
                    return ref

                self._index_nested_namespace(source_pbo, canonical)
                ref = self.assets.get(canonical)
                if ref is not None:
                    return ref

                directory = source_pbo.parent
                if directory not in self._sibling_directory_cache:
                    try:
                        self._sibling_directory_cache[directory] = tuple(directory.iterdir())
                    except OSError:
                        self._sibling_directory_cache[directory] = ()
                for sibling in sibling_namespace_pbos(
                    source_pbo,
                    canonical,
                    available_pbos=self._sibling_directory_cache[directory],
                ):
                    try:
                        sibling_resolved = sibling.resolve()
                        source_resolved = source_pbo.resolve()
                    except OSError:
                        sibling_resolved = sibling
                        source_resolved = source_pbo
                    if sibling_resolved == source_resolved:
                        continue
                    if sibling_resolved not in self.auto_sibling_pbos:
                        self.auto_sibling_pbos.add(sibling_resolved)
                        print(
                            f"[textures] auto-indexing sibling PBO for {canonical}: {sibling}",
                            file=sys.stderr,
                            flush=True,
                        )
                    self._index_pbo(sibling)
                    ref = self.assets.get(canonical)
                    if ref is not None:
                        return ref

        # Explicit --texture-source paths are included in self.inputs by the caller,
        # so the normal lazy full-index fallback handles them here.
        return super()._find_ref(canonical, source)
