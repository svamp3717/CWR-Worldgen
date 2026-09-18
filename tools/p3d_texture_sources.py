"""Texture-source discovery for P3D previews.

Adds OFP/CWA-friendly sibling PBO discovery on top of the base texture resolver.
A model in Data3D.pbo can, for example, reference ``data\\foo.pac``; in that case
we automatically try Data.pbo next to Data3D.pbo before falling back to the
configured texture search inputs.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

from p3d_texture_io import AssetRef, TextureResolver as BaseTextureResolver, _canonical


def sibling_namespace_pbo(source_pbo: Path, texture_path: str) -> Path | None:
    """Find a sibling PBO matching the first component of a canonical texture path."""
    canonical = _canonical(texture_path)
    if "\\" not in canonical:
        return None
    namespace = canonical.split("\\", 1)[0].strip()
    if not namespace or namespace in {".", ".."}:
        return None

    direct = source_pbo.with_name(f"{namespace}.pbo")
    if direct.is_file():
        return direct

    # Windows is case-insensitive, but keeping this lookup explicit also makes tests
    # and non-Windows tooling behave consistently for Data.pbo vs data.pbo.
    try:
        for child in source_pbo.parent.iterdir():
            if (
                child.is_file()
                and child.suffix.casefold() == ".pbo"
                and child.stem.casefold() == namespace.casefold()
            ):
                return child
    except OSError:
        return None
    return None


class TextureResolver(BaseTextureResolver):
    """Base resolver plus automatic sibling namespace PBO discovery."""

    def __init__(self, inputs: Sequence[Path]) -> None:
        super().__init__(inputs)
        self.auto_sibling_pbos: set[Path] = set()

    def _find_ref(self, texture_path: str, source: str) -> AssetRef | None:
        canonical = _canonical(texture_path)

        if "!" in source:
            source_pbo = Path(source.split("!", 1)[0]).expanduser()
            if source_pbo.suffix.casefold() == ".pbo":
                self._index_pbo(source_pbo)
                ref = self.assets.get(canonical)
                if ref is not None:
                    return ref

                sibling = sibling_namespace_pbo(source_pbo, canonical)
                if sibling is not None:
                    try:
                        sibling_resolved = sibling.resolve()
                        source_resolved = source_pbo.resolve()
                    except OSError:
                        sibling_resolved = sibling
                        source_resolved = source_pbo
                    if sibling_resolved != source_resolved:
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
