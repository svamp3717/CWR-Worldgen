# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent WRP object-dot, model-footprint, and forest preview layers.

The WRP stores forestry as model objects, not source forest polygons.
Classify known vegetation and generated forest clusters without requiring
mesh decoding or importing the full terrain generator.
"""
from __future__ import annotations


def is_forest_model(model_path: str | bytes) -> bool:
    """Recognize stock forest blocks, trees, bushes, and generated clusters."""
    if isinstance(model_path, bytes):
        value = model_path.split(b"\0", 1)[0].decode("ascii", "ignore")
    else:
        value = model_path
    value = value.replace("/", "\\").strip("\\").casefold()
    if not value.endswith(".p3d"):
        return False

    # Worldgen's generated forest family: interior, border, undergrowth and
    # rural groves. Exclude the separately generated ditch grass (f\g_).
    if any(f"\\f\\{prefix}_" in "\\" + value for prefix in ("c", "b", "u", "r")):
        return True

    parts = value.rsplit("\\", 1)
    if len(parts) != 2:
        return False
    directory, file_name = parts
    if directory.endswith(("\\tree", "\\trees", "\\forest", "\\forests")):
        return True
    # Stock Everon / Nogova / Malden forest blocks and individual trees.
    # Also includes the original stock understory and Vietnam vegetation
    # families commonly embedded in generated forests.
    stem = file_name[:-4]
    if directory == "data3d" and stem.startswith(
        ("les", "str", "ker", "krovi", "bush")
    ):
        return True
    if directory == "sebnam_obj" and stem.startswith(
        ("sebstr", "sebles", "sebkrovi", "sebelekrovi")
    ):
        return True
    return False


def object_marker_mode(
    show_dots: bool, show_real_size: bool,
    has_footprint: bool, footprint_pixels: float,
) -> str:
    """Determine the visible marker without coupling size mode to dots."""
    if show_real_size and has_footprint and footprint_pixels >= 2.0:
        return "footprint"
    if show_dots:
        return "dot"
    return "hidden"
