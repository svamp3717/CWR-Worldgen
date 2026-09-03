# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep painted modeler windows/doors at their true physical facade size."""
from __future__ import annotations

from typing import Mapping

_INSTALLED = False


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _exact_with_openings(
    base,
    token: str,
    texture_variant: int,
    *,
    family: str,
    front: bool,
    outbuilding_kind: str = "",
):
    from . import osm_house_modeler_texture_bridge as bridge
    from .osm_house_modeler_full_style import texture_metadata_from_token

    image = base.convert("RGB").copy()
    size = image.width
    metadata = texture_metadata_from_token(token)
    window = metadata.get("window") or {}
    door = metadata.get("door") or {}
    if not isinstance(window, Mapping):
        window = {}
    if not isinstance(door, Mapping):
        door = {}

    door_w_m = max(0.0, _number(door.get("width_m"), 0.0))
    door_h_m = max(0.0, _number(door.get("height_m"), 0.0))
    door_w = max(1, min(size - 2, int(round(size * (door_w_m or 0.95) / 4.0))))
    door_h = max(1, min(size, int(round(size * (door_h_m or 2.05) / 3.0))))
    door_x0 = (size - door_w) // 2
    door_box = (door_x0, size - door_h, door_x0 + door_w, size)

    window_w_m = max(0.0, _number(window.get("width_m"), 0.0))
    window_h_m = max(0.0, _number(window.get("height_m"), 0.0))
    sill_m = max(0.0, _number(window.get("sill_height_m"), 0.85))
    bay_m = max(0.8, _number(window.get("target_bay_spacing_m"), 4.0))
    density = max(0.0, _number(window.get("density_multiplier"), 1.0))
    if window_w_m > 0.0 and window_h_m > 0.0 and density > 0.0:
        count = max(1, min(4, int(round((4.0 / bay_m) * density))))
        win_w = max(1, min(size - 4, int(round(size * window_w_m / 4.0))))
        win_h = max(1, min(size - 2, int(round(size * window_h_m / 3.0))))
        bottom = int(round(size * (1.0 - min(2.95, sill_m) / 3.0)))
        top = max(1, bottom - win_h)
        source = bridge._window_image(metadata, token, texture_variant, max(32, win_w, win_h))
        cell = size / count
        for index in range(count):
            cx = int(round((index + 0.5) * cell))
            cell_left = int(round(index * cell + 2))
            cell_right = int(round((index + 1) * cell - 2))
            x0 = max(cell_left, cx - win_w // 2)
            x1 = min(cell_right, x0 + win_w)
            if x1 - x0 < 1:
                continue
            if front and not (x1 < door_box[0] - 2 or x0 > door_box[2] + 2):
                continue
            bridge._paste_scaled(image, source, (x0, top, x1, bottom))

    if front and (door_w_m > 0.0 or family in {
        "residential", "townhouse", "urban", "school", "shop",
        "agricultural", "industrial", "outbuilding",
    }):
        source = bridge._door_image(
            metadata,
            token,
            texture_variant,
            max(32, door_w, door_h),
            family=family,
            outbuilding_kind=outbuilding_kind,
        )
        bridge._paste_scaled(image, source, door_box)
    return image


def _exact_front_door_uv(key) -> tuple[float, float, float, float]:
    from .opening_dimension_policy import _door_metadata, _number

    door = _door_metadata(key)
    width = _number(door.get("width_m"), 0.95) or 0.95
    height = _number(door.get("height_m"), 2.05) or 2.05
    u_span = max(1.0 / 256.0, min(0.995, width / 4.0))
    v_span = max(1.0 / 256.0, min(1.0, height / 3.0))
    return ((1.0 - u_span) * 0.5, (1.0 + u_span) * 0.5, 1.0 - v_span, 1.0)


def install_opening_texture_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import opening_dimension_policy as dimensions
    from . import osm_house_modeler_texture_bridge as bridge

    bridge._with_openings = _exact_with_openings
    dimensions._front_door_uv = _exact_front_door_uv
    _INSTALLED = True
