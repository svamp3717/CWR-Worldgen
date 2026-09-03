from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"{label} anchor not found")
    return text.replace(old, new, 1)


def replace_all_required(text: str, old: str, new: str, label: str, minimum: int = 1) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"{label} anchor not found")
    count = text.count(old)
    if count < minimum:
        raise RuntimeError(f"{label} expected at least {minimum} anchors, found {count}")
    return text.replace(old, new)


def patch_full_style() -> None:
    path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_full_style.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "from functools import lru_cache\n", "import base64\nfrom functools import lru_cache\n", "base64 import")

    text = replace_once(
        text,
        '        "door_material": _text(door, "material"),\n',
        '        "door_material": (_text(door, "material") or next((str(v) for v in door.get("materials", ()) if str(v).strip()), "")),\n',
        "door material fallback",
    )

    start = text.index("def texture_style_token(choice: StyleChoice) -> str:\n")
    end = text.index("\ndef visual_style_alias(", start)
    replacement = '''def _texture_metadata(choice: StyleChoice) -> dict[str, Any]:
    window = dict(choice.window_spec or {})
    door = dict(choice.door_spec or {})
    door_material = str(door.get("material") or next(
        (value for value in door.get("materials", ()) if str(value).strip()), ""
    ))
    return {
        "window": {
            "width_m": _number(window.get("width_m"), 0.0),
            "height_m": _number(window.get("height_m"), 0.0),
            "sill_height_m": _number(window.get("sill_height_m"), 0.0),
            "target_bay_spacing_m": _number(window.get("target_bay_spacing_m"), 0.0),
            "density_multiplier": _number(window.get("density_multiplier"), 1.0),
            "type": _text(window, "type"),
            "placement_style": _text(window, "placement_style"),
            "frame_material": _text(window, "frame_material"),
        },
        "door": {
            "width_m": _number(door.get("primary_width_m"), 0.0),
            "height_m": _number(door.get("primary_height_m"), 0.0),
            "type": _text(door, "type"),
            "material": door_material,
        },
    }


def texture_style_token(choice: StyleChoice) -> str:
    palette = ",".join(str(value).strip() for value in choice.colour_palette[:6])
    encoded = base64.urlsafe_b64encode(
        json.dumps(_texture_metadata(choice), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    material = f"{str(choice.wall_material or '')}~{encoded}"
    return "|".join((str(choice.facade_style or "default"), material, palette))


def roof_texture_token(roof_style: str, roof_material: str, palette: tuple[str, ...] = ()) -> str:
    return "|".join((str(roof_style or "gabled"), str(roof_material or ""), ",".join(palette[:4])))


def split_texture_token(value: str) -> tuple[str, str, tuple[str, ...]]:
    parts = str(value or "default").split("|", 2)
    facade = parts[0] or "default"
    material_blob = parts[1] if len(parts) > 1 else ""
    material = material_blob.split("~", 1)[0]
    palette = tuple(v for v in (parts[2].split(",") if len(parts) > 2 else ()) if v)
    return facade, material, palette


def texture_metadata_from_token(value: str) -> dict[str, Any]:
    parts = str(value or "").split("|", 2)
    if len(parts) < 2 or "~" not in parts[1]:
        return {}
    encoded = parts[1].split("~", 1)[1]
    if not encoded:
        return {}
    try:
        encoded += "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        value = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}

'''
    text = text[:start] + replacement + text[end + 1:]
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_runtime() -> None:
    path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_runtime.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    tint_texture,\n    visual_style_alias,\n)",
        "    tint_texture,\n    texture_metadata_from_token,\n    visual_style_alias,\n)\nfrom .osm_house_modeler_fidelity import render_modeler_facade_texture",
        "runtime fidelity imports",
    )

    start = text.index("def _styled_image_wrapper(name: str, position: int, *, strength: float):\n")
    end = text.index("\ndef _roof_image_wrapper", start)
    replacement = '''def _styled_image_wrapper(name: str, position: int, *, strength: float):
    original = _ORIGINAL_TEXTURE_FUNCTIONS[name]
    def wrapped(*args, **kwargs):
        positional = list(args)
        token = kwargs.get("regional_style")
        if token is None and len(positional) > position:
            token = positional[position]
        token = str(token or "default")
        facade, material, palette = split_texture_token(token)
        metadata = texture_metadata_from_token(token)
        alias = visual_style_alias(facade, material)
        family = str(kwargs.get("family") or (positional[0] if positional else "residential"))
        size = int(kwargs.get("size") or (positional[1] if len(positional) > 1 else 128) or 128)
        texture_variant = int(kwargs.get("texture_variant") if "texture_variant" in kwargs else (positional[3] if len(positional) > 3 else 0))

        if (
            metadata
            and name in {"_wall_texture_image", "_front_texture_image"}
            and family in {"residential", "townhouse", "urban", "school"}
        ):
            plain = _ORIGINAL_TEXTURE_FUNCTIONS.get("_open_wall_texture_image")
            if plain is not None:
                base = plain(family, size, alias, texture_variant)
                base = tint_texture(base, palette, strength=strength)
                return render_modeler_facade_texture(
                    base,
                    metadata,
                    family=family,
                    front=name == "_front_texture_image",
                )

        if "regional_style" in kwargs:
            kwargs = dict(kwargs)
            kwargs["regional_style"] = alias
        elif len(positional) > position:
            positional[position] = alias
        image = original(*positional, **kwargs)
        return tint_texture(image, palette, strength=strength)
    return wrapped

'''
    text = text[:start] + replacement + text[end + 1:]
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_procedural_buildings() -> None:
    path = ROOT / "src" / "cwr_worldgen" / "procedural_buildings.py"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "    roof_rise = min(maximum_rise, max(1.0, main_height * 0.35))\n    interior_storeys = (\n",
        '''    roof_rise = min(maximum_rise, max(1.0, main_height * 0.35))
    if key.roof_storey:
        try:
            roof_storey_spec = json.loads(key.roof_storey_spec_json or "{}")
        except json.JSONDecodeError:
            roof_storey_spec = {}
        if not isinstance(roof_storey_spec, dict):
            roof_storey_spec = {}
        minimum_roof_height = max(
            1.2,
            float(roof_storey_spec.get("minimum_roof_height_m", 2.2) or 2.2),
        )
        usable_roof_height = max(0.8, main_height - INTERIOR_SECOND_STOREY_FLOOR_Y_M)
        roof_rise = min(
            maximum_rise,
            max(roof_rise, min(minimum_roof_height, usable_roof_height)),
        )
    interior_storeys = (
''',
        "roof storey rise",
    )
    text = replace_once(
        text,
        "    if interior_storeys >= 2:\n        minimum_eave = (\n",
        "    if interior_storeys >= 2 and not key.roof_storey:\n        minimum_eave = (\n",
        "roof storey headroom bypass",
    )

    old_foundation = '''    foundation_top = plinth_height + (
        FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0
    )
'''
    new_foundation = '''    style_plinth = max(0.0, float(getattr(key, "visible_plinth_m", 0.0) or 0.0))
    foundation_top = plinth_height + max(
        style_plinth,
        FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0,
    )
'''
    text = replace_all_required(text, old_foundation, new_foundation, "rectangular visible plinth", minimum=2)
    text = replace_once(
        text,
        '''    if depth > 0.0:
        foundation_top = FOUNDATION_VISIBLE_REVEAL_M
''',
        '''    if depth > 0.0:
        foundation_top = max(
            FOUNDATION_VISIBLE_REVEAL_M,
            max(0.0, float(getattr(key, "visible_plinth_m", 0.0) or 0.0)),
        )
''',
        "polygon visible plinth",
    )

    text = replace_once(
        text,
        '''        model_assets: list[GeneratedBuildingAsset] = []
        texture_files: list[str] = []
''',
        '''        model_assets: list[GeneratedBuildingAsset] = []
        texture_files: list[str] = []
        if selected:
            from .osm_house_modeler_fidelity import emit_detail_material_textures
            texture_files.extend(emit_detail_material_textures(self, source_dir))
''',
        "detail material emission",
    )

    # Modeler-selected light frames should receive the same real trim geometry as
    # the legacy Swedish/whitewash aliases.  Define one helper and use it in all
    # visual paths as well as write_assets.
    helper_anchor = "def _main_building_height(key: BuildingVariantKey) -> float:\n"
    helper = '''def _uses_light_window_trim(key: BuildingVariantKey) -> bool:
    frame = str(getattr(key, "window_frame_material", "") or "").casefold()
    return (
        key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
        or any(token in frame for token in ("painted", "white", "upvc", "uPVC".casefold(), "timber"))
    )


'''
    if helper not in text:
        if helper_anchor not in text:
            raise RuntimeError("window trim helper anchor not found")
        text = text.replace(helper_anchor, helper + helper_anchor, 1)
    text = text.replace("key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES", "_uses_light_window_trim(key)")
    # The replacement above also touched the helper body once; restore its direct
    # legacy-style check rather than recursively calling itself.
    text = text.replace(
        '''    return (
        _uses_light_window_trim(key)
        or any(token in frame for token in ("painted", "white", "upvc", "uPVC".casefold(), "timber"))
    )
''',
        '''    return (
        key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
        or any(token in frame for token in ("painted", "white", "upvc", "timber"))
    )
''',
        1,
    )

    path.write_text(text, encoding="utf-8", newline="\n")


def patch_upgrade() -> None:
    path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "from hashlib import sha256\n", "from hashlib import sha256\nimport json\n", "upgrade json import")
    text = replace_once(
        text,
        "from .osm_house_modeler_full_style import detail_spec_from_key\n",
        "from .osm_house_modeler_full_style import detail_spec_from_key\nfrom .osm_house_modeler_fidelity import detail_texture_path, material_texture_path\n",
        "upgrade fidelity import",
    )

    helper_anchor = "def _append_details(\n"
    helpers = r'''def _outer_eave_frames(key: _pb.BuildingVariantKey):
    if not key.footprint_vertices:
        hw, hl = key.width_m * 0.5, key.length_m * 0.5
        return (
            ((0.0, -hl), (1.0, 0.0), (0.0, -1.0), key.width_m),
            ((hw, 0.0), (0.0, 1.0), (1.0, 0.0), key.length_m),
            ((0.0, hl), (-1.0, 0.0), (0.0, 1.0), key.width_m),
            ((-hw, 0.0), (0.0, -1.0), (-1.0, 0.0), key.length_m),
        )
    ring = tuple(key.footprint_vertices)
    result = []
    for index, a in enumerate(ring):
        b = ring[(index + 1) % len(ring)]
        vx, vz = b[0] - a[0], b[1] - a[1]
        span = math.hypot(vx, vz)
        if span <= 1.0e-6:
            continue
        tangent = (vx / span, vz / span)
        outward = (tangent[1], -tangent[0])
        result.append((((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5), tangent, outward, span))
    return tuple(result)


def _append_eave_overhang(
    points, normals, faces, key, *, eave_y: float, reference_texture: str
) -> None:
    overhang = max(0.0, min(1.5, float(getattr(key, "eave_overhang_m", 0.0) or 0.0)))
    if overhang <= 0.03 or key.roof_style in {"flat", "dome", "onion"}:
        return
    material = material_texture_path(reference_texture, key.roof_material, "wood")
    for anchor, tangent, outward, span in _outer_eave_frames(key):
        centre = (
            anchor[0] + outward[0] * overhang * 0.5,
            anchor[1] + outward[1] * overhang * 0.5,
        )
        _add_box(
            points, normals, faces,
            center=centre, axis_width=tangent, axis_depth=outward,
            width=span + overhang * 2.0, depth=overhang,
            y0=eave_y - 0.07, y1=eave_y + 0.02, texture=material,
        )
        fascia = (
            anchor[0] + outward[0] * max(0.0, overhang - 0.025),
            anchor[1] + outward[1] * max(0.0, overhang - 0.025),
        )
        _add_box(
            points, normals, faces,
            center=fascia, axis_width=tangent, axis_depth=outward,
            width=span + overhang * 2.0, depth=0.05,
            y0=eave_y - 0.13, y1=eave_y + 0.02, texture=material,
        )


def _roof_storey_spec(key: _pb.BuildingVariantKey) -> dict:
    try:
        value = json.loads(str(key.roof_storey_spec_json or "{}"))
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _add_gable_window(
    points, normals, faces, *, centre_x: float, z: float, y0: float, width: float,
    height: float, glass_texture: str, trim_texture: str, front: bool,
) -> None:
    half = width * 0.5
    y1 = y0 + height
    normal_sign = -1.0 if front else 1.0
    z_face = z + normal_sign * 0.018
    if front:
        coordinates = (
            (centre_x - half, y0, z_face),
            (centre_x - half, y1, z_face),
            (centre_x + half, y1, z_face),
            (centre_x + half, y0, z_face),
        )
    else:
        coordinates = (
            (centre_x + half, y0, z_face),
            (centre_x + half, y1, z_face),
            (centre_x - half, y1, z_face),
            (centre_x - half, y0, z_face),
        )
    _add_quad(points, normals, faces, coordinates, glass_texture, u_scale=1.0, v_scale=1.0)
    frame = 0.065
    for cx, cy, fw, fh in (
        (centre_x - half - frame * 0.5, y0 + height * 0.5, frame, height + frame * 2.0),
        (centre_x + half + frame * 0.5, y0 + height * 0.5, frame, height + frame * 2.0),
        (centre_x, y0 - frame * 0.5, width + frame * 2.0, frame),
        (centre_x, y1 + frame * 0.5, width + frame * 2.0, frame),
    ):
        _add_box(
            points, normals, faces,
            center=(cx, z_face + normal_sign * 0.018),
            axis_width=(1.0, 0.0), axis_depth=(0.0, normal_sign),
            width=fw, depth=0.035, y0=cy - fh * 0.5, y1=cy + fh * 0.5,
            texture=trim_texture,
        )


def _append_roof_storey_windows(
    points, normals, faces, key, *, roof_pitch_degrees: float, reference_texture: str
) -> None:
    if not key.roof_storey or key.roof_style != "gabled" or key.footprint_vertices:
        return
    spec = _roof_storey_spec(key)
    eave_y, roof_rise, _ = _pb._gabled_profile(key, roof_pitch_degrees)
    minimum_roof = max(0.0, float(spec.get("minimum_roof_height_m", 0.0) or 0.0))
    if roof_rise + 1.0e-6 < minimum_roof:
        return
    sill_above = max(0.20, float(spec.get("sill_above_eave_m", 0.42) or 0.42))
    top_clearance = max(0.18, float(spec.get("top_clearance_m", 0.34) or 0.34))
    side_clearance = max(0.15, float(spec.get("side_clearance_m", 0.30) or 0.30))
    width = max(0.45, float(key.window_width_m or 1.0) * float(spec.get("window_width_scale", 0.82) or 0.82))
    height = max(0.50, float(key.window_height_m or 1.1) * float(spec.get("window_height_scale", 0.78) or 0.78))
    y0 = eave_y + sill_above
    height = min(height, max(0.45, eave_y + roof_rise - top_clearance - y0))
    if height < 0.45:
        return
    y1 = y0 + height
    # The gable narrows linearly towards the ridge. Require the whole window,
    # including trim, to fit at its top edge rather than clipping the roof slope.
    half_width = key.width_m * 0.5
    available_half = half_width * max(0.0, 1.0 - (y1 - eave_y) / max(0.01, roof_rise))
    usable_half = max(0.0, available_half - side_clearance)
    count = max(1, min(2, int(key.roof_storey_windows_per_gable or spec.get("windows_per_gable", 1) or 1)))
    if count == 2 and usable_half * 2.0 < width * 2.3:
        count = 1
    if usable_half * 2.0 < width + 0.12:
        return
    centres = (0.0,) if count == 1 else (-min(usable_half * 0.52, width * 0.65), min(usable_half * 0.52, width * 0.65))
    glass = detail_texture_path(reference_texture, "glass")
    trim = material_texture_path(reference_texture, key.window_frame_material, "wood")
    half_length = key.length_m * 0.5
    for front, z in ((True, -half_length), (False, half_length)):
        for centre_x in centres:
            _add_gable_window(
                points, normals, faces, centre_x=centre_x, z=z, y0=y0,
                width=width, height=height, glass_texture=glass,
                trim_texture=trim, front=front,
            )


'''
    if helpers not in text:
        if helper_anchor not in text:
            raise RuntimeError("upgrade core fidelity anchor not found")
        text = text.replace(helper_anchor, helpers + helper_anchor, 1)

    text = replace_once(
        text,
        '''    if (
        not plan.enabled
        or frame is None
        or abs(lod.resolution - 1.0) > 1.0e-6
    ):
        return lod
''',
        '''    core_style_geometry = (
        (float(getattr(key, "eave_overhang_m", 0.0) or 0.0) > 0.03 and key.roof_style not in {"flat", "dome", "onion"})
        or (key.roof_storey and key.roof_style == "gabled" and not key.footprint_vertices)
    )
    if (
        (not plan.enabled and not core_style_geometry)
        or frame is None
        or abs(lod.resolution - 1.0) > 1.0e-6
    ):
        return lod
''',
        "core style geometry gate",
    )

    text = replace_once(
        text,
        '''    detail_texture = foundation_texture or roof_texture or wall_texture
    eave_y = _roof_base_y(key, roof_pitch_degrees)

    if plan.stairs:
''',
        '''    detail_texture = foundation_texture or roof_texture or wall_texture
    eave_y = _roof_base_y(key, roof_pitch_degrees)
    reference_texture = wall_texture or roof_texture or foundation_texture
    _append_eave_overhang(
        points, normals, faces, key, eave_y=eave_y,
        reference_texture=reference_texture,
    )
    _append_roof_storey_windows(
        points, normals, faces, key,
        roof_pitch_degrees=roof_pitch_degrees,
        reference_texture=reference_texture,
    )

    if plan.stairs:
''',
        "core style geometry calls",
    )

    # Each modeler-selected secondary feature now uses a material texture that
    # matches its resolved country profile rather than borrowing the facade/roof.
    text = replace_once(
        text,
        '''        door_half, _door_height, _pivot = _pb._door_dimensions(key)
        rise = max(0.08, float(stair_spec.get("step_rise_m", 0.16) or 0.16))
''',
        '''        door_half, _door_height, _pivot = _pb._door_dimensions(key)
        stair_texture = material_texture_path(reference_texture, stair_spec.get("material"), "masonry")
        rise = max(0.08, float(stair_spec.get("step_rise_m", 0.16) or 0.16))
''',
        "stair material",
    )
    # First texture=detail_texture after stairs belongs to stair boxes.
    stair_start = text.index("    if plan.stairs:\n")
    porch_start = text.index("    if plan.porch:\n", stair_start)
    stair_block = text[stair_start:porch_start].replace("texture=detail_texture,", "texture=stair_texture,")
    text = text[:stair_start] + stair_block + text[porch_start:]

    text = replace_once(
        text,
        '''    if plan.porch:
        porch_spec = detail_spec.get("porches") or {}
        width = min(
''',
        '''    if plan.porch:
        porch_spec = detail_spec.get("porches") or {}
        porch_texture = material_texture_path(reference_texture, porch_spec.get("material"), "wood")
        porch_canopy_texture = (
            material_texture_path(reference_texture, porch_spec.get("material"), "metal")
            if "metal" in str(porch_spec.get("material", "")).casefold() or "steel" in str(porch_spec.get("material", "")).casefold()
            else roof_texture
        )
        width = min(
''',
        "porch material",
    )
    porch_start = text.index("    if plan.porch:\n")
    balcony_start = text.index("    if plan.balcony_count:\n", porch_start)
    porch_block = text[porch_start:balcony_start]
    porch_block = porch_block.replace("texture=detail_texture,", "texture=porch_texture,")
    porch_block = porch_block.replace("texture=roof_texture,", "texture=porch_canopy_texture,", 1)
    text = text[:porch_start] + porch_block + text[balcony_start:]

    text = replace_once(
        text,
        '''    if plan.balcony_count:
        balcony_spec = detail_spec.get("balconies") or {}
        floor_height = min(
''',
        '''    if plan.balcony_count:
        balcony_spec = detail_spec.get("balconies") or {}
        balcony_texture = detail_texture_path(reference_texture, "balcony")
        floor_height = min(
''',
        "balcony material",
    )
    balcony_start = text.index("    if plan.balcony_count:\n")
    chimney_start = text.index("    if plan.chimney_count:\n", balcony_start)
    balcony_block = text[balcony_start:chimney_start]
    balcony_block = balcony_block.replace("texture=foundation_texture or wall_texture,", "texture=balcony_texture,")
    balcony_block = balcony_block.replace("texture=roof_texture,", "texture=balcony_texture,")
    text = text[:balcony_start] + balcony_block + text[chimney_start:]

    text = replace_once(
        text,
        '''    if plan.chimney_count:
        chimney_spec = detail_spec.get("chimneys") or {}
        chimney_width = max(0.20, float(chimney_spec.get("width_m", 0.48) or 0.48))
''',
        '''    if plan.chimney_count:
        chimney_spec = detail_spec.get("chimneys") or {}
        chimney_texture = material_texture_path(reference_texture, chimney_spec.get("material"), "masonry")
        chimney_width = max(0.20, float(chimney_spec.get("width_m", 0.48) or 0.48))
''',
        "chimney material",
    )
    chimney_start = text.index("    if plan.chimney_count:\n")
    gutter_start = text.index("    if plan.gutters:\n", chimney_start)
    chimney_block = text[chimney_start:gutter_start].replace("texture=detail_texture,", "texture=chimney_texture,")
    text = text[:chimney_start] + chimney_block + text[gutter_start:]

    text = replace_once(
        text,
        '''    if plan.gutters:
        rainwater_spec = detail_spec.get("rainwater") or {}
        gutter = max(0.04, float(rainwater_spec.get("gutter_width_m", 0.085) or 0.085))
''',
        '''    if plan.gutters:
        rainwater_spec = detail_spec.get("rainwater") or {}
        rainwater_texture = material_texture_path(reference_texture, rainwater_spec.get("material"), "metal")
        gutter = max(0.04, float(rainwater_spec.get("gutter_width_m", 0.085) or 0.085))
''',
        "rainwater material",
    )
    gutter_start = text.index("    if plan.gutters:\n")
    end_marker = text.index("    added_points = len(points)", gutter_start)
    gutter_block = text[gutter_start:end_marker].replace("texture=roof_texture,", "texture=rainwater_texture,")
    text = text[:gutter_start] + gutter_block + text[end_marker:]

    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    patch_full_style()
    patch_runtime()
    patch_procedural_buildings()
    patch_upgrade()


if __name__ == "__main__":
    main()
