from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected patch context missing in {path}: {old[:120]!r}")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one patch context in {path}, found {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


# 1. Make modeler texture randomness/cache identity depend only on pixels that are
# actually rendered. Geometry/opening metadata remains in BuildingVariantKey and
# therefore remains exact; it simply stops multiplying unrelated PAA cache keys.
bridge = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_texture_bridge.py"
replace_once(
    bridge,
    "from hashlib import sha256\nimport random\n",
    "from hashlib import sha256\nimport json\nimport random\n",
)
replace_once(
    bridge,
    '''def _style_inputs(token: str) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:\n    facade, material, palette = split_texture_token(token)\n    return facade, material, palette, texture_metadata_from_token(token)\n\n\n@lru_cache(maxsize=_MODEL_RENDER_CACHE_SIZE)\ndef _wall_material_image(token: str, texture_variant: int, size: int) -> Image.Image:\n    \"\"\"Return one immutable cached modeler wall-material tile.\"\"\"\n    facade, material, palette, _metadata = _style_inputs(token)\n    kind, base = _upstream._choose_wall_base(facade, facade, material, palette)\n    rng = random.Random(_seed(f\"wall:{token}:{texture_variant}\"))\n    native = UPSTREAM_TEXTURE_CANONICAL_SIZE\n    return _canonical_image(_upstream._render_wall(kind, base, rng, native), int(size))\n''',
    '''def _style_inputs(token: str) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:\n    facade, material, palette = split_texture_token(token)\n    return facade, material, palette, texture_metadata_from_token(token)\n\n\ndef modeler_wall_material_identity(token: str, texture_variant: int = 0) -> str:\n    \"\"\"Return the pixel identity of the base wall material only.\n\n    Door/window geometry lives in the encoded style token but cannot affect the\n    bare wall material. Keeping it out of this identity lets sibling facade\n    textures and architecturally different buildings reuse one native 256px\n    material render whenever their visible wall material is actually identical.\n    \"\"\"\n    facade, material, palette = split_texture_token(token)\n    return json.dumps(\n        {\n            \"facade\": facade,\n            \"material\": material,\n            \"palette\": list(palette),\n            \"variant\": int(texture_variant),\n        },\n        sort_keys=True,\n        separators=(\",\", \":\"),\n    )\n\n\n@lru_cache(maxsize=_MODEL_RENDER_CACHE_SIZE)\ndef _wall_material_image_canonical(\n    facade: str,\n    material: str,\n    palette: tuple[str, ...],\n    texture_variant: int,\n    size: int,\n) -> Image.Image:\n    \"\"\"Return one immutable cached native modeler wall-material tile.\"\"\"\n    kind, base = _upstream._choose_wall_base(facade, facade, material, palette)\n    identity = json.dumps(\n        {\"facade\": facade, \"material\": material, \"palette\": list(palette)},\n        sort_keys=True,\n        separators=(\",\", \":\"),\n    )\n    rng = random.Random(_seed(f\"wall:{identity}:{int(texture_variant)}\"))\n    native = UPSTREAM_TEXTURE_CANONICAL_SIZE\n    return _canonical_image(\n        _upstream._render_wall(kind, base, rng, native), int(size)\n    )\n\n\ndef _wall_material_image(token: str, texture_variant: int, size: int) -> Image.Image:\n    facade, material, palette, _metadata = _style_inputs(token)\n    return _wall_material_image_canonical(\n        facade, material, palette, int(texture_variant), int(size)\n    )\n\n\n# Preserve the small internal cache-inspection API used by regression tests.\n_wall_material_image.cache_clear = _wall_material_image_canonical.cache_clear  # type: ignore[attr-defined]\n_wall_material_image.cache_info = _wall_material_image_canonical.cache_info  # type: ignore[attr-defined]\n''',
)
replace_once(
    bridge,
    '    rng = random.Random(_seed(f"window:{token}:{texture_variant}"))\n',
    '    window_identity = json.dumps(spec, sort_keys=True, separators=(",", ":"))\n    rng = random.Random(_seed(f"window:{window_identity}:{texture_variant}"))\n',
)
replace_once(
    bridge,
    '    rng = random.Random(_seed(f"window-frame:{token}:{texture_variant}"))\n',
    '    window_identity = json.dumps(spec, sort_keys=True, separators=(",", ":"))\n    rng = random.Random(_seed(f"window-frame:{window_identity}:{texture_variant}"))\n',
)
replace_once(
    bridge,
    '    rng = random.Random(_seed(f"door:{token}:{texture_variant}:{family}:{outbuilding_kind}"))\n',
    '    door_identity = json.dumps(\n        {\n            "type": spec["type"],\n            "material": material,\n            "family": str(family),\n            "outbuilding_kind": str(outbuilding_kind),\n        },\n        sort_keys=True,\n        separators=(",", ":"),\n    )\n    rng = random.Random(_seed(f"door:{door_identity}:{texture_variant}"))\n',
)
replace_once(
    bridge,
    '''def _number(value: object, default: float) -> float:\n    try:\n        return float(value)\n    except (TypeError, ValueError):\n        return float(default)\n\n\ndef _paste_scaled(base: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> None:\n''',
    '''def _number(value: object, default: float) -> float:\n    try:\n        return float(value)\n    except (TypeError, ValueError):\n        return float(default)\n\n\ndef _window_cache_identity(metadata: Mapping[str, Any], *, layout: bool) -> dict[str, Any]:\n    window = metadata.get("window") or {}\n    if not isinstance(window, Mapping):\n        window = {}\n    visual = _window_spec(metadata)\n    result: dict[str, Any] = {\n        "type": visual["type"],\n        "frame_material": visual["frame_material"],\n        "trim": visual["trim"],\n    }\n    if layout:\n        result.update({\n            "width_m": round(_number(window.get("width_m"), 0.0), 4),\n            "height_m": round(_number(window.get("height_m"), 0.0), 4),\n            "sill_height_m": round(_number(window.get("sill_height_m"), 0.85), 4),\n            "target_bay_spacing_m": round(_number(window.get("target_bay_spacing_m"), 4.0), 4),\n            "density_multiplier": round(_number(window.get("density_multiplier"), 1.0), 4),\n        })\n    return result\n\n\ndef _door_cache_identity(metadata: Mapping[str, Any], *, layout: bool) -> dict[str, Any]:\n    door = metadata.get("door") or {}\n    if not isinstance(door, Mapping):\n        door = {}\n    result: dict[str, Any] = {\n        "type": str(door.get("type", "panel") or "panel"),\n        "material": str(door.get("material", "timber") or "timber"),\n    }\n    if layout:\n        result.update({\n            "width_m": round(_number(door.get("width_m"), 0.0), 4),\n            "height_m": round(_number(door.get("height_m"), 0.0), 4),\n        })\n    return result\n\n\ndef modeler_texture_cache_identity(\n    kind: str,\n    *,\n    style_token: str = "default",\n    texture_variant: int = 0,\n    family: str = "",\n    outbuilding_kind: str = "",\n    roof_token: str = "",\n    size: int = 128,\n) -> str:\n    \"\"\"Return the smallest deterministic identity that fully determines pixels.\"\"\"\n    name = str(kind or "").casefold()\n    variant = int(texture_variant)\n    payload: dict[str, Any] = {\n        "kind": name,\n        "size": int(size),\n        "variant": variant,\n    }\n    if name == "roof":\n        parts = str(roof_token or "gabled").split("|", 2)\n        payload.update({\n            "shape": parts[0] or "gabled",\n            "material": parts[1] if len(parts) > 1 else "",\n        })\n    elif name == "foundation":\n        payload["foundation_type"] = "concrete foundation"\n    else:\n        metadata = texture_metadata_from_token(style_token)\n        payload["renderer_revision"] = int(\n            round(_number(metadata.get("texture_renderer_revision"), 0.0))\n        )\n        if name in {"wall", "open_wall", "interior_wall", "front"}:\n            payload["wall_material"] = modeler_wall_material_identity(\n                style_token, variant\n            )\n        if name == "wall":\n            payload["window"] = _window_cache_identity(metadata, layout=True)\n        elif name == "front":\n            payload["window"] = _window_cache_identity(metadata, layout=True)\n            payload["door"] = _door_cache_identity(metadata, layout=True)\n            payload["family"] = str(family)\n            payload["outbuilding_kind"] = str(outbuilding_kind)\n        elif name == "door":\n            payload["door"] = _door_cache_identity(metadata, layout=False)\n            payload["family"] = str(family)\n            payload["outbuilding_kind"] = str(outbuilding_kind)\n        elif name == "window_frame":\n            payload["window"] = _window_cache_identity(metadata, layout=False)\n    return json.dumps(payload, sort_keys=True, separators=(",", ":"))\n\n\ndef _paste_scaled(base: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> None:\n''',
)

# 2. Canonicalize persistent texture cache keys while leaving destination paths
# and P3D references untouched. Old texture caches are intentionally bypassed by
# a namespace suffix; P3D caches do not need invalidation because paths are stable.
budget = ROOT / "src" / "cwr_worldgen" / "building_asset_budget_policy.py"
replace_once(
    budget,
    "from typing import Any, Iterable\n",
    "from typing import Any, Iterable, Mapping\n",
)
replace_once(
    budget,
    '_TEXTURE_PARALLEL_MINIMUM = 4\n',
    '''_TEXTURE_PARALLEL_MINIMUM = 4\n_TEXTURE_CACHE_IDENTITY_REVISION = "pixel-equivalent-v1"\n_TEXTURE_NAMESPACE_KINDS = {\n    "procedural-building-wall-modeler-v1-cwa78": "wall",\n    "procedural-building-open-wall-modeler-v1-cwa78": "open_wall",\n    "procedural-building-interior-wall-modeler-v1-cwa58": "interior_wall",\n    "procedural-building-front-modeler-v1-cwa78": "front",\n    "procedural-building-door-modeler-v1-cwa84": "door",\n    "procedural-building-roof-modeler-v1-cwa78": "roof",\n    "procedural-building-foundation-modeler-v1-cwa78": "foundation",\n    "procedural-building-window-frame-modeler-v1-cwa84": "window_frame",\n}\n''',
)
replace_once(
    budget,
    '''@dataclass(frozen=True, slots=True)\nclass _TextureCacheTask:\n    cache_path: Path\n    kind: str\n    size: int\n    family: str = ""\n    style_token: str = "default"\n    texture_variant: int = 0\n    outbuilding_kind: str = ""\n    roof_token: str = ""\n    cache_enabled: bool = True\n    cache_refresh: bool = False\n\n\ndef _write_modeler_texture_cache_task(task: _TextureCacheTask) -> str:\n''',
    '''@dataclass(frozen=True, slots=True)\nclass _TextureCacheTask:\n    cache_path: Path\n    kind: str\n    size: int\n    family: str = ""\n    style_token: str = "default"\n    texture_variant: int = 0\n    outbuilding_kind: str = ""\n    roof_token: str = ""\n    cache_enabled: bool = True\n    cache_refresh: bool = False\n\n\ndef _canonical_texture_cache_request(namespace: str, payload: Any) -> tuple[str, Any]:\n    \"\"\"Collapse only texture cache keys that render byte-equivalent pixels.\"\"\"\n    kind = _TEXTURE_NAMESPACE_KINDS.get(str(namespace))\n    if kind is None or not isinstance(payload, Mapping):\n        return namespace, payload\n\n    from .osm_house_modeler_texture_bridge import modeler_texture_cache_identity\n\n    size = int(payload.get("texture_size", 128) or 128)\n    identity = modeler_texture_cache_identity(\n        kind,\n        style_token=str(payload.get("regional_style", "default") or "default"),\n        texture_variant=int(payload.get("texture_variant", 0) or 0),\n        family=str(payload.get("family", "") or ""),\n        outbuilding_kind=str(payload.get("outbuilding_kind", "") or ""),\n        roof_token=str(payload.get("roof", "") or ""),\n        size=size,\n    )\n    return (\n        f"{namespace}-{_TEXTURE_CACHE_IDENTITY_REVISION}",\n        {"identity": identity, "texture_size": size},\n    )\n\n\ndef _write_modeler_texture_cache_task(task: _TextureCacheTask) -> str:\n''',
)
replace_once(
    budget,
    '''    original_init = buildings.ProceduralBuildingLibrary.__init__\n    original_register = buildings.ProceduralBuildingLibrary.register_placement\n    original_write_assets = buildings.ProceduralBuildingLibrary.write_assets\n''',
    '''    original_init = buildings.ProceduralBuildingLibrary.__init__\n    original_register = buildings.ProceduralBuildingLibrary.register_placement\n    original_write_assets = buildings.ProceduralBuildingLibrary.write_assets\n    original_cache_key = buildings.cache_key\n\n    def canonical_texture_cache_key(namespace: str, payload):\n        revised_namespace, revised_payload = _canonical_texture_cache_request(\n            namespace, payload\n        )\n        return original_cache_key(revised_namespace, revised_payload)\n''',
)
replace_once(
    budget,
    '''    buildings.ProceduralBuildingLibrary.__init__ = budgeted_init\n    buildings.ProceduralBuildingLibrary.register_placement = register_with_final_budget\n    buildings.ProceduralBuildingLibrary.write_assets = write_assets_with_budget_progress\n''',
    '''    buildings.cache_key = canonical_texture_cache_key\n    buildings.ProceduralBuildingLibrary.__init__ = budgeted_init\n    buildings.ProceduralBuildingLibrary.register_placement = register_with_final_budget\n    buildings.ProceduralBuildingLibrary.write_assets = write_assets_with_budget_progress\n''',
)

# 3. Keep sibling wall/open/front/door tasks in the same spawned process so the
# bridge's process-local native-render caches actually get reused.
progress = ROOT / "src" / "cwr_worldgen" / "building_progress_policy.py"
replace_once(
    progress,
    '''def _worker_identity(worker: Callable[..., object]) -> tuple[str, str]:\n    return (\n        str(getattr(worker, "__module__", "")),\n        str(getattr(worker, "__name__", "")),\n    )\n\n\ndef _progress_worker_chunk(\n''',
    '''def _worker_identity(worker: Callable[..., object]) -> tuple[str, str]:\n    return (\n        str(getattr(worker, "__module__", "")),\n        str(getattr(worker, "__name__", "")),\n    )\n\n\ndef _texture_task_affinity(task: object) -> tuple[object, ...]:\n    \"\"\"Keep sibling texture outputs together without creating giant material shards.\"\"\"\n    token = str(getattr(task, "style_token", "") or "")\n    variant = int(getattr(task, "texture_variant", 0) or 0)\n    if token:\n        # Full style token is intentional here: one building style's wall/open/\n        # front/door outputs share the same native source renders. Using only the\n        # wall material could put a dominant national material on one worker and\n        # destroy load balance.\n        return ("style", token, variant)\n    roof = str(getattr(task, "roof_token", "") or "")\n    if roof:\n        parts = roof.split("|", 2)\n        return ("roof", parts[0] if parts else "", parts[1] if len(parts) > 1 else "", variant)\n    return ("kind", str(getattr(task, "kind", "") or ""), variant)\n\n\ndef _partition_indexed_tasks(\n    worker: Callable[..., object],\n    indexed_tasks: list[tuple[int, T]],\n    worker_count: int,\n) -> list[list[tuple[int, T]]]:\n    if _worker_identity(worker) != _TEXTURE_WORKER:\n        return [\n            indexed_tasks[worker_index::worker_count]\n            for worker_index in range(worker_count)\n        ]\n\n    groups: dict[tuple[object, ...], list[tuple[int, T]]] = {}\n    for item in indexed_tasks:\n        groups.setdefault(_texture_task_affinity(item[1]), []).append(item)\n\n    chunks: list[list[tuple[int, T]]] = [[] for _ in range(worker_count)]\n    loads = [0] * worker_count\n    # Largest sibling groups first, then deterministic source order. This is a\n    # small greedy bin-pack that preserves cache affinity without sacrificing all\n    # load balance when one style happens to need more output kinds than another.\n    ordered_groups = sorted(\n        groups.values(), key=lambda group: (-len(group), group[0][0])\n    )\n    for group in ordered_groups:\n        target = min(range(worker_count), key=lambda index: (loads[index], index))\n        chunks[target].extend(group)\n        loads[target] += len(group)\n    return chunks\n\n\ndef _progress_worker_chunk(\n''',
)
replace_once(
    progress,
    '''    indexed_tasks = list(enumerate(tasks))\n    chunks: list[list[tuple[int, T]]] = [\n        indexed_tasks[worker_index::worker_count]\n        for worker_index in range(worker_count)\n    ]\n''',
    '''    indexed_tasks = list(enumerate(tasks))\n    chunks = _partition_indexed_tasks(worker, indexed_tasks, worker_count)\n''',
)

# 4. Preserve the exact DXT1 encoder but remember block results across PAA files
# inside each long-lived worker process.
paa = ROOT / "src" / "cwr_worldgen" / "paa.py"
replace_once(
    paa,
    "from dataclasses import dataclass\nfrom pathlib import Path\n",
    "from dataclasses import dataclass\nfrom functools import lru_cache\nfrom pathlib import Path\n",
)
replace_once(
    paa,
    'def _compress_dxt1_rgb_bytes(raw: bytes) -> bytes:\n',
    '@lru_cache(maxsize=65_536)\ndef _compress_dxt1_rgb_bytes(raw: bytes) -> bytes:\n',
)

# Regression coverage for pixel-equivalence, cache collapse, worker affinity and
# byte-identical DXT1 memoization.
bridge_tests = ROOT / "tests" / "test_osm_house_modeler_texture_bridge.py"
text = bridge_tests.read_text(encoding="utf-8")
text = text.replace(
    "    _seed,\n    _wall_material_image,\n",
    "    _seed,\n    _wall_material_image,\n    modeler_texture_cache_identity,\n    modeler_wall_material_identity,\n",
    1,
)
text = text.replace(
    '    door_material: str = "timber",\n) -> str:\n',
    '    door_material: str = "timber",\n    door_width_m: float = 0.95,\n) -> str:\n',
    1,
)
text = text.replace(
    '            "width_m": 0.95,\n            "height_m": 2.05,\n',
    '            "width_m": door_width_m,\n            "height_m": 2.05,\n',
    1,
)
text = text.replace(
    '    native_rng = random.Random(_seed(f"wall:{token}:3"))\n',
    '    identity = modeler_wall_material_identity(token, 3)\n    identity_payload = json.loads(identity)\n    seed_basis = json.dumps(\n        {\n            "facade": identity_payload["facade"],\n            "material": identity_payload["material"],\n            "palette": identity_payload["palette"],\n        },\n        sort_keys=True,\n        separators=(",", ":"),\n    )\n    native_rng = random.Random(_seed(f"wall:{seed_basis}:3"))\n',
    1,
)
text = text.replace(
    '    wrong_rng = random.Random(_seed(f"wall:{token}:3"))\n',
    '    wrong_rng = random.Random(_seed(f"wall:{seed_basis}:3"))\n',
    1,
)
if "test_irrelevant_door_metadata_does_not_multiply_wall_cache_identity" not in text:
    text += '''\n\ndef test_irrelevant_door_metadata_does_not_multiply_wall_cache_identity() -> None:\n    narrow = _token("western_stucco", "plaster", "cream", door_width_m=0.90)\n    wide = _token("western_stucco", "plaster", "cream", door_width_m=1.40)\n\n    assert modeler_texture_cache_identity(\n        "open_wall", style_token=narrow, texture_variant=0, size=128\n    ) == modeler_texture_cache_identity(\n        "open_wall", style_token=wide, texture_variant=0, size=128\n    )\n    assert modeler_texture_cache_identity(\n        "wall", style_token=narrow, texture_variant=0, size=128\n    ) == modeler_texture_cache_identity(\n        "wall", style_token=wide, texture_variant=0, size=128\n    )\n    assert modeler_open_wall_texture_image("residential", 128, narrow, 0).tobytes() == (\n        modeler_open_wall_texture_image("residential", 128, wide, 0).tobytes()\n    )\n\n    # Front pixels really do use door width, so that identity must remain distinct.\n    assert modeler_texture_cache_identity(\n        "front", style_token=narrow, texture_variant=0, family="residential", size=128\n    ) != modeler_texture_cache_identity(\n        "front", style_token=wide, texture_variant=0, family="residential", size=128\n    )\n    assert modeler_front_texture_image("residential", 128, narrow, 0, "").tobytes() != (\n        modeler_front_texture_image("residential", 128, wide, 0, "").tobytes()\n    )\n\n\ndef test_roof_cache_identity_ignores_facade_palette_that_renderer_ignores() -> None:\n    cream = modeler_texture_cache_identity(\n        "roof", roof_token="gabled|standing-seam metal|cream", texture_variant=2, size=128\n    )\n    red = modeler_texture_cache_identity(\n        "roof", roof_token="gabled|standing-seam metal|falun red", texture_variant=2, size=128\n    )\n    assert cream == red\n    assert modeler_roof_texture_image("gabled|standing-seam metal|cream", 128, 2).tobytes() == (\n        modeler_roof_texture_image("gabled|standing-seam metal|falun red", 128, 2).tobytes()\n    )\n'''
bridge_tests.write_text(text, encoding="utf-8", newline="\n")

budget_tests = ROOT / "tests" / "test_building_asset_budget_policy.py"
text = budget_tests.read_text(encoding="utf-8")
text = text.replace(
    "from dataclasses import replace\n",
    "from dataclasses import replace\nimport base64\nimport json\n",
    1,
)
if "def _encoded_token(" not in text:
    insert = '''\n\ndef _encoded_token(*, door_width_m: float) -> str:\n    metadata = {\n        "texture_renderer_revision": 5,\n        "window": {\n            "width_m": 1.2,\n            "height_m": 1.35,\n            "sill_height_m": 0.85,\n            "target_bay_spacing_m": 3.6,\n            "density_multiplier": 1.0,\n            "type": "paired casement",\n            "frame_material": "painted timber",\n        },\n        "door": {\n            "width_m": door_width_m,\n            "height_m": 2.05,\n            "type": "panel",\n            "material": "timber",\n        },\n    }\n    encoded = base64.urlsafe_b64encode(\n        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")\n    ).decode("ascii").rstrip("=")\n    return f"western_stucco|stucco~{encoded}|cream"\n'''
    marker = '\n\ndef _placement(key: pb.BuildingVariantKey) -> pb.BuildingPlacement:\n'
    if marker not in text:
        raise RuntimeError("budget test insertion marker missing")
    text = text.replace(marker, insert + marker, 1)
if "test_texture_cache_collapses_only_pixel_equivalent_outputs" not in text:
    text += '''\n\ndef test_texture_cache_collapses_only_pixel_equivalent_outputs(tmp_path) -> None:\n    library = pb.ProceduralBuildingLibrary(\n        world_name="PixelEquivalentCache",\n        cache_dir=tmp_path,\n        cache_enabled=True,\n    )\n    narrow = replace(_variant(), texture_style_token=_encoded_token(door_width_m=0.90))\n    wide = replace(_variant(overhang=0.21), texture_style_token=_encoded_token(door_width_m=1.40))\n    library._usage[narrow] = 1\n    library._usage[wide] = 1\n\n    _total, misses = _texture_cache_tasks(library, pb)\n    kinds = [task.kind for task in misses]\n    # Door width is invisible to wall/open-wall pixels, so those cache entries\n    # collapse. The front layout uses door width and must remain two entries.\n    assert kinds.count("wall") == 1\n    assert kinds.count("open_wall") == 1\n    assert kinds.count("front") == 2\n    # Roof rendering ignores facade palette/opening metadata.\n    assert kinds.count("roof") == 1\n'''
budget_tests.write_text(text, encoding="utf-8", newline="\n")

progress_tests = ROOT / "tests" / "test_building_progress_policy.py"
text = progress_tests.read_text(encoding="utf-8")
text = text.replace(
    "import sys\n",
    "import sys\nfrom types import SimpleNamespace\n",
    1,
)
text = text.replace(
    "from cwr_worldgen.building_progress_policy import process_asset_tasks_with_progress\n",
    "from cwr_worldgen.building_progress_policy import (\n    _partition_indexed_tasks,\n    process_asset_tasks_with_progress,\n)\n",
    1,
)
if "test_texture_worker_partition_keeps_sibling_style_outputs_together" not in text:
    text += '''\n\ndef test_texture_worker_partition_keeps_sibling_style_outputs_together() -> None:\n    def texture_worker(value):\n        return value\n\n    texture_worker.__module__ = "cwr_worldgen.building_asset_budget_policy"\n    texture_worker.__name__ = "_write_modeler_texture_cache_task"\n    tasks = [\n        SimpleNamespace(style_token="style-a", texture_variant=0, kind="wall", roof_token=""),\n        SimpleNamespace(style_token="style-b", texture_variant=0, kind="wall", roof_token=""),\n        SimpleNamespace(style_token="style-a", texture_variant=0, kind="open_wall", roof_token=""),\n        SimpleNamespace(style_token="style-b", texture_variant=0, kind="front", roof_token=""),\n        SimpleNamespace(style_token="style-a", texture_variant=0, kind="front", roof_token=""),\n    ]\n    chunks = _partition_indexed_tasks(\n        texture_worker, list(enumerate(tasks)), 2\n    )\n    locations = {}\n    for chunk_index, chunk in enumerate(chunks):\n        for _source_index, task in chunk:\n            locations.setdefault(task.style_token, set()).add(chunk_index)\n    assert locations["style-a"] == {next(iter(locations["style-a"]))}\n    assert locations["style-b"] == {next(iter(locations["style-b"]))}\n'''
progress_tests.write_text(text, encoding="utf-8", newline="\n")

perf_tests = ROOT / "tests" / "test_paa_performance_cache.py"
perf_tests.write_text(
    '''from __future__ import annotations\n\nfrom cwr_worldgen.paa import _compress_dxt1_rgb_bytes\n\n\ndef test_dxt1_block_compression_is_memoized_without_changing_bytes() -> None:\n    raw = bytes((index * 17) % 256 for index in range(48))\n    _compress_dxt1_rgb_bytes.cache_clear()\n    first = _compress_dxt1_rgb_bytes(raw)\n    second = _compress_dxt1_rgb_bytes(raw)\n    assert first == second\n    info = _compress_dxt1_rgb_bytes.cache_info()\n    assert info.misses == 1\n    assert info.hits == 1\n''',
    encoding="utf-8",
    newline="\n",
)

print("Applied building texture performance optimization")
