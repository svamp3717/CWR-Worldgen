from __future__ import annotations

from cwr_worldgen import bridge_grounding_policy as grounding
from cwr_worldgen import procedural_infrastructure as infra


def test_generated_bridge_modules_do_not_publish_landcontact_lod() -> None:
    key = infra.InfrastructureModelKey(
        "bridge", "middle", width_dm=84, length_dm=292
    )
    lods = tuple(infra._bridge_lods(key, r"tinybjorsund\i\b.paa"))

    resolutions = {float(lod.resolution) for lod in lods}
    assert float(infra._LAND_CONTACT_LOD) not in resolutions
    assert float(infra._VISUAL_LOD) in resolutions
    assert float(infra._GEOMETRY_LOD) in resolutions
    assert float(infra._ROADWAY_LOD) in resolutions


def test_generated_bridge_resolution_lod_declares_bridge_semantics() -> None:
    key = infra.InfrastructureModelKey(
        "bridge", "middle", width_dm=84, length_dm=292
    )
    lods = tuple(infra._bridge_lods(key, r"tinybjorsund\i\b.paa"))
    visual = next(
        lod for lod in lods if float(lod.resolution) == float(infra._VISUAL_LOD)
    )
    properties = dict(visual.properties)

    assert properties["autocenter"] == "0"
    assert properties["class"] == "bridge"
    assert properties["map"] == "road"


def test_bridge_model_cache_namespace_is_revised_without_changing_other_assets() -> None:
    payload = {"world": "tinybjorsund", "key": {"kind": "bridge"}}
    old_bridge = grounding._OLD_BRIDGE_MODEL_CACHE_NAMESPACE
    new_bridge = grounding._NEW_BRIDGE_MODEL_CACHE_NAMESPACE

    assert infra.cache_key(old_bridge, payload) == grounding._ORIGINAL_CACHE_KEY(
        new_bridge, payload
    )
    ordinary_namespace = "procedural-infrastructure-model-v15-safe-junction-uvs"
    assert infra.cache_key(ordinary_namespace, payload) == grounding._ORIGINAL_CACHE_KEY(
        ordinary_namespace, payload
    )
