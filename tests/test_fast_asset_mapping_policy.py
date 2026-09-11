from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen.asset_mapping import OsmAssetMapping, OsmAssetRule
from cwr_worldgen.fast_asset_mapping_policy import (
    _compile_bucket,
    collect_osm_asset_requirements_fast,
)


def _rule(
    rule_id: str,
    *,
    layers=("roads",),
    geometry="line",
    match=(),
    exclude=(),
    models=(),
    textures=(),
) -> OsmAssetRule:
    return OsmAssetRule(
        rule_id=rule_id,
        layers=tuple(layers),
        geometry=geometry,
        match=tuple(match),
        exclude=tuple(exclude),
        models=tuple(models),
        textures=tuple(textures),
    )


def _dataset(*features):
    return SimpleNamespace(roads=tuple(features), gravel_roads=())


def test_indexed_mapping_preserves_wildcard_and_exclusion_semantics() -> None:
    public = SimpleNamespace(
        osm_key="way/public",
        tags={"HIGHWAY": "Service", "name": "Main Yard"},
    )
    private = SimpleNamespace(
        osm_key="way/private",
        tags={"highway": "service", "access": "PRIVATE"},
    )
    mapping = OsmAssetMapping(
        rules=(
            _rule(
                "any-road",
                match=(("highway", ("*",)),),
                models=(r"mod\any.p3d",),
            ),
            _rule(
                "public-service",
                match=(("highway", ("service", "track")),),
                exclude=(("access", ("private",)),),
                models=(r"mod\service.p3d",),
                textures=(r"mod\service.paa",),
            ),
            _rule(
                "unrelated-site",
                layers=("sites",),
                geometry="polygon",
                match=(("site", ("cemetery",)),),
                models=(r"mod\site.p3d",),
            ),
        ),
        global_textures=(r"mod\shared.paa",),
    )

    report = collect_osm_asset_requirements_fast(_dataset(public, private), mapping)

    assert report.feature_count == 2
    assert report.matched_feature_count == 2
    assert set(report.selected_models) == {r"mod\any.p3d", r"mod\service.p3d"}
    assert set(report.selected_textures) == {r"mod\service.paa", r"mod\shared.paa"}
    counts = {item.rule_id: item.feature_count for item in report.rule_matches}
    assert counts == {"any-road": 2, "public-service": 1}


def test_layer_index_does_not_offer_rules_from_unrelated_layers() -> None:
    rules = (
        _rule("road", match=(("highway", ("service",)),)),
        _rule(
            "site",
            layers=("sites",),
            geometry="polygon",
            match=(("site", ("cemetery",)),),
        ),
        _rule(
            "global",
            layers=("*",),
            geometry="any",
            match=(),
        ),
    )

    bucket = _compile_bucket(rules, "roads", "line")
    offered = set(bucket.unconditional)
    offered.update(bucket.exact[("highway", "service")])

    assert offered == {0, 2}
