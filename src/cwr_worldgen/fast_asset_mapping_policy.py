# SPDX-License-Identifier: GPL-3.0-or-later
"""Speed up OSM-to-asset rule matching and expose mapping timings."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

from . import asset_mapping as _mapping
from .assets import canonical_asset_path


_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _RuleBucket:
    """Candidate index for one OSM layer/geometry pair."""

    unconditional: tuple[int, ...]
    exact: dict[tuple[str, str], tuple[int, ...]]
    present: dict[str, tuple[int, ...]]


def _fold_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """Case-fold one feature's tags once instead of once per candidate rule."""

    return {
        str(key).casefold(): str(value).casefold()
        for key, value in tags.items()
    }


def _conditions_match_folded(
    folded: Mapping[str, str],
    conditions: Sequence[tuple[str, tuple[str, ...]]],
) -> bool:
    for key, values in conditions:
        actual = folded.get(key)
        if "*" in values:
            if actual is None or actual == "":
                return False
            continue
        if actual is None or actual not in values:
            return False
    return True


def _rule_matches_folded(
    rule: _mapping.OsmAssetRule,
    folded: Mapping[str, str],
) -> bool:
    if not _conditions_match_folded(folded, rule.match):
        return False
    if rule.exclude and _conditions_match_folded(folded, rule.exclude):
        return False
    return True


def _anchor_condition(
    rule: _mapping.OsmAssetRule,
) -> tuple[str, tuple[str, ...]] | None:
    """Choose a cheap required condition that safely narrows candidates."""

    exact = [condition for condition in rule.match if "*" not in condition[1]]
    if exact:
        return min(exact, key=lambda condition: (len(condition[1]), condition[0]))
    return rule.match[0] if rule.match else None


def _compile_bucket(
    rules: Sequence[_mapping.OsmAssetRule],
    layer: str,
    geometry: str,
) -> _RuleBucket:
    unconditional: list[int] = []
    exact: dict[tuple[str, str], list[int]] = {}
    present: dict[str, list[int]] = {}

    for index, rule in enumerate(rules):
        if "*" not in rule.layers and layer not in rule.layers:
            continue
        if rule.geometry != "any" and rule.geometry != geometry:
            continue
        anchor = _anchor_condition(rule)
        if anchor is None:
            unconditional.append(index)
            continue
        key, values = anchor
        if "*" in values:
            present.setdefault(key, []).append(index)
            continue
        for value in values:
            exact.setdefault((key, value), []).append(index)

    return _RuleBucket(
        unconditional=tuple(unconditional),
        exact={key: tuple(value) for key, value in exact.items()},
        present={key: tuple(value) for key, value in present.items()},
    )


def collect_osm_asset_requirements_fast(
    dataset: Any,
    mapping: _mapping.OsmAssetMapping,
) -> _mapping.OsmAssetMappingReport:
    """Return the historical mapping report with indexed candidate matching."""

    try:
        from .progress import report_progress

        report_progress(81, f"Mapping OSM assets with {len(mapping.rules)} indexed rules")
    except Exception:
        pass

    started = perf_counter()
    rules = mapping.rules
    matches: dict[str, list[str]] = {rule.rule_id: [] for rule in rules}
    feature_count = 0
    matched_keys: set[str] = set()
    selected_models = {canonical_asset_path(path) for path in mapping.global_models}
    selected_textures = {canonical_asset_path(path) for path in mapping.global_textures}
    canonical_assets = tuple(
        (
            tuple(canonical_asset_path(path) for path in rule.models),
            tuple(canonical_asset_path(path) for path in rule.textures),
        )
        for rule in rules
    )
    buckets: dict[tuple[str, str], _RuleBucket] = {}

    for layer, geometry, osm_key, tags in _mapping._iter_features(dataset):
        feature_count += 1
        bucket_key = (layer, geometry)
        bucket = buckets.get(bucket_key)
        if bucket is None:
            bucket = _compile_bucket(rules, layer, geometry)
            buckets[bucket_key] = bucket

        folded = _fold_tags(tags)
        candidates = set(bucket.unconditional)
        for key, actual in folded.items():
            candidates.update(bucket.exact.get((key, actual), ()))
            if actual:
                candidates.update(bucket.present.get(key, ()))

        for index in sorted(candidates):
            rule = rules[index]
            if not _rule_matches_folded(rule, folded):
                continue
            matches[rule.rule_id].append(osm_key)
            matched_keys.add(f"{layer}:{osm_key}")
            models, textures = canonical_assets[index]
            selected_models.update(models)
            selected_textures.update(textures)

    rule_matches = tuple(
        _mapping.OsmAssetRuleMatch(
            rule_id=rule.rule_id,
            feature_count=len(matches[rule.rule_id]),
            models=canonical_assets[index][0],
            textures=canonical_assets[index][1],
            sample_osm_keys=tuple(matches[rule.rule_id][:8]),
        )
        for index, rule in enumerate(rules)
        if matches[rule.rule_id]
    )
    result = _mapping.OsmAssetMappingReport(
        source=mapping.source,
        inherit_defaults=mapping.inherit_defaults,
        mapping_sha256=mapping.sha256,
        feature_count=feature_count,
        matched_feature_count=len(matched_keys),
        selected_models=tuple(sorted(selected_models)),
        selected_textures=tuple(sorted(selected_textures)),
        rule_matches=rule_matches,
    )

    try:
        from .progress import report_progress

        elapsed = perf_counter() - started
        report_progress(
            82,
            "OSM asset mapping complete "
            f"({feature_count} features, {len(rules)} rules, {elapsed:.2f}s)",
        )
    except Exception:
        pass
    return result


def install_fast_asset_mapping_policy() -> None:
    """Route generator and public mapping calls through indexed matching."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import generator

    _mapping.collect_osm_asset_requirements = collect_osm_asset_requirements_fast
    generator.collect_osm_asset_requirements = collect_osm_asset_requirements_fast
    _INSTALLED = True
