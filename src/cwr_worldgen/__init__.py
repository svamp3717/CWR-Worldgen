# SPDX-License-Identifier: GPL-3.0-or-later
"""OFP/CWA world generation utilities."""

from ._version import __version__
from .generator import BuildResult, build_milestone1, build_milestone2, build_milestone3, build_milestone4
from .model import HeightmapSpec, OsmSpec, PlayabilitySpec, WorldSpec
from .milestone6 import Milestone6Spec, build_milestone6
from .milestone7 import Milestone7Spec, build_milestone7
from .milestone8 import Milestone8Spec, build_milestone8
from .milestone9 import Milestone9Spec, build_milestone9
from .procedural_buildings import (
    BuildingGenerationResult,
    BuildingVariantKey,
    ProceduralBuildingLibrary,
    inspect_mlod,
    write_building_mlod,
)
from .normalization import (
    NormalizationSpec,
    NormalizedBundle,
    load_normalized_dataset,
    normalize_source_bundle,
    validate_normalized_bundle,
)
from .source_pipeline import (
    FrozenSourceBundle,
    Milestone5Spec,
    SourceFetchSpec,
    build_milestone5,
    fetch_sources,
    load_source_bundle,
    validate_source_bundle,
)

__all__ = [
    "__version__",
    "BuildResult",
    "HeightmapSpec",
    "OsmSpec",
    "PlayabilitySpec",
    "WorldSpec",
    "FrozenSourceBundle",
    "Milestone5Spec",
    "SourceFetchSpec",
    "Milestone6Spec",
    "Milestone7Spec",
    "Milestone8Spec",
    "Milestone9Spec",
    "BuildingGenerationResult",
    "BuildingVariantKey",
    "ProceduralBuildingLibrary",
    "NormalizationSpec",
    "NormalizedBundle",
    "build_milestone1",
    "build_milestone2",
    "build_milestone3",
    "build_milestone4",
    "build_milestone5",
    "build_milestone6",
    "build_milestone7",
    "build_milestone8",
    "build_milestone9",
    "inspect_mlod",
    "write_building_mlod",
    "fetch_sources",
    "load_source_bundle",
    "validate_source_bundle",
    "normalize_source_bundle",
    "validate_normalized_bundle",
    "load_normalized_dataset",
]
