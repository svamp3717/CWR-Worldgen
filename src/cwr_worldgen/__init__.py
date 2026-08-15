# SPDX-License-Identifier: GPL-3.0-or-later
"""OFP/CWA world generation utilities."""

import sys as _sys

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

# Milestone 9 historically used an @-prefixed staging directory. In OFP/CWA
# conventions, @ folders belong at the actual game/mod installation location,
# not inside Worldgen's build workspace. Preserve the existing Milestone 9
# implementation while translating only that legacy staging name at runtime.
_milestone9_module = _sys.modules[__name__ + ".milestone9"]
_original_milestone9_build_milestone4 = _milestone9_module.build_milestone4


def _build_milestone9_with_worldgen_runtime(*args, **kwargs):
    if kwargs.get("mod_directory_name") == "@CWR-Milestone9":
        kwargs["mod_directory_name"] = "CWR-Worldgen"
    return _original_milestone9_build_milestone4(*args, **kwargs)


_milestone9_module.build_milestone4 = _build_milestone9_with_worldgen_runtime

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
