from dataclasses import replace
from pathlib import Path

import cwr_worldgen.milestone9 as milestone9
from cwr_worldgen.milestone9 import Milestone9Spec, _Milestone9PlayabilitySpec


def test_public_milestone9_spec_accepts_advisory_object_limits() -> None:
    spec = Milestone9Spec(source_dir=Path("unused-source"), advisory_object_limits=True)
    assert spec.advisory_object_limits is True


def test_advisory_policy_is_installed_on_milestone9_build_path() -> None:
    assert getattr(milestone9.build_milestone9, "_cwr_milestone9_advisory_policy", False)
    # __init__ later wraps build_milestone4 for the runtime directory, so inspect
    # the saved wrapped function rather than requiring the marker on the outer shim.
    import cwr_worldgen
    inner = cwr_worldgen._original_milestone9_build_milestone4
    assert getattr(inner, "_cwr_milestone9_advisory_policy", False)


def test_runtime_spec_supports_the_same_advisory_policy_field() -> None:
    runtime = _Milestone9PlayabilitySpec(
        heightmap_path=Path("unused.png"),
        advisory_object_limits=True,
    )
    assert runtime.advisory_object_limits is True
    assert replace(runtime, advisory_object_limits=False).advisory_object_limits is False
