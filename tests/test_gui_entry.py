from cwr_worldgen.gui_entry import managed_replacement


def test_managed_replacement_updates_untouched_value() -> None:
    value, managed = managed_replacement("build/my_world", "build/my_world", "build/new_world")
    assert value == "build/new_world"
    assert managed == "build/new_world"


def test_managed_replacement_preserves_user_custom_value() -> None:
    value, managed = managed_replacement("D:/Worlds/custom", "build/my_world", "build/new_world")
    assert value == "D:/Worlds/custom"
    assert managed is None


def test_managed_replacement_can_compare_normalized_paths() -> None:
    value, managed = managed_replacement(
        "BUILD/My_World",
        "build/my_world",
        "build/new_world",
        normalizer=str.casefold,
    )
    assert value == "build/new_world"
    assert managed == "build/new_world"
