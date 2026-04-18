import pytest

from app.services.classification import major_plus_to_binary


@pytest.mark.parametrize(
    "label,expected",
    [
        ("no-damage", 0),
        ("minor-damage", 0),
        ("major-damage", 1),
        ("destroyed", 1),
        ("Major-Damage", 1),
        ("major", 1),
        ("no_damage", 0),
    ],
)
def test_major_plus_to_binary(label: str, expected: int) -> None:
    assert major_plus_to_binary(label) == expected


def test_major_plus_to_binary_invalid_label() -> None:
    with pytest.raises(ValueError):
        major_plus_to_binary("unknown")
