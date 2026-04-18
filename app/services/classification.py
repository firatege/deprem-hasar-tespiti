_CANONICAL_DAMAGE_TO_BINARY = {
    "no-damage": 0,
    "minor-damage": 0,
    "major-damage": 1,
    "destroyed": 1,
}

_DAMAGE_ALIASES = {
    "no_damage": "no-damage",
    "minor": "minor-damage",
    "major": "major-damage",
    "no damage": "no-damage",
    "minor damage": "minor-damage",
    "major damage": "major-damage",
}


def normalize_damage_label(label: str) -> str:
    raw = label.strip().lower()
    if raw in _CANONICAL_DAMAGE_TO_BINARY:
        return raw

    underscored = raw.replace("-", "_")
    if underscored in _DAMAGE_ALIASES:
        return _DAMAGE_ALIASES[underscored]

    spaced = raw.replace("-", " ").replace("_", " ")
    if spaced in _DAMAGE_ALIASES:
        return _DAMAGE_ALIASES[spaced]

    raise ValueError(f"Unknown damage label: {label}")


def xview2_damage_to_binary(label: str) -> int:
    normalized = normalize_damage_label(label)
    return _CANONICAL_DAMAGE_TO_BINARY[normalized]


def major_plus_to_binary(label: str) -> int:
    # Backward-compatible alias used by existing call sites/tests.
    return xview2_damage_to_binary(label)
