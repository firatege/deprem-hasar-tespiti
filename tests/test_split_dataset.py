from scripts.split_dataset import (
    _parse_ratios,
    _parse_seed_list,
    _split_rows,
    assign_events,
    select_best_seed,
)


SAMPLE_ROWS = [
    {"event_id": "e1", "tile_id": "e1_1", "binary_damage": "1"},
    {"event_id": "e1", "tile_id": "e1_2", "binary_damage": "0"},
    {"event_id": "e2", "tile_id": "e2_1", "binary_damage": "0"},
    {"event_id": "e2", "tile_id": "e2_2", "binary_damage": "0"},
    {"event_id": "e3", "tile_id": "e3_1", "binary_damage": "1"},
    {"event_id": "e4", "tile_id": "e4_1", "binary_damage": "1"},
    {"event_id": "e5", "tile_id": "e5_1", "binary_damage": "0"},
    {"event_id": "e6", "tile_id": "e6_1", "binary_damage": "1"},
]


def test_assign_events_is_deterministic() -> None:
    splits = _parse_ratios("0.7,0.15,0.15")
    a1 = assign_events(SAMPLE_ROWS, splits, seed=42)
    a2 = assign_events(SAMPLE_ROWS, splits, seed=42)
    assert a1 == a2


def test_event_level_split_has_no_leakage() -> None:
    splits = _parse_ratios("0.7,0.15,0.15")
    assignments = assign_events(SAMPLE_ROWS, splits, seed=42)
    grouped = _split_rows(SAMPLE_ROWS, assignments)

    event_to_split: dict[str, str] = {}
    for split_name, rows in grouped.items():
        for row in rows:
            event = row["event_id"]
            if event in event_to_split:
                assert event_to_split[event] == split_name
            else:
                event_to_split[event] = split_name

    assert len(event_to_split) == len({row["event_id"] for row in SAMPLE_ROWS})


def test_parse_seed_list() -> None:
    assert _parse_seed_list("42, 2026,1337") == [42, 2026, 1337]


def test_select_best_seed_returns_candidate_from_list() -> None:
    splits = _parse_ratios("0.7,0.15,0.15")
    seeds = [42, 2026, 1337]
    chosen_seed, _, grouped, summary, candidates = select_best_seed(SAMPLE_ROWS, splits, seeds)

    assert chosen_seed in seeds
    assert len(candidates) == len(seeds)
    assert chosen_seed == candidates[0]["seed"]
    assert sum(len(rows) for rows in grouped.values()) == len(SAMPLE_ROWS)
    assert {item["split"] for item in summary} == {"train", "val", "test"}


