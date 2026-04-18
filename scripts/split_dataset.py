import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitConfig:
    name: str
    ratio: float


# Objective weights (class-balance-first): pos-ratio drift has higher impact than row-ratio drift.
RATIO_WEIGHT = 1.0
POS_RATIO_WEIGHT = 4.0
ZERO_POS_PENALTY = 1_000_000.0


def _parse_seed_list(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seed-list must contain at least one integer seed")
    return seeds


def _parse_ratios(raw: str) -> list[SplitConfig]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Expected three split ratios, e.g. '0.7,0.15,0.15'")

    values = [float(part) for part in parts]
    total = sum(values)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    normalized = [value / total for value in values]
    return [
        SplitConfig("train", normalized[0]),
        SplitConfig("val", normalized[1]),
        SplitConfig("test", normalized[2]),
    ]


def _load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"event_id", "tile_id", "binary_damage"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in manifest: {sorted(missing)}")
        return [row for row in reader]


def _event_stats(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        event_id = row["event_id"]
        label = int(row["binary_damage"])
        stats[event_id]["rows"] += 1
        if label == 1:
            stats[event_id]["pos"] += 1
        else:
            stats[event_id]["neg"] += 1
    return stats


def _targets(rows: list[dict[str, str]], splits: list[SplitConfig]) -> dict[str, dict[str, float]]:
    total_rows = len(rows)
    total_pos = sum(1 for row in rows if int(row["binary_damage"]) == 1)
    total_neg = total_rows - total_pos

    targets: dict[str, dict[str, float]] = {}
    for split in splits:
        targets[split.name] = {
            "rows": total_rows * split.ratio,
            "pos": total_pos * split.ratio,
            "neg": total_neg * split.ratio,
        }
    return targets


def _score_assignment(
    split_name: str,
    event_counter: Counter[str],
    current: dict[str, Counter[str]],
    targets: dict[str, dict[str, float]],
) -> float:
    # Greedy objective: fill remaining deficits while discouraging large overshoot.
    remaining_rows = targets[split_name]["rows"] - current[split_name]["rows"]
    remaining_pos = targets[split_name]["pos"] - current[split_name]["pos"]

    row_fit = remaining_rows - event_counter["rows"]
    pos_fit = remaining_pos - event_counter["pos"]

    row_score = (1000 - row_fit) if row_fit >= 0 else (-2000 - (2 * abs(row_fit)))
    pos_score = (500 - pos_fit) if pos_fit >= 0 else (-1000 - abs(pos_fit))
    return row_score + (0.8 * pos_score)


def assign_events(
    rows: list[dict[str, str]],
    splits: list[SplitConfig],
    seed: int,
) -> dict[str, str]:
    event_stats = _event_stats(rows)
    events = list(event_stats.keys())

    # Larger groups are harder to place; assign those first.
    rng = random.Random(seed)
    rng.shuffle(events)
    events.sort(key=lambda event_id: event_stats[event_id]["rows"], reverse=True)

    targets = _targets(rows, splits)
    current = {split.name: Counter() for split in splits}
    assignments: dict[str, str] = {}

    for event_id in events:
        counter = event_stats[event_id]
        best_split = max(
            splits,
            key=lambda split: _score_assignment(split.name, counter, current, targets),
        )
        assignments[event_id] = best_split.name
        current[best_split.name]["rows"] += counter["rows"]
        current[best_split.name]["pos"] += counter["pos"]
        current[best_split.name]["neg"] += counter["neg"]

    return assignments


def _write_split_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_rows(
    rows: list[dict[str, str]], assignments: dict[str, str]
) -> dict[str, list[dict[str, str]]]:
    grouped = {"train": [], "val": [], "test": []}
    for row in rows:
        split_name = assignments[row["event_id"]]
        grouped[split_name].append(row)
    return grouped


def _summary(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        rows = grouped[split_name]
        total = len(rows)
        pos = sum(1 for row in rows if int(row["binary_damage"]) == 1)
        neg = total - pos
        summary.append(
            {
                "split": split_name,
                "rows": total,
                "positive": pos,
                "negative": neg,
                "positive_ratio": round((pos / total) if total else 0.0, 6),
            }
        )
    return summary


def _ratio_map(splits: list[SplitConfig]) -> dict[str, float]:
    return {split.name: split.ratio for split in splits}


def _evaluate_split(
    grouped: dict[str, list[dict[str, str]]],
    ratios: dict[str, float],
    global_positive_ratio: float,
) -> dict[str, Any]:
    """Evaluate one candidate split assignment.

    Objective:
    S = w1 * ratio_error + w2 * pos_ratio_error + penalty
    where class balance is intentionally weighted higher than exact split ratio.
    """
    summary = _summary(grouped)
    total_rows = sum(int(item["rows"]) for item in summary)

    ratio_error = 0.0
    pos_ratio_error = 0.0
    penalty = 0.0
    train_balance_error = 0.0

    for item in summary:
        split_name = str(item["split"])
        rows = int(item["rows"])
        pos = int(item["positive"])
        positive_ratio = float(item["positive_ratio"])

        observed_ratio = (rows / total_rows) if total_rows else 0.0
        ratio_error += abs(observed_ratio - ratios[split_name])

        drift = abs(positive_ratio - global_positive_ratio)
        pos_ratio_error += drift
        if split_name == "train":
            train_balance_error = drift

        if rows == 0 or pos == 0:
            penalty += ZERO_POS_PENALTY

    score = (RATIO_WEIGHT * ratio_error) + (POS_RATIO_WEIGHT * pos_ratio_error) + penalty
    return {
        "summary": summary,
        "score": round(score, 6),
        "ratio_error": round(ratio_error, 6),
        "pos_ratio_error": round(pos_ratio_error, 6),
        "train_balance_error": round(train_balance_error, 6),
        "penalty": round(penalty, 6),
    }


def select_best_seed(
    rows: list[dict[str, str]],
    splits: list[SplitConfig],
    seeds: list[int],
) -> tuple[int, dict[str, str], dict[str, list[dict[str, str]]], list[dict[str, Any]], list[dict[str, Any]]]:
    global_positive = sum(1 for row in rows if int(row["binary_damage"]) == 1)
    global_positive_ratio = (global_positive / len(rows)) if rows else 0.0
    ratios = _ratio_map(splits)

    candidates: list[dict[str, Any]] = []
    best: tuple[
        float,
        float,
        int,
        dict[str, str],
        dict[str, list[dict[str, str]]],
        list[dict[str, Any]],
    ] | None = None

    for seed in seeds:
        assignments = assign_events(rows, splits, seed)
        grouped = _split_rows(rows, assignments)
        eval_result = _evaluate_split(grouped, ratios, global_positive_ratio)
        summary = eval_result["summary"]
        score = float(eval_result["score"])
        train_balance_error = float(eval_result["train_balance_error"])

        candidates.append(
            {
                "seed": seed,
                "score": eval_result["score"],
                "ratio_error": eval_result["ratio_error"],
                "pos_ratio_error": eval_result["pos_ratio_error"],
                "train_balance_error": eval_result["train_balance_error"],
                "penalty": eval_result["penalty"],
                "summary": summary,
            }
        )

        # Tie-break order: (1) lower objective score, (2) better train balance, (3) deterministic seed.
        current = (score, train_balance_error, seed, assignments, grouped, summary)
        if best is None or (current[0], current[1], current[2]) < (best[0], best[1], best[2]):
            best = current

    if best is None:
        raise ValueError("No candidate seeds were evaluated")

    _, _, chosen_seed, assignments, grouped, summary = best
    candidates.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["train_balance_error"]),
            int(item["seed"]),
        )
    )
    return chosen_seed, assignments, grouped, summary, candidates


def _write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "rows", "positive", "negative", "positive_ratio"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def _compute_class_weights(train_rows: list[dict[str, str]], formula: str) -> dict[str, float]:
    n = len(train_rows)
    n_pos = sum(1 for row in train_rows if int(row["binary_damage"]) == 1)
    n_neg = n - n_pos

    if formula == "balanced":
        w0 = (n / (2 * n_neg)) if n_neg else 0.0
        w1 = (n / (2 * n_pos)) if n_pos else 0.0
    elif formula == "neg_pos":
        w0 = 1.0
        w1 = (n_neg / n_pos) if n_pos else 0.0
    elif formula == "log_smoothed":
        import math

        w0 = 1.0
        w1 = math.log1p((n_neg / n_pos)) if n_pos else 0.0
    else:
        raise ValueError(f"Unsupported weight formula: {formula}")

    return {
        "class_0": round(w0, 6),
        "class_1": round(w1, 6),
        "train_rows": n,
        "train_positive": n_pos,
        "train_negative": n_neg,
        "formula": formula,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split train_manifest.csv into train/val/test sets")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/train_manifest.csv"),
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for split outputs",
    )
    parser.add_argument(
        "--ratios",
        default="0.7,0.15,0.15",
        help="Comma-separated split ratios for train,val,test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for event ordering")
    parser.add_argument(
        "--scan-seeds",
        type=int,
        default=0,
        help="If > 0, evaluate seeds in range [seed, seed + scan_seeds - 1] and pick best",
    )
    parser.add_argument(
        "--seed-list",
        default="",
        help="Optional comma-separated seeds to scan and auto-select best split (e.g. 42,2026,1337)",
    )
    parser.add_argument(
        "--weight-formula",
        choices=["balanced", "neg_pos", "log_smoothed"],
        default="balanced",
        help="Formula used for class weights on train split",
    )
    args = parser.parse_args()

    splits = _parse_ratios(args.ratios)
    rows = _load_rows(args.input_csv)
    if not rows:
        raise ValueError("Input manifest is empty")

    if args.seed_list.strip():
        seeds = _parse_seed_list(args.seed_list)
    elif args.scan_seeds > 0:
        seeds = list(range(args.seed, args.seed + args.scan_seeds))
    else:
        seeds = [args.seed]

    chosen_seed, assignments, grouped, summary, candidates = select_best_seed(rows, splits, seeds)

    fieldnames = list(rows[0].keys())
    for split_name, file_name in (
        ("train", "train_split.csv"),
        ("val", "val_split.csv"),
        ("test", "test_split.csv"),
    ):
        _write_split_csv(args.output_dir / file_name, fieldnames, grouped[split_name])

    _write_summary_csv(args.output_dir / "class_imbalance_summary.csv", summary)

    weights = _compute_class_weights(grouped["train"], args.weight_formula)
    with (args.output_dir / "class_weights.json").open("w", encoding="utf-8") as handle:
        json.dump(weights, handle, indent=2)

    seed_report = {
        "chosen_seed": chosen_seed,
        "candidates": candidates,
        "tie_break_order": ["score", "train_balance_error", "seed"],
        "objective_weights": {
            "ratio_weight": RATIO_WEIGHT,
            "pos_ratio_weight": POS_RATIO_WEIGHT,
            "zero_pos_penalty": ZERO_POS_PENALTY,
        },
    }
    with (args.output_dir / "seed_scan_report.json").open("w", encoding="utf-8") as handle:
        json.dump(seed_report, handle, indent=2)

    payload = {
        "input_csv": str(args.input_csv.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "ratios": [split.ratio for split in splits],
        "seed": chosen_seed,
        "seed_candidates": seeds,
        "summary": summary,
        "class_weights": weights,
        "candidate_scores": candidates,
        "events": {
            "total": len(set(row["event_id"] for row in rows)),
            "train": len({row["event_id"] for row in grouped["train"]}),
            "val": len({row["event_id"] for row in grouped["val"]}),
            "test": len({row["event_id"] for row in grouped["test"]}),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()



