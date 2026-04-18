import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings
from app.services.classification import normalize_damage_label, xview2_damage_to_binary

_TILE_STEM_RE = re.compile(r"^(?P<event>.+)_(?P<tile_num>\d{8})_(?P<phase>pre|post)_disaster$")
_SEVERITY_ORDER = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


def _parse_stem(stem: str) -> tuple[str, str, str] | None:
    match = _TILE_STEM_RE.match(stem)
    if not match:
        return None
    return match.group("event"), match.group("tile_num"), match.group("phase")


def _extract_tile_label(post_label_path: Path) -> tuple[str, int, int]:
    payload = json.loads(post_label_path.read_text(encoding="utf-8"))
    lng_lat_features = payload.get("features", {}).get("lng_lat", [])

    normalized_subtypes: list[str] = []
    unknown_subtypes = 0
    for feature in lng_lat_features:
        properties = feature.get("properties", {})
        if properties.get("feature_type") != "building":
            continue

        subtype_raw = properties.get("subtype")
        if not subtype_raw:
            continue

        try:
            normalized = normalize_damage_label(str(subtype_raw))
        except ValueError:
            unknown_subtypes += 1
            continue
        normalized_subtypes.append(normalized)

    if not normalized_subtypes:
        # If no explicit damage subtype exists, default to negative class.
        return "no-damage", 0, unknown_subtypes

    worst_label = max(normalized_subtypes, key=lambda item: _SEVERITY_ORDER[item])
    return worst_label, len(normalized_subtypes), unknown_subtypes


def _iter_post_label_files(geotiffs_root: Path):
    for split_dir in sorted([p for p in geotiffs_root.iterdir() if p.is_dir()]):
        labels_dir = split_dir / "labels"
        if not labels_dir.exists():
            continue
        for post_label in sorted(labels_dir.glob("*_post_disaster.json")):
            yield split_dir.name, post_label


def build_manifest(geotiffs_root: Path, output_csv: Path) -> dict[str, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "event_id",
        "tile_id",
        "pre_image_path",
        "post_image_path",
        "pre_label_path",
        "post_label_path",
        "damage_label",
        "binary_damage",
        "building_count",
        "unknown_subtype_count",
    ]

    stats = Counter()
    label_stats = Counter()

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for split_name, post_label_path in _iter_post_label_files(geotiffs_root):
            parsed = _parse_stem(post_label_path.stem)
            if not parsed:
                stats["skipped_bad_stem"] += 1
                continue

            event_id, tile_num, _ = parsed
            tile_id = f"{event_id}_{tile_num}"
            split_root = post_label_path.parent.parent
            pre_label_path = post_label_path.with_name(post_label_path.name.replace("_post_disaster.json", "_pre_disaster.json"))
            post_image_path = split_root / "images" / post_label_path.name.replace(".json", ".tif")
            pre_image_path = split_root / "images" / pre_label_path.name.replace(".json", ".tif")

            if not pre_label_path.exists():
                stats["missing_pre_label"] += 1
                continue
            if not post_image_path.exists():
                stats["missing_post_image"] += 1
                continue
            if not pre_image_path.exists():
                stats["missing_pre_image"] += 1
                continue

            damage_label, building_count, unknown_subtype_count = _extract_tile_label(post_label_path)
            binary_damage = xview2_damage_to_binary(damage_label)

            writer.writerow(
                {
                    "split": split_name,
                    "event_id": event_id,
                    "tile_id": tile_id,
                    "pre_image_path": str(pre_image_path.resolve()),
                    "post_image_path": str(post_image_path.resolve()),
                    "pre_label_path": str(pre_label_path.resolve()),
                    "post_label_path": str(post_label_path.resolve()),
                    "damage_label": damage_label,
                    "binary_damage": binary_damage,
                    "building_count": building_count,
                    "unknown_subtype_count": unknown_subtype_count,
                }
            )
            stats["rows_written"] += 1
            label_stats[damage_label] += 1
            if binary_damage == 1:
                stats["positive_rows"] += 1
            else:
                stats["negative_rows"] += 1

    summary = {
        "rows_written": stats["rows_written"],
        "positive_rows": stats["positive_rows"],
        "negative_rows": stats["negative_rows"],
        "missing_pre_label": stats["missing_pre_label"],
        "missing_pre_image": stats["missing_pre_image"],
        "missing_post_image": stats["missing_post_image"],
        "skipped_bad_stem": stats["skipped_bad_stem"],
        "label_no_damage": label_stats["no-damage"],
        "label_minor_damage": label_stats["minor-damage"],
        "label_major_damage": label_stats["major-damage"],
        "label_destroyed": label_stats["destroyed"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build binary training manifest from xView2 GeoTIFF tiles")
    parser.add_argument(
        "--geotiffs-root",
        type=Path,
        default=settings.tiles_dir / "geotiffs",
        help="Root folder that contains split folders with images/labels",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=settings.base_dir / "data" / "processed" / "train_manifest.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    geotiffs_root = args.geotiffs_root.resolve()
    output_csv = args.output_csv.resolve()

    if not geotiffs_root.exists():
        raise FileNotFoundError(f"geotiffs root not found: {geotiffs_root}")

    summary = build_manifest(geotiffs_root, output_csv)
    payload = {
        "geotiffs_root": str(geotiffs_root),
        "output_csv": str(output_csv),
        **summary,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

