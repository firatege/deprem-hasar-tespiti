import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.scalar_baseline import build_scalar_features, compute_change_features


def _load_inference_config(path: Path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {
        "threshold": float(cfg.get("threshold", 0.5)),
        "max_bands": int(cfg.get("max_bands", 3)),
        "image_only": bool(cfg.get("image_only", False)),
        "normalize_pga": bool(cfg.get("normalize_pga", True)),
    }


def _predict(model, inf_cfg, pre_path: Path, post_path: Path):
    x = compute_change_features(pre_path, post_path, max_bands=inf_cfg["max_bands"])
    if not inf_cfg["image_only"]:
        x.extend(
            build_scalar_features(
                pga_value=0.0,
                magnitude=0.0,
                depth_km=0.0,
                acquisition_delta_days=0.0,
                building_density=0.0,
                normalize_pga=inf_cfg["normalize_pga"],
            )
        )
    x_np = np.array([x], dtype=np.float32)
    prob = float(model.predict_proba(x_np)[0, 1])
    pred = int(prob >= inf_cfg["threshold"])
    return prob, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Score already-downloaded pre/post cropped TIFF pairs")
    parser.add_argument("--pairs-dir", default="tmp/maxar_trial_batch")
    parser.add_argument("--model-path", default="models/scalar_baseline_hgb_v21/model.pkl")
    parser.add_argument("--inference-config", default="models/scalar_baseline_hgb_v21/inference_config_prod.json")
    parser.add_argument("--out-csv", default="data/processed/maxar_existing_pairs_scored.csv")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        raise RuntimeError(f"pairs dir not found: {pairs_dir}")

    with Path(args.model_path).open("rb") as f:
        model = pickle.load(f)
    inf_cfg = _load_inference_config(Path(args.inference_config))

    pre_files = sorted(pairs_dir.glob("pair_*_pre_cropped.tif"))
    if not pre_files:
        raise RuntimeError("No pair_*_pre_cropped.tif files found")

    rows = []
    for pre in pre_files:
        post = pre.with_name(pre.name.replace("_pre_cropped.tif", "_post_cropped.tif"))
        if not post.exists():
            continue
        pair_id = pre.name.replace("_pre_cropped.tif", "")
        try:
            prob, pred = _predict(model, inf_cfg, pre, post)
            rows.append(
                {
                    "rank": 0,
                    "pair_id": pair_id,
                    "probability": prob,
                    "threshold": inf_cfg["threshold"],
                    "binary_damage": pred,
                    "pre_cropped_tif": str(pre.resolve()),
                    "post_cropped_tif": str(post.resolve()),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "rank": 0,
                    "pair_id": pair_id,
                    "probability": "",
                    "threshold": inf_cfg["threshold"],
                    "binary_damage": "error",
                    "pre_cropped_tif": str(pre.resolve()),
                    "post_cropped_tif": str(post.resolve()),
                    "error": str(exc),
                }
            )

    ok_rows = [r for r in rows if isinstance(r.get("probability"), float)]
    ok_rows.sort(key=lambda x: x["probability"], reverse=True)
    top = ok_rows[: args.top_k]
    for i, row in enumerate(top, start=1):
        row["rank"] = i

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "pair_id",
        "probability",
        "threshold",
        "binary_damage",
        "pre_cropped_tif",
        "post_cropped_tif",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top)

    print(f"scored_pairs={len(ok_rows)}")
    print(f"wrote_top={len(top)}")
    print(f"out_csv={out_csv.resolve()}")
    if top:
        print(f"best_pair={top[0]['pair_id']} prob={top[0]['probability']:.4f} pred={top[0]['binary_damage']}")


if __name__ == "__main__":
    main()

