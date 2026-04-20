import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.scalar_baseline import build_scalar_features, compute_change_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one inference with scalar baseline model")
    parser.add_argument("--model-path", type=Path, default=Path("models/scalar_baseline/model.pkl"))
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=None,
        help="Optional JSON with threshold/max_bands/image_only/normalize_pga",
    )
    parser.add_argument("--pre-image", type=Path, required=True)
    parser.add_argument("--post-image", type=Path, required=True)
    parser.add_argument("--pga-value", type=float, default=0.0)
    parser.add_argument("--magnitude", type=float, default=0.0)
    parser.add_argument("--depth-km", type=float, default=0.0)
    parser.add_argument("--acquisition-delta-days", type=float, default=0.0)
    parser.add_argument("--building-density", type=float, default=0.0)
    parser.add_argument(
        "--normalize-pga",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply signed log1p normalization to PGA before model scaling",
    )
    parser.add_argument(
        "--image-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only image features (ignore scalar/context features)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-bands", type=int, default=3)
    args = parser.parse_args()

    if args.inference_config is not None:
        cfg = json.loads(args.inference_config.read_text(encoding="utf-8"))
        if "max_bands" in cfg:
            args.max_bands = int(cfg["max_bands"])
        if "image_only" in cfg:
            args.image_only = bool(cfg["image_only"])
        if "normalize_pga" in cfg:
            args.normalize_pga = bool(cfg["normalize_pga"])
        if "threshold" in cfg:
            args.threshold = float(cfg["threshold"])

    with args.model_path.open("rb") as handle:
        model = pickle.load(handle)

    x = compute_change_features(args.pre_image, args.post_image, max_bands=args.max_bands)
    if not args.image_only:
        x.extend(
            build_scalar_features(
                pga_value=args.pga_value,
                magnitude=args.magnitude,
                depth_km=args.depth_km,
                acquisition_delta_days=args.acquisition_delta_days,
                building_density=args.building_density,
                normalize_pga=args.normalize_pga,
            )
        )
    x_np = np.array([x], dtype=np.float32)

    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != int(x_np.shape[1]):
        raise ValueError(
            "Feature dimension mismatch: "
            f"model expects {int(model.n_features_in_)} but got {int(x_np.shape[1])}. "
            "Check --image-only/--max-bands and inference config."
        )

    prob = float(model.predict_proba(x_np)[0, 1])
    pred = int(prob >= args.threshold)

    print(
        json.dumps(
            {
                "probability": prob,
                "threshold": args.threshold,
                "binary_damage": pred,
                "policy": "xview2_major_destroyed_is_one",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

