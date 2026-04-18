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
    parser.add_argument("--max-bands", type=int, default=3)
    args = parser.parse_args()

    with args.model_path.open("rb") as handle:
        model = pickle.load(handle)

    x = compute_change_features(args.pre_image, args.post_image, max_bands=args.max_bands)
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

    prob = float(model.predict_proba(x_np)[0, 1])
    pred = int(prob >= 0.5)

    print(json.dumps({"probability": prob, "binary_damage": pred, "policy": "xview2_major_destroyed_is_one"}, indent=2))


if __name__ == "__main__":
    main()

