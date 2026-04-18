import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.scalar_baseline import load_dataset_from_split


SUPPORTED_MODEL_TYPES = (
    "logreg",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
)


def build_model(model_type: str):
    if model_type == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            random_state=42,
        )
    raise ValueError(f"Unsupported model_type={model_type!r}; expected one of {list(SUPPORTED_MODEL_TYPES)}")


def _evaluate(model: Pipeline, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight pre/post+scalar PGA baseline")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/train_split.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/val_split.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/test_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/scalar_baseline"))
    parser.add_argument("--pga-default", type=float, default=0.0)
    parser.add_argument("--magnitude-default", type=float, default=0.0)
    parser.add_argument("--depth-km-default", type=float, default=0.0)
    parser.add_argument("--acquisition-delta-days-default", type=float, default=0.0)
    parser.add_argument("--building-density-default", type=float, default=0.0)
    parser.add_argument(
        "--normalize-pga",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply signed log1p normalization to PGA before model scaling",
    )
    parser.add_argument("--max-bands", type=int, default=3)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=SUPPORTED_MODEL_TYPES,
        default="logreg",
        help="Classifier family to train on extracted features",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print data loading progress every N samples (0 disables progress logs)",
    )
    args = parser.parse_args()

    start = time.perf_counter()

    def _log(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[train +{elapsed:7.1f}s] {message}")

    _log("loading train split")
    t0 = time.perf_counter()

    x_train, y_train, _ = load_dataset_from_split(
        args.train_csv,
        pga_default=args.pga_default,
        magnitude_default=args.magnitude_default,
        depth_km_default=args.depth_km_default,
        acquisition_delta_days_default=args.acquisition_delta_days_default,
        building_density_default=args.building_density_default,
        max_bands=args.max_bands,
        normalize_pga=args.normalize_pga,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"train split loaded in {time.perf_counter() - t0:.1f}s; x_train={list(x_train.shape)}")

    _log("loading val split")
    t0 = time.perf_counter()
    x_val, y_val, _ = load_dataset_from_split(
        args.val_csv,
        pga_default=args.pga_default,
        magnitude_default=args.magnitude_default,
        depth_km_default=args.depth_km_default,
        acquisition_delta_days_default=args.acquisition_delta_days_default,
        building_density_default=args.building_density_default,
        max_bands=args.max_bands,
        normalize_pga=args.normalize_pga,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"val split loaded in {time.perf_counter() - t0:.1f}s; x_val={list(x_val.shape)}")

    _log("loading test split")
    t0 = time.perf_counter()
    x_test, y_test, _ = load_dataset_from_split(
        args.test_csv,
        pga_default=args.pga_default,
        magnitude_default=args.magnitude_default,
        depth_km_default=args.depth_km_default,
        acquisition_delta_days_default=args.acquisition_delta_days_default,
        building_density_default=args.building_density_default,
        max_bands=args.max_bands,
        normalize_pga=args.normalize_pga,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"test split loaded in {time.perf_counter() - t0:.1f}s; x_test={list(x_test.shape)}")

    model = build_model(args.model_type)
    _log(f"model training started (model_type={args.model_type})")
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    _log(f"model training finished in {time.perf_counter() - t0:.1f}s")

    _log("evaluating train/val/test metrics")
    t0 = time.perf_counter()
    report = {
        "train": _evaluate(model, x_train, y_train),
        "val": _evaluate(model, x_val, y_val),
        "test": _evaluate(model, x_test, y_test),
        "config": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()),
            "pga_default": args.pga_default,
            "magnitude_default": args.magnitude_default,
            "depth_km_default": args.depth_km_default,
            "acquisition_delta_days_default": args.acquisition_delta_days_default,
            "building_density_default": args.building_density_default,
            "max_bands": args.max_bands,
            "normalize_pga": args.normalize_pga,
            "model_type": args.model_type,
        },
        "shape": {
            "x_train": list(x_train.shape),
            "x_val": list(x_val.shape),
            "x_test": list(x_test.shape),
        },
    }
    _log(f"evaluation finished in {time.perf_counter() - t0:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pkl"
    metrics_path = args.output_dir / "metrics.json"

    t0 = time.perf_counter()
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"artifacts written in {time.perf_counter() - t0:.1f}s")
    _log(f"all done in {time.perf_counter() - start:.1f}s")

    print(json.dumps({"model_path": str(model_path.resolve()), "metrics_path": str(metrics_path.resolve()), **report["test"]}, indent=2))


if __name__ == "__main__":
    main()

