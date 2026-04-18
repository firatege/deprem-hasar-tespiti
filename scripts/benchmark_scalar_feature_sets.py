import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.scalar_baseline import load_dataset_from_split


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


def _evaluate_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def _to_legacy_like(x: np.ndarray, max_bands: int) -> np.ndarray:
    """Map full vector to pre-change schema: 6 stats/band + PGA only."""
    image_dim_full = max_bands * 8
    legacy_image_parts = []
    for idx in range(max_bands):
        band_start = idx * 8
        legacy_image_parts.append(x[:, band_start : band_start + 6])
    legacy_images = np.concatenate(legacy_image_parts, axis=1)
    pga_col = x[:, image_dim_full : image_dim_full + 1]
    return np.concatenate([legacy_images, pga_col], axis=1).astype(np.float32)


def _fit_and_eval(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    model = Pipeline(
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
    model.fit(x_train, y_train)
    return {
        "val": _evaluate(model, x_val, y_val),
        "test": _evaluate(model, x_test, y_test),
    }


def _fit_model(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
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
    model.fit(x_train, y_train)
    return model


def _fit_and_eval_with_sweep(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: list[float],
) -> dict[str, object]:
    model = _fit_model(x_train, y_train)
    probs_val = model.predict_proba(x_val)[:, 1]
    probs_test = model.predict_proba(x_test)[:, 1]

    baseline = {
        "val": _evaluate_probs(y_val, probs_val, threshold=0.5) | {"roc_auc": float(roc_auc_score(y_val, probs_val))},
        "test": _evaluate_probs(y_test, probs_test, threshold=0.5) | {"roc_auc": float(roc_auc_score(y_test, probs_test))},
    }

    sweep_rows: list[dict[str, float]] = []
    best = {"threshold": 0.5, "f1": baseline["val"]["f1"]}
    for thr in thresholds:
        val_m = _evaluate_probs(y_val, probs_val, threshold=thr)
        test_m = _evaluate_probs(y_test, probs_test, threshold=thr)
        sweep_rows.append(
            {
                "threshold": float(thr),
                "val_f1": val_m["f1"],
                "val_precision": val_m["precision"],
                "val_recall": val_m["recall"],
                "test_f1": test_m["f1"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
            }
        )
        if val_m["f1"] > best["f1"]:
            best = {"threshold": float(thr), "f1": float(val_m["f1"])}

    best_thr = float(best["threshold"])
    best_val = _evaluate_probs(y_val, probs_val, threshold=best_thr)
    best_test = _evaluate_probs(y_test, probs_test, threshold=best_thr)

    return {
        "metrics@0.5": baseline,
        "sweep": sweep_rows,
        "best_threshold_by_val_f1": {
            "threshold": best_thr,
            "val": best_val,
            "test": best_test,
        },
    }


def _zero_building_density(x: np.ndarray, max_bands: int) -> np.ndarray:
    x_new = x.copy()
    building_density_idx = max_bands * 8 + 4
    x_new[:, building_density_idx] = 0.0
    return x_new


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy-like vs full scalar baseline features")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/subset_500/train_split.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/subset_500/val_split.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/subset_500/test_split.csv"))
    parser.add_argument("--max-bands", type=int, default=3)
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.30,
        help="Minimum threshold value for sweep",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.70,
        help="Maximum threshold value for sweep",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Threshold step for sweep",
    )
    parser.add_argument(
        "--skip-building-density-ablation",
        action="store_true",
        help="Skip full-vs-no-building-density ablation",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print loader progress every N samples (0 disables)",
    )
    parser.add_argument("--output-json", type=Path, default=Path("models/scalar_baseline/feature_set_benchmark.json"))
    args = parser.parse_args()

    start = time.perf_counter()

    def _log(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[bench +{elapsed:7.1f}s] {message}")

    _log("loading train split")
    x_train, y_train, _ = load_dataset_from_split(
        args.train_csv,
        max_bands=args.max_bands,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"train loaded shape={list(x_train.shape)}")

    _log("loading val split")
    x_val, y_val, _ = load_dataset_from_split(
        args.val_csv,
        max_bands=args.max_bands,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"val loaded shape={list(x_val.shape)}")

    _log("loading test split")
    x_test, y_test, _ = load_dataset_from_split(
        args.test_csv,
        max_bands=args.max_bands,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"test loaded shape={list(x_test.shape)}")

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    threshold_values = [float(np.round(t, 4)) for t in thresholds]

    results_full = _fit_and_eval(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    _log("full feature model fitted/evaluated")

    x_train_legacy = _to_legacy_like(x_train, max_bands=args.max_bands)
    x_val_legacy = _to_legacy_like(x_val, max_bands=args.max_bands)
    x_test_legacy = _to_legacy_like(x_test, max_bands=args.max_bands)
    results_legacy = _fit_and_eval(
        x_train=x_train_legacy,
        y_train=y_train,
        x_val=x_val_legacy,
        y_val=y_val,
        x_test=x_test_legacy,
        y_test=y_test,
    )
    _log("legacy-like feature model fitted/evaluated")

    sweep_full = _fit_and_eval_with_sweep(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        thresholds=threshold_values,
    )
    _log("full feature threshold sweep done")

    ablation_report: dict[str, object] | None = None
    if not args.skip_building_density_ablation:
        x_train_no_bd = _zero_building_density(x_train, max_bands=args.max_bands)
        x_val_no_bd = _zero_building_density(x_val, max_bands=args.max_bands)
        x_test_no_bd = _zero_building_density(x_test, max_bands=args.max_bands)
        no_bd_metrics = _fit_and_eval(
            x_train=x_train_no_bd,
            y_train=y_train,
            x_val=x_val_no_bd,
            y_val=y_val,
            x_test=x_test_no_bd,
            y_test=y_test,
        )
        ablation_report = {
            "full": results_full,
            "full_no_building_density": no_bd_metrics,
            "delta_no_bd_minus_full": {
                "val": {k: float(no_bd_metrics["val"][k] - results_full["val"][k]) for k in results_full["val"]},
                "test": {k: float(no_bd_metrics["test"][k] - results_full["test"][k]) for k in results_full["test"]},
            },
        }
        _log("building density ablation done")

    report = {
        "data": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()),
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "test_samples": int(x_test.shape[0]),
        },
        "feature_dims": {
            "legacy_like": int(x_train_legacy.shape[1]),
            "full": int(x_train.shape[1]),
        },
        "legacy_like": results_legacy,
        "full": results_full,
        "delta_full_minus_legacy": {
            "val": {k: float(results_full["val"][k] - results_legacy["val"][k]) for k in results_full["val"]},
            "test": {k: float(results_full["test"][k] - results_legacy["test"][k]) for k in results_full["test"]},
        },
        "threshold_sweep": {
            "thresholds": threshold_values,
            "full": sweep_full,
        },
        "ablation": ablation_report,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"report written: {args.output_json}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()



