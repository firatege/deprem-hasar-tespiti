import argparse
import json
import pickle
import sys
import time
from itertools import islice, product
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.scalar_baseline import load_dataset_from_split


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics


def _threshold_sweep(y_true: np.ndarray, probs: np.ndarray, thresholds: list[float]) -> tuple[float, list[dict[str, float]]]:
    best_threshold = 0.5
    best_f1 = -1.0
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        m = _binary_metrics(y_true, probs, threshold)
        row = {
            "threshold": float(threshold),
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
        }
        rows.append(row)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = float(threshold)

    return best_threshold, rows


def _hgb_grid(preset: str) -> list[dict[str, float | int]]:
    if preset == "conservative":
        learning_rates = [0.03, 0.05]
        max_depths = [4, 5, 6]
        max_iters = [300, 500]
        min_samples_leafs = [50, 100, 150]
        l2_regs = [0.5, 1.0, 3.0]
    else:
        learning_rates = [0.03, 0.05, 0.1]
        max_depths = [6, 8, 12]
        max_iters = [300, 500]
        min_samples_leafs = [20, 50]
        l2_regs = [0.0, 0.1, 1.0]

    grid: list[dict[str, float | int]] = []
    for lr, md, mi, msl, l2 in product(learning_rates, max_depths, max_iters, min_samples_leafs, l2_regs):
        grid.append(
            {
                "learning_rate": lr,
                "max_depth": md,
                "max_iter": mi,
                "min_samples_leaf": msl,
                "l2_regularization": l2,
            }
        )
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML v2 (HGB + threshold tuning) with existing dataset features")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/train_split.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/val_split.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/test_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/scalar_baseline_hgb_v2"))
    parser.add_argument("--max-bands", type=int, default=3)
    parser.add_argument("--image-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preset", choices=["conservative", "default"], default="conservative")
    parser.add_argument("--max-candidates", type=int, default=0, help="If >0, evaluate only first N grid candidates")
    parser.add_argument("--hard-negative-mining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hnm-threshold", type=float, default=0.6)
    parser.add_argument("--hnm-weight", type=float, default=3.0)
    parser.add_argument("--hnm-max-fraction", type=float, default=0.2)
    parser.add_argument("--threshold-min", type=float, default=0.25)
    parser.add_argument("--threshold-max", type=float, default=0.65)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    start = time.perf_counter()

    def _log(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[v2 +{elapsed:7.1f}s] {message}")

    _log("loading train split")
    x_train, y_train, _ = load_dataset_from_split(args.train_csv, max_bands=args.max_bands, progress_every=args.log_every, log_fn=_log)
    _log("loading val split")
    x_val, y_val, _ = load_dataset_from_split(args.val_csv, max_bands=args.max_bands, progress_every=args.log_every, log_fn=_log)
    _log("loading test split")
    x_test, y_test, _ = load_dataset_from_split(args.test_csv, max_bands=args.max_bands, progress_every=args.log_every, log_fn=_log)

    feature_dim_before = int(x_train.shape[1])
    if args.image_only:
        image_dim = args.max_bands * 8
        x_train = x_train[:, :image_dim]
        x_val = x_val[:, :image_dim]
        x_test = x_test[:, :image_dim]
        _log(f"image-only mode enabled: feature_dim {feature_dim_before} -> {x_train.shape[1]}")

    thresholds_np = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    thresholds = [float(np.round(t, 6)) for t in thresholds_np]

    best_payload: dict[str, object] | None = None
    candidates: list[dict[str, object]] = []

    grid = _hgb_grid(args.preset)
    if args.max_candidates > 0:
        grid = list(islice(grid, args.max_candidates))
    _log(f"evaluating {len(grid)} HGB candidates (preset={args.preset})")

    for idx, params in enumerate(grid, start=1):
        model = HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            max_iter=int(params["max_iter"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=42,
        )
        model.fit(x_train, y_train)

        hnm_summary = {
            "enabled": bool(args.hard_negative_mining),
            "threshold": args.hnm_threshold,
            "weight": args.hnm_weight,
            "max_fraction": args.hnm_max_fraction,
            "selected_count": 0,
            "selected_fraction": 0.0,
        }
        if args.hard_negative_mining:
            train_probs_pre = model.predict_proba(x_train)[:, 1]
            neg_mask = y_train == 0
            candidate_idx = np.where(neg_mask & (train_probs_pre >= args.hnm_threshold))[0]
            if candidate_idx.size > 0:
                max_select = int(max(1, len(y_train) * args.hnm_max_fraction))
                if candidate_idx.size > max_select:
                    sorted_idx = candidate_idx[np.argsort(train_probs_pre[candidate_idx])[::-1]]
                    selected_idx = sorted_idx[:max_select]
                else:
                    selected_idx = candidate_idx

                sample_weight = np.ones(len(y_train), dtype=np.float64)
                sample_weight[selected_idx] = args.hnm_weight

                model = HistGradientBoostingClassifier(
                    learning_rate=float(params["learning_rate"]),
                    max_depth=int(params["max_depth"]),
                    max_iter=int(params["max_iter"]),
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    l2_regularization=float(params["l2_regularization"]),
                    random_state=42,
                )
                model.fit(x_train, y_train, sample_weight=sample_weight)
                hnm_summary["selected_count"] = int(selected_idx.size)
                hnm_summary["selected_fraction"] = float(selected_idx.size / len(y_train))

        probs_val = model.predict_proba(x_val)[:, 1]
        best_thr, sweep_rows = _threshold_sweep(y_val, probs_val, thresholds)
        val_metrics = _binary_metrics(y_val, probs_val, best_thr)

        candidate = {
            "params": params,
            "best_threshold": best_thr,
            "val_metrics": val_metrics,
            "val_sweep": sweep_rows,
            "hard_negative_mining": hnm_summary,
            "sort_key": (val_metrics["f1"], val_metrics["roc_auc"]),
        }
        candidates.append(candidate)

        _log(
            f"grid {idx}/{len(grid)} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f} threshold={best_thr:.2f}"
        )

        if best_payload is None:
            best_payload = {"model": model, **candidate}
        else:
            current_key = candidate["sort_key"]
            best_key = best_payload["sort_key"]
            if current_key > best_key:
                best_payload = {"model": model, **candidate}

    if best_payload is None:
        raise ValueError("No model candidates were evaluated")

    model = best_payload["model"]
    best_threshold = float(best_payload["best_threshold"])

    probs_train = model.predict_proba(x_train)[:, 1]
    probs_val = model.predict_proba(x_val)[:, 1]
    probs_test = model.predict_proba(x_test)[:, 1]

    metrics_05 = {
        "train": _binary_metrics(y_train, probs_train, 0.5),
        "val": _binary_metrics(y_val, probs_val, 0.5),
        "test": _binary_metrics(y_test, probs_test, 0.5),
    }
    metrics_best_threshold = {
        "train": _binary_metrics(y_train, probs_train, best_threshold),
        "val": _binary_metrics(y_val, probs_val, best_threshold),
        "test": _binary_metrics(y_test, probs_test, best_threshold),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pkl"
    metrics_path = args.output_dir / "metrics.json"
    inference_config_path = args.output_dir / "inference_config.json"
    summary_path = args.output_dir / "training_summary.json"

    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    report = {
        "selection": {
            "objective": "max val.f1, tie-breaker val.roc_auc",
            "best_params": best_payload["params"],
            "best_threshold": best_threshold,
        },
        "metrics@0.5": metrics_05,
        "metrics@best_threshold": metrics_best_threshold,
        "config": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()),
            "max_bands": args.max_bands,
            "image_only": args.image_only,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "preset": args.preset,
            "hard_negative_mining": args.hard_negative_mining,
            "hnm_threshold": args.hnm_threshold,
            "hnm_weight": args.hnm_weight,
            "hnm_max_fraction": args.hnm_max_fraction,
        },
        "shape": {
            "x_train": list(x_train.shape),
            "x_val": list(x_val.shape),
            "x_test": list(x_test.shape),
        },
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    inference_config = {
        "threshold": best_threshold,
        "max_bands": args.max_bands,
        "image_only": args.image_only,
        "normalize_pga": False,
    }
    inference_config_path.write_text(json.dumps(inference_config, indent=2), encoding="utf-8")

    compact_candidates = [
        {
            "params": item["params"],
            "best_threshold": item["best_threshold"],
            "val_f1": item["val_metrics"]["f1"],
            "val_roc_auc": item["val_metrics"]["roc_auc"],
            "hard_negative_mining": item["hard_negative_mining"],
        }
        for item in sorted(candidates, key=lambda c: c["sort_key"], reverse=True)
    ]
    summary_path.write_text(json.dumps({"top_candidates": compact_candidates[:20]}, indent=2), encoding="utf-8")

    _log("artifacts written")
    print(
        json.dumps(
            {
                "model_path": str(model_path.resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "inference_config_path": str(inference_config_path.resolve()),
                "best_threshold": best_threshold,
                **metrics_best_threshold["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


