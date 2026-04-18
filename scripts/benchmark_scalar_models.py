import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.scalar_baseline import load_dataset_from_split
from scripts.train_scalar_baseline import SUPPORTED_MODEL_TYPES, _evaluate, build_model


def _score_tuple(report: dict[str, dict[str, float]], model_name: str) -> tuple[float, float, str]:
    test = report["test"]
    return (test["f1"], test["roc_auc"], model_name)


def _parse_model_types(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in SUPPORTED_MODEL_TYPES]
    if invalid:
        raise ValueError(f"Unsupported model types: {invalid}; supported={list(SUPPORTED_MODEL_TYPES)}")
    if not items:
        raise ValueError("No model types were provided")
    # Keep order but drop duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple classifiers on scalar baseline features")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/subset_500/train_split.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/subset_500/val_split.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/subset_500/test_split.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("models/scalar_baseline/model_benchmark.json"))
    parser.add_argument("--max-bands", type=int, default=3)
    parser.add_argument(
        "--model-types",
        type=str,
        default=",".join(SUPPORTED_MODEL_TYPES),
        help="Comma-separated model list. Example: logreg,random_forest,extra_trees",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print dataset loader progress every N samples (0 disables)",
    )
    args = parser.parse_args()

    model_types = _parse_model_types(args.model_types)
    start = time.perf_counter()

    def _log(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[model-bench +{elapsed:7.1f}s] {message}")

    _log(f"loading train split: {args.train_csv}")
    x_train, y_train, _ = load_dataset_from_split(
        args.train_csv,
        max_bands=args.max_bands,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"train loaded shape={list(x_train.shape)}")

    _log(f"loading val split: {args.val_csv}")
    x_val, y_val, _ = load_dataset_from_split(
        args.val_csv,
        max_bands=args.max_bands,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"val loaded shape={list(x_val.shape)}")

    _log(f"loading test split: {args.test_csv}")
    x_test, y_test, _ = load_dataset_from_split(
        args.test_csv,
        max_bands=args.max_bands,
        progress_every=args.log_every,
        log_fn=_log,
    )
    _log(f"test loaded shape={list(x_test.shape)}")

    results: dict[str, dict[str, dict[str, float]]] = {}
    ranking_rows: list[tuple[float, float, str]] = []

    for model_type in model_types:
        _log(f"training model_type={model_type}")
        model = build_model(model_type)
        model.fit(x_train, y_train)
        report = {
            "train": _evaluate(model, x_train, y_train),
            "val": _evaluate(model, x_val, y_val),
            "test": _evaluate(model, x_test, y_test),
        }
        results[model_type] = report
        ranking_rows.append(_score_tuple(report, model_type))
        _log(
            "done "
            f"model_type={model_type} "
            f"test_f1={report['test']['f1']:.4f} "
            f"test_roc_auc={report['test']['roc_auc']:.4f}"
        )

    ranking = sorted(ranking_rows, reverse=True)
    best_model_type = ranking[0][2]

    output = {
        "data": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()),
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
        },
        "config": {
            "max_bands": args.max_bands,
            "model_types": model_types,
        },
        "models": results,
        "ranking": [
            {"model_type": model_name, "test_f1": f1, "test_roc_auc": roc_auc}
            for f1, roc_auc, model_name in ranking
        ],
        "best_model": {
            "model_type": best_model_type,
            "selection_rule": "max test.f1, tie-breaker max test.roc_auc",
            "metrics": results[best_model_type],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _log(f"benchmark written: {args.output_json}")
    print(json.dumps({"best_model": best_model_type, "output_json": str(args.output_json.resolve())}, indent=2))


if __name__ == "__main__":
    main()

