import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.siamese_data import SiamesePairDataset
from app.ml.siamese_model import SiameseDamageNet


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available")
    return torch.device(device_arg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(split_csv: Path, *, img_size: int, max_bands: int, batch_size: int, num_workers: int, shuffle: bool) -> tuple[SiamesePairDataset, DataLoader]:
    dataset = SiamesePairDataset(split_csv, img_size=img_size, max_bands=max_bands)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def evaluate(model: SiameseDamageNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    probs_all: list[float] = []
    y_all: list[int] = []

    with torch.no_grad():
        for pre, post, y in loader:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            logits = model(pre, post)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.extend(probs.tolist())
            y_all.extend(y.numpy().astype(np.int64).tolist())

    probs_np = np.array(probs_all, dtype=np.float32)
    y_np = np.array(y_all, dtype=np.int64)
    preds = (probs_np >= 0.5).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_np, preds)),
        "precision": float(precision_score(y_np, preds, zero_division=0)),
        "recall": float(recall_score(y_np, preds, zero_division=0)),
        "f1": float(f1_score(y_np, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_np, probs_np))
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Siamese visual-only baseline for damage classification")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/train_split.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/val_split.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/test_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/siamese_visual"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--max-bands", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA",
    )
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    start = time.perf_counter()

    def _log(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[siamese +{elapsed:7.1f}s] {message}")

    _log(f"device={device} amp={use_amp}")
    _log("loading datasets")

    train_ds, train_loader = build_dataloader(
        args.train_csv,
        img_size=args.img_size,
        max_bands=args.max_bands,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    _, val_loader = build_dataloader(
        args.val_csv,
        img_size=args.img_size,
        max_bands=args.max_bands,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    _, test_loader = build_dataloader(
        args.test_csv,
        img_size=args.img_size,
        max_bands=args.max_bands,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    _log(
        "dataset shapes "
        f"train={len(train_ds)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    model = SiameseDamageNet(in_channels=args.max_bands).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    labels = train_ds.labels
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(1, pos))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, (pre, post, y) in enumerate(train_loader, start=1):
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(pre, post)
                loss = criterion(logits, y)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item() * args.grad_accum_steps)
            if args.log_every and (step % args.log_every == 0):
                _log(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss / step:.4f}")

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_report = {
            "epoch": float(epoch),
            "loss": float(running_loss / max(1, len(train_loader))),
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_report)
        _log(
            f"epoch={epoch} done loss={epoch_report['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_roc_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)
            _log(f"new best checkpoint saved: epoch={epoch} val_f1={best_val_f1:.4f}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    report = {
        "train": evaluate(model, train_loader, device),
        "val": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
        "config": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "img_size": args.img_size,
            "max_bands": args.max_bands,
            "device": str(device),
            "amp": use_amp,
            "num_workers": args.num_workers,
            "seed": args.seed,
        },
        "history": history,
        "artifacts": {
            "best_checkpoint": str(best_path.resolve()),
        },
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"training complete; metrics written: {metrics_path}")

    print(
        json.dumps(
            {
                "best_checkpoint": str(best_path.resolve()),
                "metrics_path": str(metrics_path.resolve()),
                **report["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

