"""
Tier 4 Training Loop
=====================
Tier 2의 학습 패턴을 그대로 따르되, 단일 페이즈로 단순화 (Transformer는 처음부터 학습).

영길님 환경(RTX 3090/4090):
  - batch=128, seq_len=30, d_model=256
  - VRAM ~6GB
  - 5년치 일봉 (페어당 1825샘플 × 2페어 ≈ 3650) → 학습 매우 빠름 (40 epoch ≈ 1~2시간)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

from flight_mind.config import MODEL_DIR, TIER4
from flight_mind.tier4_regime.dataset import RegimeDataset, make_dataloaders
from flight_mind.tier4_regime.labeler import IDX_TO_REGIME, REGIMES
from flight_mind.tier4_regime.model import RegimeTransformer


CONSOLE = Console()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights_from_dataset(dataset: RegimeDataset, n_classes: int = 5) -> torch.Tensor:
    dist = dataset.class_distribution()
    counts = np.array([dist[REGIMES[i]] for i in range(n_classes)], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = {i: 0 for i in range(5)}
    per_class_total = {i: 0 for i in range(5)}

    with torch.no_grad():
        for x, sym, y in loader:
            x = x.to(device, non_blocking=True)
            sym = sym.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x, sym)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            for cls in range(5):
                mask = (y == cls)
                per_class_total[cls] += mask.sum().item()
                per_class_correct[cls] += ((preds == y) & mask).sum().item()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    per_class_acc = {
        IDX_TO_REGIME[c]: per_class_correct[c] / max(per_class_total[c], 1)
        for c in range(5)
    }
    return {"loss": avg_loss, "acc": acc, "per_class_acc": per_class_acc}


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, sym, y in loader:
        x = x.to(device, non_blocking=True)
        sym = sym.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x, sym)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def train(
    symbols: list[str],
    epochs: int = TIER4.epochs,
    batch_size: int = TIER4.batch_size,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    patience: int = 8,
    save_path: Path | None = None,
) -> dict:
    device = get_device()
    CONSOLE.print(f"[bold]Device: {device}[/bold]")

    save_path = save_path or (MODEL_DIR / "tier4_regime_transformer.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    CONSOLE.print(f"[cyan]Building datasets for {symbols}...[/cyan]")
    loaders = make_dataloaders(symbols, batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    train_ds = loaders["datasets"]["train"]

    n_train = len(train_ds)
    n_val = len(loaders["datasets"]["val"])
    n_test = len(loaders["datasets"]["test"])
    CONSOLE.print(f"  Train: {n_train:,}  |  Val: {n_val:,}  |  Test: {n_test:,}")

    if n_train == 0:
        raise RuntimeError("No training data — check symbol coverage and data freshness")

    dist = train_ds.class_distribution()
    CONSOLE.print(f"  Train regime distribution:")
    for k, v in dist.items():
        pct = v / max(n_train, 1) * 100
        CONSOLE.print(f"    {k:18s}: {v:>5,} ({pct:5.1f}%)")

    # Model
    from flight_mind.tier4_regime.model import RegimeTransformer
    model = RegimeTransformer(
        n_features=10,
        seq_len=TIER4.lookback_days,
        n_classes=5,
        n_pairs=len(symbols),
        d_model=TIER4.d_model,
        n_heads=TIER4.n_heads,
        n_layers=TIER4.n_layers,
    ).to(device)
    CONSOLE.print(f"  Trainable params: {model.n_trainable_params():,}")

    # Loss with class weights
    class_w = class_weights_from_dataset(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    CONSOLE.print(f"\n[bold yellow]━━━ Training {epochs} epochs ━━━[/bold yellow]")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        CONSOLE.print(
            f"  Ep {ep:>3}/{epochs}  "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f}  "
            f"val_loss={val_metrics['loss']:.4f} [bold]val_acc={val_metrics['acc']:.3f}[/bold]  "
            f"({elapsed:.1f}s)"
        )

        history.append({
            "epoch": ep,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = ep
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_metrics["acc"],
                "epoch": ep,
                "symbols": symbols,
            }, save_path)
            CONSOLE.print(f"    [green]✓ Saved checkpoint (val_acc={val_metrics['acc']:.4f})[/green]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                CONSOLE.print(f"  [yellow]Early stopping at Ep {ep} (patience={patience})[/yellow]")
                break

    # Final test evaluation
    CONSOLE.print("\n[bold yellow]━━━ Final Test Set Evaluation ━━━[/bold yellow]")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device, criterion)

    CONSOLE.print(f"  Test loss: {test_metrics['loss']:.4f}")
    CONSOLE.print(f"  Test acc:  [bold]{test_metrics['acc']:.4f}[/bold]")
    CONSOLE.print(f"  Per-class:")
    for k, v in test_metrics["per_class_acc"].items():
        CONSOLE.print(f"    {k:18s}: {v:.3f}")

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_metrics["acc"],
        "test_per_class": test_metrics["per_class_acc"],
        "history": history,
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--epochs", type=int, default=TIER4.epochs)
    p.add_argument("--batch-size", type=int, default=TIER4.batch_size)
    args = p.parse_args()

    result = train(
        symbols=args.symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    CONSOLE.print(f"\n[bold green]Training complete[/bold green]: {result}")
