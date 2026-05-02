"""
Tier 2 Training Loop
=====================
2-Phase Training:
  Phase 1: ImageNet pretrained backbone freeze, head만 학습 (5 epochs, lr=1e-3)
  Phase 2: 전체 unfreeze, fine-tuning (45 epochs, lr=3e-4)

Best Model 저장 기준: validation accuracy.
Early stopping: 7 epochs patience.

영길님의 RTX 3090/4090 환경 가정:
  - batch_size=64 → VRAM 약 8GB 사용
  - 50 epochs × 데이터 5년치 → 8~12시간
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TextColumn, TimeElapsedColumn)
from torch.utils.data import DataLoader

from flight_mind.config import MODEL_DIR, TIER2
from flight_mind.tier2_pattern.dataset import LABEL_NAMES, make_dataloaders
from flight_mind.tier2_pattern.model import FlightPatternCNN


CONSOLE = Console()


@dataclass
class TrainState:
    epoch: int = 0
    best_val_acc: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0
    history: list[dict] = field(default_factory=list)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights_from_dataset(dataset, n_classes: int = 3) -> torch.Tensor:
    """클래스 불균형 보정용 weight."""
    dist = dataset.class_distribution()
    counts = np.array([dist[name] for name in LABEL_NAMES], dtype=np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div by zero
    # inverse frequency, normalized
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = {0: 0, 1: 0, 2: 0}
    per_class_total = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            for cls in range(3):
                mask = (y == cls)
                per_class_total[cls] += mask.sum().item()
                per_class_correct[cls] += ((preds == y) & mask).sum().item()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    per_class_acc = {
        LABEL_NAMES[c]: per_class_correct[c] / max(per_class_total[c], 1)
        for c in range(3)
    }

    return {
        "loss": avg_loss,
        "acc": acc,
        "per_class_acc": per_class_acc,
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with Progress(
        TextColumn("[cyan]train[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=CONSOLE,
        transient=True,
    ) as prog:
        task = prog.add_task("batches", total=len(loader))

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            prog.advance(task)

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def train(
    symbols: list[str],
    epochs: int = TIER2.epochs,
    phase1_epochs: int = 5,
    batch_size: int = TIER2.batch_size,
    lr_phase1: float = 1e-3,
    lr_phase2: float = TIER2.lr,
    weight_decay: float = TIER2.weight_decay,
    patience: int = 7,
    save_path: Path | None = None,
) -> dict:
    """Full training pipeline. Returns final state dict + best metrics."""
    device = get_device()
    CONSOLE.print(f"[bold]Device: {device}[/bold]")

    save_path = save_path or (MODEL_DIR / "tier2_pattern_cnn.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Data
    CONSOLE.print(f"[cyan]Building datasets for {symbols}...[/cyan]")
    loaders = make_dataloaders(symbols, batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    train_ds = loaders["datasets"]["train"]
    CONSOLE.print(
        f"  Train: {len(train_ds):,}  |  "
        f"Val: {len(loaders['datasets']['val']):,}  |  "
        f"Test: {len(loaders['datasets']['test']):,}"
    )

    dist = train_ds.class_distribution()
    CONSOLE.print(f"  Train class dist: {dist}")

    # Model
    model = FlightPatternCNN(n_classes=3).to(device)
    CONSOLE.print(f"  Trainable params: {model.n_trainable_params():,}")

    # Loss with class weights
    class_w = class_weights_from_dataset(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # ===========================================================================
    # Phase 1: Backbone frozen, head only
    # ===========================================================================
    CONSOLE.print("\n[bold yellow]━━━ Phase 1: Head training (backbone frozen) ━━━[/bold yellow]")
    model.freeze_backbone()
    CONSOLE.print(f"  Trainable params (frozen backbone): {model.n_trainable_params():,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1, weight_decay=weight_decay,
    )

    state = TrainState()

    for ep in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0

        CONSOLE.print(
            f"  P1-Ep {ep:>2}/{phase1_epochs}  "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f}  "
            f"val_loss={val_metrics['loss']:.4f} [bold]val_acc={val_metrics['acc']:.3f}[/bold]  "
            f"({elapsed:.1f}s)"
        )

        state.history.append({
            "phase": 1, "epoch": ep,
            "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
        })

        if val_metrics["acc"] > state.best_val_acc:
            state.best_val_acc = val_metrics["acc"]
            state.best_epoch = ep
            torch.save({"model_state_dict": model.state_dict(),
                        "val_acc": val_metrics["acc"], "phase": 1, "epoch": ep},
                       save_path)
            CONSOLE.print(f"    [green]✓ Saved checkpoint (val_acc={val_metrics['acc']:.4f})[/green]")

    # ===========================================================================
    # Phase 2: Full unfreeze, fine-tuning
    # ===========================================================================
    CONSOLE.print("\n[bold yellow]━━━ Phase 2: Full fine-tuning (all params) ━━━[/bold yellow]")
    model.unfreeze_backbone()
    CONSOLE.print(f"  Trainable params: {model.n_trainable_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_phase2,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - phase1_epochs, eta_min=1e-6,
    )

    p2_epochs = epochs - phase1_epochs

    for ep in range(1, p2_epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        CONSOLE.print(
            f"  P2-Ep {ep:>2}/{p2_epochs}  "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f}  "
            f"val_loss={val_metrics['loss']:.4f} [bold]val_acc={val_metrics['acc']:.3f}[/bold]  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)"
        )

        state.history.append({
            "phase": 2, "epoch": ep,
            "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Early stopping
        if val_metrics["acc"] > state.best_val_acc:
            state.best_val_acc = val_metrics["acc"]
            state.best_epoch = phase1_epochs + ep
            state.patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_acc": val_metrics["acc"], "phase": 2, "epoch": ep},
                       save_path)
            CONSOLE.print(f"    [green]✓ Saved checkpoint (val_acc={val_metrics['acc']:.4f})[/green]")
        else:
            state.patience_counter += 1
            if state.patience_counter >= patience:
                CONSOLE.print(f"  [yellow]Early stopping at P2-Ep {ep} (patience={patience})[/yellow]")
                break

    # ===========================================================================
    # Final test evaluation
    # ===========================================================================
    CONSOLE.print("\n[bold yellow]━━━ Final Test Set Evaluation ━━━[/bold yellow]")
    # Load best checkpoint
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, loaders["test"], device, criterion)
    CONSOLE.print(f"  Test loss: {test_metrics['loss']:.4f}")
    CONSOLE.print(f"  Test acc:  [bold]{test_metrics['acc']:.4f}[/bold]")
    CONSOLE.print(f"  Per-class: {test_metrics['per_class_acc']}")

    return {
        "best_val_acc": state.best_val_acc,
        "best_epoch": state.best_epoch,
        "test_acc": test_metrics["acc"],
        "test_per_class": test_metrics["per_class_acc"],
        "history": state.history,
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--epochs", type=int, default=TIER2.epochs)
    p.add_argument("--batch-size", type=int, default=TIER2.batch_size)
    args = p.parse_args()

    result = train(
        symbols=args.symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    CONSOLE.print(f"\n[bold green]Training complete[/bold green]: {result}")
