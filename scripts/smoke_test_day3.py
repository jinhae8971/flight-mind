"""
Day 3 — Tier 2 Training Smoke Test
====================================
30일치 BTC + ETH 데이터로 mini training run을 돌려서
학습 파이프라인이 처음부터 끝까지 흐르는지 확인.

GPU가 없으므로 CPU에서 매우 짧게 실행:
  - 3 epochs (1 phase1 + 2 phase2)
  - batch_size=32 (CPU 메모리 절약)

실제 학습은 영길님의 RTX 3090/4090에서:
  python -m flight_mind.tier2_pattern.train --symbols BTCUSDT ETHUSDT --epochs 50
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from rich.console import Console

from flight_mind.tier2_pattern.train import train


CONSOLE = Console()


def main() -> int:
    CONSOLE.print("[bold cyan]━━━ Day 3 Tier 2 Training Smoke Test ━━━[/bold cyan]")
    CONSOLE.print(f"PyTorch: {torch.__version__}")
    CONSOLE.print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    CONSOLE.print()

    # CPU에서 짧게 실행 — 학습 파이프라인의 정상 동작만 확인
    save_path = Path("data/models/tier2_smoketest.pt")

    try:
        result = train(
            symbols=["BTCUSDT"],     # 1 페어만 (CPU 시간 절약)
            epochs=2,                # 매우 짧게 (Phase 1 + Phase 2 각 1 epoch)
            phase1_epochs=1,
            batch_size=64,           # 더 큰 batch로 CPU 효율 ↑
            patience=99,             # 짧은 학습이므로 early stop 비활성화
            save_path=save_path,
        )
    except Exception as e:
        CONSOLE.print(f"[red]Training failed: {type(e).__name__}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1

    CONSOLE.print("\n[bold green]━━━ Smoke Test Result ━━━[/bold green]")
    CONSOLE.print(f"  Best Val Acc : {result['best_val_acc']:.4f}")
    CONSOLE.print(f"  Best Epoch   : {result['best_epoch']}")
    CONSOLE.print(f"  Test Acc     : {result['test_acc']:.4f}")
    CONSOLE.print(f"  Per-class    : {result['test_per_class']}")
    CONSOLE.print(f"  Saved to     : {result['save_path']}")
    CONSOLE.print(f"  History len  : {len(result['history'])} epochs")

    # Sanity checks
    assert result["best_val_acc"] >= 0.20, "Val acc too low — model not learning?"
    assert Path(result["save_path"]).exists(), "Checkpoint not saved"

    CONSOLE.print("\n[bold green]✅ Day 3 training pipeline works end-to-end[/bold green]")
    CONSOLE.print(
        "[dim]Note: 30d × 3 epochs is too small for real performance. "
        "Run 50 epochs on 5y data on RTX 3090/4090 for production.[/dim]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
