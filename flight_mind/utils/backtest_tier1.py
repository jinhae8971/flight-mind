"""
Tier 1 Standalone Backtester
=============================
Tier 2~4 학습 전, Tier 1 룰만 단독으로 5년 데이터에 적용해보는 백테스트.

목적:
  1. Tier 1 룰의 baseline 승률/손익비 측정
  2. Tier 2~4가 추가됐을 때의 marginal lift 비교 기준선 수립
  3. 룰 자체에 버그가 있는지 조기 발견

설계:
  - Walk-forward: 매 5분봉마다 evaluate_tier1 호출
  - Position: 단순화 — 신호 발생 시 즉시 진입, +6%/-3%에서 청산
  - Fee: Binance Futures Maker 0.02% + Taker 0.04% 평균 0.05% × 2 (open+close)
  - Slippage: 0.05% (보수적)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from flight_mind.config import CAPITAL
from flight_mind.tier1_rule.engine import evaluate_tier1
from flight_mind.utils.db import fetch_ohlcv


CONSOLE = Console()

# Trading costs
FEE_PCT = 0.0005      # 0.05% per side (binance futures avg)
SLIPPAGE_PCT = 0.0005


@dataclass
class Trade:
    entry_time: datetime
    direction: str            # long / short
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_pct: float = 0.0


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    tp_pct: float = 6.0,
    sl_pct: float = -3.0,
    max_bars: int = 12,
) -> Trade:
    """진입 후 max_bars 내에 TP/SL 도달 여부를 확인."""
    entry_time = df.index[entry_idx]
    entry_price = float(df["close"].iloc[entry_idx]) * (
        1 + SLIPPAGE_PCT if direction == "long" else 1 - SLIPPAGE_PCT
    )

    end_idx = min(entry_idx + max_bars, len(df) - 1)
    for i in range(entry_idx + 1, end_idx + 1):
        bar = df.iloc[i]
        if direction == "long":
            high_pct = (bar["high"] - entry_price) / entry_price * 100
            low_pct = (bar["low"] - entry_price) / entry_price * 100
            if high_pct >= tp_pct:
                exit_price = entry_price * (1 + tp_pct / 100)
                return _close(entry_time, direction, entry_price,
                              df.index[i], exit_price, "tp")
            if low_pct <= sl_pct:
                exit_price = entry_price * (1 + sl_pct / 100)
                return _close(entry_time, direction, entry_price,
                              df.index[i], exit_price, "sl")
        else:  # short
            high_pct = (bar["high"] - entry_price) / entry_price * 100
            low_pct = (bar["low"] - entry_price) / entry_price * 100
            # short은 가격이 떨어져야 이득 — sign 반전
            if -low_pct >= tp_pct:
                exit_price = entry_price * (1 - tp_pct / 100)
                return _close(entry_time, direction, entry_price,
                              df.index[i], exit_price, "tp")
            if -high_pct <= sl_pct:  # high가 entry보다 올라가면 손실
                exit_price = entry_price * (1 - sl_pct / 100)
                return _close(entry_time, direction, entry_price,
                              df.index[i], exit_price, "sl")

    # Time exit
    exit_price = float(df["close"].iloc[end_idx]) * (
        1 - SLIPPAGE_PCT if direction == "long" else 1 + SLIPPAGE_PCT
    )
    return _close(entry_time, direction, entry_price,
                  df.index[end_idx], exit_price, "time")


def _close(entry_time, direction, entry_price, exit_time, exit_price, reason) -> Trade:
    if direction == "long":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100
    pnl_pct -= FEE_PCT * 2 * 100   # open + close
    return Trade(entry_time, direction, entry_price, exit_time, exit_price, reason, pnl_pct)


def backtest_tier1(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    cooldown_bars: int = 12,
) -> tuple[list[Trade], dict]:
    """5분봉으로 Tier 1 룰을 walk-forward 백테스트."""
    df = fetch_ohlcv(symbol, "5m", start, end)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")

    CONSOLE.print(f"[cyan]Backtesting Tier 1 on {symbol}: {len(df):,} bars[/cyan]")

    trades: list[Trade] = []
    last_signal_idx = -cooldown_bars     # 진입 직후 cooldown

    # warmup 200봉 후부터 평가 시작
    for i in range(200, len(df)):
        if i - last_signal_idx < cooldown_bars:
            continue

        # 직전 200봉 윈도우로 평가 (Tier 1은 60~120봉만 보지만 안전 마진)
        window = df.iloc[max(0, i - 200):i + 1]
        out = evaluate_tier1(window)

        # Tier 1만 단독 → score >= 0.7로 진입 (보수)
        if out.direction != "none" and out.score >= 0.7:
            trade = simulate_trade(
                df, entry_idx=i, direction=out.direction,
                tp_pct=CAPITAL.take_profit_pct,
                sl_pct=CAPITAL.stop_loss_pct,
                max_bars=CAPITAL.max_hold_bars_5m,
            )
            trades.append(trade)
            last_signal_idx = i

    metrics = compute_metrics(trades)
    return trades, metrics


def compute_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"n_trades": 0}

    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    return {
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)),
        "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
        "total_return_pct": float(pnls.sum()),
        "max_win_pct": float(pnls.max()),
        "max_loss_pct": float(pnls.min()),
        "sharpe_per_trade": float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0,
        "exit_breakdown": {
            "tp": int(sum(1 for t in trades if t.exit_reason == "tp")),
            "sl": int(sum(1 for t in trades if t.exit_reason == "sl")),
            "time": int(sum(1 for t in trades if t.exit_reason == "time")),
        },
    }


def print_report(symbol: str, metrics: dict) -> None:
    table = Table(title=f"Tier 1 Backtest — {symbol}", title_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    n = metrics.get("n_trades", 0)
    if n == 0:
        table.add_row("Trades", "0 (no signals)")
        CONSOLE.print(table)
        return

    table.add_row("Total Trades", f"{n:,}")
    table.add_row("Win Rate", f"{metrics['win_rate'] * 100:.2f}%")
    table.add_row("Profit Factor", f"{metrics['profit_factor']:.2f}")
    table.add_row("Total Return", f"{metrics['total_return_pct']:+.2f}%")
    table.add_row("Avg Win", f"{metrics['avg_win_pct']:+.2f}%")
    table.add_row("Avg Loss", f"{metrics['avg_loss_pct']:+.2f}%")
    table.add_row("Max Win / Loss", f"{metrics['max_win_pct']:+.2f}% / {metrics['max_loss_pct']:+.2f}%")
    table.add_row("Sharpe (per trade)", f"{metrics['sharpe_per_trade']:.3f}")
    table.add_row("Exit: TP / SL / Time",
                  f"{metrics['exit_breakdown']['tp']} / "
                  f"{metrics['exit_breakdown']['sl']} / "
                  f"{metrics['exit_breakdown']['time']}")
    CONSOLE.print(table)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    trades, metrics = backtest_tier1(args.symbol, args.start, args.end)
    print_report(args.symbol, metrics)
