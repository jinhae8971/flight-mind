"""
3-Tier Integrated Backtest
============================
Tier 1 (실제 룰) + Tier 2 (mock CNN) + Tier 4 (mock Regime)
→ Bayesian Confluence Fusion → 진입/청산 시뮬레이션

Day 2 Tier 1 단독 백테스트와의 차이점:
  - Day 2: evaluate_tier1() 결과만으로 진입 (단일 임계값 0.7)
  - Day 5: 3개 Tier 출력을 Fusion Layer가 통합 (임계값 0.85)
  - 일일 한도 2회, Kill-Switch -5% 등 영길님 정책 모두 적용

3가지 시나리오:
  - Optimistic / Realistic / Pessimistic
  → 영길님 PC에서 학습된 모델이 어디쯤 떨어져야 의미 있는지 가이드
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from flight_mind.config import CAPITAL, RISK
from flight_mind.fusion.layer import FusionDecision, fuse
from flight_mind.tier1_rule.engine import TierOutput, evaluate_tier1
from flight_mind.utils.db import fetch_ohlcv
from flight_mind.utils.mock_signals import (Mode, MockSignalGenerator,
                                              get_mock_generator)


CONSOLE = Console()

# Trading costs (Day 2 백테스터와 동일)
FEE_PCT = 0.0005       # 0.05% per side
SLIPPAGE_PCT = 0.0005


@dataclass
class IntegratedTrade:
    entry_time: datetime
    direction: str                    # long / short
    entry_price: float
    confluence_score: float
    tier_directions: dict             # {"T1": "long", "T2": "long", "T4": "long"}
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_pct: float = 0.0


@dataclass
class DailyState:
    """일일 거래 상태 — Kill-Switch가 참조"""
    date: pd.Timestamp
    trades_today: int = 0
    daily_pnl_pct: float = 0.0
    halted: bool = False
    halt_reason: str | None = None


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    confluence: float,
    tier_dirs: dict,
    tp_pct: float = 6.0,
    sl_pct: float = -3.0,
    max_bars: int = 12,
) -> IntegratedTrade:
    """진입 후 TP/SL 또는 시간 만료 시뮬레이션 (Day 2 백테스터와 동일 로직)."""
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
                return _close(entry_time, direction, entry_price, confluence,
                              tier_dirs, df.index[i], exit_price, "tp")
            if low_pct <= sl_pct:
                exit_price = entry_price * (1 + sl_pct / 100)
                return _close(entry_time, direction, entry_price, confluence,
                              tier_dirs, df.index[i], exit_price, "sl")
        else:  # short
            high_pct = (bar["high"] - entry_price) / entry_price * 100
            low_pct = (bar["low"] - entry_price) / entry_price * 100
            if -low_pct >= tp_pct:
                exit_price = entry_price * (1 - tp_pct / 100)
                return _close(entry_time, direction, entry_price, confluence,
                              tier_dirs, df.index[i], exit_price, "tp")
            if -high_pct <= sl_pct:
                exit_price = entry_price * (1 - sl_pct / 100)
                return _close(entry_time, direction, entry_price, confluence,
                              tier_dirs, df.index[i], exit_price, "sl")

    # Time exit
    exit_price = float(df["close"].iloc[end_idx]) * (
        1 - SLIPPAGE_PCT if direction == "long" else 1 + SLIPPAGE_PCT
    )
    return _close(entry_time, direction, entry_price, confluence,
                  tier_dirs, df.index[end_idx], exit_price, "time")


def _close(entry_time, direction, entry_price, confluence, tier_dirs,
           exit_time, exit_price, reason) -> IntegratedTrade:
    if direction == "long":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100
    pnl_pct -= FEE_PCT * 2 * 100
    return IntegratedTrade(
        entry_time, direction, entry_price, confluence, tier_dirs,
        exit_time, exit_price, reason, pnl_pct,
    )


def backtest_integrated(
    symbol: str,
    mode: Mode = "realistic",
    cooldown_bars: int = 12,
    seed: int = 42,
) -> tuple[list[IntegratedTrade], dict]:
    """
    3-Tier 통합 백테스트.

    영길님 정책 적용:
      - Confluence threshold: 0.85
      - 일일 거래 한도: 2회
      - Daily loss kill-switch: -5%
      - Cooldown after entry: 12 bars (1시간)
    """
    df = fetch_ohlcv(symbol, "5m")
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")

    CONSOLE.print(
        f"[cyan]Integrated backtest on {symbol}: {len(df):,} bars, "
        f"mode={mode}[/cyan]"
    )

    mock = get_mock_generator(mode, seed=seed)
    trades: list[IntegratedTrade] = []
    last_signal_idx = -cooldown_bars

    # 진단 통계
    stats = {
        "tier1_signals": 0,
        "tier1_t2_disagree": 0,
        "below_threshold": 0,
        "daily_limit_blocked": 0,
        "kill_switch_blocked": 0,
        "fusion_decisions_per_action": {"hold": 0, "open_long": 0, "open_short": 0},
    }

    # Daily state tracking
    daily_states: dict[pd.Timestamp, DailyState] = {}

    for i in range(200, len(df)):
        # Cooldown
        if i - last_signal_idx < cooldown_bars:
            continue

        bar_date = df.index[i].normalize()
        state = daily_states.setdefault(bar_date, DailyState(date=bar_date))

        # 일일 한도 / Kill-switch 체크
        if state.halted:
            stats["kill_switch_blocked"] += 1
            continue
        if state.trades_today >= RISK.daily_trade_limit:
            stats["daily_limit_blocked"] += 1
            continue

        # Tier 1 (실제 룰)
        window = df.iloc[max(0, i - 200):i + 1]
        t1 = evaluate_tier1(window)
        if t1.direction == "none":
            continue
        stats["tier1_signals"] += 1

        # Tier 2 (mock CNN)
        t2 = mock.generate_t2(df, i, future_horizon_bars=12, threshold_pct=0.5)

        # Tier 4 (mock Regime)
        t4 = mock.generate_t4(df, i)

        # Fusion
        decision = fuse(t1, t2, t4, available_balance_usdt=1750.0)
        stats["fusion_decisions_per_action"][decision.action] = (
            stats["fusion_decisions_per_action"].get(decision.action, 0) + 1
        )

        if decision.action == "hold":
            if decision.veto_reason and "disagree" in decision.veto_reason:
                stats["tier1_t2_disagree"] += 1
            elif decision.veto_reason and "threshold" in decision.veto_reason:
                stats["below_threshold"] += 1
            continue

        # Open position
        direction = "long" if decision.action == "open_long" else "short"
        trade = simulate_trade(
            df, entry_idx=i, direction=direction,
            confluence=decision.confluence_score,
            tier_dirs={k: v.direction for k, v in decision.tier_outputs.items()},
            tp_pct=CAPITAL.take_profit_pct,
            sl_pct=CAPITAL.stop_loss_pct,
            max_bars=CAPITAL.max_hold_bars_5m,
        )
        trades.append(trade)
        state.trades_today += 1
        state.daily_pnl_pct += trade.pnl_pct

        # Kill-switch 발동 체크
        if state.daily_pnl_pct <= RISK.daily_loss_pct:
            state.halted = True
            state.halt_reason = f"daily_loss {state.daily_pnl_pct:.2f}%"

        last_signal_idx = i

    metrics = compute_metrics(trades)
    metrics["pipeline_stats"] = stats
    metrics["mode"] = mode
    return trades, metrics


def compute_metrics(trades: list[IntegratedTrade]) -> dict:
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
        "profit_factor": (float(wins.sum() / abs(losses.sum()))
                          if len(losses) > 0 else float("inf")),
        "total_return_pct": float(pnls.sum()),
        "max_win_pct": float(pnls.max()),
        "max_loss_pct": float(pnls.min()),
        "sharpe_per_trade": (float(pnls.mean() / pnls.std())
                              if pnls.std() > 0 else 0.0),
        "exit_breakdown": {
            "tp": int(sum(1 for t in trades if t.exit_reason == "tp")),
            "sl": int(sum(1 for t in trades if t.exit_reason == "sl")),
            "time": int(sum(1 for t in trades if t.exit_reason == "time")),
        },
        "avg_confluence": float(np.mean([t.confluence_score for t in trades])),
    }


def print_comparison_table(results: dict[str, dict], symbol: str) -> None:
    """3-Tier 모드별 결과 + Day 2 Tier 1 단독 비교"""
    table = Table(
        title=f"3-Tier Integrated Backtest — {symbol}",
        title_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Tier 1 단독\n(Day 2)", justify="right", style="dim")
    table.add_column("Pessimistic", justify="right")
    table.add_column("Realistic", justify="right", style="bold yellow")
    table.add_column("Optimistic", justify="right", style="green")

    # Day 2 baseline (BTC 기준 — ETH도 비슷)
    day2_baseline = {
        "BTCUSDT": {"n_trades": 419, "win_rate": 0.2601, "total_return_pct": -72.38,
                    "profit_factor": 0.28, "exit_breakdown": {"tp": 0, "sl": 0, "time": 419}},
        "ETHUSDT": {"n_trades": 391, "win_rate": 0.2839, "total_return_pct": -77.93,
                    "profit_factor": 0.31, "exit_breakdown": {"tp": 0, "sl": 0, "time": 391}},
    }
    base = day2_baseline.get(symbol, {"n_trades": 0})

    def fmt(d, k, p="0", suffix=""):
        if d.get("n_trades", 0) == 0:
            return "—"
        v = d.get(k, 0)
        if isinstance(v, float):
            return f"{v:{p}}{suffix}"
        return str(v)

    table.add_row(
        "거래 횟수",
        fmt(base, "n_trades"),
        fmt(results["pessimistic"], "n_trades"),
        fmt(results["realistic"], "n_trades"),
        fmt(results["optimistic"], "n_trades"),
    )
    table.add_row(
        "승률 (%)",
        f"{base.get('win_rate', 0) * 100:.1f}",
        f"{results['pessimistic'].get('win_rate', 0) * 100:.1f}"
            if results['pessimistic'].get('n_trades', 0) > 0 else "—",
        f"{results['realistic'].get('win_rate', 0) * 100:.1f}"
            if results['realistic'].get('n_trades', 0) > 0 else "—",
        f"{results['optimistic'].get('win_rate', 0) * 100:.1f}"
            if results['optimistic'].get('n_trades', 0) > 0 else "—",
    )
    table.add_row(
        "Profit Factor",
        f"{base.get('profit_factor', 0):.2f}",
        f"{results['pessimistic'].get('profit_factor', 0):.2f}"
            if results['pessimistic'].get('n_trades', 0) > 0 else "—",
        f"{results['realistic'].get('profit_factor', 0):.2f}"
            if results['realistic'].get('n_trades', 0) > 0 else "—",
        f"{results['optimistic'].get('profit_factor', 0):.2f}"
            if results['optimistic'].get('n_trades', 0) > 0 else "—",
    )
    table.add_row(
        "총 수익률 (%)",
        f"{base.get('total_return_pct', 0):+.2f}",
        f"{results['pessimistic'].get('total_return_pct', 0):+.2f}"
            if results['pessimistic'].get('n_trades', 0) > 0 else "—",
        f"{results['realistic'].get('total_return_pct', 0):+.2f}"
            if results['realistic'].get('n_trades', 0) > 0 else "—",
        f"{results['optimistic'].get('total_return_pct', 0):+.2f}"
            if results['optimistic'].get('n_trades', 0) > 0 else "—",
    )

    # TP / SL / Time breakdown
    def fmt_breakdown(d):
        if d.get("n_trades", 0) == 0:
            return "—"
        b = d.get("exit_breakdown", {})
        return f"{b.get('tp', 0)}/{b.get('sl', 0)}/{b.get('time', 0)}"

    table.add_row(
        "Exit (TP/SL/Time)",
        fmt_breakdown(base),
        fmt_breakdown(results["pessimistic"]),
        fmt_breakdown(results["realistic"]),
        fmt_breakdown(results["optimistic"]),
    )

    CONSOLE.print(table)


def print_pipeline_diagnostics(symbol: str, results: dict) -> None:
    """파이프라인 단계별 차단 통계"""
    CONSOLE.print(f"\n[bold]Pipeline diagnostics — {symbol} (Realistic mode)[/bold]")
    stats = results["realistic"].get("pipeline_stats", {})

    diag = Table()
    diag.add_column("Stage")
    diag.add_column("Count", justify="right")
    diag.add_row("Tier 1 신호 발생", f"{stats.get('tier1_signals', 0):,}")
    diag.add_row("  └ T1·T2 disagree veto", f"{stats.get('tier1_t2_disagree', 0):,}")
    diag.add_row("  └ Confluence < 0.85", f"{stats.get('below_threshold', 0):,}")
    diag.add_row("  └ 일일 한도 차단", f"{stats.get('daily_limit_blocked', 0):,}")
    diag.add_row("  └ Kill-Switch 차단", f"{stats.get('kill_switch_blocked', 0):,}")
    fdpa = stats.get("fusion_decisions_per_action", {})
    diag.add_row("실제 진입 (open_long + open_short)",
                 f"{fdpa.get('open_long', 0) + fdpa.get('open_short', 0):,}")
    CONSOLE.print(diag)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--mode", default=None,
                   choices=["optimistic", "realistic", "pessimistic", "all"])
    args = p.parse_args()

    if args.mode is None or args.mode == "all":
        # Run all three modes
        results = {}
        for mode in ["pessimistic", "realistic", "optimistic"]:
            CONSOLE.print(f"\n[bold cyan]━━━ {mode.upper()} ━━━[/bold cyan]")
            try:
                _, metrics = backtest_integrated(args.symbol, mode=mode)
                results[mode] = metrics
            except Exception as e:
                CONSOLE.print(f"[red]Error in {mode}: {e}[/red]")
                results[mode] = {"n_trades": 0}

        CONSOLE.print()
        print_comparison_table(results, args.symbol)
        print_pipeline_diagnostics(args.symbol, results)
    else:
        _, metrics = backtest_integrated(args.symbol, mode=args.mode)
        CONSOLE.print(f"\n{args.mode}: {metrics}")
