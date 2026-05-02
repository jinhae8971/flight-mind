"""
Paper Trading Demo
====================
학습된 모델이 있다는 가정 하에 paper trading 단일 사이클을 시뮬레이션.

Usage:
    # 학습된 모델 사용 (영길님 PC 기준)
    python scripts/demo_paper_trading.py --symbol BTCUSDT

    # 모델 없이 mock signal 사용 (sandbox 검증용)
    python scripts/demo_paper_trading.py --symbol BTCUSDT --mock

이 스크립트가 보여주는 흐름:
  1. 시장 데이터 로드
  2. Tier 1 (실제 룰) + Tier 2/4 (또는 mock) 신호 생성
  3. Fusion Layer 의사결정
  4. ExecutionEngine paper 모드 주문 (가상 체결)
  5. Audit DB에 모든 결정 영구 기록
  6. 결과 출력
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from flight_mind.execution.engine import ExecutionEngine, Mode
from flight_mind.fusion.layer import explain, fuse
from flight_mind.tier1_rule.engine import evaluate_tier1
from flight_mind.utils.db import fetch_ohlcv


CONSOLE = Console()


def run_single_cycle(symbol: str, use_mock: bool, dry_run: bool) -> int:
    CONSOLE.print(Panel(
        f"[bold]Paper Trading Demo — {symbol}[/bold]\n"
        f"Mock: {use_mock}  |  Dry-run: {dry_run}",
        expand=False,
    ))

    # 1. Load market data
    df = fetch_ohlcv(symbol, "5m")
    if df.empty or len(df) < 200:
        CONSOLE.print(f"[red]Insufficient data for {symbol}[/red]")
        return 1

    CONSOLE.print(f"[green]✓ Loaded {len(df):,} 5m bars[/green]")

    # 2. Tier 1 (실제 룰)
    window = df.tail(201)
    t1 = evaluate_tier1(window)
    CONSOLE.print(f"\n[cyan]Tier 1:[/cyan] score={t1.score:.3f} dir={t1.direction}")
    if t1.signals:
        for k, v in list(t1.signals.items())[:5]:
            CONSOLE.print(f"  • {k}: {v}")

    # 3. Tier 2 + Tier 4
    if use_mock:
        from flight_mind.utils.mock_signals import get_mock_generator
        gen = get_mock_generator("realistic", seed=42)
        t2 = gen.generate_t2(df, len(df) - 1, future_horizon_bars=12)
        t4 = gen.generate_t4(df, len(df) - 1)
        CONSOLE.print(f"[cyan]Tier 2 (mock):[/cyan] score={t2.score:.3f} dir={t2.direction}")
        CONSOLE.print(f"[cyan]Tier 4 (mock):[/cyan] score={t4.score:.3f} dir={t4.direction}")
    else:
        # 학습된 모델 사용 시도
        try:
            from flight_mind.tier2_pattern.inference import predict_tier2
            from flight_mind.tier4_regime.inference import predict_tier4

            t2 = predict_tier2(df.tail(60))

            # Tier 4는 일봉 필요
            df_daily = df.resample("1D").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna()

            if len(df_daily) >= 230:
                t4 = predict_tier4(df_daily, symbol=symbol)
            else:
                CONSOLE.print(f"[yellow]Daily data insufficient ({len(df_daily)} < 230) — using fallback[/yellow]")
                from flight_mind.tier1_rule.engine import TierOutput
                t4 = TierOutput(0.0, "none", {"reason": "insufficient_daily"})

            CONSOLE.print(f"[cyan]Tier 2 (model):[/cyan] score={t2.score:.3f} dir={t2.direction}")
            CONSOLE.print(f"[cyan]Tier 4 (model):[/cyan] score={t4.score:.3f} dir={t4.direction}")

        except FileNotFoundError as e:
            CONSOLE.print(f"[yellow]Model not trained yet: {e}[/yellow]")
            CONSOLE.print("[yellow]Falling back to --mock mode[/yellow]")
            from flight_mind.utils.mock_signals import get_mock_generator
            gen = get_mock_generator("realistic", seed=42)
            t2 = gen.generate_t2(df, len(df) - 1, future_horizon_bars=12)
            t4 = gen.generate_t4(df, len(df) - 1)

    # 4. Fusion Layer
    decision = fuse(t1, t2, t4, available_balance_usdt=1750.0)
    CONSOLE.print()
    CONSOLE.print(Panel(explain(decision), title="Fusion Decision", expand=False))

    # 5. Market snapshot for audit
    last = df.iloc[-1]
    market_snapshot = {
        "ts_utc": last.name.isoformat(),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last["volume"]),
    }

    # 6. Execute (paper mode)
    engine = ExecutionEngine(mode=Mode.PAPER, dry_run=dry_run)
    result = engine.execute_decision(symbol, decision, market_snapshot=market_snapshot)

    # 7. Result
    CONSOLE.print(Panel(
        f"Success: {result.success}\n"
        f"Order ID: {result.order_id}\n"
        f"Filled qty: {result.filled_quantity}\n"
        f"Avg price: {result.avg_fill_price}\n"
        f"Mode: {result.mode}\n"
        f"Error: {result.error or '(none)'}",
        title="Order Result",
        expand=False,
    ))

    # 8. Show recent audit records
    from flight_mind.risk.audit import (fetch_recent_decisions,
                                          fetch_recent_trades)

    decisions = fetch_recent_decisions(symbol=symbol, limit=3)
    CONSOLE.print(f"\n[dim]Recent decisions ({len(decisions)}):[/dim]")
    for d in decisions:
        CONSOLE.print(f"  [{d['ts_utc'][:19]}] {d['action']:<11} confluence={d.get('confluence', 0):.3f}")

    return 0 if result.success else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--mock", action="store_true",
                   help="학습 모델 없이 mock signal 사용")
    p.add_argument("--dry-run", action="store_true",
                   help="시뮬레이션 — 실제 audit 기록 안 남김")
    args = p.parse_args()

    return run_single_cycle(args.symbol, args.mock, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
