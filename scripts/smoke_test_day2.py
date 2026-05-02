"""
Day 2 Pipeline Smoke Test
==========================
End-to-end 데이터 파이프라인이 처음부터 끝까지 흐르는지 검증.

실행 단계:
  1) 30일치 BTC + ETH 5분봉 다운로드 (Binance Vision)
  2) DuckDB 스키마 초기화
  3) Parquet → ohlcv 테이블 적재
  4) 피처 엔지니어링 (RSI, MA, ATR 등)
  5) Tier 1 백테스트 — 양 페어 모두

빠른 검증용 — 실제 학습/백테스트는 5년치 데이터로 별도 실행.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

from rich.console import Console

from flight_mind.utils.backtest_tier1 import backtest_tier1, print_report
from flight_mind.utils.db import db_stats, init_db, load_parquet_to_ohlcv
from flight_mind.utils.features import rebuild_all_features


CONSOLE = Console()


def step(n: int, title: str) -> None:
    CONSOLE.print(f"\n[bold cyan]━━━ Step {n}: {title} ━━━[/bold cyan]")


def main() -> int:
    # =============================================================================
    # Step 1: Download
    # =============================================================================
    step(1, "Download 30 days of 5m OHLCV (BTC + ETH)")

    # 기존 다운로드 스크립트 호출
    sys.path.insert(0, str(Path(__file__).parent))
    from download_binance_data import download_symbol

    out_dir = Path("data/raw/binance_vision")
    out_dir.mkdir(parents=True, exist_ok=True)

    end = date.today() - timedelta(days=2)   # 2일 전까지 (Binance Vision 지연 고려)
    start = end - timedelta(days=30)

    paths: dict[str, Path] = {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = download_symbol(symbol, "5m", start, end, out_dir)
        path = out_dir / f"{symbol}_5m_smoketest.parquet"
        df.to_parquet(path, compression="zstd")
        paths[symbol] = path
        CONSOLE.print(
            f"  [green]✓[/green] {symbol}: {len(df):,} rows → {path.name} "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )

    # =============================================================================
    # Step 2: Init DuckDB
    # =============================================================================
    step(2, "Initialize DuckDB schema")
    init_db()

    # =============================================================================
    # Step 3: Load Parquet → ohlcv
    # =============================================================================
    step(3, "Load Parquet → ohlcv table")
    for symbol, path in paths.items():
        n = load_parquet_to_ohlcv(path, symbol, "5m")
        CONSOLE.print(f"  [green]✓[/green] {symbol}: +{n:,} rows inserted")

    # =============================================================================
    # Step 4: Build Features
    # =============================================================================
    step(4, "Compute features (RSI, MA, ATR)")
    for symbol in paths.keys():
        result = rebuild_all_features(symbol)
        for table, count in result.items():
            CONSOLE.print(f"  [green]✓[/green] {symbol} → {table}: {count:,} rows")

    # =============================================================================
    # Step 5: Tier 1 Backtest
    # =============================================================================
    step(5, "Run Tier 1 standalone backtest")
    for symbol in paths.keys():
        try:
            trades, metrics = backtest_tier1(symbol)
            print_report(symbol, metrics)
        except Exception as e:
            CONSOLE.print(f"  [red]✗ {symbol}: {e}[/red]")

    # =============================================================================
    # Final: DB Stats
    # =============================================================================
    step(6, "Final DuckDB statistics")
    stats = db_stats()
    for table, count in stats.items():
        CONSOLE.print(f"  {table:20s}: {count:>10,}")

    CONSOLE.print("\n[bold green]✅ Day 2 smoke test completed.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
