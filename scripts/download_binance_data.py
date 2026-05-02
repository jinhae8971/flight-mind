#!/usr/bin/env python
"""
Binance Vision에서 BTC/ETH/SOL 5년치 5분봉 데이터를 다운로드.

Binance Vision은 매일 ZIP 파일로 OHLCV를 공개 (https://data.binance.vision/).
- USDT-M 선물: /futures/um/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM-DD}.zip
- 약 5년치 5m: 심볼당 ~1.5GB

Usage:
    python scripts/download_binance_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --years 5

이 스크립트는 영길님의 GitHub Actions 패턴과 호환되도록 작성됨.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console
from rich.progress import (BarColumn, Progress, TaskProgressColumn,
                           TextColumn, TimeRemainingColumn)


CONSOLE = Console()
BASE = "https://data.binance.vision/data/futures/um/daily/klines"


def download_one_day(
    symbol: str, interval: str, day: date, out_dir: Path
) -> pd.DataFrame | None:
    """하루치 ZIP을 받아 DataFrame으로 변환."""
    fname = f"{symbol}-{interval}-{day.isoformat()}.zip"
    url = f"{BASE}/{symbol}/{interval}/{fname}"

    cache_path = out_dir / symbol / fname.replace(".zip", ".parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # Binance Vision CSV는 2025년부터 헤더 포함 형식.
                # 기존 데이터는 헤더 없음 — 자동 감지.
                first_bytes = f.read(20).decode("utf-8", errors="ignore")
                f_pos = io.BytesIO()  # rewind 위해 새 buffer
            with z.open(csv_name) as f:
                has_header = first_bytes.lower().startswith("open_time")
                df = pd.read_csv(
                    f,
                    header=0 if has_header else None,
                    names=None if has_header else [
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades",
                        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
                    ],
                )

        # 컬럼명 정규화 — 신형(count) → 구형(trades)
        if "count" in df.columns:
            df = df.rename(columns={"count": "trades"})

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
        df = df[["open", "high", "low", "close", "volume", "trades"]].astype(
            {"open": "float32", "high": "float32", "low": "float32",
             "close": "float32", "volume": "float32", "trades": "int32"}
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, compression="snappy")
        return df

    except Exception as e:
        CONSOLE.print(f"[yellow]Skip {fname}: {e}[/yellow]")
        return None


def download_symbol(
    symbol: str, interval: str, start: date, end: date, out_dir: Path
) -> pd.DataFrame:
    """심볼별 일별 다운로드 후 concat."""
    days = [(start + timedelta(days=i)) for i in range((end - start).days + 1)]
    frames: list[pd.DataFrame] = []

    with Progress(
        TextColumn(f"[cyan]{symbol}[/cyan]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as prog:
        task = prog.add_task("download", total=len(days))
        for d in days:
            df = download_one_day(symbol, interval, d, out_dir)
            if df is not None:
                frames.append(df)
            prog.advance(task)

    if not frames:
        raise RuntimeError(f"No data fetched for {symbol}")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--interval", default="5m")
    p.add_argument("--years", type=int, default=5)
    p.add_argument(
        "--out", default="data/raw/binance_vision",
        help="Output directory (default: data/raw/binance_vision)",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    end = date.today() - timedelta(days=1)         # 어제까지
    start = end - timedelta(days=365 * args.years)

    CONSOLE.print(
        f"[bold]Downloading {args.symbols} | {args.interval} | "
        f"{start} ~ {end}[/bold]"
    )

    for symbol in args.symbols:
        df = download_symbol(symbol, args.interval, start, end, out_dir)
        consolidated = out_dir / f"{symbol}_{args.interval}_consolidated.parquet"
        df.to_parquet(consolidated, compression="zstd")
        CONSOLE.print(
            f"[green]✓[/green] {symbol}: {len(df):,} rows → {consolidated} "
            f"({consolidated.stat().st_size / 1e6:.1f} MB)"
        )

    CONSOLE.print("[bold green]All done.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
