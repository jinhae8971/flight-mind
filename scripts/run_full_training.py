"""
Flight-Mind Master Training Pipeline
======================================
영길님 PC에서 한 번 실행으로 전체 시스템을 학습.

실행 단계:
  ① 환경 진단
  ② 5년 데이터 다운로드 (Binance Vision)
  ③ DuckDB 적재
  ④ 피처 빌드 (RSI/MA/ATR 등)
  ⑤ Tier 2 학습 (8~12h)
  ⑥ Tier 4 학습 (1~2h)
  ⑦ Tier 1+2+4 통합 백테스트
  ⑧ 결과 리포트

각 단계는:
  - 체크포인트 파일로 진행 상태 저장 (재시작 가능)
  - 시작/완료/실패 시 Telegram 알림
  - 로그는 logs/ 디렉토리

Usage:
    # 전체 실행
    python scripts/run_full_training.py

    # 특정 단계부터 재개
    python scripts/run_full_training.py --start-from tier2

    # 환경 점검만
    python scripts/run_full_training.py --diagnose-only

    # Tier 2 학습 시 batch_size override
    python scripts/run_full_training.py --t2-batch 64
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel


CONSOLE = Console()

# Pipeline state
STATE_FILE = Path("data/.pipeline_state.json")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# =============================================================================
# State Management
# =============================================================================
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed": [], "started_at": None, "step_times": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def mark_complete(step: str, state: dict, elapsed_s: float) -> None:
    if step not in state["completed"]:
        state["completed"].append(step)
    state["step_times"][step] = round(elapsed_s, 1)
    save_state(state)


# =============================================================================
# Telegram Notification
# =============================================================================
def send_telegram(message: str, silent: bool = False) -> None:
    """영길님 Telegram 봇으로 알림 — 환경변수 필요"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return     # 환경변수 미설정 시 silent skip

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": f"🛫 Flight-Mind\n\n{message}",
            "parse_mode": "Markdown",
            "disable_notification": silent,
        }, timeout=10)
    except Exception as e:
        CONSOLE.print(f"[yellow]Telegram failed: {e}[/yellow]")


# =============================================================================
# Step Runners
# =============================================================================
def run_step(name: str, func: Callable, state: dict, force: bool = False) -> bool:
    """공통 step 실행기 — 체크포인트 + 로깅 + 알림."""
    if not force and name in state["completed"]:
        CONSOLE.print(f"[dim]Skip {name} (already completed)[/dim]")
        return True

    CONSOLE.print(Panel(f"[bold cyan]Step: {name}[/bold cyan]", expand=False))
    send_telegram(f"⏳ Starting: *{name}*", silent=True)

    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    t0 = time.time()

    try:
        func()
        elapsed = time.time() - t0
        mark_complete(name, state, elapsed)

        send_telegram(
            f"✅ *{name}* 완료\n"
            f"⏱ {timedelta(seconds=int(elapsed))}\n"
            f"📁 log: `{log_file.name}`"
        )
        CONSOLE.print(f"[bold green]✓ {name} done in {timedelta(seconds=int(elapsed))}[/bold green]\n")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log_file.write_text(f"FAILED: {name}\n\nException:\n{tb}")

        send_telegram(
            f"❌ *{name}* 실패\n"
            f"⏱ {timedelta(seconds=int(elapsed))}\n"
            f"💥 `{type(e).__name__}: {str(e)[:100]}`\n"
            f"📁 log: `{log_file.name}`"
        )
        CONSOLE.print(f"[bold red]✗ {name} FAILED[/bold red]")
        CONSOLE.print(f"[red]{tb}[/red]")
        return False


def step_diagnose():
    """① 환경 진단"""
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_env.py"],
        check=False, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Environment check failed — fix blockers and retry")


def step_download_data(years: int = 5):
    """② 5년 BTC + ETH 데이터 다운로드"""
    result = subprocess.run(
        [sys.executable, "scripts/download_binance_data.py",
         "--symbols", "BTCUSDT", "ETHUSDT",
         "--interval", "5m",
         "--years", str(years)],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed (exit {result.returncode})")


def step_load_to_db():
    """③ DuckDB 적재"""
    from flight_mind.utils.db import init_db, load_parquet_to_ohlcv

    init_db()

    raw_dir = Path("data/raw/binance_vision")
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        # consolidated 파일 우선, 없으면 일별 파일들 적재
        consolidated = raw_dir / f"{symbol}_5m_consolidated.parquet"
        if consolidated.exists():
            n = load_parquet_to_ohlcv(consolidated, symbol, "5m")
            CONSOLE.print(f"  ✓ {symbol}: +{n:,} rows")
        else:
            # Fallback: 일별 파일들
            symbol_dir = raw_dir / symbol
            if symbol_dir.exists():
                files = sorted(symbol_dir.glob("*.parquet"))
                for f in files:
                    load_parquet_to_ohlcv(f, symbol, "5m")
                CONSOLE.print(f"  ✓ {symbol}: {len(files)} files loaded")


def step_build_features():
    """④ 피처 엔지니어링"""
    from flight_mind.utils.features import rebuild_all_features

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        result = rebuild_all_features(symbol)
        for table, count in result.items():
            CONSOLE.print(f"  ✓ {symbol} → {table}: {count:,}")


def step_train_tier2(epochs: int, batch_size: int):
    """⑤ Tier 2 학습 (8~12시간)"""
    from flight_mind.tier2_pattern.train import train

    CONSOLE.print(f"[yellow]Tier 2 training start — epochs={epochs}, batch={batch_size}[/yellow]")
    CONSOLE.print("[dim]예상 시간: RTX 4090 ~8h, RTX 3090 ~12h[/dim]")

    result = train(
        symbols=["BTCUSDT", "ETHUSDT"],
        epochs=epochs,
        batch_size=batch_size,
    )
    CONSOLE.print(f"[green]Tier 2 best val_acc: {result['best_val_acc']:.4f}[/green]")
    CONSOLE.print(f"[green]Tier 2 test acc:     {result['test_acc']:.4f}[/green]")


def step_train_tier4(epochs: int, batch_size: int):
    """⑥ Tier 4 학습 (1~2시간)"""
    from flight_mind.tier4_regime.train import train

    CONSOLE.print(f"[yellow]Tier 4 training start — epochs={epochs}, batch={batch_size}[/yellow]")

    result = train(
        symbols=["BTCUSDT", "ETHUSDT"],
        epochs=epochs,
        batch_size=batch_size,
    )
    CONSOLE.print(f"[green]Tier 4 best val_acc: {result['best_val_acc']:.4f}[/green]")
    CONSOLE.print(f"[green]Tier 4 test acc:     {result['test_acc']:.4f}[/green]")


def step_integrated_backtest():
    """⑦ 통합 백테스트 — 실제 학습 모델 사용"""
    # 학습된 모델로 백테스트 (mock 아닌 진짜)
    # 추후 backtest_integrated_real.py 작성 시 대체
    from flight_mind.utils.backtest_integrated import (
        backtest_integrated, print_comparison_table, print_pipeline_diagnostics)

    results = {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        CONSOLE.print(f"\n[bold]Backtest {symbol}[/bold]")
        results[symbol] = {}
        for mode in ["pessimistic", "realistic", "optimistic"]:
            try:
                _, metrics = backtest_integrated(symbol, mode=mode)
                results[symbol][mode] = metrics
            except Exception as e:
                CONSOLE.print(f"[red]  {mode} failed: {e}[/red]")
                results[symbol][mode] = {"n_trades": 0}

        print_comparison_table(results[symbol], symbol)
        print_pipeline_diagnostics(symbol, results[symbol])

    # 결과 저장
    out = Path("data/integrated_backtest_results.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    CONSOLE.print(f"\n[dim]Results saved: {out}[/dim]")


# =============================================================================
# Main Pipeline
# =============================================================================
PIPELINE = [
    ("diagnose", "환경 진단"),
    ("download_data", "5년 데이터 다운로드"),
    ("load_to_db", "DuckDB 적재"),
    ("build_features", "피처 엔지니어링"),
    ("train_tier2", "Tier 2 (CNN) 학습"),
    ("train_tier4", "Tier 4 (Transformer) 학습"),
    ("integrated_backtest", "3-Tier 통합 백테스트"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Flight-Mind Master Training Pipeline")
    p.add_argument("--start-from", default=None,
                   help="특정 단계부터 재시작 (이전 단계 skip)")
    p.add_argument("--diagnose-only", action="store_true",
                   help="환경 진단만 실행")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--t2-epochs", type=int, default=50)
    p.add_argument("--t2-batch", type=int, default=64)
    p.add_argument("--t4-epochs", type=int, default=40)
    p.add_argument("--t4-batch", type=int, default=128)
    p.add_argument("--reset", action="store_true",
                   help="체크포인트 초기화 후 처음부터 재시작")
    args = p.parse_args()

    if args.diagnose_only:
        step_diagnose()
        return 0

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        CONSOLE.print("[yellow]State reset — starting fresh[/yellow]")

    state = load_state()
    if state["started_at"] is None:
        state["started_at"] = datetime.now().isoformat()
        save_state(state)

    # 시작 알림
    send_telegram(
        f"🚀 Flight-Mind Training Pipeline 시작\n"
        f"📅 {datetime.now():%Y-%m-%d %H:%M}\n"
        f"⚙️  Tier 2: {args.t2_epochs}ep × batch {args.t2_batch}\n"
        f"⚙️  Tier 4: {args.t4_epochs}ep × batch {args.t4_batch}"
    )

    # Skip steps until --start-from
    skip_until = args.start_from
    skipping = bool(skip_until)

    step_funcs = {
        "diagnose": step_diagnose,
        "download_data": lambda: step_download_data(args.years),
        "load_to_db": step_load_to_db,
        "build_features": step_build_features,
        "train_tier2": lambda: step_train_tier2(args.t2_epochs, args.t2_batch),
        "train_tier4": lambda: step_train_tier4(args.t4_epochs, args.t4_batch),
        "integrated_backtest": step_integrated_backtest,
    }

    for step_id, step_name in PIPELINE:
        if skipping:
            if step_id == skip_until:
                skipping = False
            else:
                CONSOLE.print(f"[dim]Skip {step_id} (--start-from)[/dim]")
                continue

        ok = run_step(step_id, step_funcs[step_id], state)
        if not ok:
            send_telegram(
                f"⛔️ Pipeline 중단 at *{step_id}*\n"
                f"`python scripts/run_full_training.py --start-from {step_id}` 로 재개"
            )
            return 1

    # 전체 완료 알림
    total_time = sum(state.get("step_times", {}).values())
    send_telegram(
        f"🎉 Flight-Mind 학습 완료!\n"
        f"⏱ 총 시간: {timedelta(seconds=int(total_time))}\n"
        f"📊 백테스트 결과: `data/integrated_backtest_results.json`"
    )
    CONSOLE.print(Panel(
        "[bold green]✅ 모든 단계 완료![/bold green]\n\n"
        f"총 시간: {timedelta(seconds=int(total_time))}\n"
        f"단계별 시간: {state.get('step_times', {})}",
        title="Pipeline Complete"
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
