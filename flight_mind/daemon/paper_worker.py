"""
Paper Trading Daemon
======================
5분마다 자동 실행되는 통합 워커 — 영길님이 PC 앞을 떠나도 시스템이 자율 운영.

사이클 (매 5분):
  1. 시장 데이터 fetch (BTC + ETH 5분봉)
  2. 오픈 포지션 체크 → TP/SL/Time exit 자동 청산
  3. 새 진입 후보 심볼 → Tier 1/2/4 신호 생성
  4. Fusion → Risk Gate → Order
  5. Audit DB 기록 + Telegram 알림
  6. 1시간마다 heartbeat
  7. 자정마다 daily summary

운영 원칙:
  - Crash-resilient: 한 사이클 실패해도 데몬 죽지 않음
  - Stateful: 재시작 시 오픈 포지션 복원
  - Observable: 모든 결정 Telegram + Audit DB 양쪽 기록
  - Graceful shutdown: SIGINT/SIGTERM 시 안전 종료

Usage:
    python -m flight_mind.daemon.paper_worker --mode paper

    # Telegram 알림 활성화
    $env:TELEGRAM_BOT_TOKEN = "..."
    $env:TELEGRAM_CHAT_ID = "..."
    python -m flight_mind.daemon.paper_worker --mode paper

    # Testnet (실제 API + 가짜 자금)
    $env:VAULT_PASSPHRASE = "..."
    python -m flight_mind.daemon.paper_worker --mode testnet
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
from rich.console import Console

from flight_mind.config import CAPITAL, EXCHANGE
from flight_mind.execution.engine import ExecutionEngine, Mode
from flight_mind.fusion.layer import explain, fuse
from flight_mind.notify.telegram import Severity, get_notifier
from flight_mind.risk.audit import (fetch_recent_decisions, fetch_recent_trades,
                                       get_audit_conn)
from flight_mind.risk.manager import compute_pnl_summary, get_risk_manager
from flight_mind.risk.position_tracker import (OpenPosition,
                                                  get_position_tracker)
from flight_mind.tier1_rule.engine import TierOutput, evaluate_tier1
from flight_mind.utils.db import fetch_ohlcv


CONSOLE = Console()


CYCLE_INTERVAL_S = 300         # 5분
HEARTBEAT_INTERVAL_S = 3600    # 1시간
DAILY_SUMMARY_HOUR_UTC = 0     # 매일 00:00 UTC


@dataclass
class DaemonStats:
    """데몬 운영 통계 — 시작 이후 누적"""
    started_at: float = field(default_factory=time.time)
    cycles_completed: int = 0
    cycles_failed: int = 0
    decisions_made: int = 0
    orders_placed: int = 0
    positions_closed: int = 0
    last_heartbeat_at: float = 0
    last_daily_summary_date: str | None = None

    def uptime_seconds(self) -> int:
        return int(time.time() - self.started_at)


# =============================================================================
# Trading Cycle
# =============================================================================
class PaperTradingDaemon:
    """5분 주기 자동 운영 워커"""

    def __init__(
        self,
        mode: Mode = Mode.PAPER,
        symbols: list[str] | None = None,
        cycle_interval_s: int = CYCLE_INTERVAL_S,
        max_consecutive_failures: int = 5,
    ):
        self.mode = mode
        self.symbols = symbols or [s.replace("/", "") for s in EXCHANGE.pairs]
        self.cycle_interval_s = cycle_interval_s
        self.max_consecutive_failures = max_consecutive_failures

        # Components (lazy-initialized)
        self.execution = ExecutionEngine(mode=mode)
        self.risk_mgr = get_risk_manager()
        self.tracker = get_position_tracker()
        self.notifier = get_notifier()

        # State
        self.running = False
        self.consecutive_failures = 0
        self.stats = DaemonStats()

    # =========================================================================
    # Main Loop
    # =========================================================================
    def run(self) -> int:
        """Main daemon loop — Ctrl+C까지 무한 실행"""
        self._setup_signal_handlers()
        self.running = True

        # Cold start: refresh open positions from DB
        n_open = self.tracker.refresh_from_db()
        CONSOLE.print(
            f"[bold green]🛫 Paper Trading Daemon starting "
            f"(mode={self.mode.value}, symbols={self.symbols}, "
            f"open={n_open})[/bold green]"
        )

        # Startup notification
        self.notifier.send(
            f"*🛫 Daemon Started*\n"
            f"Mode: `{self.mode.value}`\n"
            f"Symbols: `{', '.join(self.symbols)}`\n"
            f"Open positions: `{n_open}`\n"
            f"Cycle: every `{self.cycle_interval_s}s`",
            severity=Severity.INFO,
        )

        # Main loop
        while self.running:
            cycle_start = time.time()

            try:
                self._run_one_cycle()
                self.stats.cycles_completed += 1
                self.consecutive_failures = 0
            except Exception as e:
                self.stats.cycles_failed += 1
                self.consecutive_failures += 1
                self._handle_cycle_failure(e)

                # Too many consecutive failures → emergency stop
                if self.consecutive_failures >= self.max_consecutive_failures:
                    CONSOLE.print("[bold red]Too many consecutive failures — EMERGENCY STOP[/bold red]")
                    self.risk_mgr.trigger_emergency_stop(
                        f"daemon_failures: {self.consecutive_failures} cycles"
                    )
                    self.notifier.send_killswitch(
                        level="EMERGENCY_STOP",
                        reason=f"daemon: {self.consecutive_failures} consecutive failures",
                        metrics={},
                    )
                    break

            # Periodic tasks
            self._maybe_send_heartbeat()
            self._maybe_send_daily_summary()

            # Sleep until next 5-min boundary (drift compensation)
            elapsed = time.time() - cycle_start
            sleep_s = max(0, self.cycle_interval_s - elapsed)
            CONSOLE.print(f"[dim]Cycle done in {elapsed:.1f}s, sleeping {sleep_s:.1f}s[/dim]")

            self._sleep_interruptible(sleep_s)

        # Shutdown
        return self._shutdown()

    # =========================================================================
    # Single Cycle
    # =========================================================================
    def _run_one_cycle(self) -> None:
        """단일 사이클 — 청산 우선, 진입 차후"""
        CONSOLE.print(f"\n[cyan]━━━ Cycle @ {_now_str()} ━━━[/cyan]")

        # 1. Check exits (TP/SL/Time) for open positions
        self._check_exits()

        # 2. Try new entries for symbols without open position
        for symbol in self.symbols:
            if self.tracker.has_open_position(symbol):
                continue
            try:
                self._try_entry(symbol)
            except Exception as e:
                CONSOLE.print(f"[yellow]Entry attempt failed for {symbol}: {e}[/yellow]")
                self.notifier.send_error(
                    daemon_name="paper_worker",
                    error_type=type(e).__name__,
                    message=str(e),
                )

    def _check_exits(self) -> None:
        """오픈 포지션의 TP/SL/Time exit 확인 + 자동 청산"""
        positions = self.tracker.get_open_positions()
        if not positions:
            return

        # 현재 시세 일괄 fetch
        prices: dict[str, float] = {}
        for pos in positions:
            ticker = self.execution.fetch_ticker(pos.symbol)
            price = ticker.get("last") or ticker.get("ask")
            if price and price > 0:
                prices[pos.symbol] = price

        # 각 포지션 평가
        for pos in positions:
            current_price = prices.get(pos.symbol)
            if not current_price:
                continue

            pos.update_market_price(current_price)
            CONSOLE.print(
                f"  📊 {pos.symbol} {pos.direction}: "
                f"entry={pos.entry_price:.2f} now={current_price:.2f} "
                f"PnL={pos.unrealized_pnl_pct:+.2f}%"
            )

            exit_reason = self._should_exit(pos, current_price)
            if exit_reason:
                self._close_position(pos, current_price, exit_reason)

    def _should_exit(self, pos: OpenPosition, current_price: float) -> str | None:
        """청산 조건 평가 — TP / SL / Time exit"""
        # TP / SL based on percentage move
        if pos.is_long:
            pct_move = (current_price - pos.entry_price) / pos.entry_price * 100
        else:
            pct_move = (pos.entry_price - current_price) / pos.entry_price * 100

        if pct_move >= CAPITAL.take_profit_pct:
            return "tp"
        if pct_move <= CAPITAL.stop_loss_pct:
            return "sl"

        # Time exit (max_hold_bars × 5분)
        max_hold_minutes = CAPITAL.max_hold_bars_5m * 5
        if pos.hold_duration_minutes() >= max_hold_minutes:
            return "time"

        return None

    def _close_position(self, pos: OpenPosition, current_price: float,
                         reason: str) -> None:
        """포지션 청산 + 알림"""
        result = self.execution.close_position(
            symbol=pos.symbol,
            direction=pos.direction,
            quantity=pos.quantity,
            reason=reason,
        )

        if not result.success:
            CONSOLE.print(f"[red]Close failed for {pos.symbol}: {result.error}[/red]")
            return

        # PnL 계산 + DB 업데이트
        close_data = self.tracker.close_position(
            trade_id=pos.trade_id,
            exit_order_id=result.order_id,
            exit_price=result.avg_fill_price or current_price,
            exit_reason=reason,
        )

        self.stats.positions_closed += 1

        # Telegram 알림
        self.notifier.send_position_closed(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=result.avg_fill_price or current_price,
            pnl_usdt=close_data["realized_pnl_usdt"],
            pnl_pct=close_data["realized_pnl_pct"],
            exit_reason=reason,
            mode=self.mode.value,
        )

        CONSOLE.print(
            f"[bold green]✅ Closed {pos.symbol}[/bold green] "
            f"reason={reason} pnl={close_data['realized_pnl_usdt']:+.2f} USDT "
            f"({close_data['realized_pnl_pct']:+.2f}%)"
        )

        # Post-trade kill-switch evaluation
        new_state = self.risk_mgr.update_after_trade(mode=self.mode.value)
        if new_state.level.value != "ok":
            self.notifier.send_killswitch(
                level=new_state.level.value,
                reason=new_state.reason or "",
                metrics=new_state.metrics_at_trigger,
            )

    def _try_entry(self, symbol: str) -> None:
        """신규 진입 시도"""
        # 1. Load market data
        df = fetch_ohlcv(symbol, "5m")
        if df.empty or len(df) < 200:
            CONSOLE.print(f"[dim]{symbol}: insufficient data[/dim]")
            return

        # 2. Tier signals
        window = df.tail(201)
        t1 = evaluate_tier1(window)

        # Tier 2/4 — try real model first, fallback to mock
        t2, t4 = self._get_tier_2_and_4(df, symbol)

        # 3. Fusion
        decision = fuse(t1, t2, t4, available_balance_usdt=1750.0)
        self.stats.decisions_made += 1

        # Hold → no further action (audit는 ExecutionEngine 내부에서)
        if decision.action == "hold":
            CONSOLE.print(
                f"[dim]{symbol}: hold (confluence={decision.confluence_score:.3f})[/dim]"
            )
            # 빈도 너무 높으면 알림 스킵 (너무 시끄러움 방지)
            return

        # 4. Execute via engine (Risk Gate가 내부에서 자동 호출됨)
        last_bar = df.iloc[-1]
        market_snapshot = {
            "ts_utc": last_bar.name.isoformat(),
            "close": float(last_bar["close"]),
            "volume": float(last_bar["volume"]),
        }

        result = self.execution.execute_decision(
            symbol=symbol,
            decision=decision,
            market_snapshot=market_snapshot,
        )

        if result.success and result.filled_quantity > 0:
            self.stats.orders_placed += 1

            # 알림 — 결정 + 체결
            self.notifier.send_decision(
                symbol=symbol,
                action=decision.action,
                direction=decision.direction,
                confluence=decision.confluence_score,
                tier_signals={k: {"direction": v.direction, "score": v.score}
                              for k, v in decision.tier_outputs.items()},
            )
            self.notifier.send_order_filled(
                symbol=symbol,
                direction=decision.direction,
                quantity=result.filled_quantity,
                price=result.avg_fill_price,
                mode=self.mode.value,
            )

            CONSOLE.print(
                f"[bold green]✅ Entered {symbol} {decision.direction}[/bold green] "
                f"qty={result.filled_quantity:.5f} @ {result.avg_fill_price:.2f}"
            )

            # Tracker 갱신
            self.tracker.refresh_from_db()
        else:
            # Risk Gate 차단 또는 거래소 오류
            if result.error:
                CONSOLE.print(f"[yellow]{symbol}: {result.error}[/yellow]")

    def _get_tier_2_and_4(self, df: pd.DataFrame, symbol: str) -> tuple[TierOutput, TierOutput]:
        """학습된 모델 우선, 없으면 mock fallback"""
        # Try real models
        try:
            from flight_mind.tier2_pattern.inference import predict_tier2
            from flight_mind.tier4_regime.inference import predict_tier4

            t2 = predict_tier2(df.tail(60))

            df_daily = df.resample("1D").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

            if len(df_daily) >= 230:
                t4 = predict_tier4(df_daily, symbol=symbol)
            else:
                t4 = TierOutput(0.0, "none", {"reason": "insufficient_daily_data"})

            return t2, t4

        except FileNotFoundError:
            # 학습 미완료 — mock 사용
            from flight_mind.utils.mock_signals import get_mock_generator
            gen = get_mock_generator("realistic", seed=42)
            t2 = gen.generate_t2(df, len(df) - 1, future_horizon_bars=12)
            t4 = gen.generate_t4(df, len(df) - 1)
            return t2, t4

    # =========================================================================
    # Periodic Tasks
    # =========================================================================
    def _maybe_send_heartbeat(self) -> None:
        now = time.time()
        if now - self.stats.last_heartbeat_at >= HEARTBEAT_INTERVAL_S:
            self.notifier.send_heartbeat(
                daemon_name="paper_worker",
                uptime_seconds=self.stats.uptime_seconds(),
                n_decisions=self.stats.decisions_made,
                n_trades=self.stats.positions_closed,
            )
            self.stats.last_heartbeat_at = now

    def _maybe_send_daily_summary(self) -> None:
        """매일 자정 (UTC) 일일 리포트"""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # 자정 직후 1시간 이내 + 오늘 아직 안 보냈으면
        if now.hour != DAILY_SUMMARY_HOUR_UTC:
            return
        if self.stats.last_daily_summary_date == today:
            return

        # 어제 데이터 집계
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        with get_audit_conn() as conn:
            rows = conn.execute(
                """SELECT pnl_usdt, pnl_pct
                   FROM trades
                   WHERE DATE(exit_ts_utc) = ?""",
                [yesterday],
            ).fetchall()

        if not rows:
            self.stats.last_daily_summary_date = today
            return

        pnls_usdt = [r["pnl_usdt"] or 0 for r in rows]
        pnls_pct = [r["pnl_pct"] or 0 for r in rows]
        wins = sum(1 for p in pnls_usdt if p > 0)

        full_pnl = compute_pnl_summary()

        self.notifier.send_daily_summary({
            "date": yesterday,
            "n_trades": len(rows),
            "n_wins": wins,
            "pnl_usdt": sum(pnls_usdt),
            "pnl_pct": sum(pnls_pct),
            "best_trade_pct": max(pnls_pct) if pnls_pct else 0,
            "worst_trade_pct": min(pnls_pct) if pnls_pct else 0,
            "total_pnl_pct": full_pnl.total_realized_pct,
        })
        self.stats.last_daily_summary_date = today

    # =========================================================================
    # Shutdown / Failure
    # =========================================================================
    def _setup_signal_handlers(self) -> None:
        """SIGINT/SIGTERM → graceful shutdown"""
        def shutdown(signum, frame):
            CONSOLE.print(f"\n[yellow]Received signal {signum}, shutting down...[/yellow]")
            self.running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

    def _sleep_interruptible(self, seconds: float) -> None:
        """1초 단위로 sleep — 빠른 종료 가능"""
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    def _handle_cycle_failure(self, e: Exception) -> None:
        tb = traceback.format_exc()
        CONSOLE.print(f"[red]Cycle failed: {type(e).__name__}: {e}[/red]")
        CONSOLE.print(f"[dim]{tb}[/dim]")

        self.notifier.send_error(
            daemon_name="paper_worker",
            error_type=type(e).__name__,
            message=str(e),
        )

    def _shutdown(self) -> int:
        CONSOLE.print(f"\n[bold]Shutdown stats:[/bold]")
        CONSOLE.print(f"  Uptime:        {self.stats.uptime_seconds()}s")
        CONSOLE.print(f"  Cycles done:   {self.stats.cycles_completed}")
        CONSOLE.print(f"  Cycles failed: {self.stats.cycles_failed}")
        CONSOLE.print(f"  Decisions:     {self.stats.decisions_made}")
        CONSOLE.print(f"  Orders:        {self.stats.orders_placed}")
        CONSOLE.print(f"  Closed:        {self.stats.positions_closed}")

        self.notifier.send(
            f"*🛑 Daemon stopped*\n"
            f"Uptime: `{self.stats.uptime_seconds()}s`\n"
            f"Cycles: `{self.stats.cycles_completed}` "
            f"(failed: `{self.stats.cycles_failed}`)\n"
            f"Trades: `{self.stats.positions_closed}`",
            severity=Severity.INFO,
        )
        return 0


# =============================================================================
# Helpers
# =============================================================================
def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="Flight-Mind Paper Trading Daemon")
    p.add_argument("--mode", default="paper", choices=["paper", "testnet", "live"])
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Override symbols (default: from config)")
    p.add_argument("--cycle-s", type=int, default=CYCLE_INTERVAL_S,
                   help=f"Cycle interval seconds (default: {CYCLE_INTERVAL_S})")
    p.add_argument("--once", action="store_true",
                   help="Run single cycle and exit (testing)")
    args = p.parse_args()

    daemon = PaperTradingDaemon(
        mode=Mode(args.mode),
        symbols=args.symbols,
        cycle_interval_s=args.cycle_s,
    )

    if args.once:
        # Single cycle for testing
        CONSOLE.print("[yellow]Running single cycle (--once)[/yellow]")
        daemon.tracker.refresh_from_db()
        try:
            daemon._run_one_cycle()
            return 0
        except Exception as e:
            CONSOLE.print(f"[red]Cycle failed: {e}[/red]")
            traceback.print_exc()
            return 1

    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
