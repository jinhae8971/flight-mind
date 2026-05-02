"""
Risk Status CLI
=================
영길님이 paper trading 운영 중 한 눈에 위험 상태를 보는 진단 도구.

Usage:
    python scripts/risk_status.py                    # 전체 상태
    python scripts/risk_status.py --positions        # 오픈 포지션만
    python scripts/risk_status.py --pnl              # PnL 요약만
    python scripts/risk_status.py --killswitch       # Kill-switch 상태
    python scripts/risk_status.py --emergency-stop "이유"   # 긴급 정지
    python scripts/risk_status.py --clear            # Kill-switch 해제
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from flight_mind.config import CAPITAL, RISK
from flight_mind.risk.manager import (KillSwitchLevel, compute_pnl_summary,
                                         get_risk_manager)
from flight_mind.risk.position_tracker import get_position_tracker


CONSOLE = Console()


def show_killswitch_status() -> None:
    """현재 Kill-Switch 상태"""
    mgr = get_risk_manager()
    state = mgr.get_state()

    color_map = {
        KillSwitchLevel.OK: "green",
        KillSwitchLevel.COOLDOWN: "yellow",
        KillSwitchLevel.DAILY_HALT: "yellow",
        KillSwitchLevel.CIRCUIT_BREAKER: "red",
        KillSwitchLevel.EMERGENCY_STOP: "bold red",
    }
    color = color_map.get(state.level, "white")

    icon_map = {
        KillSwitchLevel.OK: "✅",
        KillSwitchLevel.COOLDOWN: "🟡",
        KillSwitchLevel.DAILY_HALT: "🟡",
        KillSwitchLevel.CIRCUIT_BREAKER: "🔴",
        KillSwitchLevel.EMERGENCY_STOP: "🚨",
    }
    icon = icon_map.get(state.level, "❓")

    body = f"[{color}]{icon} Level: {state.level.value.upper()}[/{color}]"
    if state.reason:
        body += f"\nReason: {state.reason}"
    if state.triggered_at:
        body += f"\nTriggered: {state.triggered_at}"
    if state.expires_at:
        body += f"\nExpires:   {state.expires_at}"

    CONSOLE.print(Panel(body, title="Kill-Switch Status", expand=False))


def show_pnl_summary() -> None:
    """PnL 요약 + 한도 비교"""
    capital = CAPITAL.total_usdt * CAPITAL.live_trading_pct
    pnl = compute_pnl_summary(capital_usdt=capital)

    table = Table(title=f"PnL Summary (capital: {capital:.2f} USDT)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Limit", justify="right", style="dim")
    table.add_column("Status", justify="center")

    def fmt_status(value: float, limit: float, lower_is_worse: bool = True) -> str:
        if lower_is_worse:
            ratio = value / limit if limit != 0 else 0
            if ratio >= 1.0:
                return "[bold red]🚫 BREACHED[/bold red]"
            if ratio >= 0.7:
                return "[yellow]⚠ APPROACHING[/yellow]"
            return "[green]✓ OK[/green]"
        return "[green]✓ OK[/green]"

    table.add_row(
        "Today realized",
        f"{pnl.today_realized_usdt:+.2f} USDT ({pnl.today_realized_pct:+.2f}%)",
        f"{RISK.daily_loss_pct:.2f}%",
        fmt_status(pnl.today_realized_pct, RISK.daily_loss_pct),
    )
    table.add_row(
        "Today trades",
        f"{pnl.today_n_trades} ({pnl.today_n_wins} wins)",
        f"{RISK.daily_trade_limit}",
        "[bold red]🚫 LIMIT[/bold red]" if pnl.today_n_trades >= RISK.daily_trade_limit
        else "[green]✓ OK[/green]",
    )
    table.add_row(
        "Week realized",
        f"{pnl.week_realized_usdt:+.2f} USDT ({pnl.week_realized_pct:+.2f}%)",
        f"{RISK.weekly_loss_pct:.2f}%",
        fmt_status(pnl.week_realized_pct, RISK.weekly_loss_pct),
    )
    table.add_row(
        "Total realized",
        f"{pnl.total_realized_usdt:+.2f} USDT ({pnl.total_realized_pct:+.2f}%)",
        "-",
        "[green]✓[/green]",
    )
    table.add_row(
        "Max Drawdown",
        f"{pnl.max_drawdown_pct:.2f}%",
        f"{RISK.max_drawdown_pct:.2f}%",
        fmt_status(pnl.max_drawdown_pct, RISK.max_drawdown_pct),
    )
    table.add_row(
        "Consecutive losses",
        str(pnl.consecutive_losses),
        "3",
        "[bold red]🚫[/bold red]" if pnl.consecutive_losses >= 3
        else "[yellow]⚠[/yellow]" if pnl.consecutive_losses >= 2
        else "[green]✓[/green]",
    )

    CONSOLE.print(table)


def show_open_positions() -> None:
    """오픈 포지션 목록"""
    tracker = get_position_tracker()
    summary = tracker.summary()

    if summary["n_positions"] == 0:
        CONSOLE.print(Panel("[dim]오픈 포지션 없음[/dim]", title="Open Positions", expand=False))
        return

    table = Table(title=f"Open Positions ({summary['n_positions']})")
    table.add_column("Trade ID", justify="right")
    table.add_column("Symbol")
    table.add_column("Direction")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Quantity", justify="right")
    table.add_column("Unrealized PnL", justify="right")
    table.add_column("Hold (min)", justify="right")

    for p in summary["positions"]:
        pnl_color = "green" if p["unrealized_pnl_usdt"] >= 0 else "red"
        pnl_str = (f"[{pnl_color}]{p['unrealized_pnl_usdt']:+.2f} USDT "
                   f"({p['unrealized_pnl_pct']:+.2f}%)[/{pnl_color}]")

        dir_color = "green" if p["direction"] == "long" else "red"
        table.add_row(
            str(p["trade_id"]),
            p["symbol"],
            f"[{dir_color}]{p['direction'].upper()}[/{dir_color}]",
            f"{p['entry_price']:.2f}",
            f"{p['current_price']:.2f}" if p['current_price'] else "-",
            f"{p['quantity']:.5f}",
            pnl_str if p['current_price'] else "[dim]no quote[/dim]",
            str(p["hold_minutes"]),
        )

    CONSOLE.print(table)
    CONSOLE.print(f"[bold]Total unrealized: "
                  f"{summary['total_unrealized_pnl_usdt']:+.2f} USDT[/bold]")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--positions", action="store_true", help="오픈 포지션만 표시")
    p.add_argument("--pnl", action="store_true", help="PnL 요약만 표시")
    p.add_argument("--killswitch", action="store_true", help="Kill-Switch 상태만 표시")
    p.add_argument("--emergency-stop", metavar="REASON",
                   help="긴급 정지 발동 (수동 해제까지 모든 거래 차단)")
    p.add_argument("--clear", action="store_true", help="Kill-Switch 해제")
    p.add_argument("--clear-force", action="store_true",
                   help="EMERGENCY_STOP 포함 강제 해제")
    args = p.parse_args()

    if args.emergency_stop:
        mgr = get_risk_manager()
        mgr.trigger_emergency_stop(args.emergency_stop)
        return 0

    if args.clear or args.clear_force:
        mgr = get_risk_manager()
        ok = mgr.clear(force=args.clear_force)
        return 0 if ok else 1

    # 단일 섹션 표시
    if args.positions:
        show_open_positions()
        return 0
    if args.pnl:
        show_pnl_summary()
        return 0
    if args.killswitch:
        show_killswitch_status()
        return 0

    # 전체 표시
    show_killswitch_status()
    CONSOLE.print()
    show_pnl_summary()
    CONSOLE.print()
    show_open_positions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
