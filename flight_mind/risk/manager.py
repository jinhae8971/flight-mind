"""
Risk Manager — Kill-Switch + Pre-trade Gate
==============================================
영길님 자본 보호의 마지막 안전 레이어.

영길님 정책 (config.py):
  - daily_loss_pct       : -5%  (하루 손실 한도)
  - weekly_loss_pct      : -10% (주간 손실 한도)
  - max_drawdown_pct     : -15% (누적 MDD 한도)
  - daily_trade_limit    : 2회   (하루 진입 횟수)
  - max_consecutive_losses : 3 (연속 손실 후 24h 거래 중단)

Kill-Switch 발동 시점:
  1. 사전 (pre-trade): execute_decision 호출 직전 RiskGate 통과 검증
  2. 사후 (post-trade): 청산 직후 누적 PnL 재계산 + 발동 조건 체크
  3. 정기 (heartbeat): 5분마다 unrealized PnL 포함 누적 손실 체크

Hierarchy (강한 것부터):
  EMERGENCY_STOP    — 모든 거래 영구 중단 (수동 해제만 가능)
  CIRCUIT_BREAKER   — 24h 거래 중단 (날짜 바뀌면 자동 해제 가능)
  DAILY_HALT        — 오늘 거래 중단 (자정 자동 해제)
  COOLDOWN          — 단순 진입 빈도 제한
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from rich.console import Console

from flight_mind.config import CAPITAL, RISK
from flight_mind.risk.audit import get_audit_conn
from flight_mind.risk.position_tracker import get_position_tracker


CONSOLE = Console()


# Kill-switch state file (영구 보존)
KILLSWITCH_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "killswitch.json"


class KillSwitchLevel(str, Enum):
    OK = "ok"
    COOLDOWN = "cooldown"
    DAILY_HALT = "daily_halt"
    CIRCUIT_BREAKER = "circuit_breaker"     # 24h
    EMERGENCY_STOP = "emergency_stop"       # 영구


@dataclass
class RiskGateResult:
    """사전 거래 검증 결과"""
    allowed: bool
    blockers: list[str] = field(default_factory=list)    # 차단 사유 (allowed=False 시)
    warnings: list[str] = field(default_factory=list)    # 진행은 가능하지만 주의 필요


@dataclass
class KillSwitchState:
    """현재 kill-switch 상태"""
    level: KillSwitchLevel = KillSwitchLevel.OK
    triggered_at: str | None = None
    expires_at: str | None = None    # 자동 해제 시각 (None = 영구)
    reason: str | None = None
    metrics_at_trigger: dict = field(default_factory=dict)


# =============================================================================
# Persistent State
# =============================================================================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_state() -> KillSwitchState:
    if not KILLSWITCH_STATE_PATH.exists():
        return KillSwitchState()
    try:
        data = json.loads(KILLSWITCH_STATE_PATH.read_text())
        return KillSwitchState(
            level=KillSwitchLevel(data.get("level", "ok")),
            triggered_at=data.get("triggered_at"),
            expires_at=data.get("expires_at"),
            reason=data.get("reason"),
            metrics_at_trigger=data.get("metrics_at_trigger", {}),
        )
    except Exception:
        return KillSwitchState()


def _save_state(state: KillSwitchState) -> None:
    KILLSWITCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    KILLSWITCH_STATE_PATH.write_text(json.dumps({
        "level": state.level.value,
        "triggered_at": state.triggered_at,
        "expires_at": state.expires_at,
        "reason": state.reason,
        "metrics_at_trigger": state.metrics_at_trigger,
    }, indent=2))


# =============================================================================
# PnL Aggregation
# =============================================================================
@dataclass
class PnLSummary:
    """누적 PnL 요약"""
    today_realized_usdt: float = 0.0
    today_realized_pct: float = 0.0
    today_n_trades: int = 0
    today_n_wins: int = 0

    week_realized_usdt: float = 0.0
    week_realized_pct: float = 0.0

    total_realized_usdt: float = 0.0
    total_realized_pct: float = 0.0
    max_drawdown_pct: float = 0.0    # 최고점 대비 최대 하락

    consecutive_losses: int = 0       # 마지막 N개 연속 손실
    last_trade_ts_utc: str | None = None


def compute_pnl_summary(capital_usdt: float | None = None) -> PnLSummary:
    """
    audit DB의 trades 테이블에서 누적 PnL 계산.

    Args:
        capital_usdt: PnL%의 기준 자본 (default: CAPITAL.live trading 비중)
    """
    if capital_usdt is None:
        capital_usdt = CAPITAL.total_usdt * CAPITAL.live_trading_pct

    summary = PnLSummary()
    now = _now_utc()
    today_str = now.strftime("%Y-%m-%d")
    week_ago_iso = (now - timedelta(days=7)).isoformat()

    with get_audit_conn() as conn:
        # 오늘
        today_rows = conn.execute(
            """SELECT pnl_usdt, pnl_pct
               FROM trades
               WHERE exit_ts_utc IS NOT NULL
                 AND DATE(exit_ts_utc) = ?""",
            [today_str],
        ).fetchall()
        summary.today_realized_usdt = sum(r["pnl_usdt"] or 0 for r in today_rows)
        summary.today_realized_pct = (summary.today_realized_usdt / capital_usdt) * 100
        summary.today_n_trades = len(today_rows)
        summary.today_n_wins = sum(1 for r in today_rows if (r["pnl_usdt"] or 0) > 0)

        # 7일
        week_rows = conn.execute(
            """SELECT pnl_usdt
               FROM trades
               WHERE exit_ts_utc IS NOT NULL
                 AND exit_ts_utc >= ?""",
            [week_ago_iso],
        ).fetchall()
        summary.week_realized_usdt = sum(r["pnl_usdt"] or 0 for r in week_rows)
        summary.week_realized_pct = (summary.week_realized_usdt / capital_usdt) * 100

        # 누적
        all_rows = conn.execute(
            """SELECT pnl_usdt, exit_ts_utc
               FROM trades
               WHERE exit_ts_utc IS NOT NULL
               ORDER BY exit_ts_utc"""
        ).fetchall()
        summary.total_realized_usdt = sum(r["pnl_usdt"] or 0 for r in all_rows)
        summary.total_realized_pct = (summary.total_realized_usdt / capital_usdt) * 100

        # MDD 계산: cumulative PnL의 peak-to-trough
        cumulative = 0.0
        peak = 0.0
        mdd = 0.0
        for r in all_rows:
            cumulative += r["pnl_usdt"] or 0
            peak = max(peak, cumulative)
            drawdown_usdt = peak - cumulative
            drawdown_pct = (drawdown_usdt / capital_usdt) * 100
            mdd = max(mdd, drawdown_pct)
        summary.max_drawdown_pct = -mdd     # 음수로 표현

        # 연속 손실
        recent_rows = conn.execute(
            """SELECT pnl_usdt, exit_ts_utc
               FROM trades
               WHERE exit_ts_utc IS NOT NULL
               ORDER BY exit_ts_utc DESC
               LIMIT 10"""
        ).fetchall()
        consec = 0
        for r in recent_rows:
            if (r["pnl_usdt"] or 0) < 0:
                consec += 1
            else:
                break
        summary.consecutive_losses = consec

        if recent_rows:
            summary.last_trade_ts_utc = recent_rows[0]["exit_ts_utc"]

    return summary


# =============================================================================
# Risk Manager (사전 게이트 + Kill-Switch)
# =============================================================================
class RiskManager:
    """모든 거래 의사결정 직전 호출되는 안전망"""

    def __init__(self, capital_usdt: float | None = None,
                 max_consecutive_losses: int = 3):
        self.capital_usdt = capital_usdt or (CAPITAL.total_usdt * CAPITAL.live_trading_pct)
        self.max_consecutive_losses = max_consecutive_losses
        self.tracker = get_position_tracker()

    # =========================================================================
    # Public API — Pre-trade Gate
    # =========================================================================
    def check(self, symbol: str, mode: str = "paper") -> RiskGateResult:
        """
        진입 직전 검증. 모든 안전 조건 체크.

        Returns:
            RiskGateResult — allowed=True 만 진입 진행
        """
        result = RiskGateResult(allowed=True)

        # 1. Kill-switch 상태 (영구/회로차단/일일정지)
        ks_blocker = self._check_killswitch()
        if ks_blocker:
            result.allowed = False
            result.blockers.append(ks_blocker)

        # 2. PnL 한도
        pnl = compute_pnl_summary(self.capital_usdt)
        pnl_blockers = self._check_pnl_limits(pnl)
        if pnl_blockers:
            result.allowed = False
            result.blockers.extend(pnl_blockers)

        # 3. 일일 거래 횟수
        if pnl.today_n_trades >= RISK.daily_trade_limit:
            result.allowed = False
            result.blockers.append(
                f"daily_trade_limit: {pnl.today_n_trades}/{RISK.daily_trade_limit}"
            )

        # 4. 연속 손실
        if pnl.consecutive_losses >= self.max_consecutive_losses:
            result.allowed = False
            result.blockers.append(
                f"consecutive_losses: {pnl.consecutive_losses} (limit: {self.max_consecutive_losses})"
            )

        # 5. 같은 심볼 오픈 포지션 (중복 진입 방지)
        if self.tracker.has_open_position(symbol):
            result.allowed = False
            result.blockers.append(f"position_already_open: {symbol}")

        # 6. 경고 (진행은 가능하지만 주의)
        if abs(pnl.today_realized_pct) >= abs(RISK.daily_loss_pct) * 0.7:
            result.warnings.append(
                f"approaching_daily_limit: {pnl.today_realized_pct:.2f}% / {RISK.daily_loss_pct}%"
            )
        if pnl.consecutive_losses >= 2:
            result.warnings.append(
                f"consecutive_losses_growing: {pnl.consecutive_losses}"
            )

        return result

    # =========================================================================
    # Public API — Post-trade Update
    # =========================================================================
    def update_after_trade(self, mode: str = "paper") -> KillSwitchState:
        """
        거래 청산 후 호출. PnL 재계산 + Kill-switch 발동 조건 체크.

        Returns:
            업데이트된 KillSwitchState (변동 있으면 영길님께 알림 권장)
        """
        pnl = compute_pnl_summary(self.capital_usdt)
        state = _load_state()

        # 자동 해제 체크 (만료 시간 지난 경우)
        if state.expires_at:
            try:
                expires_dt = datetime.fromisoformat(state.expires_at.rstrip("Z"))
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                if _now_utc() >= expires_dt:
                    CONSOLE.print(f"[green]Kill-switch expired — auto-clearing[/green]")
                    state = KillSwitchState()
                    _save_state(state)
            except Exception:
                pass

        # 새로운 발동 조건 체크 (강한 것부터)
        new_state = self._evaluate_killswitch_triggers(pnl, current_state=state)

        if new_state.level != state.level:
            _save_state(new_state)
            CONSOLE.print(
                f"[bold red]🚨 Kill-switch level changed: "
                f"{state.level.value} → {new_state.level.value}[/bold red]"
            )
            CONSOLE.print(f"   Reason: {new_state.reason}")

        return new_state

    # =========================================================================
    # Manual Controls
    # =========================================================================
    def trigger_emergency_stop(self, reason: str) -> None:
        """긴급 정지 — 영길님 수동 해제까지 모든 거래 차단"""
        state = KillSwitchState(
            level=KillSwitchLevel.EMERGENCY_STOP,
            triggered_at=_now_utc().isoformat() + "Z",
            expires_at=None,    # 수동 해제만
            reason=f"MANUAL: {reason}",
            metrics_at_trigger=compute_pnl_summary(self.capital_usdt).__dict__,
        )
        _save_state(state)
        CONSOLE.print(f"[bold red]🚨 EMERGENCY STOP: {reason}[/bold red]")

    def clear(self, force: bool = False) -> bool:
        """
        Kill-switch 해제.
        EMERGENCY_STOP은 force=True 필요.
        """
        state = _load_state()
        if state.level == KillSwitchLevel.EMERGENCY_STOP and not force:
            CONSOLE.print(
                "[yellow]EMERGENCY_STOP은 force=True 필요. 영길님 의도 확인 후 해제 권장[/yellow]"
            )
            return False

        _save_state(KillSwitchState())
        CONSOLE.print(f"[green]Kill-switch cleared (was: {state.level.value})[/green]")
        return True

    def get_state(self) -> KillSwitchState:
        return _load_state()

    # =========================================================================
    # Internal — Trigger Evaluation
    # =========================================================================
    def _check_killswitch(self) -> str | None:
        """현재 kill-switch 상태가 거래 차단하는지 체크"""
        state = _load_state()

        if state.level == KillSwitchLevel.EMERGENCY_STOP:
            return f"EMERGENCY_STOP: {state.reason} (수동 해제 필요)"

        if state.level == KillSwitchLevel.CIRCUIT_BREAKER:
            # 24h 만료 체크
            if state.expires_at:
                try:
                    expires_dt = datetime.fromisoformat(state.expires_at.rstrip("Z"))
                    expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                    if _now_utc() < expires_dt:
                        return (f"CIRCUIT_BREAKER: {state.reason} "
                                f"(해제: {expires_dt.strftime('%Y-%m-%d %H:%M UTC')})")
                except Exception:
                    return f"CIRCUIT_BREAKER: {state.reason}"

        if state.level == KillSwitchLevel.DAILY_HALT:
            # 자정 만료 체크
            if state.triggered_at:
                try:
                    trig_dt = datetime.fromisoformat(state.triggered_at.rstrip("Z"))
                    trig_dt = trig_dt.replace(tzinfo=timezone.utc)
                    if trig_dt.date() == _now_utc().date():
                        return f"DAILY_HALT: {state.reason}"
                except Exception:
                    return f"DAILY_HALT: {state.reason}"

        return None

    def _check_pnl_limits(self, pnl: PnLSummary) -> list[str]:
        """PnL 한도 위반 체크"""
        blockers = []

        if pnl.today_realized_pct <= RISK.daily_loss_pct:
            blockers.append(
                f"daily_loss_breach: {pnl.today_realized_pct:.2f}% <= {RISK.daily_loss_pct}%"
            )

        if pnl.week_realized_pct <= RISK.weekly_loss_pct:
            blockers.append(
                f"weekly_loss_breach: {pnl.week_realized_pct:.2f}% <= {RISK.weekly_loss_pct}%"
            )

        if pnl.max_drawdown_pct <= RISK.max_drawdown_pct:
            blockers.append(
                f"max_drawdown_breach: {pnl.max_drawdown_pct:.2f}% <= {RISK.max_drawdown_pct}%"
            )

        return blockers

    def _evaluate_killswitch_triggers(
        self, pnl: PnLSummary, current_state: KillSwitchState
    ) -> KillSwitchState:
        """현재 PnL 기준으로 kill-switch 레벨 결정"""
        # MDD 위반 → CIRCUIT_BREAKER (24h)
        if pnl.max_drawdown_pct <= RISK.max_drawdown_pct:
            return KillSwitchState(
                level=KillSwitchLevel.CIRCUIT_BREAKER,
                triggered_at=_now_utc().isoformat() + "Z",
                expires_at=(_now_utc() + timedelta(hours=24)).isoformat() + "Z",
                reason=f"max_drawdown breach: {pnl.max_drawdown_pct:.2f}%",
                metrics_at_trigger=pnl.__dict__,
            )

        # 주간 손실 → CIRCUIT_BREAKER (24h)
        if pnl.week_realized_pct <= RISK.weekly_loss_pct:
            return KillSwitchState(
                level=KillSwitchLevel.CIRCUIT_BREAKER,
                triggered_at=_now_utc().isoformat() + "Z",
                expires_at=(_now_utc() + timedelta(hours=24)).isoformat() + "Z",
                reason=f"weekly_loss breach: {pnl.week_realized_pct:.2f}%",
                metrics_at_trigger=pnl.__dict__,
            )

        # 일일 손실 → DAILY_HALT (자정 만료)
        if pnl.today_realized_pct <= RISK.daily_loss_pct:
            tomorrow = (_now_utc() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            return KillSwitchState(
                level=KillSwitchLevel.DAILY_HALT,
                triggered_at=_now_utc().isoformat() + "Z",
                expires_at=tomorrow.isoformat() + "Z",
                reason=f"daily_loss breach: {pnl.today_realized_pct:.2f}%",
                metrics_at_trigger=pnl.__dict__,
            )

        # 연속 손실 → DAILY_HALT
        if pnl.consecutive_losses >= self.max_consecutive_losses:
            tomorrow = (_now_utc() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            return KillSwitchState(
                level=KillSwitchLevel.DAILY_HALT,
                triggered_at=_now_utc().isoformat() + "Z",
                expires_at=tomorrow.isoformat() + "Z",
                reason=f"consecutive_losses: {pnl.consecutive_losses}",
                metrics_at_trigger=pnl.__dict__,
            )

        # 모두 OK
        return KillSwitchState()


# =============================================================================
# Module-level singleton
# =============================================================================
_manager: RiskManager | None = None


def get_risk_manager() -> RiskManager:
    global _manager
    if _manager is None:
        _manager = RiskManager()
    return _manager
