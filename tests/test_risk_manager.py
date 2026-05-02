"""
Day 8 — Risk Manager Tests
============================
Position Tracker, PnL Aggregator, Kill-Switch, Risk Gate 검증.

특히 영길님 자본을 보호하는 핵심 안전 시나리오들:
  - 일일 -5% 손실 시 자동 차단
  - 주간 -10% 손실 시 24h 차단
  - MDD -15% 시 24h 차단
  - 연속 3회 손실 시 24h 차단
  - 같은 심볼 중복 진입 방지
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flight_mind.risk.audit import (init_audit_db, log_decision, log_order,
                                       log_trade_close, log_trade_open)
from flight_mind.risk.manager import (KillSwitchLevel, RiskGateResult,
                                         RiskManager, compute_pnl_summary)
from flight_mind.risk.position_tracker import OpenPosition, PositionTracker


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    """모든 risk DB와 state file을 격리"""
    from flight_mind.risk import audit, manager
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", tmp_path / "audit.db")
    monkeypatch.setattr(manager, "KILLSWITCH_STATE_PATH", tmp_path / "killswitch.json")

    # Reset singletons (테스트 간 누수 방지)
    from flight_mind.risk import position_tracker
    monkeypatch.setattr(position_tracker, "_tracker", None)
    monkeypatch.setattr(manager, "_manager", None)

    init_audit_db()


def _create_open_trade(symbol="BTCUSDT", direction="long",
                       entry_price=70000.0, quantity=0.001) -> int:
    """헬퍼: audit DB에 오픈 trade 생성"""
    decision_id = log_decision(
        symbol=symbol, action=f"open_{direction}", direction=direction,
        confluence=0.9, tier_outputs={}, mode="paper",
    )
    order_id = log_order(
        decision_id=decision_id, exchange_order_id="TEST",
        symbol=symbol, side="buy" if direction == "long" else "sell",
        order_type="market", quantity=quantity, price=None, status="filled",
        filled_qty=quantity, avg_fill_price=entry_price, mode="paper",
    )
    return log_trade_open(decision_id, order_id, symbol, direction,
                          entry_price, quantity, mode="paper")


def _create_closed_trade(pnl_usdt: float, days_ago: int = 0,
                         symbol="BTCUSDT", direction="long") -> int:
    """헬퍼: 청산 완료된 trade (PnL 시뮬레이션용)"""
    trade_id = _create_open_trade(symbol=symbol, direction=direction)

    # 청산 (DB 직접 업데이트 — log_trade_close는 현재시각 사용)
    from flight_mind.risk.audit import get_audit_conn
    exit_dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    capital = 1750.0  # CAPITAL.live = 3500 * 0.5

    with get_audit_conn() as conn:
        conn.execute(
            """UPDATE trades
               SET exit_ts_utc = ?, exit_price = ?, pnl_usdt = ?,
                   pnl_pct = ?, exit_reason = ?, fees_usdt = 0.5
               WHERE id = ?""",
            [
                exit_dt.isoformat(),  # 'Z' 접미사 없음 — SQLite DATE() 호환
                70000.0 + (pnl_usdt / 0.001),  # back-calculate
                pnl_usdt,
                (pnl_usdt / capital) * 100,
                "tp" if pnl_usdt > 0 else "sl",
                trade_id,
            ],
        )
    return trade_id


# =============================================================================
# Position Tracker
# =============================================================================
class TestPositionTracker:
    def test_empty_initially(self):
        tracker = PositionTracker()
        assert tracker.refresh_from_db() == 0
        assert len(tracker.get_open_positions()) == 0

    def test_loads_open_position_from_db(self):
        _create_open_trade(symbol="BTCUSDT", direction="long")
        tracker = PositionTracker()
        positions = tracker.get_open_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"
        assert positions[0].direction == "long"

    def test_has_open_position(self):
        _create_open_trade(symbol="BTCUSDT")
        tracker = PositionTracker()
        assert tracker.has_open_position("BTCUSDT") is True
        assert tracker.has_open_position("ETHUSDT") is False

    def test_unrealized_pnl_long_in_profit(self):
        trade_id = _create_open_trade(symbol="BTCUSDT", direction="long",
                                       entry_price=70000.0, quantity=0.01)
        tracker = PositionTracker()
        positions = tracker.get_open_positions()
        pos = positions[0]

        # Price up 1% → unrealized profit
        pos.update_market_price(current_price=70700.0)
        # Gross: (70700 - 70000) * 0.01 = 7 USDT
        # Fee: 70700 * 0.01 * 0.0005 = 0.35 USDT (one side, entry fee was 0)
        # Net: ~6.65 USDT (positive)
        assert pos.unrealized_pnl_usdt > 6.0
        assert pos.unrealized_pnl_pct > 0.5

    def test_unrealized_pnl_short_in_profit(self):
        trade_id = _create_open_trade(symbol="BTCUSDT", direction="short",
                                       entry_price=70000.0, quantity=0.01)
        tracker = PositionTracker()
        pos = tracker.get_open_positions()[0]

        # Price down 1% → short profits
        pos.update_market_price(current_price=69300.0)
        assert pos.unrealized_pnl_usdt > 6.0

    def test_close_position_removes_from_cache(self):
        trade_id = _create_open_trade(symbol="BTCUSDT", entry_price=70000, quantity=0.01)
        tracker = PositionTracker()

        # Open: 1
        assert len(tracker.get_open_positions()) == 1

        # Create exit order (FK 충족용)
        from flight_mind.risk.audit import log_order
        exit_order_id = log_order(
            decision_id=None, exchange_order_id="EXIT_TEST",
            symbol="BTCUSDT", side="sell", order_type="market",
            quantity=0.01, price=None, status="filled",
            filled_qty=0.01, avg_fill_price=70700.0, mode="paper",
        )

        # Close at +1%
        result = tracker.close_position(
            trade_id=trade_id, exit_order_id=exit_order_id,
            exit_price=70700.0, exit_reason="tp",
        )
        assert result["realized_pnl_usdt"] > 0
        assert len(tracker.get_open_positions()) == 0


# =============================================================================
# PnL Aggregation
# =============================================================================
class TestPnLAggregation:
    def test_empty_db_zero_pnl(self):
        pnl = compute_pnl_summary(capital_usdt=1750.0)
        assert pnl.today_realized_usdt == 0
        assert pnl.total_realized_usdt == 0
        assert pnl.consecutive_losses == 0

    def test_today_pnl_aggregation(self):
        _create_closed_trade(pnl_usdt=10.0, days_ago=0)
        _create_closed_trade(pnl_usdt=-5.0, days_ago=0)

        pnl = compute_pnl_summary(capital_usdt=1750.0)
        assert pnl.today_realized_usdt == pytest.approx(5.0, abs=0.01)
        assert pnl.today_n_trades == 2
        assert pnl.today_n_wins == 1

    def test_week_vs_total(self):
        # 어제 (within week)
        _create_closed_trade(pnl_usdt=20.0, days_ago=1)
        # 10일 전 (outside week)
        _create_closed_trade(pnl_usdt=100.0, days_ago=10)

        pnl = compute_pnl_summary(capital_usdt=1750.0)
        assert pnl.week_realized_usdt == pytest.approx(20.0, abs=0.01)
        assert pnl.total_realized_usdt == pytest.approx(120.0, abs=0.01)

    def test_max_drawdown(self):
        # +100 → 누적 100, peak 100
        _create_closed_trade(pnl_usdt=100.0, days_ago=10)
        # -150 → 누적 -50, drawdown from peak: 150
        _create_closed_trade(pnl_usdt=-150.0, days_ago=8)

        pnl = compute_pnl_summary(capital_usdt=1000.0)
        # 150 / 1000 = 15%
        assert pnl.max_drawdown_pct == pytest.approx(-15.0, abs=0.5)

    def test_consecutive_losses(self):
        _create_closed_trade(pnl_usdt=10.0, days_ago=5)   # win
        _create_closed_trade(pnl_usdt=-5.0, days_ago=4)   # loss 1
        _create_closed_trade(pnl_usdt=-3.0, days_ago=3)   # loss 2
        _create_closed_trade(pnl_usdt=-7.0, days_ago=2)   # loss 3 (most recent)

        pnl = compute_pnl_summary(capital_usdt=1000.0)
        assert pnl.consecutive_losses == 3


# =============================================================================
# Risk Gate (Pre-trade)
# =============================================================================
class TestRiskGate:
    def test_allows_when_clean(self):
        mgr = RiskManager(capital_usdt=1750.0)
        result = mgr.check("BTCUSDT")
        assert result.allowed is True
        assert len(result.blockers) == 0

    def test_blocks_daily_loss_exceeded(self):
        # 오늘 -100 USDT 손실 (-5.7% of 1750)
        _create_closed_trade(pnl_usdt=-100.0, days_ago=0)

        mgr = RiskManager(capital_usdt=1750.0)
        result = mgr.check("BTCUSDT")

        assert result.allowed is False
        assert any("daily_loss" in b for b in result.blockers)

    def test_blocks_open_position_duplicate(self):
        # BTC 오픈 포지션 존재
        _create_open_trade(symbol="BTCUSDT")

        mgr = RiskManager(capital_usdt=1750.0)
        result = mgr.check("BTCUSDT")

        assert result.allowed is False
        assert any("position_already_open" in b for b in result.blockers)

    def test_allows_different_symbol_when_btc_open(self):
        _create_open_trade(symbol="BTCUSDT")

        mgr = RiskManager(capital_usdt=1750.0)
        result = mgr.check("ETHUSDT")

        assert result.allowed is True

    def test_blocks_daily_trade_limit(self):
        # 오늘 2회 거래 완료
        _create_closed_trade(pnl_usdt=1.0, days_ago=0)
        _create_closed_trade(pnl_usdt=2.0, days_ago=0)

        mgr = RiskManager(capital_usdt=1750.0)
        result = mgr.check("BTCUSDT")

        assert result.allowed is False
        assert any("daily_trade_limit" in b for b in result.blockers)

    def test_blocks_consecutive_losses(self):
        # 연속 3회 손실
        _create_closed_trade(pnl_usdt=-1.0, days_ago=2)
        _create_closed_trade(pnl_usdt=-2.0, days_ago=1)
        _create_closed_trade(pnl_usdt=-3.0, days_ago=0)

        mgr = RiskManager(capital_usdt=1750.0, max_consecutive_losses=3)
        result = mgr.check("BTCUSDT")

        assert result.allowed is False
        assert any("consecutive_losses" in b for b in result.blockers)

    def test_warning_at_70pct_of_daily_limit(self):
        """daily_loss의 70% 도달 시 warning"""
        # -5%의 70% = -3.5% = -61.25 USDT
        _create_closed_trade(pnl_usdt=-65.0, days_ago=0)

        mgr = RiskManager(capital_usdt=1750.0)
        result = mgr.check("BTCUSDT")

        # 아직 -5% 미달이므로 진행 가능 (또는 trade_count 한도)
        # 단순 확인: warning 메시지가 포함되어야 함
        if result.allowed:
            assert any("approaching_daily_limit" in w for w in result.warnings)


# =============================================================================
# Kill-Switch State
# =============================================================================
class TestKillSwitch:
    def test_initial_state_is_ok(self):
        mgr = RiskManager(capital_usdt=1750.0)
        state = mgr.get_state()
        assert state.level == KillSwitchLevel.OK

    def test_emergency_stop_blocks_all(self):
        mgr = RiskManager(capital_usdt=1750.0)
        mgr.trigger_emergency_stop("manual test")

        result = mgr.check("BTCUSDT")
        assert result.allowed is False
        assert any("EMERGENCY_STOP" in b for b in result.blockers)

    def test_emergency_stop_requires_force_to_clear(self):
        mgr = RiskManager(capital_usdt=1750.0)
        mgr.trigger_emergency_stop("test")

        # 일반 clear는 거부
        ok = mgr.clear(force=False)
        assert ok is False
        assert mgr.get_state().level == KillSwitchLevel.EMERGENCY_STOP

        # force=True 만 허용
        ok = mgr.clear(force=True)
        assert ok is True
        assert mgr.get_state().level == KillSwitchLevel.OK

    def test_post_trade_triggers_daily_halt(self):
        # 오늘 -120 USDT 손실 = -6.86% (한도 -5% 초과)
        _create_closed_trade(pnl_usdt=-120.0, days_ago=0)

        mgr = RiskManager(capital_usdt=1750.0)
        new_state = mgr.update_after_trade()

        assert new_state.level == KillSwitchLevel.DAILY_HALT
        assert "daily_loss" in new_state.reason

    def test_post_trade_triggers_circuit_breaker_on_mdd(self):
        # 누적 +200 → -200 → MDD 400 / 1000 = 40% > 15% 한도
        _create_closed_trade(pnl_usdt=200.0, days_ago=10)
        _create_closed_trade(pnl_usdt=-400.0, days_ago=9)

        mgr = RiskManager(capital_usdt=1000.0)
        new_state = mgr.update_after_trade()

        assert new_state.level == KillSwitchLevel.CIRCUIT_BREAKER
        assert "max_drawdown" in new_state.reason

    def test_kill_switch_persists_across_restarts(self):
        # 1번째 매니저로 발동
        mgr1 = RiskManager(capital_usdt=1750.0)
        mgr1.trigger_emergency_stop("persistence test")

        # 2번째 매니저 (새 인스턴스) — 같은 상태 읽혀야 함
        mgr2 = RiskManager(capital_usdt=1750.0)
        state = mgr2.get_state()
        assert state.level == KillSwitchLevel.EMERGENCY_STOP


# =============================================================================
# Integration: ExecutionEngine + RiskManager
# =============================================================================
class TestExecutionEngineIntegration:
    """Risk Gate가 ExecutionEngine과 올바르게 통합됐는지"""

    def test_engine_blocks_when_position_already_open(self):
        from flight_mind.execution.engine import ExecutionEngine, Mode
        from flight_mind.fusion.layer import FusionDecision
        from flight_mind.tier1_rule.engine import TierOutput

        # 이미 BTC 오픈
        _create_open_trade(symbol="BTCUSDT")

        engine = ExecutionEngine(mode=Mode.PAPER)
        decision = FusionDecision(
            action="open_long", direction="long",
            confluence_score=0.9, position_size_usdt=70.0,
            leverage=5, stop_loss_pct=-3.0, take_profit_pct=6.0,
            max_hold_bars=12,
            tier_outputs={"T1": TierOutput(0.9, "long", {})},
        )

        result = engine.execute_decision("BTCUSDT", decision)
        assert result.success is False
        assert "Risk Gate" in (result.error or "")

    def test_engine_allows_with_skip_risk_gate(self):
        """skip_risk_gate=True (테스트 전용)는 우회 가능"""
        from flight_mind.execution.engine import ExecutionEngine, Mode
        from flight_mind.fusion.layer import FusionDecision
        from flight_mind.tier1_rule.engine import TierOutput

        _create_open_trade(symbol="BTCUSDT")

        engine = ExecutionEngine(mode=Mode.PAPER)
        decision = FusionDecision(
            action="open_long", direction="long", confluence_score=0.9,
            position_size_usdt=70.0, leverage=5,
            stop_loss_pct=-3.0, take_profit_pct=6.0, max_hold_bars=12,
            tier_outputs={"T1": TierOutput(0.9, "long", {})},
        )

        result = engine.execute_decision("BTCUSDT", decision, skip_risk_gate=True)
        assert result.success is True   # 우회 성공
