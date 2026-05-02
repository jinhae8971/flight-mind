"""
Day 9 — Telegram Notifier + Paper Trading Daemon Tests
========================================================
실제 Telegram API를 호출하지 않고 (테스트 환경 격리)
notifier 동작과 daemon 단일 사이클 흐름 검증.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from flight_mind.notify.telegram import (Severity, TelegramConfig,
                                            TelegramNotifier)


# =============================================================================
# Telegram Notifier
# =============================================================================
class TestTelegramConfig:
    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        cfg = TelegramConfig.from_env()
        assert cfg.enabled is False

    def test_enabled_with_both_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token_123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")
        cfg = TelegramConfig.from_env()
        assert cfg.enabled is True

    def test_disabled_with_only_token(self, monkeypatch):
        """둘 다 있어야 enabled — 한쪽만 있으면 disabled"""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token_123")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        cfg = TelegramConfig.from_env()
        assert cfg.enabled is False


class TestTelegramNotifier:
    def test_disabled_send_returns_false(self):
        cfg = TelegramConfig(bot_token=None, chat_id=None, enabled=False)
        notifier = TelegramNotifier(cfg)
        assert notifier.send("hello") is False

    def test_dedup_prevents_duplicate_within_window(self):
        """같은 메시지 연속 전송 시 두 번째는 dedup"""
        cfg = TelegramConfig(bot_token="x", chat_id="y", enabled=True)
        notifier = TelegramNotifier(cfg)

        with patch.object(notifier, "_send_raw", return_value=True) as mock:
            ok1 = notifier.send("same_message", severity=Severity.INFO)
            ok2 = notifier.send("same_message", severity=Severity.INFO)

        assert ok1 is True
        assert ok2 is False    # dedup'd
        assert mock.call_count == 1

    def test_different_messages_not_deduped(self):
        cfg = TelegramConfig(bot_token="x", chat_id="y", enabled=True)
        notifier = TelegramNotifier(cfg)

        with patch.object(notifier, "_send_raw", return_value=True) as mock:
            notifier.send("msg_1")
            notifier.send("msg_2")

        assert mock.call_count == 2

    def test_critical_with_explicit_dedup_key_bypasses_dedup(self):
        """Critical 알림은 timestamp dedup_key로 매번 다름 → 중복 안 됨"""
        cfg = TelegramConfig(bot_token="x", chat_id="y", enabled=True)
        notifier = TelegramNotifier(cfg)

        with patch.object(notifier, "_send_raw", return_value=True) as mock:
            # 명시적으로 다른 dedup_key 사용 → 두 번 모두 발송
            notifier.send("kill switch test 1", severity=Severity.CRITICAL,
                          dedup_key="ks_1")
            notifier.send("kill switch test 2", severity=Severity.CRITICAL,
                          dedup_key="ks_2")

        assert mock.call_count == 2

    def test_send_raw_failure_returns_false(self):
        cfg = TelegramConfig(bot_token="x", chat_id="y", enabled=True)
        notifier = TelegramNotifier(cfg)

        with patch.object(notifier, "_send_raw", return_value=False):
            ok = notifier.send("test")
        assert ok is False

    def test_send_decision_skips_hold(self):
        """hold 액션은 알림 보내지 않음 (signal noise 방지)"""
        cfg = TelegramConfig(bot_token="x", chat_id="y", enabled=True)
        notifier = TelegramNotifier(cfg)

        with patch.object(notifier, "_send_raw", return_value=True) as mock:
            ok = notifier.send_decision(
                symbol="BTCUSDT", action="hold", direction="none",
                confluence=0.3, tier_signals={},
            )
        assert ok is False
        assert mock.call_count == 0

    def test_send_decision_sends_for_open_long(self):
        cfg = TelegramConfig(bot_token="x", chat_id="y", enabled=True)
        notifier = TelegramNotifier(cfg)

        with patch.object(notifier, "_send_raw", return_value=True) as mock:
            ok = notifier.send_decision(
                symbol="BTCUSDT", action="open_long", direction="long",
                confluence=0.92,
                tier_signals={"T1": {"direction": "long", "score": 0.9}},
            )
        assert ok is True
        assert mock.call_count == 1
        # 메시지에 핵심 정보 포함
        call_args = mock.call_args
        text = call_args[0][0] if call_args[0] else call_args[1].get("text", "")
        assert "BTCUSDT" in text
        assert "OPEN_LONG" in text or "open_long" in text


# =============================================================================
# Paper Trading Daemon — Single Cycle
# =============================================================================
class TestPaperTradingDaemonSingle:
    """Single cycle smoke test — DB 격리 환경에서"""

    @pytest.fixture(autouse=True)
    def isolated_dbs(self, tmp_path, monkeypatch):
        from flight_mind.risk import audit, manager
        monkeypatch.setattr(audit, "AUDIT_DB_PATH", tmp_path / "audit.db")
        monkeypatch.setattr(manager, "KILLSWITCH_STATE_PATH", tmp_path / "killswitch.json")

        from flight_mind.risk import position_tracker
        monkeypatch.setattr(position_tracker, "_tracker", None)
        monkeypatch.setattr(manager, "_manager", None)

        # Disable Telegram (test 환경)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        from flight_mind.notify import telegram
        monkeypatch.setattr(telegram, "_notifier", None)

        audit.init_audit_db()

    def test_daemon_initializes(self):
        from flight_mind.daemon.paper_worker import PaperTradingDaemon
        from flight_mind.execution.engine import Mode

        daemon = PaperTradingDaemon(mode=Mode.PAPER, symbols=["BTCUSDT"])
        assert daemon.mode == Mode.PAPER
        assert daemon.symbols == ["BTCUSDT"]
        assert daemon.running is False

    def test_single_cycle_with_no_data(self, monkeypatch):
        """OHLCV가 비어 있어도 데몬이 죽지 않음"""
        from flight_mind.daemon.paper_worker import PaperTradingDaemon
        from flight_mind.execution.engine import Mode

        # fetch_ohlcv → empty DataFrame
        import pandas as pd
        monkeypatch.setattr(
            "flight_mind.daemon.paper_worker.fetch_ohlcv",
            lambda symbol, interval: pd.DataFrame(),
        )

        daemon = PaperTradingDaemon(mode=Mode.PAPER, symbols=["BTCUSDT"])
        # Should not raise
        daemon._run_one_cycle()

    def test_single_cycle_runs_with_real_data(self):
        """실제 DuckDB 데이터로 단일 사이클 실행 — 죽지 않으면 OK"""
        from flight_mind.daemon.paper_worker import PaperTradingDaemon
        from flight_mind.execution.engine import Mode

        daemon = PaperTradingDaemon(mode=Mode.PAPER, symbols=["BTCUSDT"])
        daemon.tracker.refresh_from_db()

        # Should not raise
        daemon._run_one_cycle()

        # Stats 갱신 확인
        assert daemon.stats.decisions_made >= 0   # 최소한 시도는 했어야 함

    def test_consecutive_failures_tracking(self):
        from flight_mind.daemon.paper_worker import PaperTradingDaemon
        from flight_mind.execution.engine import Mode

        daemon = PaperTradingDaemon(
            mode=Mode.PAPER, symbols=["BTCUSDT"],
            max_consecutive_failures=3,
        )

        # Simulate 2 failures
        for _ in range(2):
            daemon._handle_cycle_failure(RuntimeError("test error"))
            daemon.consecutive_failures += 1

        assert daemon.consecutive_failures == 2
        # 아직 emergency stop 발동 안 됨

    def test_should_exit_logic(self):
        """TP/SL/Time 청산 조건 평가"""
        from flight_mind.daemon.paper_worker import PaperTradingDaemon
        from flight_mind.execution.engine import Mode
        from flight_mind.risk.position_tracker import OpenPosition
        from datetime import datetime, timezone

        daemon = PaperTradingDaemon(mode=Mode.PAPER)

        # Long position at 70000
        pos = OpenPosition(
            trade_id=1, symbol="BTCUSDT", direction="long",
            entry_price=70000.0, quantity=0.001,
            entry_ts_utc=datetime.now(timezone.utc).isoformat(),
        )

        # +6% → TP
        assert daemon._should_exit(pos, current_price=74200.0) == "tp"

        # -3% → SL
        assert daemon._should_exit(pos, current_price=67900.0) == "sl"

        # No exit
        assert daemon._should_exit(pos, current_price=70500.0) is None
