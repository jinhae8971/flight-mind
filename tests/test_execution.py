"""
Day 7 — Execution Layer Tests
==============================
Vault, Audit, ExecutionEngine 안전성 검증.

테스트 우선순위:
  1. Vault 암호화/복호화 무결성
  2. Live 모드 게이트 (3중 안전망)
  3. Audit 로그 영구성 (모든 결정 기록되는가)
  4. Paper 모드 흐름 (실제 주문 없이 시뮬레이션)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from flight_mind.execution.engine import (ExecutionEngine, Mode, OrderResult,
                                            is_live_authorized,
                                            make_exchange_client)
from flight_mind.fusion.layer import FusionDecision
from flight_mind.risk.audit import (AUDIT_DB_PATH, fetch_recent_decisions,
                                       fetch_recent_trades, init_audit_db,
                                       log_decision, log_order, log_trade_open)
from flight_mind.tier1_rule.engine import TierOutput
from flight_mind.vault.manager import (ApiCredential, Vault, VaultError,
                                          VaultLockedError)


# =============================================================================
# Vault Tests
# =============================================================================
class TestVault:
    """API 키 암호화/복호화 무결성"""

    @pytest.fixture
    def isolated_vault(self, tmp_path, monkeypatch):
        """격리된 vault — 영길님 실제 vault에 영향 없도록"""
        vault_path = tmp_path / "test_vault.json"
        monkeypatch.setenv("VAULT_PASSPHRASE", "test_passphrase_12345")
        return Vault(vault_path)

    def test_add_and_get_credential(self, isolated_vault):
        isolated_vault.add(
            label="test_key",
            api_key="my_api_key_123",
            secret="my_secret_456",
            permissions=["read", "trade"],
        )

        cred = isolated_vault.get("test_key")
        assert cred.api_key == "my_api_key_123"
        assert cred.secret == "my_secret_456"
        assert cred.label == "test_key"
        assert "read" in cred.permissions

    def test_wrong_passphrase_fails(self, tmp_path, monkeypatch):
        """잘못된 패스워드로는 절대 복호화 못함"""
        vault_path = tmp_path / "test_vault.json"

        # 첫 번째 패스워드로 저장
        monkeypatch.setenv("VAULT_PASSPHRASE", "correct_password")
        v1 = Vault(vault_path)
        v1.add("key1", "api_xxx", "secret_yyy")

        # 두 번째 패스워드로 시도 — 실패해야 함
        monkeypatch.setenv("VAULT_PASSPHRASE", "wrong_password_xx")
        v2 = Vault(vault_path)

        with pytest.raises(VaultLockedError):
            v2.get("key1")

    def test_no_passphrase_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VAULT_PASSPHRASE", raising=False)
        vault = Vault(tmp_path / "v.json")

        with pytest.raises(VaultError, match="VAULT_PASSPHRASE"):
            vault.add("k", "a", "b")

    def test_short_passphrase_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PASSPHRASE", "short")  # < 8 chars
        vault = Vault(tmp_path / "v.json")

        with pytest.raises(VaultError, match="too short"):
            vault.add("k", "a", "b")

    def test_overwrite_protection(self, isolated_vault):
        isolated_vault.add("k", "a1", "s1")

        # 기본은 덮어쓰기 차단
        with pytest.raises(VaultError, match="already exists"):
            isolated_vault.add("k", "a2", "s2")

        # 명시적으로 덮어쓰기 가능
        isolated_vault.add("k", "a2", "s2", overwrite=True)
        cred = isolated_vault.get("k")
        assert cred.api_key == "a2"

    def test_list_labels_no_plaintext(self, isolated_vault):
        """라벨 조회 시 평문 키 노출 안 됨"""
        isolated_vault.add("key1", "api_secret_xxx", "yyy")
        isolated_vault.add("key2", "another_secret", "zzz")

        labels = isolated_vault.list_labels()
        assert set(labels) == {"key1", "key2"}

    def test_remove(self, isolated_vault):
        isolated_vault.add("k", "a", "b")
        assert "k" in isolated_vault.list_labels()

        ok = isolated_vault.remove("k")
        assert ok is True
        assert "k" not in isolated_vault.list_labels()

        # 없는 라벨 삭제는 False 반환
        ok2 = isolated_vault.remove("nonexistent")
        assert ok2 is False


# =============================================================================
# Live Mode Gate Tests (가장 중요)
# =============================================================================
class TestLiveGate:
    """3중 안전 게이트 검증 — 실수로 라이브 주문 안 가도록"""

    def test_live_blocked_without_env_var(self, monkeypatch):
        """환경변수 없으면 Live 차단"""
        monkeypatch.delenv("FLIGHT_MIND_LIVE", raising=False)
        assert is_live_authorized() is False

    def test_live_authorized_with_env_var(self, monkeypatch):
        monkeypatch.setenv("FLIGHT_MIND_LIVE", "1")
        assert is_live_authorized() is True

    def test_engine_paper_mode_works_without_env(self, monkeypatch):
        """Paper 모드는 어떤 환경변수도 필요 없음"""
        monkeypatch.delenv("FLIGHT_MIND_LIVE", raising=False)
        engine = ExecutionEngine(mode=Mode.PAPER)
        assert engine.mode == Mode.PAPER

    def test_engine_live_mode_blocked_without_env(self, monkeypatch):
        """환경변수 없이 Live 모드 시도 → 즉시 차단"""
        monkeypatch.delenv("FLIGHT_MIND_LIVE", raising=False)
        with pytest.raises(RuntimeError, match="FLIGHT_MIND_LIVE=1"):
            ExecutionEngine(mode=Mode.LIVE)

    def test_paper_mode_no_exchange_client(self, monkeypatch):
        """Paper 모드는 CCXT 클라이언트 생성 안 함"""
        client = make_exchange_client(Mode.PAPER)
        assert client is None


# =============================================================================
# Audit Log Tests
# =============================================================================
class TestAudit:
    @pytest.fixture(autouse=True)
    def isolated_audit_db(self, tmp_path, monkeypatch):
        """격리된 audit DB"""
        from flight_mind.risk import audit
        original = audit.AUDIT_DB_PATH
        test_db = tmp_path / "test_audit.db"
        monkeypatch.setattr(audit, "AUDIT_DB_PATH", test_db)
        init_audit_db()
        yield
        # cleanup automatic via tmp_path

    def test_log_decision_returns_id(self):
        decision_id = log_decision(
            symbol="BTCUSDT",
            action="hold",
            direction="none",
            confluence=0.5,
            tier_outputs={"T1": {"score": 0.6, "direction": "long"}},
            mode="paper",
        )
        assert decision_id > 0

    def test_log_order_returns_id(self):
        decision_id = log_decision(
            symbol="BTCUSDT", action="open_long", direction="long",
            confluence=0.9, tier_outputs={}, mode="paper",
        )
        order_id = log_order(
            decision_id=decision_id,
            exchange_order_id="PAPER_123",
            symbol="BTCUSDT", side="buy", order_type="market",
            quantity=0.001, price=None, status="filled",
            filled_qty=0.001, avg_fill_price=70000.0,
            mode="paper",
        )
        assert order_id > 0

    def test_recent_decisions_query(self):
        log_decision(symbol="BTCUSDT", action="hold", direction="none",
                     confluence=0.3, tier_outputs={}, mode="paper")
        log_decision(symbol="ETHUSDT", action="open_long", direction="long",
                     confluence=0.95, tier_outputs={}, mode="paper")

        all_recent = fetch_recent_decisions(limit=10)
        assert len(all_recent) >= 2

        btc_only = fetch_recent_decisions(symbol="BTCUSDT")
        assert all(r["symbol"] == "BTCUSDT" for r in btc_only)


# =============================================================================
# Paper Mode End-to-End
# =============================================================================
class TestPaperExecution:
    """실제 주문 없이 전체 플로우 검증"""

    @pytest.fixture(autouse=True)
    def isolated_audit(self, tmp_path, monkeypatch):
        from flight_mind.risk import audit
        monkeypatch.setattr(audit, "AUDIT_DB_PATH", tmp_path / "audit.db")
        init_audit_db()

    def _make_decision(self, action: str, direction: str, confluence: float):
        return FusionDecision(
            action=action,
            direction=direction,
            confluence_score=confluence,
            position_size_usdt=70.0,
            leverage=5,
            stop_loss_pct=-3.0,
            take_profit_pct=6.0,
            max_hold_bars=12,
            tier_outputs={
                "T1": TierOutput(0.9, direction, {"mock": True}),
                "T2": TierOutput(0.85, direction, {"mock": True}),
                "T4": TierOutput(0.9, direction, {"mock": True}),
            },
        )

    def test_paper_engine_executes_open_long(self):
        engine = ExecutionEngine(mode=Mode.PAPER)
        decision = self._make_decision("open_long", "long", 0.92)

        result = engine.execute_decision("BTCUSDT", decision)

        assert result.success is True
        assert result.mode == "paper"
        assert result.filled_quantity > 0
        assert result.avg_fill_price > 0

    def test_paper_engine_executes_open_short(self):
        engine = ExecutionEngine(mode=Mode.PAPER)
        decision = self._make_decision("open_short", "short", 0.88)

        result = engine.execute_decision("BTCUSDT", decision)
        assert result.success is True
        assert result.filled_quantity > 0

    def test_paper_engine_handles_hold(self):
        """hold 결정은 주문 없이 audit만 남김"""
        engine = ExecutionEngine(mode=Mode.PAPER)
        decision = self._make_decision("hold", "none", 0.3)

        result = engine.execute_decision("BTCUSDT", decision)
        assert result.success is True
        assert result.filled_quantity == 0  # 주문 없음

    def test_dry_run_no_actual_order(self):
        """dry_run=True 시 실제 주문 없음"""
        engine = ExecutionEngine(mode=Mode.PAPER, dry_run=True)
        decision = self._make_decision("open_long", "long", 0.9)

        result = engine.execute_decision("BTCUSDT", decision)
        assert result.success is True
        assert result.exchange_order_id == "DRY_RUN"

    def test_position_size_caps_at_70_usdt(self):
        """영길님 정책: 최대 포지션 70 USDT"""
        engine = ExecutionEngine(mode=Mode.PAPER)
        sizing = engine.compute_position_size("BTCUSDT", available_usdt=100_000)

        # 100k × 2% = 2,000 USDT 였지만 hard cap 70
        assert sizing["notional_usdt"] <= 70 * 5  # leverage 5x
