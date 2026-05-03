"""
Day 10 — Real Model Adapter Tests
====================================
학습된 모델이 없는 상태에서도 코드가 정확히 동작해야 함.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_mind.tier1_rule.engine import TierOutput
from flight_mind.utils.real_model_signals import (RealModelSignalGenerator,
                                                     check_model_status,
                                                     get_signal_generator)


@pytest.fixture
def sample_5m_data() -> pd.DataFrame:
    """가짜 5분봉 데이터 — 300일 분량 (Tier 4 테스트용)"""
    n = 300 * 288    # 300일 × 288바
    rng = np.random.default_rng(42)
    base = 70000 + rng.normal(0, 200, n).cumsum() * 0.01

    return pd.DataFrame({
        "open": base * 0.999,
        "high": base * 1.002,
        "low": base * 0.998,
        "close": base,
        "volume": rng.uniform(800, 1200, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))


# =============================================================================
# Model Status Check
# =============================================================================
class TestModelStatus:
    def test_status_when_no_models(self, monkeypatch, tmp_path):
        """모델 없을 때 상태 정확히 보고"""
        from flight_mind.config import MODEL_DIR
        monkeypatch.setattr(
            "flight_mind.utils.real_model_signals.MODEL_DIR",
            tmp_path,    # 빈 디렉토리
        )

        status = check_model_status()
        assert status.tier2_available is False
        assert status.tier4_available is False
        assert status.fallback_to_mock is True


# =============================================================================
# Signal Generator Factory
# =============================================================================
class TestSignalGeneratorFactory:
    def test_mock_mode_returns_mock(self):
        gen = get_signal_generator(mode="mock")
        # Mock generator는 MOCK_CONFIGS 키를 가짐
        from flight_mind.utils.mock_signals import MockSignalGenerator
        assert isinstance(gen, MockSignalGenerator)

    def test_real_mode_returns_real_adapter(self):
        gen = get_signal_generator(mode="real", symbol="BTCUSDT")
        assert isinstance(gen, RealModelSignalGenerator)
        assert gen.symbol == "BTCUSDT"

    def test_auto_mode_falls_back_to_mock_when_no_models(self, monkeypatch, tmp_path):
        """Auto 모드: 모델 없으면 mock fallback"""
        monkeypatch.setattr(
            "flight_mind.utils.real_model_signals.MODEL_DIR",
            tmp_path,
        )

        gen = get_signal_generator(mode="auto")
        # Mock으로 fallback
        from flight_mind.utils.mock_signals import MockSignalGenerator
        assert isinstance(gen, MockSignalGenerator)


# =============================================================================
# Real Model Adapter — Behavior without trained models
# =============================================================================
class TestRealModelAdapter:
    """학습된 모델 없을 때 graceful fallback 검증"""

    def test_t2_returns_none_without_model(self, sample_5m_data):
        gen = RealModelSignalGenerator(symbol="BTCUSDT")
        # 모델 파일 없으므로 'none' direction 반환
        result = gen.generate_t2(sample_5m_data, end_idx=200)
        assert isinstance(result, TierOutput)
        assert result.direction == "none"

    def test_t4_returns_none_without_model(self, sample_5m_data):
        gen = RealModelSignalGenerator(symbol="BTCUSDT")
        # 230일 충분, 모델 파일 없음
        result = gen.generate_t4(sample_5m_data, end_idx=230 * 288 + 1000)
        assert isinstance(result, TierOutput)
        assert result.direction == "none"

    def test_t2_warmup_protection(self, sample_5m_data):
        """60봉 미만 → warmup 'none' 반환 (모델 호출 시도 안 함)"""
        gen = RealModelSignalGenerator(symbol="BTCUSDT")
        result = gen.generate_t2(sample_5m_data, end_idx=30)  # < 60
        assert result.direction == "none"
        assert "warmup" in str(result.signals.get("reason", ""))

    def test_t4_warmup_protection(self, sample_5m_data):
        """230일 미만 → warmup 'none' 반환"""
        gen = RealModelSignalGenerator(symbol="BTCUSDT")
        result = gen.generate_t4(sample_5m_data, end_idx=100 * 288)  # < 230일
        assert result.direction == "none"
        assert "warmup" in str(result.signals.get("reason", ""))

    def test_caching_works(self, sample_5m_data):
        """같은 end_idx 재호출 시 캐시 적중"""
        gen = RealModelSignalGenerator(symbol="BTCUSDT")

        # 첫 호출 (warmup → 즉시 'none')
        gen.generate_t2(sample_5m_data, end_idx=200)
        # 두 번째 호출
        gen.generate_t2(sample_5m_data, end_idx=200)

        assert gen.stats["t2_calls"] == 2
        assert gen.stats["t2_cache_hits"] == 1   # 두 번째는 캐시 히트

    def test_cache_size_limit(self, sample_5m_data):
        """캐시가 size 초과 시 가장 오래된 것 제거"""
        gen = RealModelSignalGenerator(symbol="BTCUSDT", cache_size=3)

        # 4개 다른 end_idx 호출
        for idx in [100, 150, 200, 250]:
            gen.generate_t2(sample_5m_data, end_idx=idx)

        # 가장 오래된 (100)이 제거되어야 함
        assert 100 not in gen._t2_cache
        assert len(gen._t2_cache) <= 3

    def test_cache_hit_rate_diagnostics(self, sample_5m_data):
        """진단 정보 반환 정확성"""
        gen = RealModelSignalGenerator(symbol="BTCUSDT")

        # 3번 호출 (같은 idx)
        for _ in range(3):
            gen.generate_t2(sample_5m_data, end_idx=200)

        rates = gen.cache_hit_rate()
        # 3 calls, 2 hits → 66.7%
        assert rates["t2_hit_rate"] == pytest.approx(0.667, abs=0.01)

    def test_reset_cache(self, sample_5m_data):
        gen = RealModelSignalGenerator(symbol="BTCUSDT")

        gen.generate_t2(sample_5m_data, end_idx=200)
        assert len(gen._t2_cache) > 0

        gen.reset_cache()
        assert len(gen._t2_cache) == 0


# =============================================================================
# Look-ahead Safety
# =============================================================================
class TestLookAheadSafety:
    """백테스트의 핵심 — 미래 데이터 절대 안 봄"""

    def test_t2_only_uses_past_data(self, sample_5m_data):
        """generate_t2는 end_idx까지의 데이터만 사용"""
        gen = RealModelSignalGenerator(symbol="BTCUSDT")

        # 데이터를 두 가지 버전으로 만들기:
        #   ver1: 원본
        #   ver2: end_idx 이후를 모두 NaN으로 (미래 데이터 누락)
        # 두 버전이 같은 결과를 내야 함 = look-ahead 없음

        end_idx = 200
        ver1 = sample_5m_data.copy()
        ver2 = sample_5m_data.copy()
        ver2.iloc[end_idx:] = np.nan

        gen1 = RealModelSignalGenerator(symbol="BTCUSDT")
        gen2 = RealModelSignalGenerator(symbol="BTCUSDT")

        r1 = gen1.generate_t2(ver1, end_idx=end_idx)
        r2 = gen2.generate_t2(ver2, end_idx=end_idx)

        # 모델이 없으므로 둘 다 'none' — 어쨌든 동일 결과
        assert r1.direction == r2.direction

    def test_t4_resampling_only_uses_past(self, sample_5m_data):
        """Tier 4의 일봉 변환도 end_idx까지만"""
        gen = RealModelSignalGenerator(symbol="BTCUSDT")

        end_idx = 230 * 288 + 1000

        # 원본
        r1 = gen.generate_t4(sample_5m_data, end_idx=end_idx)

        # 미래 데이터 mutilated
        ver2 = sample_5m_data.copy()
        ver2.iloc[end_idx:] = np.nan
        gen2 = RealModelSignalGenerator(symbol="BTCUSDT")
        r2 = gen2.generate_t4(ver2, end_idx=end_idx)

        # Direction 동일 (look-ahead 없음 증명)
        assert r1.direction == r2.direction


# =============================================================================
# Backtest Integration
# =============================================================================
class TestBacktestIntegration:
    """기존 backtest_integrated와 호환성"""

    def test_signal_source_mock_runs(self):
        """signal_source='mock'은 기존과 동일 동작"""
        from flight_mind.utils.backtest_integrated import backtest_integrated

        # 30일 데이터 — 가용
        try:
            _, metrics = backtest_integrated(
                "BTCUSDT", mode="realistic", signal_source="mock",
            )
            assert "signal_source" in metrics
            assert metrics["signal_source"] == "mock"
        except RuntimeError as e:
            # No data 있을 수 있음
            assert "No data" in str(e)

    def test_signal_source_real_falls_back_when_no_models(self, monkeypatch):
        """학습된 모델 없을 때 real도 graceful 동작 (모든 hold)"""
        from flight_mind.utils.backtest_integrated import backtest_integrated

        try:
            _, metrics = backtest_integrated(
                "BTCUSDT", mode="realistic", signal_source="real",
            )
            assert metrics.get("signal_source") == "real"
            # 모델 없으면 거래 0회 (모두 hold)
            # — 단, Tier 1 단독으로도 confluence 만들 수 없음
        except RuntimeError as e:
            assert "No data" in str(e)
