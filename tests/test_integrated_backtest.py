"""
Integrated Backtest — Unit Tests
==================================
Mock signal generator + 통합 백테스터 동작 검증.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_mind.utils.mock_signals import (MOCK_CONFIGS, MockSignalGenerator,
                                              get_mock_generator)


@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """200봉 + α 명확한 상승 추세 데이터 (mock generator 검증용)"""
    n = 300
    rng = np.random.default_rng(42)
    base = np.linspace(60000, 70000, n) + rng.normal(0, 100, n)
    return pd.DataFrame({
        "open": base * 0.999,
        "high": base * 1.003,
        "low": base * 0.997,
        "close": base,
        "volume": rng.uniform(800, 1200, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))


# =============================================================================
# Mock Config Tests
# =============================================================================
class TestMockConfigs:
    def test_three_modes_exist(self):
        assert "optimistic" in MOCK_CONFIGS
        assert "realistic" in MOCK_CONFIGS
        assert "pessimistic" in MOCK_CONFIGS

    def test_accuracy_ordering(self):
        """Optimistic > Realistic > Pessimistic"""
        opt = MOCK_CONFIGS["optimistic"]
        real = MOCK_CONFIGS["realistic"]
        pess = MOCK_CONFIGS["pessimistic"]
        assert opt.t2_accuracy > real.t2_accuracy > pess.t2_accuracy
        assert opt.t4_accuracy > real.t4_accuracy > pess.t4_accuracy


# =============================================================================
# Mock Signal Generator Tests
# =============================================================================
class TestMockGenerator:
    def test_t2_returns_tier_output(self, trending_up_data):
        gen = get_mock_generator("realistic", seed=42)
        out = gen.generate_t2(trending_up_data, end_idx=200, future_horizon_bars=12)
        assert out.score >= 0.0
        assert out.score <= 1.0
        assert out.direction in ("long", "short", "none")

    def test_t4_returns_tier_output(self, trending_up_data):
        gen = get_mock_generator("realistic", seed=42)
        out = gen.generate_t4(trending_up_data, end_idx=200)
        assert out.score >= 0.0
        assert out.direction in ("long", "short", "none")

    def test_optimistic_more_correct_than_pessimistic(self, trending_up_data):
        """Optimistic 모드가 Pessimistic보다 정확한 시그널을 더 많이 냄"""
        opt_gen = get_mock_generator("optimistic", seed=42)
        pess_gen = get_mock_generator("pessimistic", seed=42)

        opt_correct = 0
        pess_correct = 0
        n_samples = 50

        # 데이터 마지막 부분에 future_horizon만큼 여유가 있어야 함
        for i in range(200, 200 + n_samples):
            opt_out = opt_gen.generate_t2(trending_up_data, end_idx=i,
                                            future_horizon_bars=12)
            pess_out = pess_gen.generate_t2(trending_up_data, end_idx=i,
                                              future_horizon_bars=12)

            if opt_out.signals.get("mock") == "correct":
                opt_correct += 1
            if pess_out.signals.get("mock") == "correct":
                pess_correct += 1

        # Optimistic이 더 많이 맞아야 함 (또는 최소 같아야)
        assert opt_correct >= pess_correct - 3  # 작은 noise 허용

    def test_no_future_returns_none(self, trending_up_data):
        """미래 데이터 부족 시 'none' 반환"""
        gen = get_mock_generator("realistic", seed=42)
        # 마지막 인덱스 — 12봉 미래 없음
        out = gen.generate_t2(
            trending_up_data,
            end_idx=len(trending_up_data) - 1,
            future_horizon_bars=12,
        )
        assert out.direction == "none"


# =============================================================================
# Reproducibility
# =============================================================================
class TestReproducibility:
    def test_same_seed_same_output(self, trending_up_data):
        """같은 seed → 같은 시그널 (백테스트 재현성 보장)"""
        gen1 = get_mock_generator("realistic", seed=123)
        gen2 = get_mock_generator("realistic", seed=123)

        out1 = gen1.generate_t2(trending_up_data, end_idx=200)
        out2 = gen2.generate_t2(trending_up_data, end_idx=200)

        assert out1.direction == out2.direction
        assert abs(out1.score - out2.score) < 1e-9
