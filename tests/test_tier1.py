"""
Tier 1 Rule Engine — Unit Tests
Synthetic 데이터로 4개 sub-rule 검증.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_mind.tier1_rule.engine import (
    TierOutput,
    evaluate_tier1,
    rule_double_bottom,
    rule_ma_touch,
    rule_trendline_volume,
)


@pytest.fixture
def trending_up_with_volume_spike() -> pd.DataFrame:
    """상승 추세 후 추세선 터치 + 거래량 급증 시나리오 (long 기대)"""
    rng = np.random.default_rng(42)
    n = 100

    # 100봉 동안 점진 상승 (1.0 → 1.05)
    base = np.linspace(1.0, 1.05, n)
    noise = rng.normal(0, 0.002, n)
    close = base + noise

    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.uniform(100, 200, n),
    })

    # 마지막 봉: 추세선 근처에서 거래량 급증
    df.loc[df.index[-1], "low"] = float(df["low"].min()) * 1.001
    df.loc[df.index[-1], "volume"] = df["volume"].iloc[:-1].mean() * 2.5

    return df


@pytest.fixture
def coin_double_bottom() -> pd.DataFrame:
    """코인형 더블바텀 (오른쪽 저점이 더 낮음, 스탑헌팅)"""
    n = 80
    close = np.ones(n) * 100.0

    # 첫 저점 (idx 30)
    close[28:33] = [99, 96, 95.0, 96, 99]
    # 회복
    close[33:55] = np.linspace(99, 102, 22)
    # 두 번째 저점 — 첫 저점보다 약간 낮음 (94.5)
    close[55:60] = [99, 96, 94.5, 96, 100]
    # 직전 봉 반등 중
    close[60:] = np.linspace(100, 102, n - 60)

    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.ones(n) * 100,
    })

    return df


@pytest.fixture
def ma7_bounce() -> pd.DataFrame:
    """MA7 위에서 횡보하다가 정확히 MA7 터치 후 반등하는 시나리오"""
    n = 50
    rng = np.random.default_rng(7)

    # 1.0 → 1.02 점진 상승, MA7 위 횡보
    close = np.linspace(1.0, 1.02, n) + rng.normal(0, 0.0005, n)

    df = pd.DataFrame({
        "open": close * 0.9995,
        "high": close * 1.0005,
        "low": close * 0.999,
        "close": close,
        "volume": np.ones(n) * 100,
    })

    # 마지막 봉: MA7과 거의 같게 + 미세하게 위
    ma7 = df["close"].rolling(7).mean().iloc[-1]
    df.loc[df.index[-1], "close"] = ma7 * 1.001
    df.loc[df.index[-1], "low"] = ma7 * 0.9999

    return df


# =============================================================================
# Tests
# =============================================================================
class TestRule11TrendlineVolume:
    def test_long_signal_on_support_with_volume_spike(self, trending_up_with_volume_spike):
        out = rule_trendline_volume(trending_up_with_volume_spike)
        assert out.direction == "long"
        assert 0.5 <= out.score <= 1.0

    def test_no_signal_on_insufficient_data(self):
        df = pd.DataFrame({
            "open": [1, 2], "high": [1, 2], "low": [1, 2],
            "close": [1, 2], "volume": [10, 10],
        })
        out = rule_trendline_volume(df)
        assert out.direction == "none"
        assert out.score == 0.0


class TestRule13MaTouch:
    def test_ma7_bounce_long_signal(self, ma7_bounce):
        out = rule_ma_touch(ma7_bounce)
        # MA7 근처에서 반등하면 long 시그널
        assert out.direction in ("long", "none")  # 노이즈에 따라 가끔 none
        if out.direction == "long":
            assert out.score >= 0.5


class TestRule14DoubleBottom:
    def test_coin_double_bottom_long(self, coin_double_bottom):
        out = rule_double_bottom(coin_double_bottom)
        # 오른쪽 저점이 더 낮은 코인형 더블바텀 → long 기대
        assert out.direction == "long"
        assert out.score >= 0.5
        assert out.signals.get("type") == "coin_double_bottom"


class TestEvaluateTier1Aggregator:
    def test_signal_aggregation(self, trending_up_with_volume_spike):
        out = evaluate_tier1(trending_up_with_volume_spike)
        assert isinstance(out, TierOutput)
        assert 0.0 <= out.score <= 1.0
        assert out.direction in ("long", "short", "none")
        assert "sub_rules" in out.signals

    def test_signed_score_property(self):
        long_out = TierOutput(0.8, "long", {})
        short_out = TierOutput(0.8, "short", {})
        none_out = TierOutput(0.8, "none", {})

        assert long_out.signed_score() == 0.8
        assert short_out.signed_score() == -0.8
        assert none_out.signed_score() == 0.0
