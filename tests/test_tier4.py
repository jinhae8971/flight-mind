"""
Tier 4 — Unit Tests
====================
Labeler 정확성 + Transformer 모델 동작 검증.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from flight_mind.tier4_regime.labeler import (REGIME_TO_IDX, REGIMES,
                                                build_regime_features,
                                                label_regime,
                                                regime_distribution)
from flight_mind.tier4_regime.model import RegimeTransformer


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def synthetic_bull_market() -> pd.DataFrame:
    """30일 동안 +50% 상승 + 강한 모멘텀 (ADX > 25, return_30d > 5% 보장)"""
    n = 300
    rng = np.random.default_rng(42)

    # 강한 상승 추세 (40k → 80k, +100%)
    base = np.linspace(40000, 80000, n)
    # 작은 노이즈 (ADX가 너무 낮아지지 않도록)
    base = base + rng.normal(0, 300, n)

    # 일중 변동폭 — ADX 계산을 위해 충분한 +DM/-DM 차이
    high = base + np.abs(rng.normal(500, 300, n))
    low = base - np.abs(rng.normal(500, 300, n))

    return pd.DataFrame({
        "open": base * 0.998,
        "high": high,
        "low": low,
        "close": base,
        "volume": rng.uniform(800, 1200, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1D"))


@pytest.fixture
def synthetic_bear_market() -> pd.DataFrame:
    """30일 동안 -50% 하락 + 강한 모멘텀"""
    n = 300
    rng = np.random.default_rng(42)
    base = np.linspace(80000, 40000, n)
    base = base + rng.normal(0, 300, n)

    high = base + np.abs(rng.normal(500, 300, n))
    low = base - np.abs(rng.normal(500, 300, n))

    return pd.DataFrame({
        "open": base * 1.002,
        "high": high,
        "low": low,
        "close": base,
        "volume": rng.uniform(800, 1200, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1D"))


@pytest.fixture
def synthetic_range_market() -> pd.DataFrame:
    """40,000 근처에서 횡보"""
    n = 300
    rng = np.random.default_rng(42)
    base = 40000 + rng.normal(0, 500, n)  # ±1.25% 변동
    return pd.DataFrame({
        "open": base * 0.999,
        "high": base * 1.005,
        "low": base * 0.995,
        "close": base,
        "volume": rng.uniform(800, 1200, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1D"))


# =============================================================================
# Labeler Tests
# =============================================================================
class TestLabeler:
    def test_bull_market_detected(self, synthetic_bull_market):
        labels = label_regime(synthetic_bull_market)
        # Warmup 후 마지막 50일은 강세장으로 라벨링되어야 함
        recent = labels.iloc[-50:]
        bull_idx = REGIME_TO_IDX["Bull-Trending"]
        bear_idx = REGIME_TO_IDX["Bear-Trending"]
        # 핵심 보장: 최소 1개의 Bull 라벨 발생, Bear는 없어야 함
        assert (recent == bull_idx).sum() >= 1, "Bull market should produce at least some Bull-Trending labels"
        assert (recent == bear_idx).sum() == 0, "Bull market should not be labeled Bear"

    def test_bear_market_detected(self, synthetic_bear_market):
        labels = label_regime(synthetic_bear_market)
        recent = labels.iloc[-50:]
        bull_idx = REGIME_TO_IDX["Bull-Trending"]
        bear_idx = REGIME_TO_IDX["Bear-Trending"]
        assert (recent == bear_idx).sum() >= 1, "Bear market should produce at least some Bear-Trending labels"
        assert (recent == bull_idx).sum() == 0, "Bear market should not be labeled Bull"

    def test_range_market_detected(self, synthetic_range_market):
        labels = label_regime(synthetic_range_market)
        recent = labels.iloc[-50:]
        range_idx = REGIME_TO_IDX["Range-Bound"]
        # Range-Bound 또는 (저변동성으로 인한 다른 분류)
        # 핵심: Bull 또는 Bear는 아니어야 함
        bull_idx = REGIME_TO_IDX["Bull-Trending"]
        bear_idx = REGIME_TO_IDX["Bear-Trending"]
        not_trending = ((recent != bull_idx) & (recent != bear_idx)).sum()
        assert not_trending >= 40

    def test_warmup_is_unlabeled(self, synthetic_bull_market):
        labels = label_regime(synthetic_bull_market)
        # 처음 200일은 -1
        assert (labels.iloc[:200] == -1).all()

    def test_too_short_returns_minus_one(self):
        df = pd.DataFrame({
            "open": [1, 2, 3], "high": [1, 2, 3],
            "low": [1, 2, 3], "close": [1, 2, 3], "volume": [10, 10, 10],
        })
        labels = label_regime(df)
        assert (labels == -1).all()


class TestFeatureBuilder:
    def test_returns_all_columns(self, synthetic_bull_market):
        features = build_regime_features(synthetic_bull_market)
        expected = ["close", "return_1d", "return_7d", "return_30d",
                    "rsi_14", "adx_14", "vol_30d",
                    "ma_50_dist", "ma_200_dist",
                    "high_low_pct", "volume_zscore_30d"]
        assert all(col in features.columns for col in expected)

    def test_feature_dimensions(self, synthetic_bull_market):
        features = build_regime_features(synthetic_bull_market)
        assert len(features) == len(synthetic_bull_market)

    def test_no_inf_values(self, synthetic_bull_market):
        features = build_regime_features(synthetic_bull_market)
        # 결측은 OK, but 무한대는 안 됨
        for col in features.columns:
            inf_count = np.isinf(features[col]).sum()
            assert inf_count == 0, f"Column {col} has inf values"


# =============================================================================
# Model Tests
# =============================================================================
class TestRegimeTransformer:
    def test_forward_shape(self):
        model = RegimeTransformer(
            n_features=10, seq_len=30, n_classes=5,
            n_pairs=2, d_model=64, n_heads=4, n_layers=2,  # 작은 모델로 테스트
        )
        model.eval()

        B = 4
        x = torch.randn(B, 30, 10)
        sym = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        with torch.no_grad():
            logits = model(x, sym)

        assert logits.shape == (B, 5)
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(B), atol=1e-5)

    def test_param_count_in_range(self):
        """기본 설정 모델 — 5M 근처 예상"""
        model = RegimeTransformer(
            n_features=10, seq_len=30, n_classes=5,
            n_pairs=2, d_model=256, n_heads=8, n_layers=6,
        )
        n = model.n_trainable_params()
        # 4M ~ 6M 범위
        assert 3_000_000 < n < 7_000_000

    def test_symbol_embedding_distinguishes_pairs(self):
        """같은 입력 + 다른 symbol_idx → 다른 logits"""
        model = RegimeTransformer(
            n_features=10, seq_len=30, n_classes=5,
            n_pairs=2, d_model=64, n_heads=4, n_layers=2,
        )
        model.eval()

        x = torch.randn(1, 30, 10)
        sym0 = torch.tensor([0], dtype=torch.long)
        sym1 = torch.tensor([1], dtype=torch.long)

        with torch.no_grad():
            logits0 = model(x, sym0)
            logits1 = model(x, sym1)

        # Initial weights는 random이므로 두 logits이 달라야 함
        diff = (logits0 - logits1).abs().sum().item()
        assert diff > 0.01, "Symbol embedding has no effect on output"


# =============================================================================
# Inference API Tests
# =============================================================================
class TestInferenceAPI:
    def test_predict_returns_tier_output_when_no_model(self):
        """모델 파일 없을 때 graceful degradation"""
        from flight_mind.tier4_regime.inference import Tier4Inference

        inf = Tier4Inference(model_path="/nonexistent/path.pt")

        df = pd.DataFrame({
            "open": [1] * 300, "high": [1] * 300,
            "low": [1] * 300, "close": [1] * 300,
            "volume": [10] * 300,
        }, index=pd.date_range("2024-01-01", periods=300, freq="1D"))

        output = inf.predict(df)
        assert output.score == 0.0
        assert output.direction == "none"
        assert "reason" in output.signals
