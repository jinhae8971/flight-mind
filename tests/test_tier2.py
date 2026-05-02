"""
Tier 2 — Unit Tests
====================
GAF encoder, model, inference 동작 검증.
실제 학습은 시간이 오래 걸리므로 smoke test만 수행.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from flight_mind.tier2_pattern.encoder import GAFEncoder
from flight_mind.tier2_pattern.model import FlightPatternCNN


@pytest.fixture
def sample_window() -> pd.DataFrame:
    """60봉 샘플 OHLCV — GAF 입력 길이와 일치"""
    rng = np.random.default_rng(42)
    n = 60
    base = np.linspace(60000, 62000, n) + rng.normal(0, 100, n)
    return pd.DataFrame({
        "open": base * 0.9995,
        "high": base * 1.002,
        "low": base * 0.998,
        "close": base,
        "volume": rng.uniform(800, 1200, n),
    })


# =============================================================================
# GAFEncoder Tests
# =============================================================================
class TestGAFEncoder:
    def test_encode_shape(self, sample_window):
        encoder = GAFEncoder(image_size=60)
        img = encoder.encode(sample_window)
        assert img.shape == (3, 60, 60)
        assert img.dtype == np.float32

    def test_encode_values_in_range(self, sample_window):
        """GAF의 cosine 변환 결과는 [-1, 1] 범위"""
        encoder = GAFEncoder(image_size=60)
        img = encoder.encode(sample_window)
        assert img.min() >= -1.001
        assert img.max() <= 1.001

    def test_encode_wrong_length_raises(self, sample_window):
        encoder = GAFEncoder(image_size=64)  # default 64
        # sample_window는 길이 60, encoder는 64 기대 → 에러
        with pytest.raises(ValueError):
            encoder.encode(sample_window)

    def test_encode_batch(self, sample_window):
        encoder = GAFEncoder(image_size=60)
        batch = encoder.encode_batch([sample_window, sample_window, sample_window])
        assert batch.shape == (3, 3, 60, 60)


# =============================================================================
# Model Tests
# =============================================================================
class TestFlightPatternCNN:
    def test_model_forward_pass(self):
        """3-class 분류, 입력 (1, 3, 64, 64) → 출력 (1, 3)"""
        model = FlightPatternCNN(n_classes=3, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (2, 3)
        # softmax가 정상 동작하는지 확인
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_freeze_unfreeze(self):
        model = FlightPatternCNN(n_classes=3, pretrained=False)
        full_params = model.n_trainable_params()

        model.freeze_backbone()
        frozen_params = model.n_trainable_params()
        assert frozen_params < full_params

        model.unfreeze_backbone()
        assert model.n_trainable_params() == full_params

    def test_param_count_reasonable(self):
        """ResNet-18 + head: 11~12M 정도여야 함 (RTX 3090/4090에서 batch 64)"""
        model = FlightPatternCNN(n_classes=3, pretrained=False)
        n = model.n_trainable_params()
        # 11M ~ 12M 범위
        assert 10_000_000 < n < 13_000_000


# =============================================================================
# End-to-End Smoke Test
# =============================================================================
class TestEndToEnd:
    def test_encode_then_model_forward(self, sample_window):
        """GAF 인코딩 → 모델 추론 흐름"""
        encoder = GAFEncoder(image_size=60)
        # 60봉 input → image_size 60 사용
        img = encoder.encode(sample_window)   # (3, 60, 60)

        model = FlightPatternCNN(n_classes=3, pretrained=False)
        model.eval()

        x = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 60, 60)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (1, 3)
        probs = torch.softmax(logits, dim=1)
        assert (probs >= 0).all() and (probs <= 1).all()
