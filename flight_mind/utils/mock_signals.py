"""
Mock Tier Signal Generator
===========================
학습된 Tier 2/4 모델 없이 통합 백테스트를 위한 모의 시그널 생성.

목적:
  영길님 PC에서 실제 학습 모델로 백테스트하기 전,
  3-Tier 통합이 Tier 1 단독 대비 얼마나 개선되는지 시뮬레이션으로 측정.

3가지 모드:
  - optimistic: 학계 SOTA 가정 (T2: 85% acc, T4: 75% acc)
  - realistic : 중간 가정     (T2: 70%, T4: 65%)
  - pessimistic: 낮은 가정    (T2: 55%, T4: 50%)

생성 원리 — "Oracle + Noise":
  1. 미래 가격 변화 (truth)을 알고 있다고 가정
  2. 정확도 P에 비례해서 truth와 일치하는 시그널 생성
  3. 1-P 확률로 잘못된 시그널 또는 'none' 반환

이 방식으로 모델의 정확도가 백테스트 성능에 어떻게 매핑되는지 명확히 보임.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from flight_mind.tier1_rule.engine import TierOutput


Mode = Literal["optimistic", "realistic", "pessimistic"]


@dataclass
class MockConfig:
    """Mock 시그널 정확도 + 호출 빈도 설정"""
    name: str
    t2_accuracy: float          # 캔들 패턴 CNN 정확도
    t2_signal_rate: float       # T2가 'none' 아닌 신호 내는 빈도
    t4_accuracy: float          # 시장 국면 Transformer 정확도
    t4_block_rate: float        # T4가 'none' (Range/HighVol/Crash) 반환 빈도


MOCK_CONFIGS = {
    "optimistic": MockConfig(
        name="Optimistic (학계 SOTA 수준)",
        t2_accuracy=0.85,
        t2_signal_rate=0.30,    # 30% 시점에서 명확한 long/short 신호
        t4_accuracy=0.75,
        t4_block_rate=0.50,     # 50% 시점에서 진입 차단 (실제 시장 ~50% 횡보)
    ),
    "realistic": MockConfig(
        name="Realistic (중간 기댓값)",
        t2_accuracy=0.70,
        t2_signal_rate=0.30,
        t4_accuracy=0.65,
        t4_block_rate=0.50,
    ),
    "pessimistic": MockConfig(
        name="Pessimistic (하한선)",
        t2_accuracy=0.55,
        t2_signal_rate=0.30,
        t4_accuracy=0.50,
        t4_block_rate=0.50,
    ),
}


class MockSignalGenerator:
    """
    미래 가격을 미리 보고(oracle) 일정 정확도로 시그널 생성.

    중요: 라이브에서는 사용 불가 — 백테스트 시뮬레이션 전용.
    """

    def __init__(self, config: MockConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate_t2(
        self,
        df: pd.DataFrame,
        end_idx: int,
        future_horizon_bars: int = 12,
        threshold_pct: float = 0.5,
    ) -> TierOutput:
        """
        Tier 2 (CNN 패턴) mock — 12봉 후 가격 방향 예측.

        Logic:
          1. truth = sign(future_close - current_close)
          2. signal_rate 확률로 시그널 발생
          3. 시그널 발생 시 t2_accuracy 확률로 truth 일치, 1-acc로 반대
        """
        if end_idx + future_horizon_bars >= len(df):
            return TierOutput(0.0, "none", {"reason": "no_future"})

        # Truth: 미래 12봉 후 수익률
        current = df["close"].iloc[end_idx]
        future = df["close"].iloc[end_idx + future_horizon_bars]
        future_return_pct = (future - current) / current * 100

        if future_return_pct >= threshold_pct:
            truth = "long"
        elif future_return_pct <= -threshold_pct:
            truth = "short"
        else:
            truth = "none"   # 임계값 미만 변화 = 'hold' truth

        # Signal rate gating
        if self.rng.random() > self.config.t2_signal_rate:
            return TierOutput(0.0, "none", {"mock": "no_signal"})

        # Truth가 'none'이면 신호 안 내는 게 맞음
        if truth == "none":
            return TierOutput(0.0, "none", {"mock": "truth_is_none"})

        # 정확도 적용
        if self.rng.random() < self.config.t2_accuracy:
            # Correct prediction
            score = self.rng.uniform(0.7, 0.95)
            return TierOutput(
                score=score,
                direction=truth,
                signals={"mock": "correct", "truth": truth},
            )
        else:
            # Wrong prediction (반대 방향)
            wrong = "short" if truth == "long" else "long"
            score = self.rng.uniform(0.7, 0.95)
            return TierOutput(
                score=score,
                direction=wrong,
                signals={"mock": "wrong", "truth": truth},
            )

    def generate_t4(
        self,
        df: pd.DataFrame,
        end_idx: int,
        regime_lookback_bars: int = 30 * 288,    # 30일치 5분봉
    ) -> TierOutput:
        """
        Tier 4 (시장 국면) mock — 미래 변동성/추세를 보고 regime 결정.

        Logic:
          1. 직전 30일의 추세/변동성으로 truth_regime 결정
          2. accuracy 확률로 truth 일치, 그렇지 않으면 random regime
          3. Range/HighVol/Crash → 'none' (진입 차단), Bull → 'long', Bear → 'short'
        """
        lookback_start = max(0, end_idx - regime_lookback_bars)
        if end_idx - lookback_start < 100:
            return TierOutput(0.0, "none", {"reason": "warmup"})

        # 직전 30일 (또는 가능한 만큼)의 수익률
        window = df["close"].iloc[lookback_start:end_idx]
        period_return_pct = (window.iloc[-1] - window.iloc[0]) / window.iloc[0] * 100

        # 최근 변동성 (annualized 근사)
        log_ret = np.log(window / window.shift()).dropna()
        if len(log_ret) > 1:
            vol_annualized = log_ret.std() * np.sqrt(252 * 288) * 100
        else:
            vol_annualized = 50.0

        # Truth regime
        if period_return_pct > 5 and vol_annualized < 100:
            truth = "Bull-Trending"
        elif period_return_pct < -5 and vol_annualized < 100:
            truth = "Bear-Trending"
        elif vol_annualized > 80:
            truth = "High-Vol-Range"
        else:
            truth = "Range-Bound"

        # Accuracy 적용
        if self.rng.random() < self.config.t4_accuracy:
            predicted = truth
        else:
            # 다른 regime 무작위
            others = [r for r in ["Bull-Trending", "Bear-Trending",
                                  "Range-Bound", "High-Vol-Range"] if r != truth]
            predicted = self.rng.choice(others)

        # Regime → TierOutput 매핑 (inference.py와 동일)
        regime_map = {
            "Bull-Trending":  (0.90, "long"),
            "Bear-Trending":  (0.90, "short"),
            "Range-Bound":    (0.70, "none"),
            "High-Vol-Range": (0.90, "none"),
        }
        score, direction = regime_map[predicted]

        return TierOutput(
            score=score,
            direction=direction,
            signals={
                "mock": "ok" if predicted == truth else "wrong",
                "predicted_regime": predicted,
                "truth_regime": truth,
            },
        )


def get_mock_generator(mode: Mode = "realistic", seed: int = 42) -> MockSignalGenerator:
    """편의 함수"""
    if mode not in MOCK_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}")
    return MockSignalGenerator(MOCK_CONFIGS[mode], seed=seed)
