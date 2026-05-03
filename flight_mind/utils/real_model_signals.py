"""
Real Model Backtest Adapter
=============================
학습된 Tier 2/4 모델을 백테스트에서 호출하기 위한 어댑터.

핵심 도전:
  - 백테스트는 수만 개의 시점에서 추론 호출 → GPU 효율 + 캐싱 필수
  - Look-ahead 방지 — t시점 추론 시 t+1 이후 데이터 절대 사용 금지
  - Tier 4(일봉)는 200일 warmup 필요 — 백테스트 시작점 제한

설계 원칙:
  - LRU 캐시 (메모리 효율 + 무한 증가 방지)
  - Lazy 로드 (모델 파일 없으면 graceful fallback)
  - 일봉 변환은 윈도우당 한 번만 (Tier 4 호출 시점)
  - GPU 배치 추론은 v2에서 추가 (현재는 단일 추론)

Mock generator와 동일한 인터페이스를 제공하여
backtest_integrated가 변경 없이 실제 모델로 전환 가능.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from flight_mind.config import MODEL_DIR, TIER2, TIER4
from flight_mind.tier1_rule.engine import TierOutput


@dataclass
class ModelStatus:
    """모델 가용성 진단"""
    tier2_available: bool = False
    tier2_path: Path | None = None
    tier2_test_acc: float | None = None    # 학습 시 기록된 메트릭

    tier4_available: bool = False
    tier4_path: Path | None = None
    tier4_test_acc: float | None = None

    fallback_to_mock: bool = True


def check_model_status() -> ModelStatus:
    """학습된 모델 가용성 확인 — 영길님 PC 학습 직후 호출."""
    status = ModelStatus()

    t2_path = MODEL_DIR / "tier2_pattern_cnn.pt"
    if t2_path.exists():
        try:
            import torch
            ckpt = torch.load(t2_path, map_location="cpu", weights_only=False)
            status.tier2_available = True
            status.tier2_path = t2_path
            status.tier2_test_acc = ckpt.get("val_acc")
        except Exception:
            pass

    t4_path = MODEL_DIR / "tier4_regime_transformer.pt"
    if t4_path.exists():
        try:
            import torch
            ckpt = torch.load(t4_path, map_location="cpu", weights_only=False)
            status.tier4_available = True
            status.tier4_path = t4_path
            status.tier4_test_acc = ckpt.get("val_acc")
        except Exception:
            pass

    status.fallback_to_mock = not (status.tier2_available and status.tier4_available)
    return status


# =============================================================================
# Real Model Adapter
# =============================================================================
class RealModelSignalGenerator:
    """
    Mock generator의 인터페이스를 그대로 따라가는 실제 모델 어댑터.

    Mock의 generate_t2(df, end_idx, ...) 시그니처와 동일하게
    백테스트 코드 변경 없이 사용 가능.
    """

    def __init__(self, symbol: str = "BTCUSDT", cache_size: int = 1024):
        self.symbol = symbol
        self.cache_size = cache_size

        self._tier2_inference = None
        self._tier4_inference = None
        self._daily_cache: dict[int, pd.DataFrame] = {}    # end_idx → df_daily

        # 캐시 (end_idx → TierOutput)
        self._t2_cache: dict[int, TierOutput] = {}
        self._t4_cache: dict[int, TierOutput] = {}

        self.stats = {
            "t2_calls": 0, "t2_cache_hits": 0,
            "t4_calls": 0, "t4_cache_hits": 0,
            "t2_failures": 0, "t4_failures": 0,
        }

    # =========================================================================
    # Lazy Loaders
    # =========================================================================
    def _get_tier2(self):
        if self._tier2_inference is None:
            from flight_mind.tier2_pattern.inference import Tier2Inference
            self._tier2_inference = Tier2Inference()
        return self._tier2_inference

    def _get_tier4(self):
        if self._tier4_inference is None:
            from flight_mind.tier4_regime.inference import Tier4Inference
            self._tier4_inference = Tier4Inference()
        return self._tier4_inference

    # =========================================================================
    # Tier 2 — 60-bar 5m window
    # =========================================================================
    def generate_t2(
        self,
        df: pd.DataFrame,
        end_idx: int,
        future_horizon_bars: int = 12,    # mock 호환용 (안 쓰임)
        threshold_pct: float = 0.5,        # mock 호환용 (안 쓰임)
    ) -> TierOutput:
        """
        Tier 2 (CNN) 추론 — t시점까지의 60봉 사용.

        Look-ahead 방지: end_idx까지의 데이터만 사용 (end_idx+1 이후 절대 안 봄).
        """
        self.stats["t2_calls"] += 1

        if end_idx in self._t2_cache:
            self.stats["t2_cache_hits"] += 1
            return self._t2_cache[end_idx]

        # 60봉 윈도우 (look-ahead 안전)
        if end_idx < TIER2.input_window:
            result = TierOutput(0.0, "none", {"reason": "warmup"})
        else:
            window = df.iloc[end_idx - TIER2.input_window:end_idx]
            try:
                tier2 = self._get_tier2()
                result = tier2.predict(window)
            except FileNotFoundError:
                self.stats["t2_failures"] += 1
                result = TierOutput(0.0, "none", {"reason": "model_not_trained"})
            except Exception as e:
                self.stats["t2_failures"] += 1
                result = TierOutput(0.0, "none", {"error": str(e)[:200]})

        # 캐시 (LRU 흉내 — size 초과 시 가장 오래된 것 제거)
        if len(self._t2_cache) >= self.cache_size:
            oldest_key = next(iter(self._t2_cache))
            del self._t2_cache[oldest_key]
        self._t2_cache[end_idx] = result

        return result

    # =========================================================================
    # Tier 4 — 30-day daily window (with 200-day warmup)
    # =========================================================================
    def generate_t4(
        self,
        df: pd.DataFrame,
        end_idx: int,
        regime_lookback_bars: int = 30 * 288,    # mock 호환용
    ) -> TierOutput:
        """
        Tier 4 (Regime) 추론 — t시점까지의 일봉 230일 사용.

        5분봉 → 일봉 변환은 비용 큼 (rolling resample). 캐시 적극 활용.
        """
        self.stats["t4_calls"] += 1

        # 캐시 키 — 일봉 단위 (5분봉 288개당 1번만 변경)
        # = end_idx를 일봉 단위로 양자화
        daily_key = end_idx // 288
        cache_key = daily_key

        if cache_key in self._t4_cache:
            self.stats["t4_cache_hits"] += 1
            return self._t4_cache[cache_key]

        # Tier 4는 200일 warmup + 30일 윈도우 = 최소 230일
        # 5분봉으로는 230 × 288 = 66,240개
        min_5m_bars = 230 * 288
        if end_idx < min_5m_bars:
            result = TierOutput(0.0, "none", {"reason": "tier4_warmup"})
            self._t4_cache[cache_key] = result
            return result

        try:
            # 5분봉 → 일봉 변환 (end_idx까지만 — look-ahead 안전)
            df_5m_to_now = df.iloc[:end_idx]
            df_daily = df_5m_to_now.resample("1D").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum",
            }).dropna()

            if len(df_daily) < 230:
                result = TierOutput(0.0, "none", {"reason": "insufficient_daily"})
            else:
                tier4 = self._get_tier4()
                result = tier4.predict(df_daily, symbol=self.symbol)

        except FileNotFoundError:
            self.stats["t4_failures"] += 1
            result = TierOutput(0.0, "none", {"reason": "model_not_trained"})
        except Exception as e:
            self.stats["t4_failures"] += 1
            result = TierOutput(0.0, "none", {"error": str(e)[:200]})

        # 캐시 (작은 size — 일봉 단위라 자주 재계산 안 됨)
        if len(self._t4_cache) >= self.cache_size:
            oldest_key = next(iter(self._t4_cache))
            del self._t4_cache[oldest_key]
        self._t4_cache[cache_key] = result

        return result

    # =========================================================================
    # Diagnostics
    # =========================================================================
    def cache_hit_rate(self) -> dict[str, float]:
        t2_total = self.stats["t2_calls"]
        t4_total = self.stats["t4_calls"]
        return {
            "t2_hit_rate": (self.stats["t2_cache_hits"] / max(t2_total, 1)),
            "t4_hit_rate": (self.stats["t4_cache_hits"] / max(t4_total, 1)),
            "t2_failure_rate": (self.stats["t2_failures"] / max(t2_total, 1)),
            "t4_failure_rate": (self.stats["t4_failures"] / max(t4_total, 1)),
        }

    def reset_cache(self) -> None:
        self._t2_cache.clear()
        self._t4_cache.clear()


# =============================================================================
# Factory
# =============================================================================
def get_signal_generator(
    mode: Literal["mock", "real", "auto"] = "auto",
    symbol: str = "BTCUSDT",
    mock_mode: str = "realistic",
    seed: int = 42,
):
    """
    백테스트용 시그널 generator factory.

    mode:
      "real" : 항상 실제 모델 (없으면 에러)
      "mock" : 항상 mock (학습 검증 전)
      "auto" : 모델 있으면 real, 없으면 mock fallback
    """
    if mode == "mock":
        from flight_mind.utils.mock_signals import get_mock_generator
        return get_mock_generator(mock_mode, seed=seed)

    if mode == "real":
        return RealModelSignalGenerator(symbol=symbol)

    # auto
    status = check_model_status()
    if status.fallback_to_mock:
        from flight_mind.utils.mock_signals import get_mock_generator
        from rich.console import Console
        Console().print(
            "[yellow]Real model not available, using mock signals[/yellow]"
        )
        return get_mock_generator(mock_mode, seed=seed)

    return RealModelSignalGenerator(symbol=symbol)
