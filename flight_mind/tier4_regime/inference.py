"""
Tier 4 Inference API
=====================
Tier 1, 2와 동일한 TierOutput 인터페이스 제공.

Regime → Tier 4 Score & Direction 매핑 (가장 중요한 부분):

  Bull-Trending  → score=0.9, direction='long'    (강한 long 편향)
  Bear-Trending  → score=0.9, direction='short'   (강한 short 편향)
  Range-Bound    → score=0.7, direction='none'    (진입 자체 차단)
  High-Vol-Range → score=0.9, direction='none'    (강한 진입 차단 — 위험 회피)
  Crash          → score=1.0, direction='none'    (최강 진입 차단)

Day 2 백테스트의 핵심 진단:
  "Tier 1 단독 -72%의 가장 큰 원인은 횡보장 진입"

Tier 4의 역할은 진입 방향 추천이 아니라 진입 시점 게이팅입니다.
direction='none' + score 높음 = "지금은 어떤 방향이든 진입 금지"라는 강한 신호.

Fusion Layer가 이 신호를 어떻게 활용하는가:
  - Range-Bound/High-Vol-Range/Crash 시 T4의 signed_score = 0
  - 결과: 다른 Tier가 아무리 강한 long/short를 외쳐도
    T4의 신호 부재가 Confluence를 0.85 이하로 끌어내림 → 진입 차단
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from flight_mind.config import EXCHANGE, MODEL_DIR, TIER4
from flight_mind.tier1_rule.engine import TierOutput
from flight_mind.tier4_regime.dataset import RegimeDataset, SYMBOL_TO_IDX
from flight_mind.tier4_regime.labeler import (IDX_TO_REGIME, REGIMES,
                                                build_regime_features)
from flight_mind.tier4_regime.model import RegimeTransformer


# Regime → (score, direction) 매핑
REGIME_TO_OUTPUT = {
    "Bull-Trending":  (0.90, "long"),
    "Bear-Trending":  (0.90, "short"),
    "Range-Bound":    (0.70, "none"),    # 진입 자제
    "High-Vol-Range": (0.90, "none"),    # 강한 진입 차단
    "Crash":          (1.00, "none"),    # 최강 차단
}


class Tier4Inference:
    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str | None = None,
    ):
        self.model_path = (Path(model_path) if model_path
                           else (MODEL_DIR / "tier4_regime_transformer.pt"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: RegimeTransformer | None = None

    def _load_model(self) -> RegimeTransformer:
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run `python -m flight_mind.tier4_regime.train` first."
            )

        model = RegimeTransformer(
            n_features=10,
            seq_len=TIER4.lookback_days,
            n_classes=5,
            n_pairs=len(EXCHANGE.pairs),
            d_model=TIER4.d_model,
            n_heads=TIER4.n_heads,
            n_layers=TIER4.n_layers,
        )
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()

        self._model = model
        return model

    def predict_regime(
        self,
        df_daily: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> dict:
        """
        Args:
            df_daily: 일봉 OHLCV (최소 230일 — warmup 200 + window 30)
            symbol: 'BTCUSDT' or 'ETHUSDT'

        Returns:
            {
                'regime': str (e.g. 'Bull-Trending'),
                'probs': dict[str, float],
                'top_alt': str,
            }
        """
        features = build_regime_features(df_daily).ffill().fillna(0)

        if len(features) < TIER4.lookback_days:
            raise ValueError(
                f"Need at least {TIER4.lookback_days} daily bars, got {len(features)}"
            )

        window = features.iloc[-TIER4.lookback_days:][RegimeDataset.FEATURE_COLS]
        x = torch.from_numpy(window.values.astype(np.float32)).unsqueeze(0).to(self.device)

        symbol_idx = SYMBOL_TO_IDX.get(symbol, 0)
        sym = torch.tensor([symbol_idx], dtype=torch.long).to(self.device)

        model = self._load_model()
        with torch.no_grad():
            logits = model(x, sym)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        regime_idx = int(probs.argmax())
        regime_name = IDX_TO_REGIME[regime_idx]

        # Top 2
        top2 = np.argsort(probs)[-2:][::-1]

        return {
            "regime": regime_name,
            "probs": {REGIMES[i]: float(probs[i]) for i in range(5)},
            "top_alt": IDX_TO_REGIME[int(top2[1])] if len(top2) > 1 else None,
            "confidence": float(probs[regime_idx]),
        }

    def predict(
        self,
        df_daily: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> TierOutput:
        """Tier 4 시그널을 TierOutput 형태로 반환."""
        try:
            result = self.predict_regime(df_daily, symbol)
        except (FileNotFoundError, ValueError) as e:
            return TierOutput(0.0, "none", {"reason": str(e)})

        regime = result["regime"]
        confidence = result["confidence"]

        base_score, direction = REGIME_TO_OUTPUT[regime]

        # 모델 confidence를 score에 반영 (uncertain 한 예측은 약화)
        adjusted_score = base_score * confidence

        return TierOutput(
            score=float(adjusted_score),
            direction=direction,
            signals={
                "regime": regime,
                "confidence": confidence,
                "probs": result["probs"],
                "top_alt": result["top_alt"],
                "interpretation": _interpret(regime, direction),
            },
        )

    def is_ready(self) -> bool:
        return self.model_path.exists()


def _interpret(regime: str, direction: str) -> str:
    if direction == "none":
        if regime == "Crash":
            return "Crash detected — block all entries, consider exit if positioned"
        if regime == "High-Vol-Range":
            return "High volatility chop — entries dangerous, wait for trend"
        if regime == "Range-Bound":
            return "Range-bound market — Tier 1 signals likely false, be cautious"
    if direction == "long":
        return "Bull trend confirmed — long bias favored"
    if direction == "short":
        return "Bear trend confirmed — short bias favored"
    return "Unknown"


# Singleton
_inference_instance: Tier4Inference | None = None


def get_tier4_inference() -> Tier4Inference:
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = Tier4Inference()
    return _inference_instance


def predict_tier4(df_daily: pd.DataFrame, symbol: str = "BTCUSDT") -> TierOutput:
    return get_tier4_inference().predict(df_daily, symbol)
