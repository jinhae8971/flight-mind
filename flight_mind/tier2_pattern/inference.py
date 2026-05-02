"""
Tier 2 Inference API
=====================
Tier 1 룰 엔진과 동일한 인터페이스 (TierOutput 반환)를 가진 추론 함수.

사용법:
    from flight_mind.tier2_pattern.inference import Tier2Inference

    tier2 = Tier2Inference()
    output = tier2.predict(df_60bars)   # TierOutput
    print(output.score, output.direction)

설계:
  - 모델은 lazy load (첫 호출 시 weight 파일 로딩)
  - GPU/CPU 자동 감지
  - 추론 시 dropout 비활성화 (eval mode)
  - softmax 확률 → score 매핑:
      score = max(P_long, P_short) - P_hold  (양수일수록 confident, [-1, 1])
      direction = argmax(P_long, P_short)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from flight_mind.config import MODEL_DIR, TIER2
from flight_mind.tier1_rule.engine import TierOutput
from flight_mind.tier2_pattern.encoder import GAFEncoder
from flight_mind.tier2_pattern.model import FlightPatternCNN


class Tier2Inference:
    """Lazy-loaded Tier 2 inference."""

    def __init__(self, model_path: Path | str | None = None, device: str | None = None):
        self.model_path = Path(model_path) if model_path else (MODEL_DIR / "tier2_pattern_cnn.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: FlightPatternCNN | None = None
        self.encoder = GAFEncoder(image_size=TIER2.image_size)

    def _load_model(self) -> FlightPatternCNN:
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Run `python -m flight_mind.tier2_pattern.train` first."
            )

        model = FlightPatternCNN(n_classes=3, pretrained=False)
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()

        self._model = model
        return model

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """입력 60봉 → (3,) 확률 분포 [P_long, P_short, P_hold]."""
        if len(df) != TIER2.input_window:
            # Take last input_window bars
            df = df.iloc[-TIER2.input_window:]
            if len(df) != TIER2.input_window:
                raise ValueError(
                    f"Need {TIER2.input_window} bars, got {len(df)}"
                )

        model = self._load_model()
        gaf_img = self.encoder.encode(df)         # (3, H, W)
        x = torch.from_numpy(gaf_img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return probs    # [p_long, p_short, p_hold]

    def predict(self, df: pd.DataFrame) -> TierOutput:
        """
        Tier 2 시그널을 TierOutput 형태로 반환.

        Score mapping:
          - long_strength  = P_long - P_hold
          - short_strength = P_short - P_hold
          - winner = max(long_strength, short_strength)
          - score = clip(winner, 0, 1)  → [0, 1]
          - direction = 'long' / 'short' / 'none' (둘 다 < 0이면 none)
        """
        try:
            probs = self.predict_proba(df)
        except (FileNotFoundError, ValueError) as e:
            return TierOutput(0.0, "none", {"reason": str(e)})

        p_long, p_short, p_hold = probs

        long_strength = float(p_long - p_hold)
        short_strength = float(p_short - p_hold)

        # Winner-take-all
        if long_strength <= 0 and short_strength <= 0:
            return TierOutput(
                score=0.0,
                direction="none",
                signals={
                    "p_long": float(p_long),
                    "p_short": float(p_short),
                    "p_hold": float(p_hold),
                    "reason": "hold_dominates",
                },
            )

        if long_strength > short_strength:
            return TierOutput(
                score=float(np.clip(long_strength, 0.0, 1.0)),
                direction="long",
                signals={
                    "p_long": float(p_long),
                    "p_short": float(p_short),
                    "p_hold": float(p_hold),
                    "long_strength": long_strength,
                },
            )
        else:
            return TierOutput(
                score=float(np.clip(short_strength, 0.0, 1.0)),
                direction="short",
                signals={
                    "p_long": float(p_long),
                    "p_short": float(p_short),
                    "p_hold": float(p_hold),
                    "short_strength": short_strength,
                },
            )

    def is_ready(self) -> bool:
        """모델 체크포인트 존재 여부 확인."""
        return self.model_path.exists()


# Singleton (편의용)
_inference_instance: Tier2Inference | None = None


def get_tier2_inference() -> Tier2Inference:
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = Tier2Inference()
    return _inference_instance


def predict_tier2(df: pd.DataFrame) -> TierOutput:
    """편의 함수 — Tier 1과 동일한 호출 패턴."""
    return get_tier2_inference().predict(df)
