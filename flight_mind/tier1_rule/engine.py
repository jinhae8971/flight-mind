"""
Tier 1 — Rule Engine
====================
플라이트 매매법의 명시적 룰 부분을 구현.

4 Sub-rules:
  R1.1: 추세선 터치 + 거래량 동반 (영길님 trendline-detector 활용)
  R1.2: RSI 다이버전스 (Wilder, hidden + regular)
  R1.3: MA(7/30) 터치 + 반등 패턴
  R1.4: 더블바텀 (코인형: 오른쪽 저점이 더 낮음)

각 sub-rule은 [0, 1] 점수와 방향(long/short/none)을 반환.
최종 Tier 1 점수는 mean(active sub-rules), 방향은 majority vote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from flight_mind.config import TIER1


Direction = Literal["long", "short", "none"]


@dataclass
class TierOutput:
    """모든 Tier가 공통으로 반환하는 형태"""
    score: float                          # [0, 1]
    direction: Direction
    signals: dict                         # 디버깅 / 로깅용 메타데이터

    def signed_score(self) -> float:
        """Fusion Layer에서 쓰는 부호 있는 점수 [-1, +1]"""
        if self.direction == "long":
            return self.score
        if self.direction == "short":
            return -self.score
        return 0.0


# =============================================================================
# Sub-Rule R1.1 — 추세선 + 거래량
# =============================================================================
def rule_trendline_volume(df: pd.DataFrame) -> TierOutput:
    """
    Plait 본인 발언:
      "추세선을 확인한다. 거래량이 상승하면서 추세선을 닿을 때마다
       강하게 반등해야 건전한 상승으로 본다."

    영길님의 trendline-detector를 import하여 사용 예정.
    여기서는 placeholder — Day 5에 trendline-detector 통합.
    """
    # TODO: from trendline_detector import detect_trendlines
    # trendlines = detect_trendlines(df)

    # Placeholder logic: 직전 N봉에서 저점 추세선 가정
    lookback = TIER1.trendline_lookback
    if len(df) < lookback:
        return TierOutput(0.0, "none", {"reason": "insufficient_data"})

    recent = df.tail(lookback)
    last_close = recent["close"].iloc[-1]
    last_low = recent["low"].iloc[-1]
    last_volume = recent["volume"].iloc[-1]
    avg_volume = recent["volume"].iloc[:-1].mean()

    # 단순화된 룰: 최근 저점 근처(0.5% 이내) 터치 + 거래량 1.5배 이상
    recent_low = recent["low"].min()
    near_support = (last_low - recent_low) / recent_low < 0.005
    volume_spike = last_volume > avg_volume * TIER1.volume_spike_factor

    if near_support and volume_spike:
        return TierOutput(
            score=0.8,
            direction="long",
            signals={
                "near_support": near_support,
                "volume_spike": volume_spike,
                "support_level": float(recent_low),
            },
        )

    # TODO: 저항선 터치 + 거래량 동반 → short 시그널 추가
    return TierOutput(0.0, "none", {"near_support": near_support})


# =============================================================================
# Sub-Rule R1.2 — RSI 다이버전스
# =============================================================================
def rule_rsi_divergence(df: pd.DataFrame) -> TierOutput:
    """
    Plait 본인 발언:
      "본인 매매에서의 강점은 캔들 균형성 + 다이버전스 원리를 조합하여
       어느 지점에서 매수세, 매도세가 약해지는걸 '감'으로 찾는 것"

    PyPI rsi-divergence-detector 패키지 사용.
    """
    try:
        from rsi_divergence import calculate_rsi, find_divergences
    except ImportError:
        # 패키지 미설치 시 graceful degradation
        return TierOutput(0.0, "none", {"reason": "rsi_divergence pkg not installed"})

    if len(df) < 100:
        return TierOutput(0.0, "none", {"reason": "insufficient_data"})

    close = df["close"].astype(float)
    rsi = calculate_rsi(close, period=TIER1.rsi_period)

    divs = find_divergences(
        prices=close,
        rsi=rsi,
        rsi_period=TIER1.rsi_period,
        max_lag=3,
        include_hidden=True,
    )

    if divs.empty:
        return TierOutput(0.0, "none", {"reason": "no_divergence"})

    # 최근 5봉 내 다이버전스만 채택
    last_idx = df.index[-1]
    recent_divs = divs[divs.index >= df.index[-5]] if hasattr(divs, "index") else divs

    if recent_divs.empty:
        return TierOutput(0.0, "none", {"reason": "no_recent_divergence"})

    # bullish_divergence → long, bearish_divergence → short
    last = recent_divs.iloc[-1]
    if "bullish" in str(last.get("type", "")).lower():
        return TierOutput(0.85, "long", {"divergence": "bullish"})
    if "bearish" in str(last.get("type", "")).lower():
        return TierOutput(0.85, "short", {"divergence": "bearish"})

    return TierOutput(0.0, "none", {"reason": "ambiguous_divergence"})


# =============================================================================
# Sub-Rule R1.3 — 이평선 터치 + 반등
# =============================================================================
def rule_ma_touch(df: pd.DataFrame) -> TierOutput:
    """
    Plait 본인 발언:
      "이동평균선에 접촉할 때마다 '롱' 혹은 '숏'을 치는 방법"
      "RSI 하락 다이버전스가 생기고 7일 이평선에서 횡보하며
       힘이 약한 모습이 보여줄 때 7일 이평선이 뚫리면
       그 다음 이평선인 30일 이평선까지 하락할 것으로 예상"
    """
    if len(df) < max(TIER1.ma_periods) + 5:
        return TierOutput(0.0, "none", {"reason": "insufficient_data"})

    close = df["close"]
    last_close = close.iloc[-1]

    ma7 = close.rolling(7).mean().iloc[-1]
    ma30 = close.rolling(30).mean().iloc[-1]

    # MA7 터치 + 반등 (long)
    distance_to_ma7 = (last_close - ma7) / ma7
    near_ma7 = abs(distance_to_ma7) < 0.003  # 0.3% 이내

    # 직전 5봉이 MA7 위에서 횡보하다가 터치한 경우
    prev_5 = close.iloc[-6:-1]
    prev_above = (prev_5 > ma7).all()

    if near_ma7 and prev_above and last_close > ma7:
        return TierOutput(
            score=0.7,
            direction="long",
            signals={"ma7": float(ma7), "ma30": float(ma30), "type": "ma7_bounce"},
        )

    # MA7 이탈 + MA30 향하는 흐름 (short)
    if last_close < ma7 and ma7 > ma30 and prev_above:
        return TierOutput(
            score=0.65,
            direction="short",
            signals={"ma7": float(ma7), "ma30": float(ma30), "type": "ma7_break"},
        )

    return TierOutput(0.0, "none", {"distance_to_ma7": float(distance_to_ma7)})


# =============================================================================
# Sub-Rule R1.4 — 더블바텀 (코인형)
# =============================================================================
def rule_double_bottom(df: pd.DataFrame) -> TierOutput:
    """
    Plait 본인 발언:
      "코인은 일반적으로 더블바텀 모양이 왼쪽 저점보다 오른쪽 저점이 낮다.
       이는 보통 선물 거래를 할 때 저점에 스탑로스를 걸기 때문에
       저점만 터치하고 오르는 '스탑헌팅' 무빙이 자주 출현하기 때문"
    """
    if len(df) < 60:
        return TierOutput(0.0, "none", {"reason": "insufficient_data"})

    from scipy.signal import argrelextrema

    lows = df["low"].values
    # local minima with order=5 (5봉 좌우 최저)
    min_idx = argrelextrema(lows, np.less_equal, order=5)[0]

    if len(min_idx) < 2:
        return TierOutput(0.0, "none", {"reason": "no_double_bottom"})

    # 직전 두 저점
    last_two = min_idx[-2:]
    if len(last_two) < 2:
        return TierOutput(0.0, "none", {"reason": "no_double_bottom"})

    left_low = lows[last_two[0]]
    right_low = lows[last_two[1]]

    # 코인형: 오른쪽이 더 낮음 (스탑헌팅)
    if right_low < left_low:
        # 그러나 너무 많이 낮지는 않아야 (1% 이내)
        if (left_low - right_low) / left_low < 0.01:
            # 직전 봉이 right_low에서 반등 중인지 확인
            current_close = df["close"].iloc[-1]
            if current_close > right_low * 1.002:
                return TierOutput(
                    score=0.75,
                    direction="long",
                    signals={
                        "left_low": float(left_low),
                        "right_low": float(right_low),
                        "type": "coin_double_bottom",
                    },
                )

    return TierOutput(0.0, "none", {"reason": "no_valid_pattern"})


# =============================================================================
# Tier 1 Aggregator
# =============================================================================
def evaluate_tier1(df: pd.DataFrame) -> TierOutput:
    """
    4개 sub-rule을 모두 실행하고 결과를 통합.

    - 점수: 활성화된(score > 0) 룰들의 평균
    - 방향: signed score 합계의 부호
    """
    rules = {
        "R1.1_trendline_volume": rule_trendline_volume(df),
        "R1.2_rsi_divergence": rule_rsi_divergence(df),
        "R1.3_ma_touch": rule_ma_touch(df),
        "R1.4_double_bottom": rule_double_bottom(df),
    }

    active = [r for r in rules.values() if r.direction != "none"]
    if not active:
        return TierOutput(0.0, "none", {"sub_rules": {k: v.signals for k, v in rules.items()}})

    # signed score sum
    signed_sum = sum(r.signed_score() for r in active)
    avg_magnitude = np.mean([r.score for r in active])

    direction: Direction = "long" if signed_sum > 0 else "short" if signed_sum < 0 else "none"

    # 강한 동의(같은 방향 2개 이상) 시 부스트
    same_direction_count = sum(1 for r in active if r.direction == direction)
    if same_direction_count >= 2:
        avg_magnitude = min(1.0, avg_magnitude * 1.15)

    return TierOutput(
        score=float(avg_magnitude),
        direction=direction,
        signals={
            "active_count": len(active),
            "signed_sum": float(signed_sum),
            "sub_rules": {k: {"score": v.score, "direction": v.direction, "meta": v.signals}
                          for k, v in rules.items()},
        },
    )
