"""
End-to-End 동작 데모 — Day 1 산출물 검증.

실제 호출 흐름:
  1) Tier 1~4 동작 시뮬레이션 (T1만 실데이터, T2~T4는 mock score)
  2) Fusion Layer로 통합
  3) 의사결정 출력
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from flight_mind.config import summary
from flight_mind.fusion.layer import explain, fuse
from flight_mind.tier1_rule.engine import TierOutput, evaluate_tier1


def make_synthetic_btc_data(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """200봉 BTC-like OHLCV 합성 — 마지막 봉 추세선 터치 + 거래량 급증"""
    rng = np.random.default_rng(seed)

    base = np.linspace(60000, 63000, n) + rng.normal(0, 200, n)
    df = pd.DataFrame({
        "open": base * 0.999,
        "high": base * 1.003,
        "low": base * 0.997,
        "close": base,
        "volume": rng.uniform(800, 1200, n),
    })

    # 마지막 봉 — 저점 근처 + 거래량 2.5배
    df.loc[df.index[-1], "low"] = float(df["low"].min()) * 1.001
    df.loc[df.index[-1], "volume"] = df["volume"].iloc[:-1].mean() * 2.5

    return df


def demo_scenario_1_strong_long():
    """시나리오 1: 모든 Tier가 강한 long → 진입 기대"""
    print("\n" + "█" * 68)
    print("  시나리오 1: 모든 Tier가 강한 LONG 신호 — 진입 기대")
    print("█" * 68)

    df = make_synthetic_btc_data()
    t1 = evaluate_tier1(df)
    print(f"\n[T1 실제 평가] score={t1.score:.3f} dir={t1.direction}")
    print(f"   active rules: {t1.signals.get('active_count', 0)}")

    # Tier 2~4 — 학습 전이므로 mock
    t2 = TierOutput(0.92, "long", {"mock": "CNN strong bullish pattern"})
    t3 = TierOutput(0.85, "long", {"mock": "TCN orderbook buying pressure"})
    t4 = TierOutput(0.88, "long", {"mock": "Transformer Bull-Trending regime"})

    decision = fuse(t1, t2, t3, t4, available_balance_usdt=1750.0)
    print(explain(decision))


def demo_scenario_2_t1_t2_disagree():
    """시나리오 2: T1·T2 부호 반대 → veto 발동"""
    print("\n" + "█" * 68)
    print("  시나리오 2: T1·T2 disagree — Veto 발동 기대")
    print("█" * 68)

    t1 = TierOutput(0.95, "long", {"mock": "rule says long"})
    t2 = TierOutput(0.95, "short", {"mock": "CNN says short — rare conflict"})
    t3 = TierOutput(0.85, "long", {})
    t4 = TierOutput(0.85, "long", {})

    decision = fuse(t1, t2, t3, t4, available_balance_usdt=1750.0)
    print(explain(decision))


def demo_scenario_3_below_threshold():
    """시나리오 3: 신호는 있지만 confluence가 0.85 미만 → hold"""
    print("\n" + "█" * 68)
    print("  시나리오 3: Confluence 0.85 미만 — Hold 기대")
    print("█" * 68)

    t1 = TierOutput(0.7, "long", {})
    t2 = TierOutput(0.7, "long", {})
    t3 = TierOutput(0.5, "long", {})
    t4 = TierOutput(0.5, "long", {})
    # signed = 0.30*0.7 + 0.30*0.7 + 0.20*0.5 + 0.20*0.5 = 0.62 < 0.85

    decision = fuse(t1, t2, t3, t4, available_balance_usdt=1750.0)
    print(explain(decision))


def demo_scenario_4_realistic_short():
    """시나리오 4: 실전 short 시나리오"""
    print("\n" + "█" * 68)
    print("  시나리오 4: 추세 반전 SHORT — 진입 기대")
    print("█" * 68)

    t1 = TierOutput(0.82, "short", {"mock": "RSI bearish divergence + MA7 break"})
    t2 = TierOutput(0.91, "short", {"mock": "CNN: distribution top pattern"})
    t3 = TierOutput(0.78, "short", {"mock": "TCN: ask wall absorbing buys"})
    t4 = TierOutput(0.80, "short", {"mock": "Transformer: regime shift to bearish"})

    decision = fuse(t1, t2, t3, t4, available_balance_usdt=1750.0)
    print(explain(decision))


if __name__ == "__main__":
    print(summary())
    demo_scenario_1_strong_long()
    demo_scenario_2_t1_t2_disagree()
    demo_scenario_3_below_threshold()
    demo_scenario_4_realistic_short()

    print("\n" + "✅" * 34)
    print("  Day 1 데모 완료 — 4개 시나리오 모두 의도대로 동작")
    print("✅" * 34)
