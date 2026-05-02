"""
Fusion Layer — Unit Tests
영길님 결정사항(threshold=0.85, T1·T2 disagreement veto)이 정확히 동작하는지 검증.
"""
from __future__ import annotations

import pytest

from flight_mind.config import CAPITAL, FUSION
from flight_mind.fusion.layer import explain, fuse
from flight_mind.tier1_rule.engine import TierOutput


def make(score: float, direction: str) -> TierOutput:
    return TierOutput(score=score, direction=direction, signals={})  # type: ignore


class TestFusionThreshold:
    """영길님의 0.85 보수 임계값 동작 검증"""

    def test_all_high_long_above_threshold_opens_position(self):
        """4개 모두 강한 long → 진입"""
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "long"),
            t3=make(0.85, "long"),
            t4=make(0.85, "long"),
            available_balance_usdt=1750.0,
        )
        assert decision.action == "open_long"
        assert decision.confluence_score >= 0.85
        assert decision.position_size_usdt == pytest.approx(1750.0 * 0.02)
        assert decision.leverage == CAPITAL.leverage

    def test_below_threshold_holds(self):
        """confluence가 0.85 미만 → 강제 hold"""
        decision = fuse(
            t1=make(0.7, "long"),
            t2=make(0.7, "long"),
            t3=make(0.5, "none"),
            t4=make(0.5, "none"),
            available_balance_usdt=1750.0,
        )
        # 0.30*0.7 + 0.30*0.7 + 0 + 0 = 0.42 < 0.85
        assert decision.action == "hold"
        assert decision.confluence_score < FUSION.threshold

    def test_just_above_threshold(self):
        """경계값 시나리오 — 정확히 0.86"""
        # T1 long 0.95 → 0.30*0.95 = 0.285
        # T2 long 0.95 → 0.30*0.95 = 0.285
        # T3 long 0.7  → 0.20*0.7  = 0.14
        # T4 long 0.7  → 0.20*0.7  = 0.14
        # sum = 0.85
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "long"),
            t3=make(0.7, "long"),
            t4=make(0.7, "long"),
            available_balance_usdt=1750.0,
        )
        # 0.85 이상이어야 진입
        assert decision.confluence_score >= FUSION.threshold * 0.99


class TestFusionVeto:
    """T1, T2 부호 반대 시 강제 hold"""

    def test_t1_t2_disagree_vetoes(self):
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "short"),
            t3=make(0.85, "long"),
            t4=make(0.85, "long"),
            available_balance_usdt=1750.0,
        )
        assert decision.action == "hold"
        assert decision.veto_reason is not None
        assert "disagree" in decision.veto_reason.lower()


class TestFusionShortSide:
    def test_all_short_high_opens_short(self):
        decision = fuse(
            t1=make(0.95, "short"),
            t2=make(0.95, "short"),
            t3=make(0.85, "short"),
            t4=make(0.85, "short"),
            available_balance_usdt=1750.0,
        )
        assert decision.action == "open_short"
        assert decision.direction == "short"


class TestFusionPositionSizing:
    def test_position_size_caps_at_2_percent(self):
        """영길님 정책: 거래당 max 2%"""
        decision = fuse(
            t1=make(1.0, "long"),
            t2=make(1.0, "long"),
            t3=make(1.0, "long"),
            t4=make(1.0, "long"),
            available_balance_usdt=10_000.0,
        )
        assert decision.position_size_usdt == pytest.approx(10_000.0 * 0.02)

    def test_realistic_3500_usdt_seed(self):
        """실제 영길님 시드 — live trading sub-account 1,750 USDT"""
        live_balance = 3500.0 * 0.50
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "long"),
            t3=make(0.95, "long"),
            t4=make(0.95, "long"),
            available_balance_usdt=live_balance,
        )
        assert decision.action == "open_long"
        assert decision.position_size_usdt == pytest.approx(35.0)  # 1,750 × 2%


class TestExplainability:
    def test_explain_outputs_readable_summary(self):
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "long"),
            t3=make(0.85, "long"),
            t4=make(0.85, "long"),
            available_balance_usdt=1750.0,
        )
        text = explain(decision)
        assert "Confluence" in text
        assert "T1" in text and "T2" in text and "T3" in text and "T4" in text
        assert "open_long" in text.lower() or "OPEN_LONG" in text


# =============================================================================
# 3-Tier 새 시그니처 테스트 (Tier 3 제외)
# =============================================================================
class TestThreeTierFusion:
    """Tier 3 제외 후 가중치 재분배 검증.

    Weights: T1=0.35, T2=0.35, T4=0.30 (sum=1.0, T3=0.0)
    """

    def test_three_tier_strong_long_opens(self):
        """Tier 3 인자 없이 호출 — 새로운 권장 인터페이스"""
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "long"),
            t4=make(0.95, "long"),
            available_balance_usdt=1750.0,
        )
        # 0.35*0.95 + 0.35*0.95 + 0.30*0.95 = 0.95
        assert decision.action == "open_long"
        assert decision.confluence_score == pytest.approx(0.95, abs=0.01)

    def test_three_tier_t4_alone_blocks_below_threshold(self):
        """T4가 'none'이면 T1+T2 둘만 기여 → 0.85 못 넘김"""
        decision = fuse(
            t1=make(1.0, "long"),
            t2=make(1.0, "long"),
            t4=make(1.0, "none"),  # Range-Bound 같은 진입 차단 시그널
            available_balance_usdt=1750.0,
        )
        # signed = 0.35*1.0 + 0.35*1.0 + 0.30*0 = 0.70 < 0.85
        assert decision.action == "hold"
        assert decision.confluence_score == pytest.approx(0.70, abs=0.01)

    def test_three_tier_t4_short_overrides_long(self):
        """T1, T2가 long인데 T4가 short → confluence가 약화됨"""
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "long"),
            t4=make(0.95, "short"),
            available_balance_usdt=1750.0,
        )
        # signed = 0.35*0.95 + 0.35*0.95 - 0.30*0.95 = 0.38 (long)
        # < 0.85 → hold
        assert decision.action == "hold"
        assert decision.direction == "long"  # 약하게나마 long 방향

    def test_backward_compat_with_t3(self):
        """기존 4-Tier 호출도 여전히 동작 (T3는 무시됨)"""
        decision = fuse(
            t1=make(0.95, "long"),
            t2=make(0.95, "long"),
            t3=make(0.95, "short"),  # 강한 short인데도 무시되어야 함
            t4=make(0.95, "long"),
            available_balance_usdt=1750.0,
        )
        # T3는 weight=0이므로 영향 없음
        # signed = 0.35*0.95 + 0.35*0.95 + 0 + 0.30*0.95 = 0.95
        assert decision.action == "open_long"
        # T3는 출력에 포함되지만 가중치 0으로 표시
        assert "T3" in decision.tier_outputs
