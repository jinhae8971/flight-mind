"""
Bayesian Confluence Fusion Layer
=================================
4개 Tier의 출력을 통합하여 최종 진입/청산 결정을 내림.

영길님 결정사항:
  - Threshold: 0.85 (Conservative, 하루 1~2회)
  - Tier weights: T1=0.30, T2=0.30, T3=0.20, T4=0.20
  - T1, T2 부호 반대 시 강제 hold
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from flight_mind.config import CAPITAL, FUSION
from flight_mind.tier1_rule.engine import TierOutput


Action = Literal["open_long", "open_short", "hold", "close_position"]


@dataclass
class FusionDecision:
    """Fusion Layer의 최종 출력"""
    action: Action
    confluence_score: float          # [0, 1] 절댓값
    direction: Literal["long", "short", "none"]
    position_size_usdt: float = 0.0
    leverage: int = 1
    stop_loss_pct: float = -3.0
    take_profit_pct: float = 6.0
    max_hold_bars: int = 12

    # 근거 — observability
    tier_outputs: dict[str, TierOutput] = field(default_factory=dict)
    veto_reason: str | None = None


def _to_signed(o: TierOutput) -> float:
    """Tier output을 [-1, +1]로 정규화"""
    return o.signed_score()


def fuse(
    t1: TierOutput,
    t2: TierOutput,
    t4: TierOutput,
    available_balance_usdt: float,
    t3: TierOutput | None = None,    # Tier 3 제거됨 — backward-compat용
) -> FusionDecision:
    """
    3-Tier Bayesian Confluence Fusion (Tier 3 호가창 제외).

    confluence_signed = w1·T1 + w2·T2 + w4·T4   ∈ [-1, +1]
    confluence_score  = |confluence_signed|      ∈ [0, +1]
    direction         = sign(confluence_signed)

    Tier 3는 ROI 대비 인프라 부담이 너무 커서 제외 (영길님 결정, 2026-05-02).
    가중치는 T1/T2/T4로 재분배 (0.35/0.35/0.30).
    """
    tier_outputs = {"T1": t1, "T2": t2, "T4": t4}
    if t3 is not None:
        tier_outputs["T3"] = t3   # 로깅/관측용으로만 보존

    # 1) Veto 룰 — T1, T2 부호 반대 시 강제 hold
    if FUSION.require_t1_t2_agree:
        if t1.direction != "none" and t2.direction != "none":
            if t1.direction != t2.direction:
                return FusionDecision(
                    action="hold",
                    confluence_score=0.0,
                    direction="none",
                    tier_outputs=tier_outputs,
                    veto_reason=f"T1({t1.direction}) vs T2({t2.direction}) disagree",
                )

    # 2) 가중합 — 3-Tier (Tier 3 제외)
    signed = (
        FUSION.w_tier1_rule * _to_signed(t1)
        + FUSION.w_tier2_pattern * _to_signed(t2)
        + FUSION.w_tier4_regime * _to_signed(t4)
    )

    score = abs(signed)
    direction = "long" if signed > 0 else "short" if signed < 0 else "none"

    # 3) Threshold 체크 — 영길님 결정: 0.85
    if score < FUSION.threshold or direction == "none":
        return FusionDecision(
            action="hold",
            confluence_score=score,
            direction=direction,
            tier_outputs=tier_outputs,
            veto_reason=f"confluence {score:.3f} < threshold {FUSION.threshold}",
        )

    # 4) Position sizing — Kelly fraction (1/4 Kelly)
    position_size = available_balance_usdt * CAPITAL.max_position_pct  # max 2%

    return FusionDecision(
        action="open_long" if direction == "long" else "open_short",
        confluence_score=score,
        direction=direction,
        position_size_usdt=position_size,
        leverage=CAPITAL.leverage,
        stop_loss_pct=CAPITAL.stop_loss_pct,
        take_profit_pct=CAPITAL.take_profit_pct,
        max_hold_bars=CAPITAL.max_hold_bars_5m,
        tier_outputs=tier_outputs,
    )


def explain(decision: FusionDecision) -> str:
    """의사결정 근거를 사람이 읽을 수 있는 형태로 출력"""
    lines = [
        f"\n{'=' * 60}",
        f"Action: {decision.action.upper()}",
        f"Confluence: {decision.confluence_score:.3f} (threshold: {FUSION.threshold})",
        f"Direction: {decision.direction}",
        "-" * 60,
    ]
    for name, t in decision.tier_outputs.items():
        weight = {
            "T1": FUSION.w_tier1_rule,
            "T2": FUSION.w_tier2_pattern,
            "T3": FUSION.w_tier3_microstr,   # 0.0 (제외됨)
            "T4": FUSION.w_tier4_regime,
        }[name]
        contribution = weight * t.signed_score()
        # Tier 3는 weight=0이므로 표시할 때 명시
        suffix = "  [excluded]" if name == "T3" else ""
        lines.append(
            f"  {name} (w={weight:.2f}): score={t.score:.3f} dir={t.direction:<5} "
            f"→ contribution={contribution:+.3f}{suffix}"
        )
    lines.append("-" * 60)
    if decision.veto_reason:
        lines.append(f"Veto: {decision.veto_reason}")
    if decision.action.startswith("open"):
        lines.append(f"Position: {decision.position_size_usdt:.2f} USDT @ {decision.leverage}x")
        lines.append(f"SL/TP: {decision.stop_loss_pct}% / +{decision.take_profit_pct}%")
    lines.append("=" * 60)
    return "\n".join(lines)
