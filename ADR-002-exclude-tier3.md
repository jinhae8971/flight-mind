# ADR-002: Tier 3 (Microstructure TCN) 제외 결정

**Status**: Accepted
**Date**: 2026-05-02
**Author**: Younggil (decision), Claude (analysis)
**Supersedes Section of**: ADR-001 (4-Tier → 3-Tier 시스템 변경)

---

## 1. Context

ADR-001에서 정의된 4-Tier 시스템 중 Tier 3 (호가창 TCN)의 ROI를 면밀히 검토한 결과,
**다른 Tier들 대비 인프라 부담이 과도하게 큼**에도 불구하고 marginal lift가 가장 작다는
결론에 도달했다.

## 2. Decision

**Tier 3 (호가창 TCN)을 시스템에서 영구 제외한다.**

가중치 재분배:
```
기존 (4-Tier):  T1=0.30, T2=0.30, T3=0.20, T4=0.20  (합 1.0)
신규 (3-Tier):  T1=0.35, T2=0.35, T3=0.00, T4=0.30  (합 1.0)
```

코드 인터페이스:
- `fuse(t1, t2, t4, available_balance)` — 새로운 권장 시그니처
- `fuse(t1, t2, t3=..., t4, available_balance)` — backward-compat 유지
- `config.FUSION.w_tier3_microstr = 0.0` — 가중치 0으로 자연스러운 무력화

## 3. Rationale (정량 근거)

### 3.1. 비교 매트릭스

| 항목 | Tier 3 (호가창) | Tier 1 | Tier 2 | Tier 4 |
|------|------|------|------|------|
| 데이터 수집 | **WebSocket 24/7, 12GB/월** | Binance Vision 일괄 | Binance Vision 일괄 | Binance Vision 일괄 |
| 학습 시간 (RTX 4090) | **36~48h** | 즉시 | 8~12h | 1~2h |
| 정확도 (학계 SOTA) | 65~71% | N/A (룰) | 80~90% | 70~80% |
| Marginal Lift (예상) | **+5~10%p** | baseline | +15~25%p | 횡보 차단 효과 |
| 운영 복잡도 | **매우 높음** | 낮음 | 중간 | 낮음 |
| 가중치 (4-Tier ADR) | 0.20 | 0.30 | 0.30 | 0.20 |

**핵심 결론**: Tier 3는 0.20의 가중치를 위해 시스템 복잡도가 2배가 되며, 다른 Tier 대비
marginal lift가 가장 낮다.

### 3.2. 영길님의 보수 정책과의 적합성

영길님의 결정사항:
- Capital: 3,500 USDT (소액 검증)
- Confluence threshold: 0.85 (보수)
- Daily trade limit: 1~2회

이 정책에서는 Tier 3가 제공하는 "초단기 진입 타이밍 최적화"가 본질적으로 불필요하다.
보수적 진입은 "정확한 타이밍"보다 "잘못된 타이밍 회피"가 우선이며, 이는 이미 Tier 4 (국면)가 담당.

### 3.3. 3-Tier 시스템의 자연스러운 보수성 강화

3개 Tier에서 0.85 임계값을 넘으려면 **3개 모두 강한 합의**가 필요:
- T1 + T2 동의(같은 방향) + T4 동의: 가능
- T1 + T2 동의 + T4가 'none' (Range/Crash): confluence ≤ 0.70 → **자동 차단**

즉, **Tier 4가 횡보 게이트 역할을 더 강력하게 수행**한다.
이는 Day 2 백테스트의 핵심 진단 — "Tier 1 단독 -72%의 가장 큰 원인은 횡보장 진입" — 을
직접적으로 해결한다.

## 4. Consequences

### Positive

- **운영 단순화**: WebSocket 24/7 데몬 불필요, GitHub Actions 패턴 100% 활용 가능
- **저장 부담 절감**: 12GB/월 → 0GB (BTC + ETH 5분봉만)
- **학습 시간 단축**: 36~48h × N개월 학습 데이터 → 0
- **시스템 신뢰도 향상**: 컴포넌트 수 감소 → 장애 표면 감소
- **영길님 시간 절약**: 인프라 운영 부담 ↓, 알파 검증에 집중 가능

### Negative & Mitigation

| 리스크 | 완화 방안 |
|------|------|
| 미세 진입 타이밍 알파 손실 | 영길님 보수 정책상 영향 미미 (하루 1~2회 진입) |
| Day 2 -72% 개선 폭 감소 가능성 | Tier 4 가중치 +0.10으로 횡보 차단 강화 (실제 가장 큰 효과) |
| 학계 SOTA 대비 알파 갭 | 추후 v2.0에서 sentiment 분석(Tier 5)로 대체 검토 |

### Out of Scope

- ❌ 호가창 데이터 수집 인프라 — 영구 제외
- ❌ 실시간 LOB 분석 — 제외
- ✅ 미래 옵션: sentiment 분석 (Twitter/X) → 새로운 Tier 3 후보로 재검토 가능

## 5. Implementation Status

- [x] `flight_mind/tier3_microstr/` 디렉토리 정리 (모든 .py 삭제, `__init__.py`만 남김)
- [x] `config.py`: Tier3Config 클래스 제거, 가중치 재분배
- [x] `fusion/layer.py`: 새 3-Tier 시그니처 + backward-compat
- [x] `tests/test_fusion.py`: 3-Tier 테스트 4개 추가
- [x] 38개 테스트 모두 통과

## 6. Forward Compatibility

만약 향후 Tier 3 (또는 sentiment Tier)를 다시 추가하게 되면:
- `FUSION.w_tier3_microstr`를 0보다 큰 값으로 설정
- 다른 가중치 비례 조정
- 기존 fuse() 호출은 코드 변경 없이 자동으로 새 가중치 반영

이 ADR의 결정은 **시스템에서 "복잡도 vs 알파"의 trade-off를 명시적으로 관리**하는 선례를
남긴다. 향후 모든 Tier 추가/제거 결정에 동일한 정량 분석을 적용할 것.

---

**Sign-off**:
- [x] Younggil — Tier 3 제외 결정
- [x] Claude — 코드 정리 및 ADR 작성 완료
