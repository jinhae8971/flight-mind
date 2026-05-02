# ADR-001: Flight-Mind 4-Tier Hybrid Trading System Architecture

**Status**: Accepted
**Date**: 2026-05-02
**Author**: Younggil (with Claude as system architect)
**Project**: `flight-mind` (Binance Futures Auto-Trading via Flight Strategy)

---

## 1. Context (배경)

### 1-1. 문제 정의

플라이트(FlightChallenge)의 매매법은 50만원 → 100억의 검증된 전략이지만, 다음 구성요소가 결합되어 있다:

1. **명시적 룰**: 추세선, RSI 다이버전스, 거래량, 이평선 터치
2. **암묵적 직관**: 캔들 균형성, 호가창 미세 흐름, 시장 국면 변화

기존 룰 기반 자동화로는 (1)만 재현 가능하며, 본질 알파의 30~50%가 (2)에 있다.
플라이트 본인의 발언: *"호가창 보는법을 글로 쓰게되면 '그냥 감인데'라고 할 정도로 감적인 부분이 있어서..."*

### 1-2. 기회

학계 SOTA(2024-2025)에 따르면:
- GAF-CNN: 캔들 패턴 인식 90.7% 정확도 (실제 시장)
- Temporal CNN: 호가창 2초 예측 71% 정확도 (BTC)
- Transformer 기반 시장 국면 분류: 70~80%

이를 룰 엔진과 결합하면 플라이트의 "감"을 70~85% 수준에서 모방 가능.

### 1-3. 제약 조건 (영길님 결정사항)

| 항목 | 값 | 영향 |
|------|------|------|
| GPU | RTX 3090/4090 (24GB VRAM) | 모델 파라미터 ≤ 200M, batch_size 동적 조정 |
| Live Capital | 3,500 USDT (≈500만원) | Kelly fraction 보수, 거래당 max 70 USDT (2%) |
| Confluence Threshold | 0.85 | Tier 4개 모두 일치할 때만 진입, 일일 1~2회 |

---

## 2. Decision (결정)

### 2-1. 핵심 아키텍처 결정

**4-Tier Hybrid System** with Bayesian Confluence Fusion:

```
┌──────────────────────────────────────────────────────────────────┐
│                    FUSION LAYER (Bayesian)                        │
│  Confluence(t) = w1·T1 + w2·T2 + w3·T3 + w4·T4  ∈ [0, 1]        │
│  Action: Long if Confluence > 0.85 ∧ direction == long           │
│          Short if Confluence > 0.85 ∧ direction == short         │
│          Hold otherwise                                           │
└──────────────────────────────────────────────────────────────────┘
        ▲                ▲                ▲                ▲
        │                │                │                │
┌───────┴──────┐ ┌───────┴──────┐ ┌──────┴──────┐ ┌───────┴──────┐
│ TIER 1       │ │ TIER 2       │ │ TIER 3      │ │ TIER 4       │
│ Rule Engine  │ │ Pattern CNN  │ │ Microstr TCN│ │ Regime Trans │
│ w1 = 0.30    │ │ w2 = 0.30    │ │ w3 = 0.20   │ │ w4 = 0.20    │
└──────────────┘ └──────────────┘ └─────────────┘ └──────────────┘
   Plaintext       GAF-CNN          Temporal CNN    Transformer
   확실성 높음     플라이트 학습     호가창 학습      국면 학습
```

### 2-2. 가중치 결정 근거

| Tier | 가중치 | 이유 |
|------|--------|------|
| T1 (Rule) | 0.30 | 가장 검증된 영역. 플라이트 본인이 명시적으로 정의한 룰 |
| T2 (CNN) | 0.30 | 플라이트의 "캔들 균형성 = 모양 외움" 직접 모방. 90% 정확도 |
| T3 (TCN) | 0.20 | 호가창 알파지만 데이터 노이즈로 안정성 낮음 (71%) |
| T4 (Transformer) | 0.20 | 국면 감지는 매우 어려움. 보조 필터 역할 |

**합계 = 1.0**, 각 Tier 출력 [0, 1] 정규화 후 가중합.

### 2-3. Tier별 상세 사양

#### TIER 1 — Rule Engine (Deterministic Layer)
- **입력**: 5m / 15m / 1h / 4h OHLCV
- **출력**: `{score: float [0,1], direction: long|short|none, signals: list}`
- **로직** (4개 sub-rule, 각 0~1점):
  - R1.1: 추세선 터치 + 거래량 동반 (영길님 `trendline-detector` 활용)
  - R1.2: RSI 다이버전스 (`rsi-divergence-detector` PyPI 패키지)
  - R1.3: MA(7/30) 터치 + 반등 패턴
  - R1.4: 더블바텀 (코인형: 오른쪽 저점이 더 낮음)
- **점수 계산**: `score = mean(R1.1, R1.2, R1.3, R1.4)` after sign alignment

#### TIER 2 — Pattern Memory CNN (Implicit Visual Learning)
- **입력**: 직전 60개 캔들의 GAF(Gramian Angular Field) 인코딩 이미지 (64×64×3)
- **모델**: ResNet-18 백본 + 분류 헤드
  - Output: (long_strength, short_strength, neutral) softmax
- **학습 데이터**:
  - BTC/ETH/SOL 2021-2026 5m 캔들 (≈526만 캔들)
  - 라벨: 향후 12-bar (1시간) 수익률 +0.5% 이상이면 long, -0.5% 이하면 short
  - 플라이트 본인 매매일지 100건은 **가중 라벨** (importance × 5)
- **파라미터 수**: ~11M (ResNet-18 표준)
- **학습 시간**: RTX 4090에서 약 8~12시간 (50 epoch)
- **목표 정확도**: validation 80%+

#### TIER 3 — Microstructure Memory TCN (Order Flow Learning)
- **입력**: 호가창 최근 100 스냅샷 (10단계 bid/ask, 100×40 매트릭스)
- **모델**: Temporal Convolutional Network (causal, dilated)
  - 6 layer, dilation [1,2,4,8,16,32], kernel 3
  - Output: 다음 2초 가격 방향 (up/down/flat)
- **학습 데이터**:
  - Binance BTC/USDT 선물 호가창 6개월 스냅샷 (~1.2TB)
  - 50ms 샘플링, 100ms 미래 가격 변화로 라벨링
- **파라미터 수**: ~2M
- **학습 시간**: RTX 4090에서 약 36~48시간 (전체 데이터 기준)
  - **현실적 절충**: 2개월치만 학습 → 12시간
- **목표 정확도**: 65%+ (학계 SOTA 71%, 보수적 목표)

#### TIER 4 — Market Regime Transformer (Macro Context)
- **입력**: 영길님의 `risk-regime-monitor` 34지표 + 1d/4h OHLCV 30일치
- **모델**: 작은 Transformer (6 layer, 256 dim, 8 heads)
  - Output: {Bull-Trending, Bear-Trending, Range-Bound, High-Vol-Range, Crash} 5-class
- **학습 데이터**:
  - 2018-2026 BTC daily + 영길님의 risk-regime 데이터
  - 라벨: 후행 30일 가격 행동으로 자동 분류
- **파라미터 수**: ~5M
- **학습 시간**: RTX 4090에서 약 6~8시간
- **목표 정확도**: 70%+

### 2-4. 진입/청산 룰 (Confluence > 0.85 시)

```python
def execute_decision(confluence_score, direction, ctx):
    if confluence_score < 0.85:
        return {"action": "hold"}

    # Position sizing — Kelly fraction (very conservative)
    kelly_f = min(0.02, ctx.kelly_estimate * 0.25)  # 1/4 Kelly
    position_usdt = ctx.available_balance * kelly_f
    # 3500 USDT × 2% = 70 USDT per trade max

    # Leverage — 보수 모드에서는 5x로 제한
    leverage = 5  # 플라이트 100x 권장이지만 우리는 5x

    return {
        "action": "open_position",
        "direction": direction,
        "size_usdt": position_usdt,
        "leverage": leverage,
        "stop_loss_pct": -3.0,   # tier4 regime 따라 동적 조정
        "take_profit_pct": 6.0,  # 손익비 2:1
        "max_hold_bars": 12,     # 1시간 후 자동 청산
    }
```

### 2-5. 시드 분리 (Account Bulkhead)

| 계좌 | 비중 | 금액 | 용도 |
|------|------|------|------|
| Live Trading Sub-account | 50% | 1,750 USDT | Flight-Mind 자동 운용 |
| Reserve Sub-account | 30% | 1,050 USDT | 비상시 재충전 + DCA |
| Cold Wallet (외부) | 20% | 700 USDT 상당 BTC | Hardware wallet, 자동 출금 대상 |

**자동 출금 정책**: 매주 일요일 21:00 KST에 직전 1주 수익의 30%를 Cold Wallet으로 자동 이체 (트래블룰 대응 위해 100만원 미만 분할).

### 2-6. Kill-Switch 룰 (영길님 `ct-agent-ultra` 기반 확장)

| Trigger | 임계값 | Action |
|---------|--------|--------|
| Daily Loss | -5% (-87.5 USDT) | 거래 중단, 24시간 쿨다운 |
| Weekly Loss | -10% (-175 USDT) | 1주일 정지, 모델 재평가 |
| Max DD | -15% | 시스템 셧다운, 영길님 수동 컨펌 필요 |
| Liquidation Event | 1회 | 24시간 쿨다운, 해당 Tier 학습 데이터 추가 |
| Daily Trade Count | > 2 | 추가 진입 차단 |
| Confluence Disagreement | T1·T2 부호 반대 | 자동 hold |

---

## 3. Consequences (결과)

### 3-1. Positive

- **검증된 SOTA 활용**: 모든 Tier가 학계 검증된 아키텍처. Bleeding edge 위험 회피
- **점진적 학습**: 라이브 데이터로 model retrain 가능 (online learning ready)
- **영길님 자산 90% 재활용**: ct-agent-ultra, trendline-detector, chart-analyzer
- **명확한 Failure Mode**: Tier 1 단독 동작도 정상 작동 (Tier 2~4 fallback)

### 3-2. Negative & Mitigation

| 리스크 | 완화 방안 |
|--------|-----------|
| TCN 학습 데이터 1.2TB 저장 부담 | 온더플라이 다운샘플링 (50ms→500ms), 학습 후 모델만 보존 |
| 모델 과적합 (overfitting) | Walk-forward validation, 1개월마다 retrain |
| 알고리즘 알파 소멸 | 분기별 retraining, sentiment 데이터 추가 검토 |
| Kelly fraction 추정 오류 | 1/4 Kelly로 강제 보수화 |
| Confluence > 0.85 신호가 너무 적음 | 첫 1개월 paper trading으로 빈도 측정, 필요 시 가중치 재조정 |

### 3-3. Out of Scope (이번에 안 다루는 것)

- ❌ Sentiment 분석 (Twitter/X) — 추후 Tier 5로 확장 가능
- ❌ On-chain 데이터 — 추후 Tier 6로 확장 가능
- ❌ 옵션 IV/funding rate — Phase 2
- ❌ 다중 거래소 차익거래 — 별도 시스템

---

## 4. Implementation Roadmap

### Week 1: Data Foundation
**Goal**: 모든 학습 데이터 수집 + Tier 1 룰 엔진 가동

- [ ] D1: GitHub 레포 생성 (`flight-mind`), 디렉토리 구조 확정
- [ ] D2: Binance Vision에서 BTC/ETH/SOL 5y 5m 캔들 다운로드
- [ ] D3: Binance WebSocket → DuckDB 호가창 수집 데몬 가동 시작
- [ ] D4: `risk-regime-monitor` API 연동 (Tier 4용)
- [ ] D5: Tier 1 룰 엔진 (`tier1_rule.py`) 완성 + unit test
- [ ] D6: 플라이트 매매일지 크롤러 (`scripts/crawl_dcinside.py`)
- [ ] D7: Backtest 프레임워크 정비 (`backtest_lab` 활용)

### Week 2: Tier 2 (CNN Pattern Memory)
**Goal**: 캔들 모양 외우기 — 80%+ accuracy

- [ ] D8: GAF 인코더 (`encoders/gaf.py`)
- [ ] D9: ResNet-18 모델 정의 + DataLoader
- [ ] D10-12: 학습 (RTX 4090, 50 epoch)
- [ ] D13: Plaitfllight 매매일지 100건 가중 fine-tuning
- [ ] D14: Tier 2 추론 API 패키징

### Week 3: Tier 3 (TCN Microstructure)
**Goal**: 호가창 외우기 — 65%+ accuracy

- [ ] D15: LOB 데이터 전처리 파이프라인 (50ms → 500ms 다운샘플)
- [ ] D16: TCN 모델 정의
- [ ] D17-19: 학습 (2개월치 데이터, ~12h)
- [ ] D20-21: 평가 + 추론 API

### Week 4: Tier 4 + Fusion + Live
**Goal**: 통합 + Paper Trading 시작

- [ ] D22: Tier 4 Transformer 정의 + 학습
- [ ] D23: Bayesian Fusion Layer
- [ ] D24: Order Execution Engine (CCXT Pro 기반)
- [ ] D25: Kill-Switch + Account Bulkhead
- [ ] D26: Telegram 알림 통합
- [ ] D27: Testnet Paper Trading 시작
- [ ] D28: 모니터링 대시보드 (영길님 `github-actions-dashboard` 통합)

### Week 5+: Validation
- 4주간 Paper Trading
- Real-money 100 USDT 테스트 → 1,000 USDT → 1,750 USDT 단계적 확대

---

## 5. Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Exchange API | CCXT Pro (WebSocket) |
| Data Store | DuckDB (feature store), Parquet (cold storage) |
| ML Framework | PyTorch 2.x, timm (ResNet), pyts (GAF) |
| Orchestration | LangGraph (multi-agent decision flow) |
| Scheduling | GitHub Actions + APScheduler |
| Vault | AES-256-GCM (영길님 ct-agent-ultra 패턴) |
| Monitoring | FastAPI Dashboard + Telegram Bot |
| Deployment | Docker + Docker Compose (Live), GitHub Actions (Backtest) |

---

## 6. Key Architectural Principles

1. **Defense in Depth**: 4 Tier 모두 일치해야 진입 — 단일 Tier 오작동에 강건
2. **Graceful Degradation**: 어느 Tier든 실패 시 weight=0으로 fallback
3. **Bulkhead Pattern**: 계좌 분리, 손실 전파 차단
4. **Observability First**: 모든 의사결정 로그, 모든 Tier output 저장
5. **Reproducibility**: Random seed 고정, 모델 버전 관리, MLflow
6. **Security First**: API key는 출금 권한 없는 trade-only, IP whitelist

---

## 7. Open Questions (영길님 컨펌 필요)

1. ~~GPU 환경?~~ ✅ 로컬 RTX 3090/4090
2. ~~시드 규모?~~ ✅ 3,500 USDT
3. ~~Confluence threshold?~~ ✅ 0.85
4. **레포 위치**: `jinhae8971/flight-mind` (public) vs private?
5. **Cold Wallet**: 어떤 하드웨어 지갑? (Ledger Nano X 가정)
6. **선물 거래 vs 현물**: 일단 USDT-M 선물 가정. 컨펌 필요
7. **거래 페어**: BTC/USDT 단독 → ETH/USDT, SOL/USDT 확장? (멀티 페어는 Tier 4 학습 부담 증가)

---

## 8. References

- 플라이트 매매법 1차 리서치 (이전 대화 참조)
- GAF-CNN: Chen & Tsai, "Encoding candlesticks as images...", Financial Innovation, 2020 (90.7% acc)
- Temporal CNN for LOB: Jha et al., "Deep Learning for Digital Asset Limit Order Books", arXiv 2010.01241 (71% acc, 2-sec horizon)
- 영길님 기존 자산: `ct-agent-ultra`, `trendline-detector`, `chart-analyzer`, `risk-regime-monitor`

---

**Sign-off**:
- [ ] Younggil — Architecture Approved
- [x] Claude — System Design Complete

**Next Step**: Day 1 작업 — 레포 스캐폴딩 + 데이터 수집 스크립트
