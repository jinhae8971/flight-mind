# Day 4 — Tier 4 (시장 국면 Transformer) 구현 완료 보고

## 결과 요약

**플라이트의 "시장 국면 직관"을 5-class Transformer로 모방하는 시스템 완성.**

```
✅ Regime Labeler      (Bull/Bear/Range/HighVol/Crash 자동 라벨링)
✅ Feature Builder     (10개 피처: returns, RSI, ADX, vol, MA dist 등)
✅ PyTorch Dataset     (페어 임베딩, look-ahead 안전 split)
✅ Transformer Model   (6-layer, 8-head, 256-dim, ~5M params)
✅ Training Pipeline   (class weight + cosine LR + early stop)
✅ Inference API       (Tier 1, 2와 동일 인터페이스)
✅ Unit Tests          (12 tests, 합성 시장 + 모델 검증)
```

전체 테스트 스위트: **34/34 통과** (Day 1: 14 + Day 3: 8 + Day 4: 12)

## Tier 4의 가장 중요한 설계 결정

### 1. 진입 차단 게이트로서의 역할

Tier 4는 다른 Tier와 본질적으로 다른 역할을 합니다:

| Tier | 주된 역할 |
|------|----------|
| Tier 1 (Rule) | 진입 시그널 발생 |
| Tier 2 (CNN) | 진입 시그널 강화/필터 |
| Tier 3 (TCN) | 진입 타이밍 최적화 |
| **Tier 4 (Transformer)** | **진입 자체 차단 게이트** |

```python
REGIME_TO_OUTPUT = {
    "Bull-Trending":  (0.90, "long"),     # 강한 long 편향
    "Bear-Trending":  (0.90, "short"),    # 강한 short 편향
    "Range-Bound":    (0.70, "none"),     # 진입 자제
    "High-Vol-Range": (0.90, "none"),     # 강한 진입 차단
    "Crash":          (1.00, "none"),     # 최강 차단
}
```

`direction='none'` + 높은 score는 Fusion Layer에서 **다른 Tier의 신호를 무력화**시킵니다. Day 2에서 발견한 "Tier 1 단독 -72%의 가장 큰 원인은 횡보장 진입"이라는 진단의 직접적 해결책입니다.

### 2. 5-Class 설계 근거

| 국면 | 정의 | 플라이트의 행동 |
|------|------|---------------|
| Bull-Trending | 30일 +5% & ADX > 25 & MA200 위 | 장투 long 가능 |
| Bear-Trending | 30일 -5% & ADX > 25 & MA200 아래 | 단발성 short |
| Range-Bound | ADX < 20 & vol < 50% | 진입 회피 |
| High-Vol-Range | ADX < 25 & vol > 80% | **절대 진입 회피** |
| Crash | 7일 -15% or 1일 -8% | 모든 포지션 청산 |

**Crash는 절대적 우선순위**를 가집니다. 다른 어떤 조건과 무관하게 즉시 라벨링됩니다.

### 3. 페어 임베딩 (BTC vs ETH)

같은 30일 윈도우라도 BTC와 ETH는 다른 국면 특성을 보입니다 (특히 알트시즌 vs BTC 단독 상승):

```python
# Symbol embedding을 시퀀스의 모든 timestep에 broadcast
sym_emb = self.symbol_embed(symbol_idx).unsqueeze(1)  # (B, 1, d_model)
h = h + sym_emb                                         # (B, S, d_model)
```

테스트에서 검증: **같은 입력 + 다른 symbol_idx → 다른 logits**

## 영길님 환경에서 학습 실행 가이드

```bash
# 데이터 준비 (이미 완료된 경우 skip)
python scripts/download_binance_data.py --symbols BTCUSDT ETHUSDT --years 5
python scripts/smoke_test_day2.py  # DuckDB 적재

# Tier 4 학습 (RTX 3090/4090, 1~2시간)
python -m flight_mind.tier4_regime.train \
    --symbols BTCUSDT ETHUSDT \
    --epochs 40 \
    --batch-size 128
```

### 예상 학습 통계

| 항목 | 값 |
|------|------|
| 총 샘플 수 | 약 3,200 (페어당 1,600) |
| 모델 파라미터 | ~5M |
| VRAM 사용 | ~6GB (batch=128) |
| 학습 시간 | 1~2시간 (RTX 4090) |
| 예상 정확도 | 70~80% |

Tier 2(8~12시간) 대비 훨씬 빠릅니다. **데이터셋이 작기 때문**(일봉 5년 = 1,825일/페어).

## 검증된 동작

### 합성 강세장 → Bull-Trending 라벨

```python
# 40k → 80k 강한 상승 + 노이즈
synthetic_bull_market = 300일 데이터
labels = label_regime(df)
# 결과: Bull-Trending ≥ 1, Bear-Trending = 0
```

### 합성 약세장 → Bear-Trending 라벨

대칭적으로 검증됨.

### 모델 forward pass

```python
model = RegimeTransformer(...)
x = (B, 30, 10)              # 30일 × 10 features
sym = (B,)                   # symbol_idx
logits = model(x, sym)       # (B, 5)
probs = softmax(logits)      # 5-class 확률 분포
```

### Graceful Degradation

```python
inf = Tier4Inference(model_path="/nonexistent.pt")
output = inf.predict(df)
# → TierOutput(score=0.0, direction="none", signals={"reason": "..."})
```

모델 파일 없을 때 시스템 전체가 멈추지 않고 **Tier 4 신호를 무시**하고 다른 Tier로 진행 가능.

## 4-Tier 시스템 진척도

```
Tier 1 (Rule):        ✅ 완성 (Day 1)
Tier 2 (CNN):         ✅ 완성 (Day 3)
Tier 3 (TCN):         ⏳ 미구현
Tier 4 (Transformer): ✅ 완성 (Day 4)
Fusion Layer:         ✅ 완성 (Day 1)
Backtest Harness:     ✅ Tier 1 단독 (Day 2)
```

**4-Tier 중 3개 완성** + 통합 Fusion 작동 가능 상태.

## 다음 단계 후보

| 옵션 | 작업량 | ROI |
|------|------|-----|
| **Tier 1+2+4 통합 백테스트** | 1일 | 매우 높음 — 학계 가설 정량 검증 |
| Tier 3 (호가창 TCN) 구현 | 5~7일 | 높음 — 마지막 알파 레이어 |
| GitHub Actions CI 세팅 | 0.5일 | 중간 — 영길님 PC 작업 안전 |
| 5년 데이터 다운로드 + 학습 | 1일 데이터 + 12시간 학습 | 매우 높음 — 실제 모델 가동 |

영길님 권장: **Tier 1+2+4 통합 백테스트**가 가장 가치 있습니다. Day 2에서 측정한 Tier 1 단독 -72%가 Tier 2, 4 추가 시 얼마나 개선되는지를 정량 측정하면, **이 프로젝트의 본질적 가치를 데이터로 증명**할 수 있습니다.
