# Flight-Mind 학습 가이드 (영길님 PC용)

> **목표**: 영길님 PC(Windows + RTX 3090/4090)에서 5년 BTC + ETH 데이터로 Tier 2/4를 학습하고
> 통합 백테스트를 거쳐 paper trading 직전까지 완성하는 완전 자동화 절차.

---

## 0. 사전 준비 체크리스트

| 항목 | 요구 사항 |
|------|----------|
| OS | Windows 10/11 (영길님 환경) 또는 WSL2 |
| Python | 3.11+ |
| GPU | NVIDIA RTX 3090 / 4090 (24GB VRAM) |
| NVIDIA Driver | 525.x 이상 (CUDA 12.1 지원) |
| 디스크 | C: 또는 D:\ 드라이브에 **15 GB 여유** |
| 네트워크 | data.binance.vision 접속 가능 |
| 시간 | 약 12~15시간 (대부분 학습 대기) |

### 핵심: 영길님 환경 변수 확인

영길님 메모리에 따르면 GitHub PAT, Telegram 봇이 이미 설정되어 있습니다.
PowerShell에서 확인:

```powershell
# Telegram 알림용 (학습 진행 상황 알림)
$env:TELEGRAM_BOT_TOKEN = "<영길님 봇 토큰>"
$env:TELEGRAM_CHAT_ID = "<영길님 채팅 ID>"

# 영구 저장 (재부팅 후에도 유지)
[Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "...", "User")
[Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "...", "User")
```

---

## 1. 레포 클론 (5분)

```powershell
# 작업 디렉토리 (디스크 여유 많은 곳)
cd D:\projects   # 또는 영길님 선호 경로
git clone https://github.com/jinhae8971/flight-mind.git
cd flight-mind
```

---

## 2. 한 번에 실행하기 (추천)

```powershell
.\setup_and_train.ps1
```

이 한 줄이 다음을 자동으로 수행합니다:

1. ✅ Python venv 생성 + 의존성 설치 (~5분)
2. ✅ PyTorch CUDA 12.1 설치 (~5분)
3. ✅ 환경 진단 (GPU, 디스크, 네트워크 점검)
4. ✅ GPU VRAM에 따른 batch_size 자동 결정
5. ✅ 5년 BTC + ETH 데이터 다운로드 (~1시간)
6. ✅ DuckDB 적재 + 피처 빌드 (~30분)
7. ✅ Tier 2 학습 (RTX 4090: 8시간 / RTX 3090: 12시간)
8. ✅ Tier 4 학습 (1~2시간)
9. ✅ 3-Tier 통합 백테스트 (30분)

**중간 단계마다 Telegram으로 진행 상황 알림이 옵니다.**

---

## 3. 단계별 옵션 (필요시)

### 환경 진단만

```powershell
.\setup_and_train.ps1 -DiagnoseOnly
```

### 특정 단계부터 재시작 (실패 시)

```powershell
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2
```

가능한 단계 ID:
- `diagnose` — 환경 점검
- `download_data` — 데이터 다운로드
- `load_to_db` — DuckDB 적재
- `build_features` — 피처 빌드
- `train_tier2` — Tier 2 (CNN) 학습
- `train_tier4` — Tier 4 (Transformer) 학습
- `integrated_backtest` — 통합 백테스트

### Batch size 수동 지정

```powershell
# RTX 3090 (24GB) 기준 Tier 2 batch=64
.\setup_and_train.ps1 -SkipSetup -T2Batch 64 -T4Batch 128
```

### 처음부터 완전 재시작

```powershell
.\setup_and_train.ps1 -Reset
```

---

## 4. 단계별 예상 시간 + 진행 모니터링

### 예상 시간 (RTX 4090 기준)

```
Step 1: 환경 진단              ~30초
Step 2: 데이터 다운로드        ~1시간 (네트워크 의존)
Step 3: DuckDB 적재           ~10분
Step 4: 피처 빌드             ~20분
Step 5: Tier 2 학습           ~8시간  ← 가장 김
Step 6: Tier 4 학습           ~1~2시간
Step 7: 통합 백테스트         ~30분
                            ───────
총합:                          ~12시간
```

### 진행 모니터링

**터미널**:
```powershell
# 다른 PowerShell 창에서
Get-Content logs\*.log -Tail 50 -Wait
```

**Telegram**: 자동 알림
- 각 단계 시작/완료
- 실패 시 즉시 알림 + 재시작 명령어

**파일 시스템**:
- `data/.pipeline_state.json` — 현재 진행 상황
- `data/models/tier2_pattern_cnn.pt` — Tier 2 체크포인트
- `data/models/tier4_regime_transformer.pt` — Tier 4 체크포인트
- `data/integrated_backtest_results.json` — 최종 결과

---

## 5. 학습 결과 해석

학습 완료 후 `data/integrated_backtest_results.json`에서 확인 가능한 핵심 지표:

```json
{
  "BTCUSDT": {
    "realistic": {
      "n_trades": 200,         ← 5년 거래 횟수
      "win_rate": 0.62,        ← 승률 (목표 ≥ 55%)
      "total_return_pct": 35.0, ← 5년 누적 수익률
      "profit_factor": 1.8,    ← 1.5 이상 권장
      "max_loss_pct": -2.8     ← 최대 단일 거래 손실
    }
  }
}
```

### 평가 기준

| Metric | 합격선 | 양호 | 우수 |
|--------|--------|------|------|
| Test Acc (Tier 2) | ≥ 55% | ≥ 65% | ≥ 75% |
| Test Acc (Tier 4) | ≥ 50% | ≥ 60% | ≥ 70% |
| Win Rate (BT) | ≥ 50% | ≥ 60% | ≥ 70% |
| Profit Factor | ≥ 1.2 | ≥ 1.5 | ≥ 2.0 |
| Max DD | ≥ -10% | ≥ -7% | ≥ -5% |

**합격선 미달 시**: Tier 2 라벨링 임계값 조정 (0.5% → 0.3%) 또는 future_horizon 확장 (12봉 → 24봉)

---

## 6. 학습 실패 대응 가이드

### 자주 발생할 수 있는 문제

#### 6.1. "CUDA out of memory"
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB...
```

**해결책**: batch_size 줄이기
```powershell
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2 -T2Batch 32
```

#### 6.2. "No data fetched for BTCUSDT"

Binance Vision API 변경 또는 네트워크 문제. 로그 확인:
```powershell
Get-Content logs\download_data_*.log -Tail 100
```

#### 6.3. 학습 도중 멈춤 (PC 재부팅 등)

체크포인트가 자동 저장되므로 같은 명령어로 재개 가능:
```powershell
.\setup_and_train.ps1 -SkipSetup
# (이미 완료된 단계 자동 skip)
```

#### 6.4. Test Accuracy가 너무 낮음 (< 50%)

라벨링 분포 확인 (영길님 메모리: hold가 76% 많음):
```powershell
& .venv\Scripts\python.exe -c "
from flight_mind.tier2_pattern.dataset import OhlcvWindowDataset
ds = OhlcvWindowDataset(['BTCUSDT', 'ETHUSDT'], 'train')
print(ds.class_distribution())
"
```

장 적은 클래스에 추가 weight 적용 또는 미래 horizon 확장 검토.

---

## 7. 학습 완료 후 다음 단계

학습이 정상 완료되고 백테스트가 양호하면:

### 7.1. 결과 검토 (1일)
- `data/integrated_backtest_results.json` 분석
- 만족스러우면 다음 단계, 아니면 하이퍼파라미터 튜닝

### 7.2. Paper Trading 준비 (1~2일)
*다음 작업으로 영길님과 함께 진행 예정*:
- Execution Engine + Vault 구현 (CCXT 기반 주문 실행)
- Risk Manager (Kill-Switch, Daily PnL 감시)
- Telegram 알림 통합

### 7.3. Binance Testnet (1개월)
- 실제 주문 흐름 검증
- API 키 발급 (testnet, 출금 권한 OFF)
- 일일 PnL 모니터링

### 7.4. Live (소액 → 풀 시드)
영길님 결정사항에 따른 단계적 확대:
- 100 USDT 1주 안정 → 1,000 USDT
- 1,000 USDT 1주 안정 → 1,750 USDT (full live)

---

## 8. 주요 명령 요약 카드

```powershell
# 처음 실행
.\setup_and_train.ps1

# 환경만 점검
.\setup_and_train.ps1 -DiagnoseOnly

# 학습 도중 실패 → 재시작
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2

# 처음부터 완전히 재시작
.\setup_and_train.ps1 -Reset

# 학습된 모델로 백테스트만 다시
& .venv\Scripts\python.exe -m flight_mind.utils.backtest_integrated --symbol BTCUSDT --mode all

# Tier 2만 단독 학습 (튜닝 시)
& .venv\Scripts\python.exe -m flight_mind.tier2_pattern.train --symbols BTCUSDT ETHUSDT --epochs 50

# Tier 4만 단독 학습
& .venv\Scripts\python.exe -m flight_mind.tier4_regime.train --symbols BTCUSDT ETHUSDT --epochs 40
```

---

## 9. FAQ

**Q. 학습 도중 PC를 다른 작업에도 쓸 수 있나요?**
A. GPU 사용량이 거의 100%이므로 게임/3D 작업은 부적합. 일반 사무 작업은 가능.

**Q. 학습 시 전기료는 얼마나 나오나요?**
A. RTX 4090 약 450W × 12시간 = 5.4 kWh ≈ 한국 전기료 약 1,000원.

**Q. 모델 체크포인트만 따로 백업할 수 있나요?**
A. 네 — `data/models/` 디렉토리만 백업. 약 60 MB.
   ```powershell
   Copy-Item -Recurse data\models D:\backup\flight-mind-models-$(Get-Date -Format "yyyyMMdd")
   ```

**Q. 학습 결과가 좋지 않으면 데이터를 어떻게 확장하나요?**
A. SOL/USDT 등 추가 페어 학습:
   ```powershell
   & .venv\Scripts\python.exe scripts\download_binance_data.py --symbols SOLUSDT --years 5
   & .venv\Scripts\python.exe -m flight_mind.tier2_pattern.train --symbols BTCUSDT ETHUSDT SOLUSDT --epochs 50
   ```
   다만 영길님 결정 (BTC + ETH 2-pair) 기준이 우선.

---

## 10. 영길님께 솔직히 말씀드리는 두 가지

### 10.1. 학습 성공률은 100%가 아닙니다

학습 데이터는 5년 시장 데이터지만, 모델이 의미있는 패턴을 잡을지는 사후 검증 후에야 알 수 있습니다.
Day 5 mock 백테스트에서 보여드린 Pessimistic 시나리오 (T2: 55%, T4: 50%)가 실제 학습 결과가 될 수도
있습니다. 그 경우 break-even 근처 — 즉 큰 손실은 없지만 큰 수익도 없는 상태.

### 10.2. 시장 국면 의존성

만약 학습 데이터(2021~2026)에 강세장이 많이 포함되면 모델이 long bias를 가질 가능성이 있습니다.
Tier 4가 이를 부분적으로 보정하지만, **2026년 5월 이후의 시장이 학습 데이터와 다른 국면**이면
실전 성과가 백테스트와 다를 수 있습니다.

이게 영길님의 보수 정책 (Confluence 0.85, 하루 1~2회, 5x leverage, 70 USDT max position)이
가치를 발휘하는 지점입니다 — **잘못된 모델로도 큰 손실 안 나도록 시스템 자체가 보호**합니다.
