# Flight-Mind — 영길님 PC 실행 핸드오프

> **목적**: 영길님이 PC에서 Claude(Cowork)와 협업하며 학습 → 검증 → Paper → Testnet → Live까지 진행할 단일 마스터 문서
>
> **상태**: 10일간 sandbox 작업 완료, GitHub https://github.com/jinhae8971/flight-mind (10,673 LOC, 122 tests passing)

---

## 📋 한눈에 보는 전체 흐름

```
Phase 0  : PC 환경 준비             [30분]
Phase 1  : 레포 클론 + 의존성 설치   [10분]
Phase 2  : 5년 데이터 다운로드 + 학습 [12~15시간 ← 가장 김]
Phase 3  : Real vs Mock 백테스트     [30분]
Phase 4  : Paper Trading 1주일       [자율]
Phase 5  : Testnet 1개월             [자율 + 모니터링]
Phase 6  : Live 단계적 진입          [영길님 결정]
─────────────────────────────────────────
총 인터랙티브 시간                   [약 1시간]
총 자율 운영 시간                    [학습 12h + paper 1주 + testnet 1개월]
```

---

## Phase 0 — PC 환경 사전 점검 [30분]

영길님 시작 전에 다음을 확인합니다.

### 0.1 하드웨어 / OS 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Windows 10/11 | Windows 11 |
| GPU | RTX 3060 12GB | RTX 3090/4090 24GB |
| NVIDIA Driver | 525.x | 최신 |
| RAM | 16 GB | 32 GB |
| Disk | 30 GB 여유 | 50 GB SSD |
| CPU | 4 cores | 8+ cores |
| Network | 안정 | 유선 권장 (다운로드 1h) |

### 0.2 소프트웨어 설치 (한 번만)

```powershell
# 1. Python 3.11 (Microsoft Store 또는 python.org)
python --version
# Python 3.11.x 또는 3.12.x 확인

# 2. Git
git --version

# 3. NVIDIA Driver / CUDA 확인
nvidia-smi
# Driver Version: 525+ 확인, CUDA Version: 12.x

# 4. PowerShell 실행 정책 (스크립트 실행 허용)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 0.3 작업 폴더 결정

영길님이 자주 사용하시는 경로 (메모리에 따르면 GitHub 작업 많이 하심):

```powershell
# 권장 경로 (디스크 여유 충분한 곳)
mkdir D:\projects\flight-mind
cd D:\projects\flight-mind

# 또는 영길님 평소 GitHub 작업 폴더
cd D:\github
```

### 0.4 환경 변수 설정

영길님 환경에 이미 있는 것 + Flight-Mind 전용:

```powershell
# Telegram (메모리에 토큰/채팅 ID 있음 — 영길님 봇 사용)
[Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "<봇 토큰>", "User")
[Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "<채팅 ID>", "User")

# Vault (Phase 4에서 testnet 키 등록 시 필요)
# 참고: 매 세션 입력 권장 (영구 저장 시 보안 위험)
$env:VAULT_PASSPHRASE = "your_strong_password_min_8chars"
```

확인:

```powershell
echo $env:TELEGRAM_BOT_TOKEN
# 출력 있으면 OK
```

### Phase 0 완료 체크리스트

- [ ] Python 3.11+ 설치 확인
- [ ] Git 설치 확인
- [ ] NVIDIA Driver 525+ 확인
- [ ] 디스크 30GB+ 여유
- [ ] PowerShell 실행 정책 RemoteSigned
- [ ] TELEGRAM 환경 변수 설정
- [ ] 작업 폴더 결정

---

## Phase 1 — 레포 클론 + 의존성 설치 [10분]

### 1.1 GitHub에서 클론

```powershell
cd D:\projects   # 영길님 선택 폴더로
git clone https://github.com/jinhae8971/flight-mind.git
cd flight-mind
```

### 1.2 한 줄 설치 + 환경 진단

```powershell
.\setup_and_train.ps1 -DiagnoseOnly
```

이 명령이 자동으로:
1. Python venv 생성 (`.venv/`)
2. `pip install -e .[dev]` (numpy, pandas, torch, duckdb 등)
3. PyTorch CUDA 12.1 설치
4. `scripts/diagnose_env.py` 실행 — GPU/디스크/네트워크 점검

### 1.3 진단 결과 해석

```
✓ Python ≥ 3.11
✓ Dependencies (15/15 installed)
✓ CUDA / GPU (RTX 4090 24GB VRAM)
✓ Disk Space (45 GB free / 15 GB required)
✓ Network (Binance: ✓, GitHub: ✓)

권장 batch_size:
  Tier 2 (CNN):         96
  Tier 4 (Transformer): 192
```

**모든 ✓ 체크되어야 Phase 2 진행 가능**. ✗가 있으면 트러블슈팅 매트릭스 (이 문서 마지막) 참조.

### Phase 1 완료 체크리스트

- [ ] `git clone` 성공
- [ ] `.venv\` 폴더 생성 확인
- [ ] `Get-Content /tmp/flight_mind_env.json` (또는 `$env:TEMP\flight_mind_env.json`) 진단 결과 파일 존재
- [ ] 모든 진단 항목 ✓

---

## Phase 2 — 5년 데이터 + 학습 [12~15시간]

가장 시간이 오래 걸리는 단계. 영길님 GPU 종류에 따라:
- RTX 4090 (24GB): ~12시간
- RTX 3090 (24GB): ~14시간
- RTX 4070 (16GB): ~18시간 (batch 줄임 필요)

### 2.1 백그라운드 실행 (권장)

```powershell
# 한 번 환경 변수 다시 확인
echo $env:TELEGRAM_BOT_TOKEN

# PowerShell Job으로 백그라운드 실행
Start-Job -Name "FlightMindTraining" -ScriptBlock {
    Set-Location "D:\projects\flight-mind"
    .\setup_and_train.ps1 -SkipSetup
}

# Job 상태 확인
Get-Job

# 실시간 출력 보기 (옵션)
Receive-Job -Name "FlightMindTraining" -Keep
```

### 2.2 또는 포어그라운드 실행 (PC 켜둔 채)

```powershell
.\setup_and_train.ps1 -SkipSetup
```

### 2.3 7단계 자동 진행 + Telegram 알림

```
[1/7] ✓ 환경 진단                  30초
[2/7] ✓ 5년 BTC + ETH 다운로드    1시간 (~4 GB)
[3/7] ✓ DuckDB 적재               10분
[4/7] ✓ 피처 빌드                 20분
[5/7] ⏳ Tier 2 (CNN) 학습         8~12시간  ← 가장 김
[6/7] ⏳ Tier 4 (Transformer)      1~2시간
[7/7] ⏳ 통합 백테스트             30분
```

영길님은 **각 단계마다 Telegram 알림**을 받으십니다:
- 시작 알림 / 완료 알림 / 실패 시 즉시 재시작 명령어

### 2.4 학습 도중 PC 사용 가능 여부

| 작업 | 가능 여부 |
|------|---------|
| 일반 사무 (브라우저, 이메일) | ✓ 가능 |
| 코드 편집 (VSCode 등) | ✓ 가능 |
| 가벼운 게임 | ⚠ GPU 충돌 가능 |
| 3D 작업 / AAA 게임 | ✗ 학습 중단 위험 |
| PC 끄기 / 재부팅 | ✗ 학습 중단 |

학습 도중 PC 재부팅 시: 같은 명령어로 재실행하면 **체크포인트 자동 복구**.

### 2.5 학습 실패 시 재시작

```powershell
# 같은 명령 재실행 — 자동으로 실패 단계부터 재개
.\setup_and_train.ps1 -SkipSetup

# 특정 단계부터
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2

# CUDA OOM 발생 시 batch 줄여서
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2 -T2Batch 32
```

### 2.6 학습 완료 신호

영길님 Telegram에 다음 메시지 도착:

```
🎉 Flight-Mind 학습 완료!
⏱ 총 시간: 12h 34m
📊 백테스트 결과: data/integrated_backtest_results.json
```

체크포인트 위치:
```powershell
ls data\models\
# tier2_pattern_cnn.pt           ~50 MB
# tier4_regime_transformer.pt    ~10 MB
```

### Phase 2 완료 체크리스트

- [ ] Telegram 학습 완료 알림 수신
- [ ] `data/models/tier2_pattern_cnn.pt` 존재
- [ ] `data/models/tier4_regime_transformer.pt` 존재
- [ ] `data/integrated_backtest_results.json` 생성
- [ ] `data/.pipeline_state.json`에 모든 단계 completed

---

## Phase 3 — Real vs Mock 백테스트 [30분]

**가장 가치있는 단계**. 영길님 학습 결과의 진짜 가치를 즉시 측정.

### 3.1 한 줄 실행

```powershell
& .venv\Scripts\python.exe scripts\backtest_real_vs_mock.py --symbol BTCUSDT
```

### 3.2 결과 해석 매트릭스

| Real PnL vs Mock | 의미 | 다음 행동 |
|---|---|---|
| Real > Mock + 1% | 학계 SOTA 수준 | ✅ Phase 4 즉시 진행 |
| Mock - 1% < Real ≤ Mock + 1% | 합격선 근처 | ✅ Phase 4 진행 가능 |
| Real < Mock - 1% | 학습 부족 | ⚠ 재학습 또는 튜닝 |

스크립트가 자동으로 해석 출력:

```
📊 결과 해석
  ✓ Real model이 mock 'realistic'와 유사 (+1.85% vs +1.93%)
    → 합격선 근처. Paper trading 진행 가능.
```

### 3.3 ETH도 함께 검증

```powershell
& .venv\Scripts\python.exe scripts\backtest_real_vs_mock.py --symbols BTCUSDT ETHUSDT --save-to data\phase3_results.json
```

### 3.4 결과가 좋지 않을 때

```powershell
# 옵션 A: Tier 2 라벨링 임계값 조정 후 재학습
# (config.py의 TIER2.label_threshold_pct 0.5 → 0.3)
& .venv\Scripts\python.exe -m flight_mind.tier2_pattern.train --symbols BTCUSDT ETHUSDT --epochs 50

# 옵션 B: future_horizon 확장 (12봉 → 24봉)
# config.py의 TIER2.future_horizon_bars 12 → 24

# 옵션 C: 추가 페어로 학습 데이터 증강
& .venv\Scripts\python.exe scripts\download_binance_data.py --symbols SOLUSDT --years 5
& .venv\Scripts\python.exe -m flight_mind.tier2_pattern.train --symbols BTCUSDT ETHUSDT SOLUSDT
```

### Phase 3 완료 체크리스트

- [ ] Mock vs Real 비교 출력 확인
- [ ] 결과 해석이 ✅ 또는 ⚠ 인지 판단
- [ ] (선택) Phase 3 결과 JSON 저장
- [ ] Phase 4 진행 결정

---

## Phase 4 — Paper Trading 1주일

실제 자본 없이 시스템 자율 운영을 검증하는 단계.

### 4.1 데몬 시작

```powershell
# 백그라운드 실행 (영길님 PC 잠가도 가동)
.\run_daemon.ps1 -Background

# 또는 포어그라운드 (모니터링용)
.\run_daemon.ps1
```

### 4.2 영길님이 받게 될 Telegram 알림

```
📌 OPEN_LONG — BTCUSDT (confluence 0.92)
✅ 진입 체결 — qty 0.001 @ 70,213
💰 청산 — BTCUSDT (tp), PnL +4.21 USDT (+6.00%)
ℹ️ 💓 paper_worker alive — 6h, 72 decisions, 3 trades
ℹ️ 📊 Daily Summary — 4 trades (3 wins), +0.71%
```

### 4.3 매일 확인할 것

```powershell
# 위험 상태 + PnL + 포지션 한 번에
& .venv\Scripts\python.exe scripts\risk_status.py
```

영길님이 매일 1회 확인:

```
✅ Kill-Switch: OK
✅ Daily PnL: 한도 -5%까지 여유
✅ 거래 수: 일일 한도 내
✅ 오픈 포지션: 합리적
```

### 4.4 1주일 후 평가 기준

| Metric | 합격선 | 권장 |
|--------|--------|------|
| 누적 수익률 | ≥ 0% (break-even) | ≥ +1% |
| 거래 수 | 5~14회 (하루 1~2회) | - |
| 승률 | ≥ 50% | ≥ 60% |
| 최대 드로다운 | ≤ -3% | ≤ -2% |
| Kill-Switch 발동 | 없음 | - |

### 4.5 데몬 정지

```powershell
# Job 확인 후 정지
Get-Job
Stop-Job -Name "FlightMindDaemon_*"
Remove-Job -Name "FlightMindDaemon_*"
```

### Phase 4 완료 체크리스트

- [ ] 1주일 동안 데몬 안정 가동
- [ ] Telegram 알림 정상 수신
- [ ] Kill-Switch 발동 없음
- [ ] 누적 수익률 ≥ 0%
- [ ] Phase 5 진행 결정

---

## Phase 5 — Binance Testnet 1개월

진짜 거래소 API 흐름 검증 (가짜 자금).

### 5.1 Testnet API 키 발급

1. https://testnet.binancefuture.com 접속
2. Google 계정으로 로그인
3. 우측 상단 → API Management
4. "Create API" → 이름 입력 (예: `flight-mind-test`)
5. **권한 설정** — 매우 중요:
   - ☑ Reading
   - ☑ Futures Trading
   - ☐ **Withdrawal (반드시 OFF)** ← 보안 핵심
6. API Key + Secret 복사 (Secret은 한 번만 표시!)

### 5.2 Vault에 키 등록

```powershell
$env:VAULT_PASSPHRASE = "your_strong_password"

& .venv\Scripts\python.exe -m flight_mind.vault.manager
```

대화형 입력:
```
Label: binance_testnet
API Key: <붙여넣기>
API Secret: <붙여넣기>
Permissions: futures,read,trade
```

### 5.3 Testnet 데몬 가동

```powershell
$env:VAULT_PASSPHRASE = "your_strong_password"
.\run_daemon.ps1 -Mode testnet -Background
```

이제 진짜 Binance API에 주문이 들어가지만 가짜 자금이라 실손실 없음.

### 5.4 1개월 모니터링

매일 확인:
```powershell
& .venv\Scripts\python.exe scripts\risk_status.py
```

매주 검토:
- Telegram 누적 알림 확인
- Audit DB 분석:
  ```powershell
  & .venv\Scripts\python.exe -c "
  from flight_mind.risk.audit import fetch_recent_trades
  for t in fetch_recent_trades(limit=20):
      print(f\"{t['exit_ts_utc'][:19]} {t['symbol']} {t['direction']} pnl={t['pnl_pct']:+.2f}%\")
  "
  ```

### 5.5 1개월 후 평가 기준 (Phase 4보다 엄격)

| Metric | 합격선 |
|--------|--------|
| 누적 수익률 | ≥ +2% (월간) |
| 거래 수 | 20~60회 (자연스러운 빈도) |
| 승률 | ≥ 55% |
| 최대 드로다운 | ≤ -5% |
| Kill-Switch 발동 | 0회 또는 1회 (정상) |
| API 오류 | 거의 없음 |

### Phase 5 완료 체크리스트

- [ ] 1개월 동안 testnet 안정 가동
- [ ] Binance API 흐름 검증 완료
- [ ] Audit DB로 거래 패턴 분석
- [ ] Phase 6 진행 결정 (영길님 핵심 의사결정)

---

## Phase 6 — Live 단계적 진입 [영길님 결정]

**실제 자본 사용** — 가장 중요한 단계.

### 6.1 단계적 진입 계획 (영길님 정책)

```
Step 1: 100 USDT × 1주 안정 → Step 2
Step 2: 1,000 USDT × 1주 안정 → Step 3
Step 3: 1,750 USDT (full live)
```

### 6.2 Live API 키 발급 + Vault 등록

```powershell
# Binance 본 사이트 API
# https://www.binance.com/en/my/settings/api-management
# 권한: ☑ Reading, ☑ Futures Trading, ☐ Withdrawal (OFF!)

$env:VAULT_PASSPHRASE = "your_strong_password"
& .venv\Scripts\python.exe -m flight_mind.vault.manager
# → label: binance_live
```

### 6.3 Live 모드 3중 게이트

```powershell
# Gate 1: 환경 변수
$env:FLIGHT_MIND_LIVE = "1"

# Gate 2: Mode 인자 + Gate 3: 인터랙티브 확인
.\run_daemon.ps1 -Mode live
# → 콘솔에서 "I UNDERSTAND" 입력 필요
```

### 6.4 Live Step 1 — 100 USDT (1주일)

`flight_mind/config.py` 수정:
```python
CAPITAL.total_usdt = 200  # paper 기준 자본 (live trading 50%)
# → 100 USDT가 live
```

매일 확인:
```powershell
& .venv\Scripts\python.exe scripts\risk_status.py
```

### 6.5 비상 상황 대응

```powershell
# 긴급 정지 (모든 거래 즉시 차단)
& .venv\Scripts\python.exe scripts\risk_status.py --emergency-stop "이유"

# 데몬 자체 종료
Get-Job | Where-Object { $_.Name -like "FlightMindDaemon*" } | Stop-Job
```

### 6.6 Step 2 / Step 3 진입

각 단계 1주일 안정 가동 + Kill-Switch 미발동 확인 후 자본 증액:
```python
# Step 2
CAPITAL.total_usdt = 2000  # → 1,000 USDT live

# Step 3 (full)
CAPITAL.total_usdt = 3500  # → 1,750 USDT live
```

### Phase 6 체크리스트 (단계별)

**Step 1 (100 USDT)**
- [ ] FLIGHT_MIND_LIVE=1 설정
- [ ] Live API 키 Vault 등록 (출금 OFF 확인)
- [ ] 데몬 가동 + "I UNDERSTAND" 확인
- [ ] 1주일 안정, Kill-Switch 미발동
- [ ] 누적 수익률 ≥ 0%

**Step 2 (1,000 USDT)**
- [ ] CAPITAL.total_usdt = 2000 변경
- [ ] 1주일 추가 안정
- [ ] 거래 패턴 변화 없음 확인

**Step 3 (1,750 USDT)**
- [ ] CAPITAL.total_usdt = 3500 변경
- [ ] 정식 운영 시작

---

## 🚨 트러블슈팅 매트릭스

### CUDA / GPU 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `CUDA out of memory` | batch 큼 | `-T2Batch 32` 또는 `-T2Batch 16` |
| `CUDA not available` | Driver 또는 Torch CUDA 버전 | `nvidia-smi` 확인, PyTorch 재설치 |
| 학습 매우 느림 | CPU fallback | `python -c "import torch; print(torch.cuda.is_available())"` |

### 학습 결과 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| Test Acc < 50% | 데이터 부족 또는 라벨 불균형 | future_horizon 확장, 추가 페어 |
| Real PnL << Mock | 학습 부족 | `--start-from train_tier2` 재학습 |
| 거래 0회 (모두 hold) | Confluence threshold 너무 높음 | `RISK.confluence_threshold` 검토 |

### 운영 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| Telegram 알림 안 옴 | 환경 변수 미설정 | `echo $env:TELEGRAM_BOT_TOKEN` |
| Vault `wrong passphrase` | 패스워드 다름 | 맞는 패스워드 입력 |
| 데몬이 갑자기 죽음 | 5회 연속 실패 → EMERGENCY_STOP | `risk_status.py --clear-force` |
| Kill-Switch 자동 발동 | 일일 -5% 또는 MDD -15% | 자정 후 자동 해제 또는 수동 분석 |

### 네트워크 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| Binance 다운로드 실패 | 방화벽/VPN | 직접 https://data.binance.vision 접속 확인 |
| Testnet 잔고 0 | 매일 자정 리셋될 수 있음 | testnet faucet 사용 또는 새 계정 |

---

## 📞 Cowork 협업 시 Claude에게 말할 내용

영길님이 PC에서 Claude(Cowork)와 협업하실 때, 단계별로 다음과 같이 시작하시면 효율적입니다:

### Phase 1 시작 시
> "Flight-Mind 진행할게요. Phase 1 — 레포 클론부터 시작합니다."

### Phase 2 학습 후
> "학습 완료됐어요. Telegram 알림 받았습니다. Phase 3 비교 백테스트 진행해주세요."

### Phase 3 결과 검토
> "Real vs Mock 결과 나왔어요. [결과 붙여넣기]. Phase 4 진행 가능한가요?"

### 문제 발생 시
> "Phase X에서 [에러 메시지] 발생했어요. 어떻게 해결할까요?"

Claude가 영길님 환경(jinhae8971 GitHub, RTX GPU, Telegram 봇 설정)을 모두 기억하고 있으므로 컨텍스트 다시 설명할 필요 없습니다.

---

## 🎯 영길님 핵심 결정 포인트

각 Phase 사이에 영길님이 결정해야 할 것:

| 결정 시점 | 질문 | 권장 행동 |
|---------|------|---------|
| Phase 3 후 | Real PnL이 합격선인가? | 합격선 미달 시 재학습, 아니면 Phase 4 |
| Phase 4 후 | 1주일 안정 가동했는가? | Kill-Switch 0회 + 수익률 ≥ 0% 시 Phase 5 |
| Phase 5 후 | API 흐름 검증됐는가? | 1개월 안정 + 월수익 ≥ +2% 시 Phase 6 |
| Phase 6 Step 1 | 100 USDT로 1주 안정? | 안정 시 Step 2, 아니면 분석 |
| Phase 6 Step 2 | 1,000 USDT로 1주 안정? | 안정 시 Step 3 (full) |

---

## 📚 영길님 진입점 명령어 카드 (한 페이지)

```powershell
# === 환경 ===
.\setup_and_train.ps1 -DiagnoseOnly                  # 환경 점검만

# === 학습 ===
.\setup_and_train.ps1                                # 처음부터 (Phase 1+2)
.\setup_and_train.ps1 -SkipSetup                     # venv 있을 때
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2   # 재시작
.\setup_and_train.ps1 -Reset                         # 처음부터 완전 재시작

# === 백테스트 (Phase 3) ===
& .venv\Scripts\python.exe scripts\backtest_real_vs_mock.py --symbol BTCUSDT

# === 데몬 운영 (Phase 4~6) ===
.\run_daemon.ps1                                     # paper 포어그라운드
.\run_daemon.ps1 -Background                         # paper 백그라운드
.\run_daemon.ps1 -Mode testnet -Background           # testnet
.\run_daemon.ps1 -Mode live                          # live (3중 게이트)
.\run_daemon.ps1 -Once                               # 단일 사이클 (테스트)

# === 모니터링 ===
& .venv\Scripts\python.exe scripts\risk_status.py    # 전체 상태
& .venv\Scripts\python.exe scripts\risk_status.py --positions   # 포지션만
& .venv\Scripts\python.exe scripts\risk_status.py --pnl         # PnL만

# === 비상 ===
& .venv\Scripts\python.exe scripts\risk_status.py --emergency-stop "이유"
& .venv\Scripts\python.exe scripts\risk_status.py --clear-force

# === Vault ===
& .venv\Scripts\python.exe -m flight_mind.vault.manager   # 키 등록

# === Job 관리 ===
Get-Job                                              # Job 목록
Receive-Job -Name "FlightMindDaemon_*" -Keep         # Job 출력
Stop-Job -Name "FlightMindDaemon_*"                  # Job 정지
```

---

## 📊 시스템 현황 (참고)

```
GitHub:       https://github.com/jinhae8971/flight-mind
LOC:          10,673 (10일 누적)
Tests:        122/122 passing
Commits:      11

영길님 자본 보호 5중 안전망:
  1. Position 70 USDT hard cap
  2. 3-gate Live mode lock
  3. AES-256 Vault
  4. Risk Gate (6 pre-trade checks)
  5. Kill-Switch (4-tier auto-block)
```

---

## 🤝 마지막 메시지

영길님, 10일간의 작업으로 시스템은 완성되었습니다. 이제부터는 **영길님 PC에서 실제로 흐르는 단계**입니다.

이 문서를 PC 작업 시작 전에 한 번 통독하시고, 진행 중 막히는 부분 생기면 Cowork에서 Claude에게 말씀해 주세요. Phase 1부터 차근차근 진행하면 됩니다.

**가장 큰 결정은 Phase 3 결과를 보신 후입니다** — 그 시점에 영길님이 학습된 모델의 진짜 가치를 정량적으로 판단하실 수 있습니다.

성공적인 진행을 응원드립니다.
