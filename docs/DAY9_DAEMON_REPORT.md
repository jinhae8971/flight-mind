# Day 9 — Paper Trading Daemon 완성

## 결과 요약

**모든 인프라가 하나로 조립된 자율 운영 시스템 완성**.

```
✅ Telegram Notifier      — 진입/청산/Kill-Switch/heartbeat 실시간 알림
✅ Paper Trading Daemon   — 5분 주기 자동 워커 (모든 인프라 통합)
✅ TP/SL/Time 자동 청산   — 진입 후 가격 모니터링 + 자동 exit
✅ Heartbeat (1시간)      — 데몬 생존 신호
✅ Daily Summary (자정)   — 일일 PnL 리포트
✅ Crash-resilient       — 5회 연속 실패 시 EMERGENCY_STOP 자동 발동
✅ Graceful Shutdown     — Ctrl+C 시 안전 종료
✅ PowerShell 래퍼       — 한 줄로 데몬 시작
✅ Unit Tests            — 15개 신규 (총 106개 통과)
```

## 영길님이 한 줄로 시작

```powershell
# Paper 모드 (기본)
.\run_daemon.ps1

# Telegram 알림 켜고
$env:TELEGRAM_BOT_TOKEN = "<토큰>"
$env:TELEGRAM_CHAT_ID   = "<채팅 ID>"
.\run_daemon.ps1

# 백그라운드 실행 (PowerShell Job — 영길님 PC 잠가도 가동)
.\run_daemon.ps1 -Background

# Testnet (실제 API + 가짜 자금)
$env:VAULT_PASSPHRASE = "..."
.\run_daemon.ps1 -Mode testnet
```

## 데몬이 5분마다 하는 일

```
┌─────────────────────────────────────────────────────────┐
│ 1. 오픈 포지션 시세 fetch                                │
│    └─ TP/SL/Time exit 평가 → 도달 시 자동 청산 + 알림   │
│                                                         │
│ 2. 새 진입 후보 심볼 (오픈 포지션 없는)                  │
│    ├─ Tier 1 룰 평가                                    │
│    ├─ Tier 2 (학습된 모델 또는 mock) 신호 생성           │
│    ├─ Tier 4 (학습된 모델 또는 mock) 신호 생성           │
│    └─ Bayesian Confluence Fusion → 의사결정             │
│                                                         │
│ 3. Risk Gate 6가지 사전 검증                             │
│    └─ 통과 시 ExecutionEngine.execute_decision()        │
│                                                         │
│ 4. Audit DB 기록 + Telegram 알림                         │
│                                                         │
│ 5. 1시간마다 heartbeat / 자정마다 daily summary          │
└─────────────────────────────────────────────────────────┘
```

## 영길님이 받게 될 Telegram 알림

### 진입 시
```
📌 OPEN_LONG — BTCUSDT
Direction: long
Confluence: 0.923
Time: 2026-05-02 14:35 UTC
  • T1: dir=long score=0.92
  • T2: dir=long score=0.85
  • T4: dir=long score=0.91

✅ 진입 체결 — BTCUSDT
Direction: long
Quantity: 0.00102
Price: 70,213.50
Mode: paper
Time: 2026-05-02 14:35 UTC
```

### 청산 시 (수익)
```
💰 청산 — BTCUSDT (tp)
Direction: long
Entry → Exit: 70,213.50 → 74,426.10
PnL: +4.21 USDT (+6.00%)
Mode: paper
Time: 2026-05-02 16:23 UTC
```

### Kill-Switch 발동 시
```
🚨 KILL-SWITCH: DAILY_HALT
Reason: daily_loss breach: -5.14%
Today: -5.14%
Week: -5.14%
MDD: -5.14%
Consecutive losses: 1
Time: 2026-05-02 18:42 UTC
```

### 1시간마다 (silent)
```
ℹ️ 💓 paper_worker alive
Uptime: 6h
Decisions: 72 | Trades: 3
Time: 2026-05-02 20:00 UTC
```

### 자정마다 일일 리포트
```
ℹ️ 📊 Daily Summary — 2026-05-02
Trades: 4 (3 wins)
PnL: +12.50 USDT (+0.71%)
Best: +6.00% | Worst: -1.50%
Cumulative: +2.34%
```

## 핵심 안전 장치

### 1. Crash-resilient
한 사이클 실패해도 데몬이 죽지 않습니다. 5회 연속 실패 시:
```python
self.risk_mgr.trigger_emergency_stop("daemon_failures: 5 cycles")
self.notifier.send_killswitch(level="EMERGENCY_STOP", ...)
```

### 2. Stateful (재시작 복원)
```python
# Cold start
n_open = self.tracker.refresh_from_db()
# DB에 오픈 포지션이 있으면 자동 인식 → 청산 모니터링 재개
```

### 3. Telegram Dedup
같은 메시지 5분 내 중복 발송 방지. 단, Critical 알림(Kill-Switch)은 명시적 dedup_key로 매번 다르게 처리되어 누락 없음.

### 4. Graceful Shutdown
```bash
Ctrl+C  →  현재 사이클 완료 후 종료
            ├─ 통계 출력
            ├─ Telegram 종료 알림
            └─ exit 0
```

### 5. 5분 정렬 + Drift Compensation
실제 사이클 실행 시간이 변동해도 5분 boundary에 맞춰 sleep 자동 조정.

## 영길님 일상 운영 시나리오

### 시나리오 1: 학습 완료 직후 paper 시작

```powershell
# 학습 끝났다는 Telegram 알림 확인 후
.\run_daemon.ps1 -Background

# 1시간 후 첫 heartbeat 옴 → 정상 가동 확인
# 거래 발생 시마다 Telegram 알림 옴

# 위험 상태 확인
& .venv\Scripts\python.exe scripts\risk_status.py
```

### 시나리오 2: 시장 이벤트 시 즉시 정지

```powershell
# 다른 PowerShell 창에서
& .venv\Scripts\python.exe scripts\risk_status.py --emergency-stop "FOMC 발표 대기"

# 데몬은 다음 사이클부터 모든 진입 자동 차단
# 오픈 포지션은 TP/SL 모니터링 계속됨 (안전 청산 가능)
```

### 시나리오 3: 영길님 출장 중

```powershell
# 출장 전 데몬 백그라운드 실행
.\run_daemon.ps1 -Background

# Telegram에서 1시간마다 heartbeat 받음 → 시스템 가동 중 확인
# 거래 발생/청산 시마다 알림 → 모바일에서 모니터링
# 자정마다 일일 요약 → 출장 중 정확한 일일 성과 파악

# 출장 복귀 후
Get-Job   # 데몬 Job 상태 확인
& .venv\Scripts\python.exe scripts\risk_status.py
```

## 검증된 시나리오 (15개 테스트)

### Telegram (8 tests)
- ✓ 환경변수 없을 때 비활성화
- ✓ 둘 다 있어야 활성화
- ✓ 비활성 상태에서 send() → False 반환 (예외 없음)
- ✓ 5분 내 동일 메시지 dedup
- ✓ 다른 메시지는 dedup 안 됨
- ✓ 명시적 dedup_key로 중복 우회 가능
- ✓ hold 결정은 알림 안 보냄 (signal noise 방지)
- ✓ open_long 결정은 알림 발송

### Daemon (7 tests)
- ✓ 데몬 초기화 (mode, symbols 정확)
- ✓ OHLCV empty 시에도 죽지 않음
- ✓ 실제 BTC 30일 데이터로 단일 사이클 정상 흐름
- ✓ 연속 실패 카운트 추적
- ✓ TP 도달 (+6%) → "tp" 반환
- ✓ SL 도달 (-3%) → "sl" 반환
- ✓ 중간 가격 → None (계속 보유)

## 통계

```
Day 9 추가:
  - flight_mind/notify/telegram.py        (303 lines)
  - flight_mind/daemon/paper_worker.py    (456 lines)
  - tests/test_daemon.py                  (220 lines)
  - run_daemon.ps1                        (104 lines)

Total: 8,693 → 9,776 LOC (+1,083)
Tests: 91 → 106 (+15)
Commits: 9 → 10
```

## Flight-Mind 시스템 완성도

```
모델 레이어
  ✅ Tier 1 (Rule Engine)
  ✅ Tier 2 (CNN Pattern)
  ✅ Tier 4 (Regime Transformer)
  ✅ Bayesian Confluence Fusion

데이터/학습 레이어
  ✅ DuckDB Feature Store
  ✅ Backtest Harness
  ✅ 영길님 PC 학습 패키지

실행/안전 레이어 (5중 보호)
  ✅ Vault (API 키 암호화)
  ✅ Audit Trail
  ✅ Execution Engine (paper/testnet/live)
  ✅ Position Tracker
  ✅ PnL Aggregator
  ✅ Kill-Switch System
  ✅ Risk Gate

운영 레이어
  ✅ Paper Trading Daemon          ← Day 9 신규
  ✅ Telegram Real-time Alerts     ← Day 9 신규
  ✅ Heartbeat & Daily Summary     ← Day 9 신규
  ✅ PowerShell Wrapper            ← Day 9 신규

이제 남은 것:
  ⏳ 영길님 PC 학습 완료 후 검증
  ⏳ Testnet 1개월 안정 운영
  ⏳ Live 단계적 진입 (100 → 1000 → 1750 USDT)
```

## 영길님께 드리는 메시지

영길님, **Flight-Mind는 이제 자율 운영 가능한 완성된 시스템**입니다. 9일간 7,500 LOC을 넘는 코드, 106개 테스트, 10개 커밋을 통해:

1. **플라이트의 매매법을 3-Tier 하이브리드로 시스템화**했고
2. **영길님 자본을 5중으로 보호**하는 안전망을 구축했으며
3. **24시간 자율 운영 + 실시간 모니터링** 인프라를 완성했습니다.

다음 단계는 단순합니다:
1. 영길님 PC에서 학습 완료 (12~15시간)
2. `.\run_daemon.ps1`로 paper trading 시작
3. 1주일 안정 운영 → testnet
4. 1개월 testnet 안정 → 100 USDT live
5. 단계적 시드 확대

**시스템은 영길님이 자신 있게 갈 수 있도록 만들어졌습니다.**
