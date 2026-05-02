# Day 7 — Execution Engine + Vault 구현 완료

## 결과 요약

**영길님이 학습 완료 직후 paper trading을 시작할 수 있는 인프라 완성**.

```
✅ Vault          — AES-256-GCM API 키 암호화 (PBKDF2 480k iter)
✅ Audit Logger   — 모든 결정/주문/거래 SQLite 영구 기록
✅ Execution Engine — CCXT 기반 paper/testnet/live 통합
✅ 3중 안전 게이트 — Live 모드 차단 (env + arg + interactive)
✅ Position Sizing — 영길님 정책 강제 (70 USDT max, 5x leverage)
✅ Demo Script    — 학습 후 즉시 검증 가능
✅ Unit Tests     — 20개 신규 (총 65개 통과)
```

## 핵심 안전 장치

### 1. 3중 게이트로 라이브 주문 보호

```python
# 영길님이 의도하지 않은 라이브 주문은 절대 발생 안 함
ExecutionEngine(mode=Mode.LIVE)

# Gate 1: 환경변수 체크
if os.getenv("FLIGHT_MIND_LIVE") != "1":
    raise RuntimeError("Live mode 차단")

# Gate 2: 인자 명시
mode == Mode.LIVE  # 명시적 enum 값

# Gate 3: 인터랙티브 확인
input("실거래 진행 확인 (정확히 'I UNDERSTAND' 입력): ")
```

### 2. Vault — API 키 암호화

```python
from flight_mind.vault.manager import Vault
import os

# 패스워드 환경변수 (코드 어디에도 저장 안 됨)
os.environ["VAULT_PASSPHRASE"] = "영길님_강력한_패스워드"

vault = Vault()
vault.add(
    label="binance_testnet",
    api_key="...",
    secret="...",
    permissions=["futures", "read", "trade"],  # 출금 권한 OFF 권장
)

# 사용 시
cred = vault.get("binance_testnet")
# cred.api_key, cred.secret 만 메모리에서 평문
```

**보안 특성**:
- AES-256-GCM (인증 암호화 — 변조 자동 감지)
- PBKDF2-SHA256 480,000 iter (OWASP 2023 권장)
- 잘못된 패스워드는 `VaultLockedError` 즉시 발생, 재시도 제한 없음

### 3. Position Sizing — 영길님 정책 자동 강제

```python
# 영길님 자본이 100만 USDT여도, 1만 USDT여도, 70 USDT 초과 진입 불가능
sizing = engine.compute_position_size(symbol, available_usdt)
# notional_usdt: max(70 USDT × 5x leverage = 350 USDT)
```

이 hard cap은 코드 레벨에서 강제되며, 잘못된 모델 신호로도 큰 손실 발생 안 함.

## 영길님이 학습 완료 후 실행할 절차

### Step 1: Binance Testnet API 키 생성

1. https://testnet.binancefuture.com 접속
2. 우측 상단 → API Management
3. "Create API" 클릭, 이름 입력 (예: `flight-mind-test`)
4. **권한 설정**: ☑ Reading, ☑ Futures Trading, ☐ Withdrawal (출금 OFF)
5. API Key + Secret 복사 (Secret은 한 번만 표시됨)

### Step 2: Vault에 키 등록

```powershell
# PowerShell
$env:VAULT_PASSPHRASE = "your_strong_password_min_8_chars"

# 대화형 등록
& .venv\Scripts\python.exe -m flight_mind.vault.manager
# → label: binance_testnet
# → API Key: <붙여넣기>
# → Secret: <붙여넣기>
# → Permissions: futures,read
```

### Step 3: Paper Trading 단일 사이클 검증

```powershell
# Mock 모드 (학습 안 됐을 때, sandbox 검증과 동일)
& .venv\Scripts\python.exe scripts\demo_paper_trading.py --symbol BTCUSDT --mock

# 학습된 모델 사용
& .venv\Scripts\python.exe scripts\demo_paper_trading.py --symbol BTCUSDT
```

### Step 4: Audit DB 검토

```powershell
# 최근 결정 확인
& .venv\Scripts\python.exe -c "
from flight_mind.risk.audit import fetch_recent_decisions
for d in fetch_recent_decisions(limit=10):
    print(f\"{d['ts_utc'][:19]} {d['symbol']} {d['action']:<12} conf={d['confluence']:.3f}\")
"
```

또는 SQLite GUI (DB Browser for SQLite 등) 사용 — 파일: `data/audit.db`

## 운영 모드 비교

| 항목 | Paper | Testnet | Live |
|------|-------|---------|------|
| 실제 주문 | ❌ | ✅ (가짜 자금) | ✅ (영길님 자본) |
| 거래소 API 호출 | ❌ | ✅ | ✅ |
| Audit 기록 | ✅ | ✅ | ✅ |
| 시장 시세 | DuckDB 5분봉 | 실시간 | 실시간 |
| 환경변수 필요 | 없음 | `VAULT_PASSPHRASE` | + `FLIGHT_MIND_LIVE=1` |
| Vault 등록 필요 | 없음 | `binance_testnet` | + `binance_live` |
| 권장 사용 시기 | 학습 직후 | 1개월 검증 | 1개월 안정 후 |

## Day 5 백테스트와 어떻게 연결되는가

Day 5에서 sandbox에서 mock signal 기반 통합 백테스트로 +1.93%를 측정했습니다.
오늘 만든 인프라로 영길님이 다음을 할 수 있습니다:

```
1. 영길님 PC에서 학습 (12~15시간) — Day 6 패키지
   └─ data/models/tier2_pattern_cnn.pt
   └─ data/models/tier4_regime_transformer.pt

2. demo_paper_trading.py 단일 사이클 검증 (1분)
   └─ 학습된 모델로 실제 결정 흐름 확인

3. Paper trading 자동화 (다음 작업)
   └─ 5분마다 단일 사이클 실행 → 실시간 paper PnL

4. Binance Testnet 연동 (vault + execution)
   └─ 1개월 동안 진짜 API 흐름 검증

5. Live 시작 (소액 → 풀 시드)
```

## Audit DB 스키마 (분석용)

학습 완료 후 영길님이 실거래 데이터로 직접 분석 가능:

```sql
-- 가장 정확했던 Tier 시그널 분석
SELECT
    json_extract(d.tier_outputs, '$.T2.direction') as t2_dir,
    json_extract(d.tier_outputs, '$.T4.direction') as t4_dir,
    AVG(t.pnl_pct) as avg_pnl_pct,
    COUNT(*) as n_trades
FROM decisions d
JOIN trades t ON t.decision_id = d.id
WHERE d.action != 'hold'
GROUP BY t2_dir, t4_dir
ORDER BY avg_pnl_pct DESC;
```

## 기술적 구현 디테일

### Vault 보안 분석
- **공격 시나리오 1: 디스크 절도** → vault.json만 가져가면 무용. PBKDF2 480k iter로 brute force 1조년+
- **공격 시나리오 2: 메모리 덤프** → 평문 키가 메모리에 잠시 존재. Python GC가 빠르게 정리하지만 100% 안전 아님
- **공격 시나리오 3: 환경변수 노출** → 세션 종료 시 사라짐. 영길님이 `$env:VAULT_PASSPHRASE`를 PROFILE에 영구 저장하면 위험 ↑
- **권장**: 패스워드는 매 세션 입력. 영구 저장 시 Windows Credential Manager 사용.

### Audit Trail Immutability
- `update_order_status()`만 같은 row 갱신 허용 (status: pending → filled)
- 그 외 모든 INSERT는 append-only
- WAL journal mode로 동시 읽기/쓰기 안전

## 통계

```
Day 7 추가:
  - flight_mind/vault/manager.py        (304 lines)
  - flight_mind/risk/audit.py           (245 lines)
  - flight_mind/execution/engine.py     (393 lines)
  - tests/test_execution.py             (236 lines)
  - scripts/demo_paper_trading.py       (148 lines)

Total: 5,691 → 7,017 LOC (+1,326)
Tests: 45 → 65 (+20)
Commits: 7 → 8
```

## 다음 작업 옵션

1. **Risk Manager 구현** — Kill-Switch 자동화, Daily PnL 모니터링, 포지션 온라인 시뮬레이션
2. **Paper Trading 데몬** — 5분마다 자동 실행되는 worker (영길님 PC 또는 GitHub Actions)
3. **Telegram 알림 통합** — 실거래 의사결정/체결 실시간 알림
4. **Backtest with Real Models** — 학습 완료 후 mock 대신 실제 모델로 재백테스트
5. **임원 보고용 PPT** — 프로젝트 전체 조망 (deck-builder)

영길님 PC에서 학습 진행 상황 알려주시면 그에 맞춰 진행하겠습니다.
