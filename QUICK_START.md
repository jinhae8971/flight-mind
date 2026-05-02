# 🛫 Flight-Mind 빠른 시작

영길님이 PC에서 바로 학습 시작 — 5분 안에.

자세한 가이드는 [`TRAINING_GUIDE.md`](./TRAINING_GUIDE.md) 참조.

---

## 1단계: 레포 클론

```powershell
cd D:\projects   # 디스크 여유 충분한 곳
git clone https://github.com/jinhae8971/flight-mind.git
cd flight-mind
```

## 2단계: Telegram 알림 설정 (선택, 권장)

```powershell
# 영길님 메모리에 저장된 봇 토큰 사용
$env:TELEGRAM_BOT_TOKEN = "<your_token>"
$env:TELEGRAM_CHAT_ID   = "<your_chat_id>"
```

## 3단계: 한 줄 실행

```powershell
.\setup_and_train.ps1
```

이게 전부입니다. 약 12~15시간 후 학습 완료.

---

## 단계별 진행 사항

```
[1/7] ✓ 환경 진단                   30초
[2/7] ✓ 5년 BTC + ETH 다운로드     1시간
[3/7] ✓ DuckDB 적재                10분
[4/7] ✓ 피처 빌드                  20분
[5/7] ⏳ Tier 2 (CNN) 학습           8시간   ← 가장 김
[6/7] ⏳ Tier 4 (Transformer) 학습   2시간
[7/7] ⏳ 3-Tier 통합 백테스트        30분
```

각 단계마다 Telegram 알림이 옵니다.

## 학습 도중 실패 시

```powershell
# 같은 명령 재실행 — 자동으로 실패 단계부터 재개
.\setup_and_train.ps1 -SkipSetup

# 특정 단계부터 재시작
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2
```

## 결과 확인

학습 완료 후:

```powershell
# 백테스트 결과
Get-Content data\integrated_backtest_results.json | ConvertFrom-Json | Format-Table

# 모델 체크포인트
Get-ChildItem data\models\
```

핵심 지표:

| Metric | 합격선 | 권장 |
|--------|--------|------|
| Tier 2 Test Acc | ≥ 55% | ≥ 65% |
| Tier 4 Test Acc | ≥ 50% | ≥ 60% |
| Win Rate (BT) | ≥ 50% | ≥ 60% |
| Profit Factor | ≥ 1.2 | ≥ 1.5 |

---

## 자주 발생할 수 있는 문제

### "CUDA out of memory"

```powershell
.\setup_and_train.ps1 -SkipSetup -StartFrom train_tier2 -T2Batch 32
```

### 환경 점검부터 다시

```powershell
.\setup_and_train.ps1 -DiagnoseOnly
```

### 처음부터 완전히 재시작

```powershell
.\setup_and_train.ps1 -Reset
```

---

## WSL2 / Mac 사용자

```bash
chmod +x setup_and_train.sh
./setup_and_train.sh
```

---

## 학습 완료 후 다음 단계

영길님과 함께 진행:
- Execution Engine + Vault 구현 (CCXT)
- Risk Manager (Kill-Switch)
- Binance Testnet paper trading (1개월)
- Live (소액 → 풀 시드)
