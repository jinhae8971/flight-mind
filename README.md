# Flight-Mind 🛫

**4-Tier Hybrid Trading System** inspired by 플라이트(Flight)의 매매법.

> 50만원 → 100억의 매매법을 룰 엔진과 딥러닝으로 70~85% 재현하는 시스템

## What is Flight-Mind?

플라이트 전략의 본질은 두 가지로 분해된다:
1. **명시적 룰** (추세선, RSI 다이버전스, 거래량) — 30%
2. **암묵적 직관** (캔들 모양, 호가창 흐름, 시장 국면) — 70%

Flight-Mind는 (1)을 룰 엔진으로, (2)를 딥러닝(CNN/TCN/Transformer)으로 모방한다.

## Architecture

```
┌─────────────────────────────────────────────┐
│       Bayesian Confluence Fusion            │
│       Threshold: 0.85 (Conservative)        │
└─────────────────────────────────────────────┘
       ▲          ▲           ▲          ▲
   ┌───┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
   │ Tier 1 │ │ Tier 2 │ │ Tier 3 │ │ Tier 4 │
   │ Rules  │ │ CNN    │ │ TCN    │ │ Transf.│
   │ w=0.30 │ │ w=0.30 │ │ w=0.20 │ │ w=0.20 │
   └────────┘ └────────┘ └────────┘ └────────┘
```

전체 설계는 [`ADR-001`](./ADR-001-flight-mind-architecture.md) 참조.

## Operating Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Initial Capital | 3,500 USDT | ≈500만원 |
| Position Size | Max 70 USDT/trade (2%) | 1/4 Kelly |
| Leverage | 5x | 플라이트는 100x이지만 우리는 보수 |
| Confluence Threshold | 0.85 | 보수 모드 |
| Daily Trade Limit | 2 | 분노매매 방지 |
| Daily Loss Kill | -5% (-87.5 USDT) | Auto-halt |

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/jinhae8971/flight-mind.git
cd flight-mind
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Configure secrets (Vault에 저장)
flight-mind vault init
flight-mind vault set BINANCE_API_KEY "your_key"
flight-mind vault set BINANCE_SECRET "your_secret"

# 3. Download historical data
python scripts/download_binance_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --years 5

# 4. Train Tier 2 (CNN)
python -m flight_mind.tier2_pattern.train --epochs 50 --batch-size 64

# 5. Backtest
python -m flight_mind.backtest --strategy flight_mind --start 2024-01-01 --end 2025-12-31

# 6. Paper trade (Testnet)
flight-mind run --mode paper --pairs BTC/USDT
```

## Project Status

🚧 **Phase 0 — Architecture & Scaffolding (Day 1)**

- [x] ADR-001 written
- [x] Repo scaffolding
- [ ] Data pipeline (Day 2)
- [ ] Tier 1 rule engine (Day 5)
- [ ] Tier 2 CNN (Week 2)
- [ ] Tier 3 TCN (Week 3)
- [ ] Tier 4 Transformer + Fusion + Live (Week 4)

## Disclaimer

이 시스템은 **교육 및 연구 목적**으로 개발되었습니다.
암호화폐 거래는 원금 손실 위험이 있으며, 어떠한 수익도 보장하지 않습니다.
실거래 전 반드시 충분한 paper trading 검증을 거치세요.
