"""
Flight-Mind 시스템 설정 — 영길님 결정사항 인코딩

- GPU: RTX 3090/4090 (24GB VRAM 가정)
- Capital: 3,500 USDT
- Confluence Threshold: 0.85 (Conservative)
"""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Paths
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
VAULT_PATH = ROOT / ".vault.enc"

for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Capital & Risk
# =============================================================================
class CapitalConfig(BaseModel):
    """영길님 결정: 3,500 USDT 소액 실전 검증"""
    total_usdt: float = 3500.0
    live_trading_pct: float = 0.50    # 1,750 USDT
    reserve_pct: float = 0.30          # 1,050 USDT
    cold_wallet_pct: float = 0.20      # 700 USDT (Cold storage)

    # Position sizing
    max_position_pct: float = 0.02     # per-trade 2% = 70 USDT max
    kelly_fraction: float = 0.25       # 1/4 Kelly (conservative)
    leverage: int = 5                  # 플라이트는 100x이지만 우리는 5x

    # Stop loss / take profit
    stop_loss_pct: float = -3.0
    take_profit_pct: float = 6.0       # 손익비 2:1
    max_hold_bars_5m: int = 12         # 1시간 후 자동 청산


class RiskConfig(BaseModel):
    """Kill-switch 룰 — 영길님의 ct-agent-ultra 패턴 확장"""
    daily_loss_pct: float = -5.0       # -87.5 USDT 도달 시 24h 쿨다운
    weekly_loss_pct: float = -10.0
    max_drawdown_pct: float = -15.0    # 도달 시 셧다운, 수동 컨펌

    daily_trade_limit: int = 2         # 영길님 결정: 하루 1~2회 진입
    cooldown_after_liquidation_h: int = 24


# =============================================================================
# Tier Weights & Thresholds
# =============================================================================
class FusionConfig(BaseModel):
    """Bayesian Confluence Layer — 영길님 결정: 0.85 (Conservative)"""
    threshold: float = 0.85

    # Tier weights — sum to 1.0
    w_tier1_rule: float = 0.30
    w_tier2_pattern: float = 0.30
    w_tier3_microstr: float = 0.20
    w_tier4_regime: float = 0.20

    # Disagreement penalty
    require_t1_t2_agree: bool = True   # T1, T2 부호 반대 시 강제 hold


# =============================================================================
# Per-Tier Configs
# =============================================================================
class Tier1Config(BaseModel):
    """Rule Engine — 4 sub-rules"""
    timeframes: list[str] = ["5m", "15m", "1h", "4h"]
    rsi_period: int = 14
    ma_periods: list[int] = [7, 30, 120]
    trendline_lookback: int = 60       # bars
    volume_spike_factor: float = 1.5   # 최근 20봉 평균 대비


class Tier2Config(BaseModel):
    """Pattern Memory CNN — GAF + ResNet-18"""
    backbone: str = "resnet18"
    input_window: int = 60             # 60 캔들
    image_size: int = 60               # GAF는 정사각형, input_window와 일치 필수
    output_classes: int = 3            # long / short / neutral

    # Training
    epochs: int = 50
    batch_size: int = 64               # RTX 4090 24GB 기준
    lr: float = 3e-4
    weight_decay: float = 1e-5

    # Label
    future_horizon_bars: int = 12      # 1시간 후 수익률
    long_threshold: float = 0.005      # +0.5%
    short_threshold: float = -0.005

    # Plait fine-tuning
    plait_log_weight: float = 5.0


class Tier3Config(BaseModel):
    """Microstructure TCN"""
    lob_levels: int = 10               # 10단계 bid/ask
    snapshot_window: int = 100         # 직전 100 스냅샷
    sample_interval_ms: int = 500      # 50ms → 500ms 다운샘플 (저장 부담 완화)
    prediction_horizon_ms: int = 2000  # 2초 후 예측

    tcn_layers: int = 6
    tcn_kernel: int = 3
    tcn_channels: int = 64

    epochs: int = 30
    batch_size: int = 256


class Tier4Config(BaseModel):
    """Market Regime Transformer"""
    timeframes: list[str] = ["1d", "4h"]
    lookback_days: int = 30
    n_indicators: int = 34             # 영길님 risk-regime-monitor 활용
    regimes: list[str] = ["Bull-Trending", "Bear-Trending", "Range-Bound",
                           "High-Vol-Range", "Crash"]

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    epochs: int = 40
    batch_size: int = 128


# =============================================================================
# Exchange Config
# =============================================================================
class ExchangeConfig(BaseModel):
    name: Literal["binance"] = "binance"
    market_type: Literal["future", "spot"] = "future"  # USDT-M 선물
    pairs: list[str] = ["BTC/USDT", "ETH/USDT"]   # 영길님 결정: BTC + ETH 2-pair
    testnet: bool = True               # Phase 1은 Testnet


# =============================================================================
# Master Settings
# =============================================================================
class Settings(BaseSettings):
    """환경변수 또는 .env에서 읽는 비밀 정보"""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Vault — actual keys live in encrypted vault, not env
    vault_password: str = Field(default="", alias="VAULT_PASSWORD")

    # Telegram
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # Modes
    mode: Literal["backtest", "paper", "live"] = "paper"
    log_level: str = "INFO"


# =============================================================================
# Singleton
# =============================================================================
CAPITAL = CapitalConfig()
RISK = RiskConfig()
FUSION = FusionConfig()
TIER1 = Tier1Config()
TIER2 = Tier2Config()
TIER3 = Tier3Config()
TIER4 = Tier4Config()
EXCHANGE = ExchangeConfig()
SETTINGS = Settings()


def summary() -> str:
    """현재 설정 요약 출력"""
    return f"""
Flight-Mind Configuration Summary
==================================
Capital      : {CAPITAL.total_usdt} USDT (live: {CAPITAL.total_usdt * CAPITAL.live_trading_pct})
Position Max : {CAPITAL.max_position_pct * 100}% per trade ({CAPITAL.total_usdt * CAPITAL.max_position_pct} USDT)
Leverage     : {CAPITAL.leverage}x
Stop / TP    : {CAPITAL.stop_loss_pct}% / +{CAPITAL.take_profit_pct}%

Confluence   : {FUSION.threshold} (Conservative)
Tier Weights : T1={FUSION.w_tier1_rule}, T2={FUSION.w_tier2_pattern}, T3={FUSION.w_tier3_microstr}, T4={FUSION.w_tier4_regime}

Daily Limit  : {RISK.daily_trade_limit} trades / day
Kill-Switch  : daily {RISK.daily_loss_pct}% / weekly {RISK.weekly_loss_pct}% / max-DD {RISK.max_drawdown_pct}%

Mode         : {SETTINGS.mode}
Pairs        : {EXCHANGE.pairs}
Testnet      : {EXCHANGE.testnet}
"""


if __name__ == "__main__":
    print(summary())
