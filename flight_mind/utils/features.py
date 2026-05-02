"""
Feature Engineering Pipeline
=============================
Raw OHLCV → 사전 계산된 지표 (features_5m, features_1h, features_1d).

설계 원칙:
  - Idempotent: 같은 데이터에 여러 번 실행해도 동일 결과
  - Vectorized: pandas + numpy로 일괄 계산 (TA-Lib 의존 회피)
  - Streaming-friendly: 신규 데이터만 증분 계산 가능

영길님의 가상 메트롤로지 경험 — 동일한 패턴을 차트에 적용한 셈.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from flight_mind.utils.db import fetch_ohlcv, get_conn


# =============================================================================
# Indicator Functions (Vectorized)
# =============================================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI — TA-Lib과 호환되는 평탄화 방식."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing — alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift()

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """거래량 z-score — 거래량 급증 탐지 (Plait Tier 1.1)"""
    rolling = volume.rolling(window, min_periods=window)
    return (volume - rolling.mean()) / rolling.std()


# =============================================================================
# Feature Builders
# =============================================================================
def build_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    """5분봉 OHLCV → features_5m 스키마"""
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["rsi_14"] = rsi(df["close"], 14)
    out["ma_7"] = df["close"].rolling(7, min_periods=7).mean()
    out["ma_30"] = df["close"].rolling(30, min_periods=30).mean()
    out["ma_120"] = df["close"].rolling(120, min_periods=120).mean()
    out["atr_14"] = atr(df, 14)
    out["volume_ma20"] = df["volume"].rolling(20, min_periods=20).mean()
    out["volume_zscore"] = volume_zscore(df["volume"], 20)
    return out


def build_features_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
    """5분봉 → 1시간봉 리샘플 + 지표"""
    df_1h = pd.DataFrame({
        "open": df_5m["open"].resample("1h").first(),
        "high": df_5m["high"].resample("1h").max(),
        "low": df_5m["low"].resample("1h").min(),
        "close": df_5m["close"].resample("1h").last(),
        "volume": df_5m["volume"].resample("1h").sum(),
    }).dropna()

    out = pd.DataFrame(index=df_1h.index)
    out["close"] = df_1h["close"]
    out["rsi_14"] = rsi(df_1h["close"], 14)
    out["ma_50"] = df_1h["close"].rolling(50, min_periods=50).mean()
    out["ma_200"] = df_1h["close"].rolling(200, min_periods=200).mean()
    out["atr_14"] = atr(df_1h, 14)
    return out


def build_features_1d(df_5m: pd.DataFrame) -> pd.DataFrame:
    """5분봉 → 일봉 + 시장 국면 지표 (Tier 4 입력)"""
    df_1d = pd.DataFrame({
        "open": df_5m["open"].resample("1D").first(),
        "high": df_5m["high"].resample("1D").max(),
        "low": df_5m["low"].resample("1D").min(),
        "close": df_5m["close"].resample("1D").last(),
        "volume": df_5m["volume"].resample("1D").sum(),
    }).dropna()

    out = pd.DataFrame(index=df_1d.index)
    out["close"] = df_1d["close"]
    out["return_1d"] = df_1d["close"].pct_change()
    out["volatility_30d"] = out["return_1d"].rolling(30, min_periods=30).std() * np.sqrt(365)
    out["rsi_14"] = rsi(df_1d["close"], 14)
    out["ma_50"] = df_1d["close"].rolling(50, min_periods=50).mean()
    out["ma_200"] = df_1d["close"].rolling(200, min_periods=200).mean()
    return out


# =============================================================================
# Persistence
# =============================================================================
def upsert_features(df: pd.DataFrame, table: str, symbol: str) -> int:
    """DataFrame을 features 테이블에 INSERT OR REPLACE."""
    df = df.dropna(how="all").copy()
    if df.empty:
        return 0

    df = df.reset_index().rename(columns={"index": "open_time"})
    df["symbol"] = symbol

    cols = ["symbol", "open_time"] + [c for c in df.columns
                                        if c not in ("symbol", "open_time")]
    df = df[cols]

    with get_conn() as conn:
        # DuckDB의 가장 안정적인 upsert 패턴: temp register + INSERT OR REPLACE
        conn.register("__staging", df)
        conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM __staging")
        conn.unregister("__staging")
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE symbol = '{symbol}'"
        ).fetchone()[0]

    return n


def rebuild_all_features(symbol: str) -> dict[str, int]:
    """심볼 하나에 대해 모든 features 테이블 재계산."""
    df_5m = fetch_ohlcv(symbol, "5m")
    if df_5m.empty:
        return {"error": f"No OHLCV data for {symbol}"}

    f_5m = build_features_5m(df_5m)
    f_1h = build_features_1h(df_5m)
    f_1d = build_features_1d(df_5m)

    return {
        "features_5m": upsert_features(f_5m, "features_5m", symbol),
        "features_1h": upsert_features(f_1h, "features_1h", symbol),
        "features_1d": upsert_features(f_1d, "features_1d", symbol),
    }


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    result = rebuild_all_features(symbol)
    print(f"Features rebuilt for {symbol}:")
    for k, v in result.items():
        print(f"  {k:20s}: {v:>10,}")
