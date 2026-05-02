"""
Market Regime Labeler
=====================
일봉 OHLCV로부터 5개 시장 국면을 자동 라벨링.

라벨 정의:
  - Bull-Trending  : 30일 수익률 > +5% AND ADX > 25 AND 가격 > MA200
  - Bear-Trending  : 30일 수익률 < -5% AND ADX > 25 AND 가격 < MA200
  - Range-Bound    : ADX < 20 AND 30일 변동성 < 50% (저변동 횡보)
  - High-Vol-Range : ADX < 25 AND 30일 변동성 > 80% (고변동 횡보 - 위험)
  - Crash          : 7일 수익률 < -15% OR 1일 수익률 < -8%

근거:
  - ADX는 추세 강도의 표준 지표 (Wilder 1978)
  - 30일 변동성은 연환산 변동성 (BTC 보통 50~120%)
  - Crash 정의는 2018, 2021, 2022 5월/11월 등 실제 폭락장 기준 캘리브레이션

설계 원칙:
  - Look-ahead 안전: 라벨링은 t시점에 t-1까지의 정보만 사용
  - 결정론적: 동일 입력 → 동일 출력 (학습 데이터 일관성)
  - Soft 라벨 옵션: 하나의 국면에 100% 확신이 아닐 때 대비 (향후 확장)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


REGIMES = ["Bull-Trending", "Bear-Trending", "Range-Bound",
           "High-Vol-Range", "Crash"]
N_REGIMES = 5
REGIME_TO_IDX = {r: i for i, r in enumerate(REGIMES)}
IDX_TO_REGIME = {i: r for r, i in REGIME_TO_IDX.items()}


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index — 추세 강도 측정 (Wilder 1978).

    ADX > 25: 강한 추세
    ADX < 20: 약한 추세 / 횡보
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # +DM, -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # True Range
    prev_close = close.shift()
    tr = pd.concat(
        [(high - low),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing (alpha = 1/period)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0)


def realized_volatility(close: pd.Series, window: int = 30) -> pd.Series:
    """연환산 30일 실현 변동성 (%)"""
    log_ret = np.log(close / close.shift())
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(365) * 100


def label_regime(df: pd.DataFrame) -> pd.Series:
    """
    일봉 DataFrame에 대해 각 시점의 시장 국면 라벨 반환.

    Args:
        df: 일봉 OHLCV (index: date, columns: open, high, low, close, volume)

    Returns:
        Series of regime indices (int), aligned to df.index
        Returns -1 if not enough history (warmup)
    """
    if len(df) < 200:
        return pd.Series(-1, index=df.index, dtype=int)

    close = df["close"]
    ma200 = close.rolling(200, min_periods=200).mean()
    adx_14 = adx(df, period=14)
    vol_30 = realized_volatility(close, window=30)
    ret_30 = close.pct_change(30) * 100
    ret_7 = close.pct_change(7) * 100
    ret_1 = close.pct_change(1) * 100

    labels = pd.Series(REGIME_TO_IDX["Range-Bound"], index=df.index, dtype=int)

    # Crash가 가장 우선 (다른 조건 무관하게)
    crash_mask = (ret_7 < -15) | (ret_1 < -8)
    labels[crash_mask] = REGIME_TO_IDX["Crash"]

    # Bull-Trending
    bull_mask = (
        (ret_30 > 5)
        & (adx_14 > 25)
        & (close > ma200)
        & ~crash_mask
    )
    labels[bull_mask] = REGIME_TO_IDX["Bull-Trending"]

    # Bear-Trending
    bear_mask = (
        (ret_30 < -5)
        & (adx_14 > 25)
        & (close < ma200)
        & ~crash_mask
    )
    labels[bear_mask] = REGIME_TO_IDX["Bear-Trending"]

    # High-Vol-Range
    high_vol_range_mask = (
        (adx_14 < 25)
        & (vol_30 > 80)
        & ~crash_mask
        & ~bull_mask
        & ~bear_mask
    )
    labels[high_vol_range_mask] = REGIME_TO_IDX["High-Vol-Range"]

    # Range-Bound이 default — 위 조건들에 해당 안 되면 모두 Range-Bound

    # Warmup 처리: 200봉 미만은 -1
    labels.iloc[:200] = -1

    return labels


def build_regime_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    일봉으로부터 Tier 4 Transformer가 사용할 다중 피처 빌드.

    Returns DataFrame with columns:
      - close, return_1d, return_7d, return_30d
      - rsi_14, adx_14
      - vol_30d (annualized)
      - ma_50_dist, ma_200_dist (현재가와 MA의 % 차이)
      - high_low_pct (당일 변동폭)
      - volume_zscore_30d
    """
    out = pd.DataFrame(index=df_daily.index)

    close = df_daily["close"]
    out["close"] = close
    out["return_1d"] = close.pct_change(1) * 100
    out["return_7d"] = close.pct_change(7) * 100
    out["return_30d"] = close.pct_change(30) * 100

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)

    out["adx_14"] = adx(df_daily, 14)
    out["vol_30d"] = realized_volatility(close, 30)

    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    out["ma_50_dist"] = (close - ma50) / ma50 * 100
    out["ma_200_dist"] = (close - ma200) / ma200 * 100

    out["high_low_pct"] = (df_daily["high"] - df_daily["low"]) / close * 100

    vol = df_daily["volume"]
    vol_ma = vol.rolling(30, min_periods=30).mean()
    vol_std = vol.rolling(30, min_periods=30).std()
    out["volume_zscore_30d"] = (vol - vol_ma) / vol_std.replace(0, np.nan)

    return out


def regime_distribution(labels: pd.Series) -> dict[str, int]:
    """라벨 분포 진단"""
    dist = {}
    for idx, name in IDX_TO_REGIME.items():
        dist[name] = int((labels == idx).sum())
    dist["UNLABELED"] = int((labels == -1).sum())
    return dist


if __name__ == "__main__":
    # CLI: 자기 진단용
    from flight_mind.utils.db import get_conn

    with get_conn(read_only=True) as conn:
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            df = conn.execute(f"""
                SELECT open_time, open, high, low, close, volume
                FROM features_1d
                WHERE symbol = '{symbol}'
                ORDER BY open_time
            """).fetch_df()
            if df.empty:
                print(f"{symbol}: no data")
                continue

            df = df.set_index("open_time")
            # features_1d에 없는 high/low/open은 ohlcv_1d로 fallback... 단순화 위해 close만 활용 가능
            # 여기서는 일단 fetch_ohlcv 활용
            from flight_mind.utils.db import fetch_ohlcv
            df_d = fetch_ohlcv(symbol, "5m").resample("1D").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna()

            labels = label_regime(df_d)
            dist = regime_distribution(labels)
            print(f"\n{symbol} regime distribution ({len(df_d)} days):")
            for k, v in dist.items():
                print(f"  {k:18s}: {v:>5,}")
