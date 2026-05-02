"""
Gramian Angular Field (GAF) Encoder
====================================
시계열 데이터를 2D 이미지로 변환하여 CNN이 학습 가능한 형태로 만든다.

학술적 근거:
  - Chen & Tsai (2020) "Encoding candlesticks as images for pattern classification
    using convolutional neural networks", Financial Innovation
  - 실제 시장 데이터에서 90.7% 정확도 달성 (LSTM 대비 우수)

구현 전략:
  - pyts 라이브러리의 GramianAngularField 사용 (검증된 구현)
  - OHLC 4개 채널 중 3개를 RGB로 매핑: (close, high-low spread, volume)
  - GASF (summation) 방식 사용 — GADF보다 양/음 변화에 더 민감

입력: pd.DataFrame [N rows, columns: open, high, low, close, volume]
출력: np.ndarray [3, image_size, image_size] — PyTorch (C, H, W) 형식
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyts.image import GramianAngularField


class GAFEncoder:
    """
    OHLCV 시계열을 RGB GAF 이미지로 변환.

    Channel 매핑:
      R: close 가격 (정규화)      — 추세 정보
      G: high-low spread (변동성) — 변동성 정보
      B: volume                   — 참여도 정보

    이렇게 하면 단순한 close GAF보다 더 풍부한 정보를 한 이미지에 압축 가능.
    """

    def __init__(self, image_size: int = 64, method: str = "summation"):
        """
        Args:
            image_size: 출력 이미지 크기 (NxN)
            method: 'summation' (GASF) | 'difference' (GADF)
        """
        self.image_size = image_size
        self.method = method

        # pyts API: image_size는 입력 길이와 같아야 하므로 None으로 두고 후처리
        # 실제로는 입력 시퀀스 길이를 image_size에 맞춰 사용
        self.gaf = GramianAngularField(
            image_size=image_size,
            method=method,
            sample_range=(-1, 1),
        )

    def _normalize_channel(self, x: np.ndarray) -> np.ndarray:
        """채널을 [-1, 1]로 정규화 (GAF가 cosine 변환을 쓰므로 필수)."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x

        # min-max scaling to [-1, 1]
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if x_max - x_min < 1e-12:
            return np.zeros_like(x)
        return 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    def _to_gaf(self, series: np.ndarray) -> np.ndarray:
        """단일 1D 시계열 → 2D GAF 이미지."""
        # pyts는 입력이 (n_samples, n_timestamps) 형태를 기대
        series = series.reshape(1, -1)
        try:
            img = self.gaf.fit_transform(series)
            return img[0]  # (image_size, image_size)
        except Exception:
            # 길이 불일치 등의 경우 zero fallback
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """
        OHLCV DataFrame을 (3, H, W) 텐서로 변환.

        Args:
            df: 60봉 OHLCV DataFrame (length must == image_size)

        Returns:
            np.ndarray of shape (3, image_size, image_size), dtype float32
        """
        if len(df) != self.image_size:
            raise ValueError(
                f"Input length {len(df)} != image_size {self.image_size}. "
                "Expected exact match for GAF encoding."
            )

        # Channel 1 (R): close 추세
        close = self._normalize_channel(df["close"].values)
        ch_r = self._to_gaf(close)

        # Channel 2 (G): high-low spread (변동성)
        spread = (df["high"] - df["low"]).values
        spread_norm = self._normalize_channel(spread)
        ch_g = self._to_gaf(spread_norm)

        # Channel 3 (B): volume (참여도)
        # volume은 log scale로 변환 후 정규화 (long-tail 분포 완화)
        log_vol = np.log1p(df["volume"].values)
        vol_norm = self._normalize_channel(log_vol)
        ch_b = self._to_gaf(vol_norm)

        # Stack: (3, H, W)
        rgb = np.stack([ch_r, ch_g, ch_b], axis=0).astype(np.float32)
        return rgb

    def encode_batch(self, dfs: list[pd.DataFrame]) -> np.ndarray:
        """배치 인코딩 — (B, 3, H, W) 반환."""
        return np.stack([self.encode(df) for df in dfs], axis=0)
