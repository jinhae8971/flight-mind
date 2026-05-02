"""
Tier 2 PyTorch Dataset
=======================
DuckDB의 ohlcv 테이블에서 직접 윈도우를 슬라이딩하며
GAF 이미지 + 라벨을 생성.

설계 원칙:
  - Lazy Loading: DuckDB 쿼리는 __init__ 시점에 인덱스만 빌드, 실제 OHLCV는 __getitem__에서 fetch
    → 메모리 효율적 (5년 데이터 풀로딩 방지)
  - In-memory cache: 자주 접근하는 윈도우는 LRU 캐싱
  - Label Look-ahead Safe: 라벨 생성에 미래 12봉 사용하므로,
    학습/검증 분할 시 시간순으로 분할 필수 (look-ahead bias 방지)

라벨 정의:
  - long  (class 0): 미래 12봉 후 close가 +0.5% 이상 상승
  - short (class 1): 미래 12봉 후 close가 -0.5% 이하 하락
  - hold  (class 2): 그 사이 (작은 변동)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from flight_mind.config import TIER2
from flight_mind.tier2_pattern.encoder import GAFEncoder
from flight_mind.utils.db import fetch_ohlcv


LABEL_NAMES = ["long", "short", "hold"]
N_CLASSES = 3


class OhlcvWindowDataset(Dataset):
    """
    Sliding window dataset over DuckDB ohlcv table.

    Each item: (gaf_image [3, H, W], label [int])
    """

    def __init__(
        self,
        symbols: list[str],
        split: Literal["train", "val", "test"],
        window_size: int = TIER2.input_window,
        future_horizon: int = TIER2.future_horizon_bars,
        long_threshold: float = TIER2.long_threshold,
        short_threshold: float = TIER2.short_threshold,
        image_size: int | None = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        # test_ratio = 1 - train_ratio - val_ratio = 0.15
    ):
        self.symbols = symbols
        self.split = split
        self.window_size = window_size
        self.future_horizon = future_horizon
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

        # GAF는 정사각형 이미지 → image_size는 window_size와 일치해야 함
        # 명시적으로 다른 값을 넣으면 ValueError 처리
        effective_image_size = image_size if image_size is not None else window_size
        if effective_image_size != window_size:
            raise ValueError(
                f"GAF requires image_size == window_size. "
                f"Got image_size={effective_image_size}, window_size={window_size}"
            )

        self.encoder = GAFEncoder(image_size=effective_image_size)

        # Build (symbol, end_idx) index for each valid window
        self.index: list[tuple[str, int]] = []
        self._symbol_data: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            df = fetch_ohlcv(symbol, "5m")
            if df.empty:
                continue
            self._symbol_data[symbol] = df

            n = len(df)
            # Valid range: window_size <= end_idx < n - future_horizon
            min_end = window_size
            max_end = n - future_horizon

            # Time-based split (no shuffle to prevent look-ahead bias)
            train_end = int(min_end + (max_end - min_end) * train_ratio)
            val_end = int(min_end + (max_end - min_end) * (train_ratio + val_ratio))

            if split == "train":
                end_indices = range(min_end, train_end)
            elif split == "val":
                end_indices = range(train_end, val_end)
            else:  # test
                end_indices = range(val_end, max_end)

            for end_idx in end_indices:
                self.index.append((symbol, end_idx))

    def __len__(self) -> int:
        return len(self.index)

    def _make_label(self, df: pd.DataFrame, end_idx: int) -> int:
        """미래 future_horizon 봉 후 가격 변화로 라벨 결정."""
        current_close = float(df["close"].iloc[end_idx - 1])
        future_close = float(df["close"].iloc[end_idx - 1 + self.future_horizon])

        future_return = (future_close - current_close) / current_close

        if future_return >= self.long_threshold:
            return 0  # long
        if future_return <= self.short_threshold:
            return 1  # short
        return 2  # hold

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        symbol, end_idx = self.index[idx]
        df = self._symbol_data[symbol]

        # Window: [end_idx - window_size, end_idx)
        window = df.iloc[end_idx - self.window_size:end_idx].copy()

        # Encode to GAF
        gaf_img = self.encoder.encode(window)
        gaf_tensor = torch.from_numpy(gaf_img)  # (3, H, W) float32

        # Label
        label = self._make_label(df, end_idx)

        return gaf_tensor, label

    def class_distribution(self) -> dict[str, int]:
        """라벨 분포 — 클래스 불균형 진단용."""
        counts = {name: 0 for name in LABEL_NAMES}
        for symbol, end_idx in self.index:
            df = self._symbol_data[symbol]
            label = self._make_label(df, end_idx)
            counts[LABEL_NAMES[label]] += 1
        return counts


def make_dataloaders(
    symbols: list[str],
    batch_size: int = TIER2.batch_size,
    num_workers: int = 0,
):
    """학습/검증/테스트 DataLoader 한번에 생성."""
    from torch.utils.data import DataLoader

    train_ds = OhlcvWindowDataset(symbols, "train")
    val_ds = OhlcvWindowDataset(symbols, "val")
    test_ds = OhlcvWindowDataset(symbols, "test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "datasets": {"train": train_ds, "val": val_ds, "test": test_ds},
    }
