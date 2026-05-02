"""
Tier 4 PyTorch Dataset
=======================
일봉 OHLCV에서 30일 윈도우를 슬라이딩하며
(features, symbol_idx, regime_label) 튜플 생성.

설계:
  - 일봉이라 sample 수가 적음 (5년 = 1825일/페어)
  - Look-ahead 안전: 시간순 split (train: ~70%, val: 15%, test: 15%)
  - Class weight: regime 분포가 매우 불균형이므로 학습 시 가중치 적용 필요
  - Multi-pair: BTC, ETH 모두 사용 → 페어 임베딩으로 모델이 구분 학습
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from flight_mind.config import EXCHANGE, TIER4
from flight_mind.tier4_regime.labeler import (build_regime_features,
                                                label_regime,
                                                regime_distribution)
from flight_mind.utils.db import fetch_ohlcv


# Symbol → index mapping
SYMBOL_TO_IDX = {pair.replace("/", ""): i for i, pair in enumerate(EXCHANGE.pairs)}


class RegimeDataset(Dataset):
    """30일 윈도우 → 시장 국면 분류"""

    FEATURE_COLS = [
        "return_1d", "return_7d", "return_30d",
        "rsi_14", "adx_14", "vol_30d",
        "ma_50_dist", "ma_200_dist",
        "high_low_pct", "volume_zscore_30d",
    ]   # 10 features

    def __init__(
        self,
        symbols: list[str],
        split: Literal["train", "val", "test"],
        seq_len: int = TIER4.lookback_days,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        self.split = split
        self.seq_len = seq_len

        # Build (symbol, end_idx) index
        self.index: list[tuple[str, int]] = []
        self._features: dict[str, pd.DataFrame] = {}
        self._labels: dict[str, pd.Series] = {}

        for symbol in symbols:
            # 5분봉 → 일봉 리샘플 (DuckDB에 저장된 features_1d는 정보 부족 — 직접 빌드)
            df_5m = fetch_ohlcv(symbol, "5m")
            if df_5m.empty or len(df_5m) < 200 * 288:  # 200일 미만이면 skip
                continue

            df_1d = df_5m.resample("1D").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last",
                "volume": "sum",
            }).dropna()

            features = build_regime_features(df_1d)
            labels = label_regime(df_1d)

            # 결측 처리
            features = features.ffill().fillna(0)

            # 유효 인덱스: features와 labels 모두 valid한 시점
            valid_mask = (labels >= 0) & features[self.FEATURE_COLS].notna().all(axis=1)
            valid_idx = np.where(valid_mask.values)[0]

            if len(valid_idx) < seq_len + 50:   # 최소 검증 데이터 보장
                continue

            self._features[symbol] = features
            self._labels[symbol] = labels

            # Sliding window: end_idx >= seq_len + 200 (warmup), end_idx valid
            min_end = seq_len + 200
            max_end = len(df_1d)
            valid_ends = [i for i in range(min_end, max_end) if labels.iloc[i - 1] >= 0]

            n = len(valid_ends)
            train_n = int(n * train_ratio)
            val_n = int(n * val_ratio)

            if split == "train":
                slice_ends = valid_ends[:train_n]
            elif split == "val":
                slice_ends = valid_ends[train_n:train_n + val_n]
            else:  # test
                slice_ends = valid_ends[train_n + val_n:]

            for end_idx in slice_ends:
                self.index.append((symbol, end_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        symbol, end_idx = self.index[idx]
        features = self._features[symbol]
        labels = self._labels[symbol]

        # Window: [end_idx - seq_len, end_idx)
        window = features.iloc[end_idx - self.seq_len:end_idx][self.FEATURE_COLS]
        x = torch.from_numpy(window.values.astype(np.float32))

        symbol_idx = SYMBOL_TO_IDX.get(symbol, 0)
        sym = torch.tensor(symbol_idx, dtype=torch.long)

        label = int(labels.iloc[end_idx - 1])
        return x, sym, label

    def class_distribution(self) -> dict[str, int]:
        from flight_mind.tier4_regime.labeler import IDX_TO_REGIME
        counts = {name: 0 for name in IDX_TO_REGIME.values()}
        for symbol, end_idx in self.index:
            label = int(self._labels[symbol].iloc[end_idx - 1])
            if label >= 0:
                counts[IDX_TO_REGIME[label]] += 1
        return counts


def make_dataloaders(
    symbols: list[str],
    batch_size: int = TIER4.batch_size,
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader

    train_ds = RegimeDataset(symbols, "train")
    val_ds = RegimeDataset(symbols, "val")
    test_ds = RegimeDataset(symbols, "test")

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
        "datasets": {"train": train_ds, "val": val_ds, "test": test_ds},
    }
