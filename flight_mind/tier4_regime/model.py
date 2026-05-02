"""
Tier 4 — Market Regime Transformer
====================================
30일치 다중 피처 시계열 → 5-class 시장 국면 예측.

Architecture:
  Input  : (B, seq_len=30, n_features=10)
         + (B,) symbol_idx     ← 페어 임베딩 (BTC, ETH)
  Embed  : Linear(n_features, d_model) + PositionalEncoding + SymbolEmbedding
  Encoder: 6 × TransformerEncoderLayer (d_model=256, n_heads=8, dim_feedforward=512)
  Head   : LayerNorm → Mean Pooling → Linear(d_model, 5)
  Output : (B, 5) logits

Why 6-layer Transformer?
  - 학계 SOTA: TFT, Autoformer 등은 6-layer 권장
  - 30일 시퀀스는 짧으므로 layer 수가 deep할 필요 없음
  - 약 5M params — RTX 3090/4090에서 batch=128 학습 시 VRAM 6GB
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from flight_mind.config import EXCHANGE


N_PAIRS = len(EXCHANGE.pairs)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Transformer 원본 논문)"""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class RegimeTransformer(nn.Module):
    """
    Multi-pair, multi-feature regime classifier.

    BTC와 ETH는 같은 시장 국면을 다르게 반영할 수 있으므로
    페어별 임베딩을 추가하여 모델이 페어 특성을 학습.
    """

    def __init__(
        self,
        n_features: int = 10,
        seq_len: int = 30,
        n_classes: int = 5,
        n_pairs: int = N_PAIRS,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Feature embedding: n_features → d_model
        self.feature_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)

        # Symbol embedding (BTC vs ETH)
        self.symbol_embed = nn.Embedding(n_pairs, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,             # (B, seq_len, n_features)
        symbol_idx: torch.Tensor,    # (B,) long tensor
    ) -> torch.Tensor:
        """
        Returns: (B, n_classes) logits
        """
        B, S, F = x.shape
        assert S == self.seq_len, f"seq_len={S} != expected {self.seq_len}"
        assert F == self.n_features, f"n_features={F} != expected {self.n_features}"

        # Feature projection
        h = self.feature_proj(x)                  # (B, S, d_model)
        h = self.pos_enc(h)                        # add positional

        # Symbol embedding을 시퀀스의 모든 timestep에 broadcast
        sym_emb = self.symbol_embed(symbol_idx).unsqueeze(1)  # (B, 1, d_model)
        h = h + sym_emb                            # (B, S, d_model)

        # Transformer encoding
        h = self.encoder(h)                        # (B, S, d_model)

        # Mean pooling across time
        h = self.norm(h)
        h = h.mean(dim=1)                          # (B, d_model)

        # Classification
        logits = self.head(h)                      # (B, n_classes)
        return logits

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_default_model() -> RegimeTransformer:
    """기본 설정으로 모델 생성 (config.TIER4 활용)"""
    from flight_mind.config import TIER4
    return RegimeTransformer(
        n_features=10,                  # build_regime_features의 column 수
        seq_len=TIER4.lookback_days,
        n_classes=5,
        d_model=TIER4.d_model,
        n_heads=TIER4.n_heads,
        n_layers=TIER4.n_layers,
    )
