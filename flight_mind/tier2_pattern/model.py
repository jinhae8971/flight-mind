"""
Tier 2 Model — ResNet-18 + Classification Head
================================================
GAF 이미지를 입력받아 (long, short, hold) 3-class 확률 분포를 출력.

선택 근거:
  - ResNet-18은 11M 파라미터로 RTX 4090 24GB에서 batch 64 학습 시 VRAM 8GB 사용
  - GAF-CNN 학계 SOTA에서 ResNet 계열이 가장 안정적 결과 보고
  - timm 라이브러리로 ImageNet pretrained weights 활용 가능 (transfer learning)

학습 전략:
  - Phase 1: ImageNet pretrained backbone freeze, head만 학습 (빠른 수렴)
  - Phase 2: 전체 unfreeze, low LR로 fine-tuning
  - Class imbalance: WeightedCrossEntropyLoss (hold 클래스가 다수일 가능성)
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class FlightPatternCNN(nn.Module):
    """
    플라이트 캔들 패턴 학습 모델.

    Architecture:
      Input  : (B, 3, 64, 64) — RGB GAF
      Backbone: ResNet-18 (timm) → (B, 512)
      Head   : Dropout → Linear(512, 128) → ReLU → Dropout → Linear(128, 3)
      Output : (B, 3) logits
    """

    def __init__(
        self,
        n_classes: int = 3,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        backbone_dim = self._build_backbone(backbone, pretrained)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def _build_backbone(self, backbone: str, pretrained: bool) -> int:
        """3-tier fallback: timm → torchvision → 자체 구현 ResNet-18.

        Returns: backbone output feature dim
        """
        # Tier 1: timm (preferred — 가장 풍부한 모델 zoo)
        if HAS_TIMM:
            try:
                self.backbone = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
                return self.backbone.num_features
            except Exception:
                pass

        # Tier 2: torchvision
        try:
            from torchvision.models import resnet18
            net = resnet18(weights="DEFAULT" if pretrained else None)
            self.backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
            return 512
        except ImportError:
            pass

        # Tier 3: 자체 구현 (pretrained 없음, 학습 처음부터)
        self.backbone = self._build_minimal_resnet()
        return 512

    @staticmethod
    def _build_minimal_resnet() -> nn.Module:
        """torchvision/timm 없을 때 사용하는 최소 ResNet-18 구현."""
        class BasicBlock(nn.Module):
            def __init__(self, in_c, out_c, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_c)
                self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_c)
                self.shortcut = nn.Identity()
                if stride != 1 or in_c != out_c:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_c),
                    )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = self.shortcut(x)
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += identity
                return self.relu(out)

        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            BasicBlock(64, 64), BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2), BasicBlock(128, 128),
            BasicBlock(128, 256, stride=2), BasicBlock(256, 256),
            BasicBlock(256, 512, stride=2), BasicBlock(512, 512),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) GAF images

        Returns:
            (B, n_classes) logits
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self) -> None:
        """Phase 1: backbone freeze."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Phase 2: 전체 unfreeze."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
