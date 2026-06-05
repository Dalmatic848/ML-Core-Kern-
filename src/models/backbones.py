"""
Универсальные backbone'ы для одиночного и двупотокового обучения.

Поддерживаемые архитектуры:
  resnet18      (512d)
  resnet50      (2048d)
  efficientnet_b3 (1536d) — хороший баланс качество/скорость
  efficientnet_b4 (1792d) — тяжелее
  convnext_tiny  (768d)  — современный сверточный
  swin_t         (768d)  — Swin Transformer (если веса скачаны)

Для двупотокового: DualStreamModel(backbone='resnet50', num_classes=6)
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# Экстракторы признаков (backbone без головы)
# Каждый возвращает (B, feat_dim) тензор
# ─────────────────────────────────────────────────────────────────────────────

FEAT_DIMS = {
    'resnet18':        512,
    'resnet50':        2048,
    'efficientnet_b3': 1536,
    'efficientnet_b4': 1792,
    'convnext_tiny':   768,
    'swin_t':          768,
}


def _make_extractor(arch: str) -> nn.Module:
    """Создаёт backbone без классификационной головы."""
    if arch == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        return m

    if arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        return m

    if arch == 'efficientnet_b3':
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        return _EfficientNetExtractor(m)

    if arch == 'efficientnet_b4':
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        return _EfficientNetExtractor(m)

    if arch == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        return _ConvNeXtExtractor(m)

    if arch == 'swin_t':
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Identity()
        return m

    raise ValueError(f'Неизвестная архитектура: {arch!r}. Доступны: {list(FEAT_DIMS)}')


class _EfficientNetExtractor(nn.Module):
    """EfficientNet без classifier: features → avgpool → flatten → (B, feat_dim)."""
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool  = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(self.features(x)).flatten(1)


class _ConvNeXtExtractor(nn.Module):
    """ConvNeXt без Linear: features → avgpool → LayerNorm → flatten → (B, feat_dim)."""
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool  = model.avgpool
        self.norm     = model.classifier[0]   # LayerNorm2d
        self.flatten  = model.classifier[1]   # Flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.norm(self.avgpool(self.features(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Одиночный классификатор (single-stream)
# ─────────────────────────────────────────────────────────────────────────────

class SingleStreamModel(nn.Module):
    """Любой backbone + dropout + Linear."""

    def __init__(self, arch: str, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        self.backbone  = _make_extractor(arch)
        feat_dim       = FEAT_DIMS[arch]
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def create_single(arch: str, num_classes: int,
                  freeze_mode: str = 'none', dropout_p: float = 0.5) -> SingleStreamModel:
    return SingleStreamModel(arch, num_classes, dropout_p)


# ─────────────────────────────────────────────────────────────────────────────
# Двупотоковый (dual-stream)
# ─────────────────────────────────────────────────────────────────────────────

class DualStreamModel(nn.Module):
    """Два независимых backbone (ДС + УФ) → конкатенация → classifier.

    head: (feat_dim*2) → bottleneck → num_classes
    """

    def __init__(self, arch: str, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        feat_dim           = FEAT_DIMS[arch]
        self.backbone_ds   = _make_extractor(arch)
        self.backbone_uv   = _make_extractor(arch)
        bottleneck         = min(feat_dim, 1024)
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim * 2, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(bottleneck, num_classes),
        )

    def forward(self, x_ds: torch.Tensor, x_uv: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.backbone_ds(x_ds), self.backbone_uv(x_uv)], dim=1)
        return self.head(feat)

    def load_from_single_checkpoints(
        self,
        ds_ckpt_path: Optional[Path] = None,
        uv_ckpt_path: Optional[Path] = None,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        for ckpt_path, backbone in [(ds_ckpt_path, self.backbone_ds),
                                     (uv_ckpt_path, self.backbone_uv)]:
            if ckpt_path and Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                # Отфильтровываем слои головы (разный размер у single vs dual)
                state = {k: v for k, v in state.items()
                         if not any(k.startswith(p) for p in ('fc.', 'head.', 'classifier.'))}
                missing, unexpected = backbone.load_state_dict(state, strict=False)
                print(f'  ← {Path(ckpt_path).name}  '
                      f'missing={len(missing)} unexpected={len(unexpected)}')


def create_dual(arch: str, num_classes: int,
                freeze_mode: str = 'none', dropout_p: float = 0.5) -> DualStreamModel:
    return DualStreamModel(arch, num_classes, dropout_p)
